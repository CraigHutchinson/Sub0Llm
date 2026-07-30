// tokenizer.cpp — implementation of the reusable BPE tokenizer (sub0::tok).
//
// Single source of truth for the truecasing + BPE vocabulary: the corpus scan
// (passes 1-2), the BPE merge learning, and the encode/detokenize/serialize the
// engine and the configurator both build on. Free of any dependency on the
// generated config so the configurator can use it.

#include "sub0/tokenizer.hpp"
#include "sub0/unigram.hpp"

#include <algorithm>
#include <cassert>
#include <istream>
#include <limits>
#include <ostream>
#include <queue>
#include <sstream>
#include <unordered_set>

namespace sub0::tok {

using namespace sub0::casing;

namespace {

// Serialize a symbol sequence to a string key for the unique-word map. Symbols are
// byte values (0..255), so one byte per symbol is a lossless, compact key — short
// enough to stay in std::string's small buffer (no heap) for typical words.
std::string seq_key(const std::vector<int>& s) {
    std::string k(s.size(), '\0');
    for (std::size_t i = 0; i < s.size(); ++i) k[i] = static_cast<char>(s[i] & 0xFF);
    return k;
}

// Unigram word encoder: Viterbi-segment the byte run stream[lo,hi) into the min Σ −log p piece-id
// sequence (piece_index / piece_logp). Every base byte is a candidate, so dp[L] is always reachable;
// the explicit fallback keeps it total even if a byte were somehow absent. The runtime tokenizer's
// only word encoder (see tokenizer.hpp's Tokenizer/learn() doc comments).
void viterbi_encode_word(const Tokenizer& t, std::span<const int> stream, std::size_t lo, std::size_t hi,
                         std::vector<int>& out) {
    const int    L   = static_cast<int>(hi - lo);
    const double INF = std::numeric_limits<double>::infinity();
    std::vector<double> dp(static_cast<std::size_t>(L) + 1, INF);
    std::vector<int>    bi(static_cast<std::size_t>(L) + 1, -1), bid(static_cast<std::size_t>(L) + 1, -1);
    dp[0] = 0.0;
    std::string key;
    for (int a = 0; a < L; ++a) {
        if (dp[static_cast<std::size_t>(a)] == INF) continue;
        const int lmax = std::min(t.max_piece, L - a);
        key.resize(0);
        for (int l = 1; l <= lmax; ++l) {
            key.push_back(static_cast<char>(stream[lo + static_cast<std::size_t>(a + l - 1)] & 0xFF));
            const auto it = t.piece_index.find(key);
            if (it == t.piece_index.end()) continue;
            const double cost = dp[static_cast<std::size_t>(a)] - static_cast<double>(t.piece_logp[static_cast<std::size_t>(it->second)]);
            if (cost < dp[static_cast<std::size_t>(a + l)]) {
                dp[static_cast<std::size_t>(a + l)]  = cost;
                bi[static_cast<std::size_t>(a + l)]  = a;
                bid[static_cast<std::size_t>(a + l)] = it->second;
            }
        }
    }
    if (dp[static_cast<std::size_t>(L)] == INF) {                       // unreachable -> raw bytes
        for (int a = 0; a < L; ++a) out.push_back(stream[lo + static_cast<std::size_t>(a)] & 0xFF);  // base id == byte value
        return;
    }
    std::vector<int> rev;
    for (int j = L; j > 0;) { rev.push_back(bid[static_cast<std::size_t>(j)]); j = bi[static_cast<std::size_t>(j)]; }
    for (auto r = rev.rbegin(); r != rev.rend(); ++r) out.push_back(*r);
}

}  // namespace

// ============================================================================
//  Scan — passes 1 and 2 over an in-memory chunk
// ============================================================================

void Scan::add_names(std::string_view chunk) {
    raw_bytes += chunk.size();
    long qr = 0;
    static thread_local std::string norm;             // reused per worker thread -- no per-chunk temp/alloc
    normalize_text(chunk, qr, norm);
    quote_repl += qr;
    norm_bytes += norm.size();

    // A capital following a lowercase word mid-sentence ("...to Spot") is a name use.
    auto preceded_by_lowercase = [&](std::size_t start) {
        std::size_t k = start;
        while (k > 0 && (norm[k - 1] == ' ' || norm[k - 1] == '\t')) --k;
        return k > 0 && is_lower(static_cast<unsigned char>(norm[k - 1]));
    };
    for (std::size_t i = 0, n = norm.size(); i < n;) {
        const unsigned char c = static_cast<unsigned char>(norm[i]);
        if (!is_alpha(c)) { ++i; continue; }
        std::size_t j = i;
        bool all_lower = true, rest_lower = true;
        while (j < n && is_alpha(static_cast<unsigned char>(norm[j]))) {
            const unsigned char ch = static_cast<unsigned char>(norm[j]);
            if (!is_lower(ch)) all_lower = false;
            if (j > i && !is_lower(ch)) rest_lower = false;
            ++j;
        }
        const std::string w = norm.substr(i, j - i);
        if (all_lower) {
            lower_count[w] += 1;
        } else if (is_upper(static_cast<unsigned char>(w[0])) && rest_lower) {  // "Spot", "The"
            if (preceded_by_lowercase(i)) {
                std::string lw = w;
                lw[0] = static_cast<char>(to_lower(static_cast<unsigned char>(lw[0])));
                midcap_count[lw] += 1;
            }
        }
        i = j;
    }
}

void Scan::merge_names(Scan& other) {
    raw_bytes  += other.raw_bytes;
    norm_bytes += other.norm_bytes;
    quote_repl += other.quote_repl;
    for (const auto& [w, c] : other.lower_count)  lower_count[w]  += c;   // sums commute -> order-free
    for (const auto& [w, c] : other.midcap_count) midcap_count[w] += c;
    other.lower_count = {};
    other.midcap_count = {};
    other.raw_bytes = other.norm_bytes = 0;
    other.quote_repl = 0;
}

void Scan::add_words(std::string_view chunk, const std::unordered_set<std::string>& attested) {
    long qr = 0;
    static thread_local std::string      norm;        // reused per worker thread (string_view in, no temp)
    static thread_local std::vector<int> stream;      // reused id stream -- no per-chunk 4B/symbol alloc
    normalize_text(chunk, qr, norm);
    modality::add_modality(modality, norm);   // D2: per-byte spacing modality, over the SAME normalized view
    truecase_tokenize(norm, attested, &st, stream);
    for (std::size_t i = 0, n = stream.size(); i < n;) {
        const std::size_t end = word_unit_end(stream, i);
        if (end == i) {                                   // standalone symbol
            const int s = stream[i];
            if (s == TOK_CAP)      { used_cap = true; ++i; continue; }
            if (s == TOK_UP)       { used_up = true; ++i; continue; }
            if (is_symbol_piece_byte(s)) {                // collect a plain-symbol RUN as a learnable unit
                std::size_t se = i + 1;
                while (se < n && is_symbol_piece_byte(stream[se])) ++se;
                if (se - i >= 2) {                        // a run the Unigram can mint as a symbol piece
                    std::string skey(se - i, '\0');
                    for (std::size_t k = i; k < se; ++k) skey[k - i] = static_cast<char>(stream[k] & 0xFF);
                    const auto sit = index.find(skey);
                    if (sit == index.end()) {
                        std::vector<int> seq(stream.begin() + static_cast<std::ptrdiff_t>(i),
                                             stream.begin() + static_cast<std::ptrdiff_t>(se));
                        for (int bb : seq) byte_used[static_cast<std::size_t>(bb)] = 1;
                        index.emplace(std::move(skey), static_cast<int>(word_syms.size()));
                        word_syms.push_back(std::move(seq));
                        word_freq.push_back(1);
                    } else {
                        word_freq[static_cast<std::size_t>(sit->second)] += 1;
                    }
                    i = se;
                    continue;
                }
            }
            byte_used[static_cast<std::size_t>(s)] = 1;
            ++i;
            continue;
        }
        // Build the lookup key from the byte run; only allocate the word's symbol
        // vector / mark its bytes on a cache MISS.
        std::string key(end - i, '\0');
        for (std::size_t k = i; k < end; ++k) key[k - i] = static_cast<char>(stream[k] & 0xFF);
        const auto it = index.find(key);
        if (it == index.end()) {
            std::vector<int> seq(stream.begin() + static_cast<std::ptrdiff_t>(i),
                                 stream.begin() + static_cast<std::ptrdiff_t>(end));
            for (int bb : seq) byte_used[static_cast<std::size_t>(bb)] = 1;
            index.emplace(std::move(key), static_cast<int>(word_syms.size()));
            word_syms.push_back(std::move(seq));
            word_freq.push_back(1);
        } else {
            word_freq[static_cast<std::size_t>(it->second)] += 1;
        }
        i = end;
    }
}

void Scan::merge_words(Scan& other) {
    for (int bb = 0; bb < 256; ++bb) if (other.byte_used[static_cast<std::size_t>(bb)]) byte_used[static_cast<std::size_t>(bb)] = 1;
    used_cap = used_cap || other.used_cap;
    used_up  = used_up  || other.used_up;
    st.words += other.st.words; st.cap += other.st.cap; st.up += other.st.up; st.names += other.st.names;
    modality::merge(modality, other.modality);   // D2: fold the per-byte modality (order-free)
    for (std::size_t i = 0; i < other.word_syms.size(); ++i) {
        const std::string k = seq_key(other.word_syms[i]);
        const auto it = index.find(k);
        if (it == index.end()) {
            index.emplace(k, static_cast<int>(word_syms.size()));
            word_syms.push_back(std::move(other.word_syms[i]));
            word_freq.push_back(other.word_freq[i]);
        } else {
            word_freq[static_cast<std::size_t>(it->second)] += other.word_freq[i];
        }
    }
    other.index = {};
    other.word_syms = {};
    other.word_freq = {};
}

void Scan::rebuild_index() {
    index.clear();
    index.reserve(word_syms.size());
    for (std::size_t i = 0; i < word_syms.size(); ++i)
        index.emplace(seq_key(word_syms[i]), static_cast<int>(i));
}

std::unordered_set<std::string> derive_attested(const Scan& s, long long* withheld) {
    // A lowercase form is attested unless its mid-sentence-capital ("name") uses
    // dominate its lowercase uses.
    std::unordered_set<std::string> attested;
    long long n_withheld = 0;
    for (const auto& [w, lc] : s.lower_count) {
        const auto it = s.midcap_count.find(w);
        const long long mid = (it == s.midcap_count.end()) ? 0 : it->second;
        if (mid > lc) { ++n_withheld; continue; }
        attested.insert(w);
    }
    if (withheld) *withheld = n_withheld;
    return attested;
}

// ============================================================================
//  learn — base alphabet + Unigram LM word-piece vocabulary (the runtime tokenizer)
// ============================================================================

// The hardcoded per-byte glue FLOOR (casing::glue_default): a prose-derived default that every
// tokenizer starts from and a corpus can only refine.
std::array<casing::GlueDefault, 256> default_glue_table() {
    std::array<casing::GlueDefault, 256> g{};
    for (int b = 0; b < 256; ++b) g[static_cast<std::size_t>(b)] = casing::glue_default(b);
    return g;
}

// v2 (schemeV4, D2): corpus-derived per-byte glue. The hardcoded table is the floor; a byte with
// DECISIVE, unimodal modality evidence (>= 500 samples, second combo < 25%) is overridden to its
// dominant (glued-before -> lead, glued-after -> trail). So a code corpus makes `=`/`.`/`/` glue
// where prose leaves them spaced, while bimodal or sparse bytes keep the floor (their ambiguity is
// for markers/JOIN, not a single default). encode + decode both read the baked result -- lossless
// for any table, since a deviation still pays a JOIN or a literal space.
std::array<casing::GlueDefault, 256> derive_glue(const modality::ModalityStats& ms) {
    std::array<casing::GlueDefault, 256> g = default_glue_table();
    for (const auto& [cp, cm] : ms.chars) {
        if (cp >= 256 || cm.total() < 500 || cm.bimodal()) continue;
        const int d = cm.dominant();                      // Combo bit1=glued-before, bit0=glued-after
        g[static_cast<std::size_t>(cp)] = { (d & 2) != 0, (d & 1) != 0 };
    }
    return g;
}

Tokenizer learn(Scan& scan, const std::unordered_set<std::string>& attested,
                const LearnOptions& opts) {
    Tokenizer t;
    t.glue = derive_glue(scan.modality);   // D2: corpus-adaptive per-byte spacing (hardcoded floor + evidence)

    // Fix the base alphabet.
    //
    // Every id in this block is a SCHEME constant, not something discovered from the corpus: base
    // id == symbol code for the whole fixed alphabet -- the 256 raw bytes (no offset) followed by
    // the markers in exactly TokenId's declared order (TOK_EOS..TOK_MARKER_COUNT-1), which is why
    // this is just two counting loops rather than one line per marker: base_symbol[i] == i holds
    // uniformly across both ranges (verified the same way on load, see deserialize()).
    for (int b = 0; b < 256; ++b) t.base_symbol.push_back(b);
    for (int m = TOK_EOS; m < TOK_MARKER_COUNT; ++m) t.base_symbol.push_back(m);
    t.n_base = static_cast<int>(t.base_symbol.size());
    assert(t.n_base == TOK_MARKER_COUNT && "base alphabet layout drifted from TokenId's declared markers");

    t.expansion.resize(static_cast<std::size_t>(t.n_base));
    for (int id = 0; id < t.n_base; ++id)
        t.expansion[static_cast<std::size_t>(id)] = {t.base_symbol[static_cast<std::size_t>(id)]};

    // Unigram LM word vocabulary. Learn the pieces top-down from the RAW word table (no remap),
    // then map them onto the base alphabet: a single-byte piece reuses its base id, a multi-byte
    // piece becomes a new id. Word encoding is then Viterbi over piece_index/piece_logp (set up
    // here) -- globally occurrence-optimal, no dead slots (unlike greedy BPE, see
    // learn_bpe_analysis() below, which this runtime path no longer uses).
    std::vector<std::pair<std::string, long long>> words;
    words.reserve(scan.word_syms.size());
    for (std::size_t w = 0; w < scan.word_syms.size(); ++w) {
        std::string s;
        s.reserve(scan.word_syms[w].size());
        for (int b : scan.word_syms[w]) s.push_back(static_cast<char>(b & 0xFF));
        words.push_back({std::move(s), scan.word_freq[w]});
    }
    UnigramOptions uo;
    uo.target          = opts.vocab_target;
    uo.min_count       = opts.min_merge;
    uo.min_word_freq   = opts.min_word_freq;      // learn-set reduction (huge corpora)
    uo.max_learn_words = opts.max_learn_words;
    uo.verbose         = opts.verbose;
    const Unigram u = learn_unigram(words, uo);

    double min_lp = 0.0;
    for (double lp : u.logp) min_lp = std::min(min_lp, lp);
    const float floor_lp = static_cast<float>(min_lp - 5.0);    // unseen bytes: usable but a last resort
    t.piece_logp.assign(static_cast<std::size_t>(t.n_base), floor_lp);   // markers stay at floor (not in index)
    for (int id = 0; id < u.size(); ++id) {
        const std::string& s = u.token[static_cast<std::size_t>(id)];
        const float lp = static_cast<float>(u.logp[static_cast<std::size_t>(id)]);
        if (s.size() == 1) {
            const int bid = static_cast<unsigned char>(s[0]);   // base id == byte value
            t.piece_logp[static_cast<std::size_t>(bid)] = lp;
            t.piece_index[s] = bid;
        } else {
            const int pid = static_cast<int>(t.expansion.size());
            std::vector<int> codes;
            codes.reserve(s.size());
            for (char c : s) codes.push_back(static_cast<unsigned char>(c));
            t.expansion.push_back(std::move(codes));
            t.piece_logp.push_back(lp);
            t.piece_index.emplace(s, pid);
        }
    }
    // Every base byte must be a Viterbi candidate so any input segments, including a byte the
    // corpus never showed (floor-scored single token).
    for (int b = 0; b < 256; ++b) t.piece_index.emplace(std::string(1, static_cast<char>(b)), b);
    t.max_piece = std::max(1, u.max_len);
    t.vocab     = static_cast<int>(t.expansion.size());
    // Remap the scan word table to the Viterbi piece-id segmentation, so the configurator emits /
    // reports read the final tokenized ids directly.
    for (std::vector<int>& w : scan.word_syms) {
        std::vector<int> ids;
        viterbi_encode_word(t, w, 0, w.size(), ids);
        w = std::move(ids);
    }
    t.attested = attested;
    t.loaded   = true;
    return t;
}

Tokenizer learn(std::string_view corpus, const LearnOptions& opts) {
    Scan s;
    s.add_names(corpus);
    const std::unordered_set<std::string> attested = derive_attested(s);
    s.add_words(corpus, attested);
    return learn(s, attested, opts);
}

// ============================================================================
//  learn_bpe_analysis — offline greedy-merge BPE (--dump-vocab A/B + curve only)
// ============================================================================

BpeAnalysisVocab learn_bpe_analysis(Scan& scan, int vocab_target, int min_merge) {
    BpeAnalysisVocab t;

    // Same fixed base alphabet as learn() above (see its comment) -- duplicated, not shared, since
    // Tokenizer and BpeAnalysisVocab are different types with no common base; this is the one
    // remaining place BPE tie-breaking needs base id == symbol code to be deterministic.
    for (int b = 0; b < 256; ++b) t.base_symbol.push_back(b);
    for (int m = TOK_EOS; m < TOK_MARKER_COUNT; ++m) t.base_symbol.push_back(m);
    t.n_base = static_cast<int>(t.base_symbol.size());
    assert(t.n_base == TOK_MARKER_COUNT && "base alphabet layout drifted from TokenId's declared markers");
    t.expansion.resize(static_cast<std::size_t>(t.n_base));
    for (int id = 0; id < t.n_base; ++id)
        t.expansion[static_cast<std::size_t>(id)] = {t.base_symbol[static_cast<std::size_t>(id)]};

    // Greedy merge over the word table (already base ids -- base id == byte value, so
    // scan.word_syms's raw bytes need no remap).
    int vocab = t.n_base;

    std::vector<std::vector<int>>& word_syms = scan.word_syms;
    const std::vector<long long>&  word_freq = scan.word_freq;

    std::unordered_map<std::pair<int, int>, long long, PairHash> pc;     // pair -> count
    std::unordered_map<std::pair<int, int>, std::vector<int>, PairHash> pair_words;  // pair -> word ids
    // Heap top = MAX count, then SMALLEST pair. Lazy deletion: an entry is valid only
    // while its count still equals pc[pair]; stale ones are skipped on pop.
    struct HeapItem { long long count; std::pair<int, int> pr; };
    auto worse = [](const HeapItem& a, const HeapItem& b) {
        return a.count != b.count ? a.count < b.count : a.pr > b.pr;
    };
    std::priority_queue<HeapItem, std::vector<HeapItem>, decltype(worse)> heap(worse);

    for (std::size_t w = 0; w < word_syms.size(); ++w) {
        const std::vector<int>& s = word_syms[w];
        const long long f = word_freq[w];
        for (std::size_t k = 0; k + 1 < s.size(); ++k) {
            const std::pair<int, int> p{s[k], s[k + 1]};
            pc[p] += f;
            pair_words[p].push_back(static_cast<int>(w));
        }
    }
    for (const auto& [p, c] : pc) heap.push({c, p});

    std::unordered_set<std::pair<int, int>, PairHash> touched;
    std::unordered_map<std::pair<int, int>, int, PairHash> wd;

    while (vocab < vocab_target) {
        std::pair<int, int> best{0, 0}; long long best_c = -1;
        while (!heap.empty()) {
            const HeapItem top = heap.top();
            const auto it = pc.find(top.pr);
            if (it != pc.end() && it->second == top.count && top.count > 0) { best = top.pr; best_c = top.count; break; }
            heap.pop();
        }
        if (best_c < min_merge) break;

        const int new_id = vocab++;
        t.merges.push_back(best);
        t.merge_count.push_back(best_c);   // tokens this merge removes from the corpus (vocab-curve benefit)
        std::vector<int> exp = t.expansion[static_cast<std::size_t>(best.first)];
        exp.insert(exp.end(), t.expansion[static_cast<std::size_t>(best.second)].begin(),
                   t.expansion[static_cast<std::size_t>(best.second)].end());
        t.expansion.push_back(std::move(exp));

        std::vector<int> wl;
        if (auto pit = pair_words.find(best); pit != pair_words.end()) wl = std::move(pit->second);
        pair_words.erase(best);
        std::sort(wl.begin(), wl.end());
        wl.erase(std::unique(wl.begin(), wl.end()), wl.end());
        touched.clear();
        for (const int w : wl) {
            std::vector<int>& s = word_syms[static_cast<std::size_t>(w)];
            std::vector<int> ns; ns.reserve(s.size());
            bool changed = false;
            for (std::size_t k = 0; k < s.size();) {
                if (k + 1 < s.size() && s[k] == best.first && s[k + 1] == best.second) { ns.push_back(new_id); k += 2; changed = true; }
                else { ns.push_back(s[k]); ++k; }
            }
            if (!changed) continue;                          // stale index entry
            const long long f = word_freq[static_cast<std::size_t>(w)];
            wd.clear();
            for (std::size_t k = 0; k + 1 < s.size();  ++k) wd[{s[k], s[k + 1]}]   += 1;
            for (std::size_t k = 0; k + 1 < ns.size(); ++k) wd[{ns[k], ns[k + 1]}] -= 1;
            for (const auto& [p, d] : wd) {
                if (d == 0) continue;
                pc[p] -= static_cast<long long>(d) * f;
                touched.insert(p);
                if (d < 0) pair_words[p].push_back(w);
            }
            s.swap(ns);
        }
        for (const auto& p : touched) {
            const auto it = pc.find(p);
            if (it != pc.end() && it->second > 0) heap.push({it->second, p});
        }
        pc.erase(best);
    }

    t.vocab = vocab;
    return t;
}

// ============================================================================
//  encode / detokenize
// ============================================================================

namespace {

// True iff `s` is a whitespace BYTE (not a marker). The JOIN encoder classifies the
// whitespace run between content tokens; the decoder emits such bytes verbatim.
inline bool is_ws_byte(int s) { return s == ' ' || s == '\t' || s == '\n' || s == '\r'; }

// Pending re-casing the decoder applies to the next content, set by a CAP/UP marker.
// CapFirst capitalises the first alpha letter then clears; UpWord upper-cases the alpha
// run and clears at the first non-alpha or the word's end.
enum class Recase { None, CapFirst, UpWord };

// Literal-matched structural markers: text the encoder recognizes whole, never case/BPE-processed.
// NOT every marker is here -- only the ones identified by an exact text match (EOS, the turn
// boundaries); JOIN/NEWLINE/quotes/SPELL/SPACE*/TAB* are derived from spacing CONTEXT instead (see
// encode_join's tile_ws/quote-handling), so they don't belong in this table.
struct LiteralMarker { std::string_view text; int id; };
constexpr std::array<LiteralMarker, 3> kLiteralMarkers = {{
    {"<|endoftext|>", TOK_EOS}, {"<|im_start|>", TOK_TURN_START}, {"<|im_end|>", TOK_TURN_END},
}};

// WS5b bracket-glue: byte -> marker id, one row per bracket, keyed by the literal byte value (NOT a
// linear-scan table like kLiteralMarkers -- brackets are single bytes, not multi-byte literals, so a
// direct switch/lookup is both simpler and cheaper). Angle brackets `<`/`>` are deliberately excluded
// (too overloaded -- comparisons/generics/HTML, see docs/TOKENIZER_REVIEW.md §5.3).
constexpr int glue_marker_for(int byte) {
    switch (byte) {
        case '(': return TOK_GLUE_OPAREN;   case ')': return TOK_GLUE_CPAREN;
        case '[': return TOK_GLUE_OBRACKET; case ']': return TOK_GLUE_CBRACKET;
        case '{': return TOK_GLUE_OBRACE;   case '}': return TOK_GLUE_CBRACE;
        default:  return -1;
    }
}

// Single source of truth for what each JOIN-scheme marker DOES: the exact bytes it decodes to and its
// effect on the decoder's three state flags. detokenize_join walks this table instead of a per-marker
// switch, and encode_join derives each marker's pending-space effect from the SAME row (emit_marker),
// so the two halves of the round-trip can no longer silently disagree about a marker -- the exact bug
// class the fuzz net exists to catch (e.g. the WS5b open/close `dps` inversion, §5.9). Adding a special
// becomes one row here plus its encode-side selection, not mirrored edits in two switch statements.
// See docs/TOKENIZER_REVIEW.md §2.
//
// Indexed by `id - TOK_EOS` over the whole marker range [TOK_EOS, TOK_MARKER_COUNT). Reserved-headroom
// ids (no assigned meaning yet) stay default-constructed = INERT: no bytes, no state change -- exactly
// the old reserved-headroom no-op, so a model that samples one can't emit a wrapped-around byte.
// State sentinels: dps / in_spell are -1 = "leave unchanged", else 0/1 to set; recase is -1 =
// unchanged, else a Recase value. `lead_space` emits one pending space before the literal iff `dps`.
struct MarkerSpec {
    std::string_view literal;              // bytes emitted on decode ("" for pure state markers / inert)
    bool             lead_space = false;   // emit ' ' before `literal` if a space is pending
    signed char      dps        = -1;      // pending-space AFTER (-1 keep, 0 false, 1 true)
    signed char      recase     = -1;      // -1 keep, else static_cast<Recase>(recase)
    signed char      in_spell   = -1;      // spell-group state AFTER (-1 keep, 0 false, 1 true)
};

constexpr auto kMarkerSpecs = [] {
    constexpr signed char keep = -1, off = 0, on = 1;
    constexpr signed char none = static_cast<signed char>(Recase::None);
    constexpr signed char capf = static_cast<signed char>(Recase::CapFirst);
    constexpr signed char upw  = static_cast<signed char>(Recase::UpWord);
    std::array<MarkerSpec, static_cast<std::size_t>(TOK_MARKER_COUNT - TOK_EOS)> a{};
    auto row = [&](int id, MarkerSpec s) { a[static_cast<std::size_t>(id - TOK_EOS)] = s; };
    //          marker            literal          lead   dps   recase in_spell
    row(TOK_JOIN,         {"",              false, off,  keep, keep});   // cancel a pending space (glue)
    row(TOK_NEWLINE,      {"\n",            false, off,  none, keep});
    row(TOK_PARA,         {"\n\n",          false, off,  none, keep});
    row(TOK_EOS,          {"<|endoftext|>", true,  off,  none, keep});
    row(TOK_TURN_START,   {"<|im_start|>",  true,  off,  none, keep});
    row(TOK_TURN_END,     {"<|im_end|>",    true,  off,  none, keep});
    row(TOK_SPACE2,       {"  ",            false, off,  none, keep});
    row(TOK_SPACE4,       {"    ",          false, off,  none, keep});
    row(TOK_TAB2,         {"\t\t",          false, off,  none, keep});
    row(TOK_TAB4,         {"\t\t\t\t",      false, off,  none, keep});
    row(TOK_CAP,          {"",              false, keep, capf, keep});   // recase only; no bytes, dps untouched
    row(TOK_UP,           {"",              false, keep, upw,  keep});
    row(TOK_ODQUOTE,      {"\"",            true,  off,  none, keep});
    row(TOK_CDQUOTE,      {"\"",            false, on,   none, keep});
    row(TOK_SPELL_START,  {"",              true,  off,  keep, on});     // recase carries INTO the group
    row(TOK_SPELL_END,    {"",              false, on,   none, off});
    row(TOK_GLUE_OPAREN,  {"(",             false, off,  keep, keep});   // recase deliberately untouched (§5.9)
    row(TOK_GLUE_CPAREN,  {")",             false, on,   keep, keep});
    row(TOK_GLUE_OBRACKET,{"[",             false, off,  keep, keep});
    row(TOK_GLUE_CBRACKET,{"]",             false, on,   keep, keep});
    row(TOK_GLUE_OBRACE,  {"{",             false, off,  keep, keep});
    row(TOK_GLUE_CBRACE,  {"}",             false, on,   keep, keep});
    return a;
}();

// JOIN-scheme encode. `stream` is the truecased byte+marker stream. The encoder mirrors the
// decoder's pending-space state `dps` so it emits exactly the tokens that reconstruct the text:
//  - a single inter-content space is implicit (no token); JOIN cancels a pending space (glue);
//  - a lone '\n' -> NEWLINE, "\n\n" -> PARA, any other whitespace -> verbatim byte tokens;
//  - a double quote with ` "x` spacing -> OPEN_DQUOTE, `x" ` -> CLOSE_DQUOTE (bundles the space);
//  - a word of N word-piece sub-tokens: N=1 bare, N>=2 SPELL_START sub.. SPELL_END (schemeV3+ -- a
//    bare JOIN between exactly two ids used to mark a 2-piece word, but that shape is indistinguishable
//    from an ordinary word glued to trailing punctuation via the SAME general-glue JOIN below; measured
//    93.9% of real occurrences were that false positive, not a genuine split -- see docs/TOKENIZER_DESIGN.md).
// See docs/TOKENIZER_DESIGN.md §2-§5.
void encode_join(const Tokenizer& t, std::span<const int> stream, std::vector<int>& out) {
    const std::size_t n = stream.size();
    bool dps = false;                 // decoder's pending_space after the last emitted token
    // Word-encode cache (gigatoken's central lever): a 32 MB corpus chunk (configurator Pass 3)
    // repeats its common words thousands of times, and per-word Viterbi is by far this pass's
    // dominant per-occurrence cost -- so memoise each word's id sequence by its raw byte key and
    // collapse the Zipf head to a hash lookup. Keying on word bytes ALONE is correct: a word unit
    // carries no case/spacing markers (those are handled outside this branch) and viterbi_encode_word
    // is a pure function of (t, bytes). Both the cache and `word_key` scratch are call-local, so they
    // never outlive the one tokenizer `t` and add no cross-call state (the runtime single-prompt path
    // pays only one empty-map construction). `word_key` is reused across words (no per-word alloc).
    std::unordered_map<std::string, std::vector<int>> word_cache;
    // Pre-size the cache so a big chunk fills it without the ~log2(unique) incremental rehashes, each
    // of which re-inserts every element so far. On a DIVERSE 4 MB chunk (~20k unique words) this is a
    // measured ~12% off the full encode; n/32 sits comfortably above the unique-word count for real
    // corpora (Heaps' law keeps uniques well below n/32), and the one over-sized bucket array is a
    // cheap zeroed alloc next to the encode itself. Guarded so the runtime single-prompt path (tiny n)
    // pays nothing. See benchmarks/frontend_bench.cpp's diverse-corpus benchmark.
    if (n > 1024) word_cache.reserve(n / 32);
    std::string word_key;
    auto emit_byte = [&](int b) { out.push_back(b); };  // base id == byte value
    // Emit a marker and apply its pending-space effect from the SAME kMarkerSpecs row the decoder
    // walks -- so the encoder can never assume a `dps` outcome the decoder won't reproduce (e.g. the
    // open/close bracket direction, which used to be a hand-written is_close_bracket() on this side).
    auto emit_marker = [&](int id) {
        out.push_back(id);
        if (const signed char d = kMarkerSpecs[static_cast<std::size_t>(id - TOK_EOS)].dps; d >= 0) dps = (d != 0);
    };
    // Realize the whitespace run [lo,hi) (mirrors the decoder). A single inter-word space is
    // implicit (free) under a pending space; an empty inter-content gap glues with JOIN; any
    // other run is tiled into NEWLINE/PARA + run-length SPACE2/4 / TAB2/4 tokens, with a verbatim
    // byte for an odd remainder ('\r', a lone space/tab, ...). `inter_content` is false for the
    // trailing run (no following content), where a single space must stay literal, not implicit.
    auto tile_ws = [&](std::size_t lo, std::size_t hi, bool inter_content, bool target_lead_glue = false) {
        const std::size_t gap = hi - lo;
        if (gap == 0) {
            if (target_lead_glue) return;                           // v2: glue is this byte's default -- no JOIN
            if (dps) { out.push_back(TOK_JOIN); dps = false; }       // glue: cancel the pending space
            return;
        }
        if (inter_content && gap == 1 && stream[lo] == ' ' && dps) {
            if (target_lead_glue) { emit_byte(' '); dps = false; }   // v2 deviation: force the space the default omits
            return;                                                  // else: implicit single space (free)
        }
        for (std::size_t k = lo; k < hi;) {
            const int c = stream[k];
            if (c == '\n' && k + 1 < hi && stream[k + 1] == '\n') { out.push_back(TOK_PARA);    k += 2; }
            else if (c == '\n')                                   { out.push_back(TOK_NEWLINE); k += 1; }
            else if (c == ' ') {
                std::size_t run = 0; while (k + run < hi && stream[k + run] == ' ')  ++run;
                for (; run >= 4; run -= 4, k += 4) out.push_back(TOK_SPACE4);
                for (; run >= 2; run -= 2, k += 2) out.push_back(TOK_SPACE2);
                if (run == 1) { emit_byte(' '); ++k; }
            } else if (c == '\t') {
                std::size_t run = 0; while (k + run < hi && stream[k + run] == '\t') ++run;
                for (; run >= 4; run -= 4, k += 4) out.push_back(TOK_TAB4);
                for (; run >= 2; run -= 2, k += 2) out.push_back(TOK_TAB2);
                if (run == 1) { emit_byte('\t'); ++k; }
            } else { emit_byte(c); ++k; }                            // '\r' or other whitespace byte
        }
        dps = false;
    };
    std::size_t i = 0;
    while (i < n) {
        std::size_t g = i;
        while (g < n && is_ws_byte(stream[g])) ++g;     // whitespace gap [i, g)
        if (g >= n) {                                   // trailing whitespace (no content follows)
            tile_ws(i, g, /*inter_content=*/false);
            break;
        }
        // Literal structural markers (the standard GPT-2/3 `<|endoftext|>` stop signal, plus the
        // ChatML-adopted turn markers, §5.6) -- all ordinary ASCII, so without this check they'd
        // fall through to normal word/byte encoding (an opaque, meaningless multi-token sequence to
        // the model). Checked whole -- not case/BPE-processed -- so each collapses to exactly one
        // token. All three literals start with '<', which nothing else in ordinary prose does at a
        // word-start position anywhere near as often -- gate the table scan behind that one byte
        // compare so the common (non-'<') case is CHEAPER than the old unconditional EOS check, not
        // more expensive, despite checking 3 literals instead of 1.
        if (stream[g] == '<') {
            bool matched = false;
            for (const LiteralMarker& lm : kLiteralMarkers) {
                if (g + lm.text.size() <= n &&
                    std::equal(lm.text.begin(), lm.text.end(), stream.begin() + static_cast<std::ptrdiff_t>(g))) {
                    tile_ws(i, g, /*inter_content=*/true);
                    emit_marker(lm.id);
                    i = g + lm.text.size();
                    matched = true;
                    break;
                }
            }
            if (matched) continue;
        }
        // Directional double quote: bundle the common spacing into one OPEN/CLOSE token.
        if (stream[g] == '"') {
            const std::size_t gap = g - i;
            const bool after_glue = (g + 1 < n && !is_ws_byte(stream[g + 1]));
            if (gap == 1 && stream[i] == ' ' && dps && after_glue) {  // ` "x` -> OPEN
                emit_marker(TOK_ODQUOTE); i = g + 1; continue;
            }
            // Line-initial opening quote (measured on real TinyStories dialogue: 55389/576762 = 9.6%
            // of ALL quote occurrences -- 87.8% of the fallback total, by far the dominant miss, see
            // docs/TOKENIZER_REVIEW.md §5.8). A quote right after a single '\n' is OPEN-shaped
            // (glued to the content that follows) but the plain OPEN check above requires the
            // preceding whitespace to literally BE a space, which a newline never is. The newline
            // still needs its own NEWLINE token (its exact identity must survive the round-trip --
            // reconstructing it as a plain space would silently flatten line structure), but the
            // quote itself doesn't need `dps` to be true first: TOK_ODQUOTE's decoder only emits a
            // leading space `if (dps)`, and `dps` is already false right after a NEWLINE (decode's
            // own NEWLINE case sets it), so this is a pure win with NO decode-side change needed --
            // it just recognizes a case the generic OPEN check couldn't reach.
            if (gap == 1 && stream[i] == '\n' && after_glue) {  // "\n\"x" -> NEWLINE, then OPEN
                emit_marker(TOK_NEWLINE);
                emit_marker(TOK_ODQUOTE); i = g + 1; continue;
            }
            if (gap == 0 && dps && !after_glue) {                     // `x" ` -> CLOSE
                emit_marker(TOK_CDQUOTE); i = g + 1; continue;
            }
            // else: fall through to the bare-quote path
        }
        // Bracket glue (WS5b): collapse the JOIN tax on a bracket glued directly to what precedes it
        // (`f(`, `)y`, `((`, ...) into ONE token, instead of [JOIN, byte]. Unlike the quote markers,
        // this never needs to disambiguate direction (the byte itself already says open vs. close) --
        // it exists purely to eliminate the JOIN. Only fires on the GENUINELY glued case (gap==0 &&
        // dps): a bracket preceded by a real space ("f (x)") was already free before this and is left
        // alone, falling through unchanged to the ordinary byte path below. An OPEN marker also clears
        // `dps` afterward (mirrors TOK_ODQUOTE/TOK_SPELL_START: content right inside a bracket is
        // typically glued too, "f(x" / "[i" -- one marker absorbs the JOIN on BOTH sides); a CLOSE
        // marker leaves `dps` true (closing brackets are typically followed by a space in real
        // prose/code, ") the" / ") {"). See docs/TOKENIZER_REVIEW.md §5.9.
        if (const int gm = glue_marker_for(stream[g]); gm >= 0 && g == i && dps) {
            emit_marker(gm);   // spec row carries the open/close pending-space direction
            i = g + 1;
            continue;
        }
        tile_ws(i, g, /*inter_content=*/true, t.glue_lead(stream[g]));
        i = g;
        // Case markers prefix the word and do not affect spacing.
        while (i < n && (stream[i] == TOK_CAP || stream[i] == TOK_UP)) {
            out.push_back(stream[i]);
            ++i;
        }
        if (i >= n) break;
        const std::size_t end = word_unit_end(stream, i);
        if (end == i) {                                 // a standalone byte (punctuation, digit, bare quote, ...)
            // v2 (schemeV4, point 3): a run of plain symbols whose WHOLE maximal extent is a single
            // learned piece (`://`, `->`, `==`) collapses to one token -- its internal JOINs vanish and
            // it glues by its boundary bytes (piece_lead/trail_glue). A partial run (no whole-run piece)
            // stays byte-by-byte via the fall-through below -- still lossless, just not collapsed.
            if (is_symbol_piece_byte(stream[i])) {
                std::size_t se = i + 1;
                while (se < n && is_symbol_piece_byte(stream[se])) ++se;
                if (se - i >= 2) {
                    word_key.assign(se - i, '\0');
                    for (std::size_t k = i; k < se; ++k) word_key[k - i] = static_cast<char>(stream[k] & 0xFF);
                    const auto pit = t.piece_index.find(word_key);
                    if (pit != t.piece_index.end() && pit->second >= t.n_base) {
                        out.push_back(pit->second);
                        dps = !t.glue_trail(stream[se - 1]);
                        i = se;
                        continue;
                    }
                }
            }
            emit_byte(stream[i]); dps = !t.glue_trail(stream[i]); ++i;   // trail=glue -> next glues free ($5)
        } else {
            // Look up (or Viterbi-encode + memoise) this word's piece-id sequence by its byte key.
            word_key.assign(end - i, '\0');
            for (std::size_t k = i; k < end; ++k) word_key[k - i] = static_cast<char>(stream[k] & 0xFF);
            const auto [it, miss] = word_cache.try_emplace(word_key);
            if (miss) viterbi_encode_word(t, stream, i, end, it->second);   // fills the fresh empty slot
            const std::vector<int>& sub = it->second;
            const std::size_t N = sub.size();
            // SPELL-encapsulate any multi-piece word (N>=2), not just N>=3 (schemeV3+): a bare JOIN
            // between two ids is indistinguishable from an ordinary single-piece word glued to trailing
            // punctuation (measured on real corpus text: 93.9% of the old N==2 bare-JOIN shape was
            // exactly that false positive, not a genuine 2-piece split -- see docs/TOKENIZER_DESIGN.md).
            // TOK_JOIN now means general glue ONLY, never a word boundary.
            if (N >= 2) {
                emit_marker(TOK_SPELL_START);
                for (int id : sub) out.push_back(id);
                emit_marker(TOK_SPELL_END);
            } else {                                    // common single-token word
                for (int id : sub) out.push_back(id);
            }
            dps = true;
            i = end;
        }
    }
}

// Per-piece glue: a learned piece inherits the lead-glue of its FIRST expansion byte and the
// trail-glue of its LAST -- so a symbol piece (`://`, `->`) glues exactly as its boundary bytes
// would, and decode (keyed by piece id) agrees with encode (keyed by the unit's boundary bytes).
// A no-op for word pieces (letters default space-both) and identical to the byte rule for id < 256.
inline bool piece_lead_glue(const Tokenizer& t, int id) {
    if (id < 256) return t.glue_lead(id);
    const std::vector<int>& e = t.expansion[static_cast<std::size_t>(id)];
    return !e.empty() && t.glue_lead(e.front());
}
inline bool piece_trail_glue(const Tokenizer& t, int id) {
    if (id < 256) return t.glue_trail(id);
    const std::vector<int>& e = t.expansion[static_cast<std::size_t>(id)];
    return !e.empty() && t.glue_trail(e.back());
}

// JOIN-scheme decode FSM (token level): reconstruct bytes with a single implicit space between
// content tokens (`dps`), cancelled by JOIN; OPEN/CLOSE_DQUOTE emit a quote with their bundled
// spacing; SPELL_START..SPELL_END is a spaceless group (`in_spell`) whose sub-tokens glue with no
// internal spaces; case markers re-case the upcoming word (CAP = first letter, UP = whole word,
// carried across its JOINed / SPELL sub-tokens). detokenize_join(encode_join(x)) == normalize_text(x).
std::string detokenize_join(const Tokenizer& t, std::span<const int> ids) {
    std::string out;
    bool   dps = false;            // pending space before the next content token
    bool   in_spell = false;       // inside a SPELL_START..SPELL_END spaceless group
    Recase recase = Recase::None;  // pending re-casing for the next content
    const std::size_t m = ids.size();
    for (std::size_t k = 0; k < m; ++k) {
        const int id = ids[k];
        // Every id in the fixed marker range [TOK_EOS, TOK_MARKER_COUNT) is fully described by
        // kMarkerSpecs: decode is a walk of {leading space?, literal bytes, dps/recase/in_spell
        // effect} -- the SAME table encode_join reads via emit_marker, so the two halves can't drift.
        // Reserved-headroom ids are INERT rows (no bytes, no state change) = the old reserved-headroom
        // no-op, so a model that samples one emits nothing, never a wrapped-around byte. Ordinary
        // content (piece ids >= n_base, or a raw byte < 256) falls straight past this to the tail.
        if (id >= TOK_EOS && id < TOK_MARKER_COUNT) {
            const MarkerSpec& s = kMarkerSpecs[static_cast<std::size_t>(id - TOK_EOS)];
            if (s.lead_space && dps) out += ' ';
            out += s.literal;
            if (s.dps      >= 0) dps      = (s.dps != 0);
            if (s.recase   >= 0) recase   = static_cast<Recase>(s.recase);
            if (s.in_spell >= 0) in_spell = (s.in_spell != 0);
            continue;
        }
        if (id >= 0 && id < 256 && is_ws_byte(id)) {    // verbatim whitespace byte
            out += static_cast<char>(id); dps = false; recase = Recase::None; continue;
        }
        if (id < 0 || id >= static_cast<int>(t.expansion.size())) continue;   // out-of-range guard
        // v2 (schemeV4): a lead-glue-default byte (sentence punctuation) glues to what precedes by
        // default, so it suppresses the pending space -- `word,` reconstructs with no space. A learned
        // symbol piece glues by its first byte's default via piece_lead_glue.
        const bool lead_glue = piece_lead_glue(t, id);
        if (!in_spell && dps && !lead_glue) out += ' ';               // leading implicit space (never inside a SPELL group)
        for (int code : t.expansion[static_cast<std::size_t>(id)]) {
            const unsigned char c = static_cast<unsigned char>(code);
            if      (recase == Recase::CapFirst && is_alpha(c)) { out += static_cast<char>(to_upper(c)); recase = Recase::None; }
            else if (recase == Recase::UpWord   && is_alpha(c)) { out += static_cast<char>(to_upper(c)); }
            else if (recase == Recase::UpWord)                  { out += static_cast<char>(c); recase = Recase::None; }  // UP ends at a non-alpha (the ' in "NASA's")
            else                                                { out += static_cast<char>(c); }
        }
        if (!in_spell) {
            dps = !piece_trail_glue(t, id);   // trail-glue byte ($) / piece leaves no pending space
            // UP spans the whole word: keep it across JOINed sub-tokens, reset at the word's end.
            if (recase == Recase::UpWord && !(k + 1 < m && ids[k + 1] == TOK_JOIN)) recase = Recase::None;
        }
    }
    return out;
}

}  // namespace

std::vector<int> encode(const Tokenizer& t, const std::string& text) {
    std::vector<int> out;
    if (!t.loaded) return out;
    long replaced = 0;
    static thread_local std::string      norm;        // reused per calling thread (no per-call alloc)
    static thread_local std::vector<int> stream;
    normalize_text(text, replaced, norm);
    truecase_tokenize(norm, t.attested, nullptr, stream);

    out.reserve(stream.size());
    encode_join(t, stream, out);               // the JOIN encoder (handles word -> Viterbi internally)
    return out;
}

std::string detokenize(const Tokenizer& t, std::span<const int> ids) {
    return detokenize_join(t, ids);
}

void scan_doc_boundaries(std::span<const std::int32_t> toks, std::uint64_t base_index,
                         const Tokenizer&, int&, std::vector<std::uint64_t>& doc_starts) {
    constexpr std::int32_t eos_id = static_cast<std::int32_t>(TOK_EOS);
    for (std::size_t li = 0; li < toks.size(); ++li) {
        if (toks[li] == eos_id) {   // explicit marker: the NEXT token starts a new document
            // A trailing EOS as the corpus's very last token pushes a boundary equal to the final
            // token count -- harmless: sample_window's uniform draw never reaches that exact
            // position, so it only ever resolves back to the preceding (real) document's own end.
            doc_starts.push_back(base_index + li + 1);
        }
    }
}

// ============================================================================
//  Serialization (tokenizer.tok "S0TF" -- bumped from "S0TE": Stage 2 (docs/TOKENIZER_REVIEW.md
//  §5.8) extends TOK_MARKER_COUNT from 14 to 32 (turn markers + reserved headroom), which
//  shifts where learned piece ids start (n_base changes) -- an old "S0TE" file's ids would silently
//  misalign under the new scheme rather than fail to load, so it must be REJECTED outright, the
//  same discipline as the earlier "S0TZ"->"S0TE" bump. See the TOK_EOS comment in casing.hpp and
//  the base_symbol verification below. Carries kSchemeVersion right after the magic (see its own
//  doc comment in casing.hpp) so a future transition-rule-only change has somewhere to signal from
//  without needing another magic bump.)
// ============================================================================

void serialize(const Tokenizer& t, std::ostream& os) {
    auto wu32 = [&](std::uint32_t v) { os.write(reinterpret_cast<const char*>(&v), sizeof v); };
    auto wu16 = [&](std::uint16_t v) { os.write(reinterpret_cast<const char*>(&v), sizeof v); };
    auto wf32 = [&](float v)         { os.write(reinterpret_cast<const char*>(&v), sizeof v); };
    wu32(0x46543053u);  // "S0TF"
    wu32(kSchemeVersion);
    wu32(static_cast<std::uint32_t>(t.vocab));
    wu32(static_cast<std::uint32_t>(t.n_base));
    for (int code : t.base_symbol) wu16(static_cast<std::uint16_t>(code));
    // Word-vocabulary kind: always 1 (Unigram, pieces + log-probs) -- the only runtime-loadable
    // kind (see deserialize(), which rejects anything else outright). The discriminator itself
    // stays in the format for forward compatibility, not because this build ever writes another
    // value.
    wu32(1u);
    wu32(static_cast<std::uint32_t>(t.max_piece));
    for (int id = 0; id < t.vocab; ++id)
        wf32(id < static_cast<int>(t.piece_logp.size()) ? t.piece_logp[static_cast<std::size_t>(id)] : -1e30f);
    for (int id = t.n_base; id < t.vocab; ++id) {              // each piece is a byte sequence
        const std::vector<int>& e = t.expansion[static_cast<std::size_t>(id)];
        wu16(static_cast<std::uint16_t>(e.size()));
        for (int code : e) os.put(static_cast<char>(code & 0xFF));
    }
    // Sorted so the artifact is byte-deterministic regardless of set iteration order;
    // membership is order-independent anyway.
    std::vector<std::string> att(t.attested.begin(), t.attested.end());
    std::sort(att.begin(), att.end());
    wu32(static_cast<std::uint32_t>(att.size()));
    for (const std::string& w : att) {
        wu16(static_cast<std::uint16_t>(w.size()));
        os.write(w.data(), static_cast<std::streamsize>(w.size()));
    }
    // v2 (schemeV4, D2): the corpus-derived per-byte glue table, appended last. A gracefully-
    // degrading trailing section (AGENTS.md §3.2): an older file without it loads with the hardcoded
    // floor, so the magic need not bump. 1 byte/entry: bit1 = lead-glue, bit0 = trail-glue.
    for (int b = 0; b < 256; ++b)
        os.put(static_cast<char>((t.glue[static_cast<std::size_t>(b)].lead ? 2 : 0) |
                                 (t.glue[static_cast<std::size_t>(b)].trail ? 1 : 0)));
}

std::uint64_t fingerprint(const Tokenizer& t) {
    std::ostringstream os(std::ios::binary);   // the serialized form IS the identity; hash it
    serialize(t, os);
    const std::string s = os.str();
    std::uint64_t h = 1469598103934665603ull;  // FNV-1a (64-bit) offset basis
    for (unsigned char c : s) { h ^= c; h *= 1099511628211ull; }
    return h;
}

bool deserialize(Tokenizer& out, std::istream& is) {
    auto ru32 = [&] { std::uint32_t v{}; is.read(reinterpret_cast<char*>(&v), sizeof v); return v; };
    auto ru16 = [&] { std::uint16_t v{}; is.read(reinterpret_cast<char*>(&v), sizeof v); return v; };
    auto rf32 = [&] { float v{};         is.read(reinterpret_cast<char*>(&v), sizeof v); return v; };
    if (ru32() != 0x46543053u) return false;  // "S0TF" (a pre-Stage-2 "S0TE" file is rejected, not misread)
    (void)ru32();  // kSchemeVersion: nothing to branch on yet (this build understands exactly one
                    // version); read to keep the stream cursor aligned for the fields that follow.

    Tokenizer t;
    t.vocab  = static_cast<int>(ru32());
    t.n_base = static_cast<int>(ru32());
    // The fixed scheme (256 raw bytes + the markers, TOK_EOS..TOK_MARKER_COUNT-1) must be fully
    // present -- a corrupt/foreign/stale-version file is rejected outright, not partially read.
    if (t.n_base < TOK_MARKER_COUNT) return false;
    t.base_symbol.resize(static_cast<std::size_t>(t.n_base));
    t.expansion.resize(static_cast<std::size_t>(t.n_base));
    for (int i = 0; i < t.n_base; ++i) {
        const int code = static_cast<int>(ru16());
        t.base_symbol[static_cast<std::size_t>(i)] = code;
        t.expansion[static_cast<std::size_t>(i)] = {code};
        // base id == symbol code across the WHOLE fixed scheme (bytes 0..255, then the markers) --
        // a single uniform VERIFY, not a byte-range check plus a separate 14-way marker check (see
        // learn(): base_symbol[i]==i holds identically for both ranges by construction). A corrupt
        // or foreign file fails here, loudly, rather than silently misassigning ids -- e.g. this is
        // exactly what stops a pre-EOS file's TOK_CAP=256 from being misread as TOK_EOS=256, on top
        // of the magic-number bump above already rejecting it. Ids past TOK_MARKER_COUNT (learned
        // pieces, or a future file's reserved-marker headroom this build doesn't know about yet)
        // aren't checked here -- this build doesn't need to understand them to load correctly.
        if (i < TOK_MARKER_COUNT && code != i) return false;
    }
    // Word-vocabulary kind: only 1 (Unigram, pieces + log-probs) is loadable at runtime -- a
    // pre-WS2 file's legacy BPE-merge encoding (kind 0) is rejected outright, not decoded, since
    // this build's encode()/detokenize() no longer has a BPE word encoder to use it with.
    const int kind = static_cast<int>(ru32());
    if (kind != 1) return false;
    t.max_piece = static_cast<int>(ru32());
    t.piece_logp.resize(static_cast<std::size_t>(t.vocab));
    for (int id = 0; id < t.vocab; ++id) t.piece_logp[static_cast<std::size_t>(id)] = rf32();
    t.expansion.reserve(static_cast<std::size_t>(t.vocab));
    for (int id = t.n_base; id < t.vocab; ++id) {
        const int len = static_cast<int>(ru16());
        std::vector<int> codes(static_cast<std::size_t>(len));
        std::string s(static_cast<std::size_t>(len), '\0');
        for (int k = 0; k < len; ++k) { const int c = static_cast<unsigned char>(is.get()); codes[static_cast<std::size_t>(k)] = c; s[static_cast<std::size_t>(k)] = static_cast<char>(c); }
        t.expansion.push_back(std::move(codes));
        t.piece_index.emplace(std::move(s), id);
    }
    for (int b = 0; b < 256; ++b)                                 // single bytes are candidates too
        t.piece_index.emplace(std::string(1, static_cast<char>(b)), b);   // base id == byte value
    const int n_words = static_cast<int>(ru32());
    for (int i = 0; i < n_words; ++i) {
        const int len = static_cast<int>(ru16());
        std::string w(static_cast<std::size_t>(len), '\0');
        is.read(w.data(), len);
        t.attested.insert(std::move(w));
    }
    if (!is) return false;
    // v2 (schemeV4, D2): optional trailing per-byte glue table (see serialize). Absent in a legacy
    // file -> the hardcoded floor, so an older tokenizer.tok still loads and encodes identically.
    t.glue = default_glue_table();
    char gt[256];
    if (is.read(gt, 256).gcount() == 256)
        for (int b = 0; b < 256; ++b) {
            const unsigned v = static_cast<unsigned char>(gt[b]);
            t.glue[static_cast<std::size_t>(b)] = { (v & 2) != 0, (v & 1) != 0 };
        }
    is.clear();   // a short/absent table leaves EOF/fail bits set on `is` -- expected, not an error
    t.loaded = true;
    out = std::move(t);
    return true;
}

}  // namespace sub0::tok
