// tokenizer.cpp — implementation of the reusable BPE tokenizer (sub0::tok).
//
// Single source of truth for the truecasing + BPE vocabulary: the corpus scan
// (passes 1-2), the BPE merge learning, and the encode/detokenize/serialize the
// engine and the configurator both build on. Free of any dependency on the
// generated config so the configurator can use it.

#include "sub0/tokenizer.hpp"
#include "sub0/unigram.hpp"

#include <algorithm>
#include <istream>
#include <limits>
#include <ostream>
#include <queue>
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

// Apply learned merges to one pre-token word (a sequence of base ids), lowest rank
// first, then append the resulting ids to `out`. Reproduces the corpus tokenization
// for the same word so prompts stay in-distribution with training.
//
// TODO(perf, hot): this is O(N^2 * merges) per word -- each merge rescans every adjacent
// pair for the lowest rank. It runs for EVERY word occurrence during corpus.tok emission
// (configurator), but a corpus is Zipfian -- the same few thousand words dominate. The big
// win is MEMOIZATION: cache word-bytes -> encoded ids (the unique-word table already exists
// in Scan during learn; the emission path re-encodes from scratch). With a cache the
// per-unique-word cost amortizes to ~zero. Secondary: a priority-queue / linked-list BPE
// (as learn() already uses) removes the inner rescan for the rare long/OOV word.
void bpe_encode_word(const Tokenizer& t, std::vector<int>& seq, std::vector<int>& out) {
    while (seq.size() >= 2) {
        int best_rank = std::numeric_limits<int>::max(), best_pos = -1;
        for (std::size_t k = 0; k + 1 < seq.size(); ++k) {
            const auto it = t.merge_rank.find({seq[k], seq[k + 1]});
            if (it != t.merge_rank.end() && it->second < best_rank) {
                best_rank = it->second;
                best_pos  = static_cast<int>(k);
            }
        }
        if (best_pos < 0) break;
        seq[static_cast<std::size_t>(best_pos)] = t.n_base + best_rank;
        seq.erase(seq.begin() + best_pos + 1);
    }
    for (int id : seq) out.push_back(id);
}

// Unigram word encoder: Viterbi-segment the byte run stream[lo,hi) into the min Σ −log p piece-id
// sequence (piece_index / piece_logp). Every base byte is a candidate, so dp[L] is always reachable;
// the explicit fallback keeps it total even if a byte were somehow absent. Replaces bpe_encode_word
// when the tokenizer is in Unigram mode (t.max_piece > 0).
void viterbi_encode_word(const Tokenizer& t, const std::vector<int>& stream, std::size_t lo, std::size_t hi,
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
        for (int a = 0; a < L; ++a) out.push_back(t.byte_base[static_cast<std::size_t>(stream[lo + static_cast<std::size_t>(a)] & 0xFF)]);
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
    const std::string norm = normalize_text(std::string(chunk), qr);
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
    const std::string      norm   = normalize_text(std::string(chunk), qr);
    const std::vector<int> stream = truecase_tokenize(norm, attested, &st);
    for (std::size_t i = 0, n = stream.size(); i < n;) {
        const std::size_t end = word_unit_end(stream, i);
        if (end == i) {                                   // standalone symbol
            const int s = stream[i];
            if (s == TOK_CAP)      used_cap = true;
            else if (s == TOK_UP)  used_up = true;
            else                   byte_used[static_cast<std::size_t>(s)] = 1;
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
//  learn — base alphabet + incremental BPE
// ============================================================================

Tokenizer learn(Scan& scan, const std::unordered_set<std::string>& attested,
                const LearnOptions& opts) {
    Tokenizer t;

    // Fix the base alphabet. BPE tie-breaking is by (count, pair) on these byte-derived
    // ids, so the merge sequence is deterministic and order-independent of the scan.
    t.byte_base.fill(-1);
    auto add_marker = [&](int code) {                 // append a marker, return its base id
        const int id = static_cast<int>(t.base_symbol.size());
        t.base_symbol.push_back(code);
        return id;
    };
    // The COMPLETE 256-byte alphabet (a total tokenizer -- any byte is encodable, no silent drop of
    // out-of-corpus characters) followed by the spacing/case markers, ALWAYS present so the base
    // alphabet is corpus-independent and the ids are stable. This is the only scheme (the legacy
    // corpus-derived "space-as-a-byte-token" alphabet was retired with the old models).
    for (int b = 0; b < 256; ++b) { t.byte_base[static_cast<std::size_t>(b)] = b; t.base_symbol.push_back(b); }
    t.cap_id        = add_marker(TOK_CAP);
    t.up_id         = add_marker(TOK_UP);
    t.join_id       = add_marker(TOK_JOIN);
    t.newline_id    = add_marker(TOK_NEWLINE);
    t.para_id       = add_marker(TOK_PARA);
    t.odquote_id    = add_marker(TOK_ODQUOTE);
    t.cdquote_id    = add_marker(TOK_CDQUOTE);
    t.spell_start_id = add_marker(TOK_SPELL_START);
    t.spell_end_id  = add_marker(TOK_SPELL_END);
    t.space2_id     = add_marker(TOK_SPACE2);
    t.space4_id     = add_marker(TOK_SPACE4);
    t.tab2_id       = add_marker(TOK_TAB2);
    t.tab4_id       = add_marker(TOK_TAB4);
    t.n_base = static_cast<int>(t.base_symbol.size());

    t.expansion.resize(static_cast<std::size_t>(t.n_base));
    for (int id = 0; id < t.n_base; ++id)
        t.expansion[static_cast<std::size_t>(id)] = {t.base_symbol[static_cast<std::size_t>(id)]};

    // Unigram LM word vocabulary (the DEFAULT). Learn the pieces top-down from the RAW word table
    // (no remap), then map them onto the base alphabet: a single-byte piece reuses its base id, a
    // multi-byte piece becomes a new id. Word encoding is then Viterbi over piece_index/piece_logp
    // (set up here) instead of the greedy merge replay -- globally occurrence-optimal, no dead slots.
    if (opts.method == LearnOptions::Method::Unigram) {
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
                const int bid = t.byte_base[static_cast<unsigned char>(s[0])];
                if (bid >= 0) { t.piece_logp[static_cast<std::size_t>(bid)] = lp; t.piece_index[s] = bid; }
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
        for (int b = 0; b < 256; ++b) {
            const int bid = t.byte_base[static_cast<std::size_t>(b)];
            if (bid >= 0) t.piece_index.emplace(std::string(1, static_cast<char>(b)), bid);
        }
        t.max_piece = std::max(1, u.max_len);
        t.vocab     = static_cast<int>(t.expansion.size());
        // Remap the scan word table to the Viterbi piece-id segmentation (parallels the BPE remap), so
        // the configurator emits / reports read final ids uniformly for both methods.
        for (std::vector<int>& w : scan.word_syms) {
            std::vector<int> ids;
            viterbi_encode_word(t, w, 0, w.size(), ids);
            w = std::move(ids);
        }
        t.attested  = attested;
        t.loaded    = true;
        return t;
    }

    // --- BPE (method == BPE): remap the word table to base ids, then greedy merge. Kept for the
    //     vocabulary A/B + the bytes/token curve; no longer the default word encoder. ---
    for (std::vector<int>& w : scan.word_syms)
        for (int& s : w) s = t.byte_base[static_cast<std::size_t>(s)];
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

    while (vocab < opts.vocab_target) {
        std::pair<int, int> best{0, 0}; long long best_c = -1;
        while (!heap.empty()) {
            const HeapItem top = heap.top();
            const auto it = pc.find(top.pr);
            if (it != pc.end() && it->second == top.count && top.count > 0) { best = top.pr; best_c = top.count; break; }
            heap.pop();
        }
        if (best_c < opts.min_merge) break;

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
    for (std::size_t i = 0; i < t.merges.size(); ++i)
        t.merge_rank.emplace(t.merges[i], static_cast<int>(i));
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
//  encode / detokenize
// ============================================================================

namespace {

// True iff `s` is a whitespace BYTE (not a marker). The JOIN encoder classifies the
// whitespace run between content tokens; the decoder emits such bytes verbatim.
inline bool is_ws_byte(int s) { return s == ' ' || s == '\t' || s == '\n' || s == '\r'; }

// Pending re-casing the decoder applies to the next content, set by a CAP/UP marker.
// CapFirst capitalises the first alpha letter then clears; UpWord upper-cases the alpha
// run and clears at the first non-alpha or the word's end (mirrors casing::detokenize).
enum class Recase { None, CapFirst, UpWord };

// JOIN-scheme encode. `stream` is the truecased byte+marker stream. The encoder mirrors the
// decoder's pending-space state `dps` so it emits exactly the tokens that reconstruct the text:
//  - a single inter-content space is implicit (no token); JOIN cancels a pending space (glue);
//  - a lone '\n' -> NEWLINE, "\n\n" -> PARA, any other whitespace -> verbatim byte tokens;
//  - a double quote with ` "x` spacing -> OPEN_DQUOTE, `x" ` -> CLOSE_DQUOTE (bundles the space);
//  - a word of N BPE sub-tokens: N=1 bare, N=2 sub JOIN sub, N>=3 SPELL_START sub.. SPELL_END.
// See docs/TOKENIZER_DESIGN.md §2-§5.
void encode_join(const Tokenizer& t, const std::vector<int>& stream, std::vector<int>& out) {
    const std::size_t n = stream.size();
    bool dps = false;                 // decoder's pending_space after the last emitted token
    std::vector<int> seq, sub;
    auto emit_byte = [&](int b) { out.push_back(t.byte_base[static_cast<std::size_t>(b)]); };
    // Realize the whitespace run [lo,hi) (mirrors the decoder). A single inter-word space is
    // implicit (free) under a pending space; an empty inter-content gap glues with JOIN; any
    // other run is tiled into NEWLINE/PARA + run-length SPACE2/4 / TAB2/4 tokens, with a verbatim
    // byte for an odd remainder ('\r', a lone space/tab, ...). `inter_content` is false for the
    // trailing run (no following content), where a single space must stay literal, not implicit.
    auto tile_ws = [&](std::size_t lo, std::size_t hi, bool inter_content) {
        const std::size_t gap = hi - lo;
        if (gap == 0) {
            if (dps) { out.push_back(t.join_id); dps = false; }       // glue: cancel the pending space
            return;
        }
        if (inter_content && gap == 1 && stream[lo] == ' ' && dps) return;  // implicit single space
        for (std::size_t k = lo; k < hi;) {
            const int c = stream[k];
            if (c == '\n' && k + 1 < hi && stream[k + 1] == '\n') { out.push_back(t.para_id);    k += 2; }
            else if (c == '\n')                                   { out.push_back(t.newline_id); k += 1; }
            else if (c == ' ') {
                std::size_t run = 0; while (k + run < hi && stream[k + run] == ' ')  ++run;
                for (; run >= 4; run -= 4, k += 4) out.push_back(t.space4_id);
                for (; run >= 2; run -= 2, k += 2) out.push_back(t.space2_id);
                if (run == 1) { emit_byte(' '); ++k; }
            } else if (c == '\t') {
                std::size_t run = 0; while (k + run < hi && stream[k + run] == '\t') ++run;
                for (; run >= 4; run -= 4, k += 4) out.push_back(t.tab4_id);
                for (; run >= 2; run -= 2, k += 2) out.push_back(t.tab2_id);
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
        // Directional double quote: bundle the common spacing into one OPEN/CLOSE token.
        if (stream[g] == '"') {
            const std::size_t gap = g - i;
            const bool after_glue = (g + 1 < n && !is_ws_byte(stream[g + 1]));
            if (gap == 1 && stream[i] == ' ' && dps && after_glue) {  // ` "x` -> OPEN
                out.push_back(t.odquote_id); dps = false; i = g + 1; continue;
            }
            if (gap == 0 && dps && !after_glue) {                     // `x" ` -> CLOSE
                out.push_back(t.cdquote_id); dps = true; i = g + 1; continue;
            }
            // else: fall through to the bare-quote path
        }
        tile_ws(i, g, /*inter_content=*/true);
        i = g;
        // Case markers prefix the word and do not affect spacing.
        while (i < n && (stream[i] == TOK_CAP || stream[i] == TOK_UP)) {
            out.push_back(stream[i] == TOK_CAP ? t.cap_id : t.up_id);
            ++i;
        }
        if (i >= n) break;
        const std::size_t end = word_unit_end(stream, i);
        if (end == i) {                                 // a standalone byte (punctuation, digit, bare quote, ...)
            emit_byte(stream[i]); dps = true; ++i;
        } else {
            sub.clear();
            if (t.max_piece > 0) {                      // Unigram: Viterbi over the raw word bytes
                viterbi_encode_word(t, stream, i, end, sub);
            } else {                                    // BPE: remap to base ids, greedy merge
                seq.assign(stream.begin() + static_cast<std::ptrdiff_t>(i),
                           stream.begin() + static_cast<std::ptrdiff_t>(end));
                for (int& s : seq) s = t.byte_base[static_cast<std::size_t>(s)];
                bpe_encode_word(t, seq, sub);
            }
            const std::size_t N = sub.size();
            if (N >= 3) {                               // SPELL-encapsulate: 2 delimiters <= N-1 JOINs
                out.push_back(t.spell_start_id);
                for (int id : sub) out.push_back(id);
                out.push_back(t.spell_end_id);
            } else if (N == 2) {                        // sun+day: a single JOIN between the two
                out.push_back(sub[0]); out.push_back(t.join_id); out.push_back(sub[1]);
            } else {                                    // common single-token word
                for (int id : sub) out.push_back(id);
            }
            dps = true;
            i = end;
        }
    }
}

// JOIN-scheme decode FSM (token level): reconstruct bytes with a single implicit space between
// content tokens (`dps`), cancelled by JOIN; OPEN/CLOSE_DQUOTE emit a quote with their bundled
// spacing; SPELL_START..SPELL_END is a spaceless group (`in_spell`) whose sub-tokens glue with no
// internal spaces; case markers re-case the upcoming word (CAP = first letter, UP = whole word,
// carried across its JOINed / SPELL sub-tokens). detokenize_join(encode_join(x)) == normalize_text(x).
std::string detokenize_join(const Tokenizer& t, const std::vector<int>& ids) {
    std::string out;
    bool   dps = false;            // pending space before the next content token
    bool   in_spell = false;       // inside a SPELL_START..SPELL_END spaceless group
    Recase recase = Recase::None;  // pending re-casing for the next content
    const std::size_t m = ids.size();
    for (std::size_t k = 0; k < m; ++k) {
        const int id = ids[k];
        if (id == t.join_id)         { dps = false; continue; }
        if (id == t.newline_id)      { out += '\n';   dps = false; recase = Recase::None; continue; }
        if (id == t.para_id)         { out += "\n\n"; dps = false; recase = Recase::None; continue; }
        if (id == t.space2_id)       { out += "  ";       dps = false; recase = Recase::None; continue; }
        if (id == t.space4_id)       { out += "    ";     dps = false; recase = Recase::None; continue; }
        if (id == t.tab2_id)         { out += "\t\t";     dps = false; recase = Recase::None; continue; }
        if (id == t.tab4_id)         { out += "\t\t\t\t"; dps = false; recase = Recase::None; continue; }
        if (id == t.cap_id)          { recase = Recase::CapFirst; continue; }
        if (id == t.up_id)           { recase = Recase::UpWord;   continue; }
        if (id == t.odquote_id)      { if (dps) out += ' '; out += '"'; dps = false; recase = Recase::None; continue; }
        if (id == t.cdquote_id)      { out += '"'; dps = true; recase = Recase::None; continue; }
        if (id == t.spell_start_id)  { if (dps) out += ' '; in_spell = true; dps = false; continue; }
        if (id == t.spell_end_id)    { in_spell = false; dps = true; recase = Recase::None; continue; }
        if (id >= 0 && id < 256 && is_ws_byte(id)) {    // verbatim whitespace byte
            out += static_cast<char>(id); dps = false; recase = Recase::None; continue;
        }
        if (id < 0 || id >= static_cast<int>(t.expansion.size())) continue;   // out-of-range guard
        if (!in_spell && dps) out += ' ';               // leading implicit space (never inside a SPELL group)
        for (int code : t.expansion[static_cast<std::size_t>(id)]) {
            const unsigned char c = static_cast<unsigned char>(code);
            if      (recase == Recase::CapFirst && is_alpha(c)) { out += static_cast<char>(to_upper(c)); recase = Recase::None; }
            else if (recase == Recase::UpWord   && is_alpha(c)) { out += static_cast<char>(to_upper(c)); }
            else if (recase == Recase::UpWord)                  { out += static_cast<char>(c); recase = Recase::None; }  // UP ends at a non-alpha (the ' in "NASA's")
            else                                                { out += static_cast<char>(c); }
        }
        if (!in_spell) {
            dps = true;
            // UP spans the whole word: keep it across JOINed sub-tokens, reset at the word's end.
            if (recase == Recase::UpWord && !(k + 1 < m && ids[k + 1] == t.join_id)) recase = Recase::None;
        }
    }
    return out;
}

}  // namespace

std::vector<int> encode(const Tokenizer& t, const std::string& text) {
    std::vector<int> out;
    if (!t.loaded) return out;
    long replaced = 0;
    const std::string      norm   = normalize_text(text, replaced);
    const std::vector<int> stream = truecase_tokenize(norm, t.attested, nullptr);

    out.reserve(stream.size());
    encode_join(t, stream, out);               // the JOIN encoder (handles word -> Viterbi/BPE internally)
    return out;
}

std::string detokenize(const Tokenizer& t, const std::vector<int>& ids) {
    return detokenize_join(t, ids);
}

// ============================================================================
//  Serialization (tokenizer.bin "S0TZ")
// ============================================================================

void serialize(const Tokenizer& t, std::ostream& os) {
    auto wu32 = [&](std::uint32_t v) { os.write(reinterpret_cast<const char*>(&v), sizeof v); };
    auto wu16 = [&](std::uint16_t v) { os.write(reinterpret_cast<const char*>(&v), sizeof v); };
    auto wf32 = [&](float v)         { os.write(reinterpret_cast<const char*>(&v), sizeof v); };
    wu32(0x5A543053u);  // "S0TZ"
    wu32(static_cast<std::uint32_t>(t.vocab));
    wu32(static_cast<std::uint32_t>(t.n_base));
    for (int code : t.base_symbol) wu16(static_cast<std::uint16_t>(code));
    // Word-vocabulary kind: 1 = Unigram (pieces + log-probs, the default), 0 = legacy BPE merges.
    if (t.max_piece > 0) {
        wu32(1u);
        wu32(static_cast<std::uint32_t>(t.max_piece));
        for (int id = 0; id < t.vocab; ++id)
            wf32(id < static_cast<int>(t.piece_logp.size()) ? t.piece_logp[static_cast<std::size_t>(id)] : -1e30f);
        for (int id = t.n_base; id < t.vocab; ++id) {              // each piece is a byte sequence
            const std::vector<int>& e = t.expansion[static_cast<std::size_t>(id)];
            wu16(static_cast<std::uint16_t>(e.size()));
            for (int code : e) os.put(static_cast<char>(code & 0xFF));
        }
    } else {
        wu32(0u);
        wu32(static_cast<std::uint32_t>(t.merges.size()));
        for (const auto& [a, b] : t.merges) { wu32(static_cast<std::uint32_t>(a)); wu32(static_cast<std::uint32_t>(b)); }
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
}

bool deserialize(Tokenizer& out, std::istream& is) {
    auto ru32 = [&] { std::uint32_t v{}; is.read(reinterpret_cast<char*>(&v), sizeof v); return v; };
    auto ru16 = [&] { std::uint16_t v{}; is.read(reinterpret_cast<char*>(&v), sizeof v); return v; };
    auto rf32 = [&] { float v{};         is.read(reinterpret_cast<char*>(&v), sizeof v); return v; };
    if (ru32() != 0x5A543053u) return false;  // "S0TZ"

    Tokenizer t;
    t.vocab  = static_cast<int>(ru32());
    t.n_base = static_cast<int>(ru32());
    t.byte_base.fill(-1);
    t.base_symbol.resize(static_cast<std::size_t>(t.n_base));
    t.expansion.resize(static_cast<std::size_t>(t.n_base));
    for (int i = 0; i < t.n_base; ++i) {
        const int code = static_cast<int>(ru16());
        t.base_symbol[static_cast<std::size_t>(i)] = code;
        t.expansion[static_cast<std::size_t>(i)] = {code};
        if      (code == TOK_CAP)         t.cap_id = i;
        else if (code == TOK_UP)          t.up_id  = i;
        else if (code == TOK_JOIN)        t.join_id = i;
        else if (code == TOK_NEWLINE)     t.newline_id = i;
        else if (code == TOK_PARA)        t.para_id    = i;
        else if (code == TOK_ODQUOTE)     t.odquote_id = i;
        else if (code == TOK_CDQUOTE)     t.cdquote_id = i;
        else if (code == TOK_SPELL_START) t.spell_start_id = i;
        else if (code == TOK_SPELL_END)   t.spell_end_id   = i;
        else if (code == TOK_SPACE2)      t.space2_id = i;
        else if (code == TOK_SPACE4)      t.space4_id = i;
        else if (code == TOK_TAB2)        t.tab2_id   = i;
        else if (code == TOK_TAB4)        t.tab4_id   = i;
        else                              t.byte_base[static_cast<unsigned char>(code)] = i;
    }
    const int kind = static_cast<int>(ru32());
    if (kind == 1) {                                                   // Unigram: pieces + log-probs
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
            if (t.byte_base[static_cast<std::size_t>(b)] >= 0)
                t.piece_index.emplace(std::string(1, static_cast<char>(b)), t.byte_base[static_cast<std::size_t>(b)]);
    } else {                                                          // legacy BPE merges
        const int n_merges = static_cast<int>(ru32());
        t.expansion.reserve(static_cast<std::size_t>(t.n_base + n_merges));
        t.merges.reserve(static_cast<std::size_t>(n_merges));
        for (int i = 0; i < n_merges; ++i) {
            const int a = static_cast<int>(ru32());
            const int b = static_cast<int>(ru32());
            t.merges.emplace_back(a, b);
            t.merge_rank.emplace(std::pair{a, b}, i);
            std::vector<int> exp = t.expansion[static_cast<std::size_t>(a)];
            exp.insert(exp.end(), t.expansion[static_cast<std::size_t>(b)].begin(),
                       t.expansion[static_cast<std::size_t>(b)].end());
            t.expansion.push_back(std::move(exp));
        }
    }
    const int n_words = static_cast<int>(ru32());
    for (int i = 0; i < n_words; ++i) {
        const int len = static_cast<int>(ru16());
        std::string w(static_cast<std::size_t>(len), '\0');
        is.read(w.data(), len);
        t.attested.insert(std::move(w));
    }
    if (!is) return false;
    t.loaded = true;
    out = std::move(t);
    return true;
}

}  // namespace sub0::tok
