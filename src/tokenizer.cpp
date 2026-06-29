// tokenizer.cpp — implementation of the reusable BPE tokenizer (sub0::tok).
//
// Single source of truth for the truecasing + BPE vocabulary: the corpus scan
// (passes 1-2), the BPE merge learning, and the encode/detokenize/serialize the
// engine and the configurator both build on. Free of any dependency on the
// generated config so the configurator can use it.

#include "sub0/tokenizer.hpp"

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
    if (opts.join_scheme) {
        // JOIN scheme: the COMPLETE 256-byte alphabet (a total tokenizer -- any byte is
        // encodable, no silent drop of out-of-corpus characters) followed by the markers,
        // ALWAYS present so the base alphabet is corpus-independent and the ids are stable.
        for (int b = 0; b < 256; ++b) { t.byte_base[static_cast<std::size_t>(b)] = b; t.base_symbol.push_back(b); }
        t.cap_id     = add_marker(TOK_CAP);
        t.up_id      = add_marker(TOK_UP);
        t.join_id    = add_marker(TOK_JOIN);
        t.newline_id = add_marker(TOK_NEWLINE);
        t.para_id    = add_marker(TOK_PARA);
        t.join_scheme = true;
    } else {
        // Legacy scheme: only the byte values the corpus actually used (in byte order) then
        // the markers that were actually emitted -- a compact, corpus-derived alphabet.
        for (int b = 0; b < 256; ++b)
            if (scan.byte_used[static_cast<std::size_t>(b)]) {
                t.byte_base[static_cast<std::size_t>(b)] = static_cast<int>(t.base_symbol.size());
                t.base_symbol.push_back(b);
            }
        if (scan.used_cap) { t.cap_id = static_cast<int>(t.base_symbol.size()); t.base_symbol.push_back(TOK_CAP); }
        if (scan.used_up)  { t.up_id  = static_cast<int>(t.base_symbol.size()); t.base_symbol.push_back(TOK_UP); }
    }
    t.n_base = static_cast<int>(t.base_symbol.size());

    // Remap the word table from raw byte symbols to base ids (a bijection on used
    // bytes, preserving pair frequencies and tie-breaking -> identical merges). After
    // this, scan.word_syms holds base ids; BPE below rewrites them to final token ids.
    for (std::vector<int>& w : scan.word_syms)
        for (int& s : w) s = t.byte_base[static_cast<std::size_t>(s)];

    t.expansion.resize(static_cast<std::size_t>(t.n_base));
    for (int id = 0; id < t.n_base; ++id)
        t.expansion[static_cast<std::size_t>(id)] = {t.base_symbol[static_cast<std::size_t>(id)]};
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

// JOIN-scheme encode. `stream` is the truecased byte+marker stream. A single inter-content
// space is implicit (no token); JOIN suppresses the implicit space (glued punctuation, and
// the splits inside a multi-token word); NEWLINE/PARA encode the common newline runs; any
// other whitespace is emitted verbatim as literal byte tokens (lossless, rare in prose).
// See docs/TOKENIZER_DESIGN.md §2,§4,§5.
void encode_join(const Tokenizer& t, const std::vector<int>& stream, std::vector<int>& out) {
    const std::size_t n = stream.size();
    bool prev_content = false;
    std::vector<int> seq, sub;
    std::size_t i = 0;
    while (i < n) {
        std::size_t g = i;
        while (g < n && is_ws_byte(stream[g])) ++g;     // whitespace gap [i, g)
        const std::size_t gap = g - i;
        const bool content_after = (g < n);
        if (gap > 0) {
            const bool inter = prev_content && content_after;
            if (inter && gap == 1 && stream[i] == ' ') {
                // implicit single inter-content space -- emit nothing (the common case, free)
            } else if (gap == 1 && stream[i] == '\n') {
                out.push_back(t.newline_id);
            } else if (gap == 2 && stream[i] == '\n' && stream[i + 1] == '\n') {
                out.push_back(t.para_id);
            } else {                                    // verbatim: leading/trailing/multi/tab/CR runs
                for (std::size_t k = i; k < g; ++k) out.push_back(t.byte_base[static_cast<std::size_t>(stream[k])]);
            }
        } else if (prev_content && content_after) {
            out.push_back(t.join_id);                   // adjacent content with no space -> glue
        }
        i = g;
        if (i >= n) break;
        // Content item: any case markers, then a word unit (BPE + intra-word JOINs) or one byte.
        while (i < n && (stream[i] == TOK_CAP || stream[i] == TOK_UP)) {
            out.push_back(stream[i] == TOK_CAP ? t.cap_id : t.up_id);
            ++i;
        }
        if (i >= n) break;
        const std::size_t end = word_unit_end(stream, i);
        if (end == i) {                                 // a standalone byte (punctuation, digit, ...)
            out.push_back(t.byte_base[static_cast<std::size_t>(stream[i])]);
            ++i;
        } else {
            seq.assign(stream.begin() + static_cast<std::ptrdiff_t>(i),
                       stream.begin() + static_cast<std::ptrdiff_t>(end));
            for (int& s : seq) s = t.byte_base[static_cast<std::size_t>(s)];
            sub.clear();
            bpe_encode_word(t, seq, sub);
            for (std::size_t k = 0; k < sub.size(); ++k) {
                if (k > 0) out.push_back(t.join_id);    // intra-word: no implicit space between sub-tokens
                out.push_back(sub[k]);
            }
            i = end;
        }
        prev_content = true;
    }
}

// JOIN-scheme decode FSM (token level): reconstruct bytes with a single implicit space
// between content tokens, cleared by JOIN; whitespace/markers emit their literal spacing;
// case markers re-case the upcoming word (CAP = first letter, UP = whole word across its
// JOINed sub-tokens). detokenize_join(encode_join(x)) == normalize_text(x) by construction.
std::string detokenize_join(const Tokenizer& t, const std::vector<int>& ids) {
    std::string out;
    bool pending_space = false;
    int  case_mode = 0;     // 0 none, 1 cap-first-letter, 2 upper-whole-word
    const std::size_t m = ids.size();
    for (std::size_t k = 0; k < m; ++k) {
        const int id = ids[k];
        if (id == t.join_id)    { pending_space = false; continue; }
        if (id == t.newline_id) { out += '\n';   pending_space = false; case_mode = 0; continue; }
        if (id == t.para_id)    { out += "\n\n"; pending_space = false; case_mode = 0; continue; }
        if (id == t.cap_id)     { case_mode = 1; continue; }
        if (id == t.up_id)      { case_mode = 2; continue; }
        if (id >= 0 && id < 256 && is_ws_byte(id)) {    // verbatim whitespace byte
            out += static_cast<char>(id); pending_space = false; case_mode = 0; continue;
        }
        if (id < 0 || id >= static_cast<int>(t.expansion.size())) continue;   // out-of-range guard
        if (pending_space) out += ' ';
        for (int code : t.expansion[static_cast<std::size_t>(id)]) {
            const unsigned char c = static_cast<unsigned char>(code);
            if (case_mode == 1 && is_alpha(c))      { out += static_cast<char>(to_upper(c)); case_mode = 0; }
            else if (case_mode == 2 && is_alpha(c)) { out += static_cast<char>(to_upper(c)); }
            else                                    { out += static_cast<char>(c); }
        }
        pending_space = true;
        // UP applies to the whole word; keep it across the word's JOINed sub-tokens, reset at end.
        if (case_mode == 2 && !(k + 1 < m && ids[k + 1] == t.join_id)) case_mode = 0;
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
    if (t.join_scheme) { encode_join(t, stream, out); return out; }
    for (std::size_t i = 0, n = stream.size(); i < n;) {
        const std::size_t end = word_unit_end(stream, i);
        if (end == i) {
            const int id = t.sym_to_base(stream[i]);
            if (id >= 0) out.push_back(id);
            ++i;
            continue;
        }
        std::vector<int> seq;
        for (std::size_t k = i; k < end; ++k) {
            const int id = t.sym_to_base(stream[k]);
            if (id >= 0) seq.push_back(id);
        }
        bpe_encode_word(t, seq, out);
        i = end;
    }
    return out;
}

std::string detokenize(const Tokenizer& t, const std::vector<int>& ids) {
    if (t.join_scheme) return detokenize_join(t, ids);
    std::vector<int> stream;
    for (int id : ids) {
        if (id < 0 || id >= static_cast<int>(t.expansion.size())) continue;
        const std::vector<int>& e = t.expansion[static_cast<std::size_t>(id)];
        stream.insert(stream.end(), e.begin(), e.end());
    }
    return casing::detokenize(stream);
}

// ============================================================================
//  Serialization (tokenizer.bin "S0TZ")
// ============================================================================

void serialize(const Tokenizer& t, std::ostream& os) {
    auto wu32 = [&](std::uint32_t v) { os.write(reinterpret_cast<const char*>(&v), sizeof v); };
    auto wu16 = [&](std::uint16_t v) { os.write(reinterpret_cast<const char*>(&v), sizeof v); };
    wu32(0x5A543053u);  // "S0TZ"
    wu32(static_cast<std::uint32_t>(t.vocab));
    wu32(static_cast<std::uint32_t>(t.n_base));
    for (int code : t.base_symbol) wu16(static_cast<std::uint16_t>(code));
    wu32(static_cast<std::uint32_t>(t.merges.size()));
    for (const auto& [a, b] : t.merges) { wu32(static_cast<std::uint32_t>(a)); wu32(static_cast<std::uint32_t>(b)); }
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
        if      (code == TOK_CAP)     t.cap_id = i;
        else if (code == TOK_UP)      t.up_id  = i;
        else if (code == TOK_JOIN)    { t.join_id    = i; t.join_scheme = true; }  // join scheme detected
        else if (code == TOK_NEWLINE) t.newline_id = i;
        else if (code == TOK_PARA)    t.para_id    = i;
        else                          t.byte_base[static_cast<unsigned char>(code)] = i;
    }
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
