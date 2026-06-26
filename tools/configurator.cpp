// configurator.cpp — stage 1 of the build chain ("sub0-configure").
//
// This tool is intentionally INDEPENDENT of the engine: it is the program that
// *produces* the compile-time configuration the engine is built against, so it
// cannot itself depend on that configuration. It is the single source of truth
// for tokenization: it reads the corpus, applies corpus-aware truecasing, learns
// a BPE vocabulary, tokenizes the whole corpus, and writes three artifacts:
//
//   - <out>            the constexpr header (model dims + final VOCAB + paths)
//   - corpus.tok       the tokenized corpus (int32 stream) consumed by training
//   - tokenizer.bin    the runtime tokenizer (base alphabet + merges + attested
//                      words) used only to encode prompts and detokenize output
//
// Training and generation therefore consume integer tokens directly; only
// generation maps token ids back to text for the user.
//
// Usage:
//   sub0-configure --corpus <path> -o <generated_config.hpp>
//                  [--dmodel N --layers N --heads N --seq N --ternary 0|1]
//                  [--vocab N --min-merge N]

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <print>
#include <queue>
#include <string>
#include <string_view>
#include <system_error>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include <CLI/CLI.hpp>

#include "sub0/casing.hpp"

namespace {

// Hash for an adjacent-symbol pair, used by the BPE merge counter.
struct PairHash {
    std::size_t operator()(const std::pair<int, int>& p) const noexcept {
        return (static_cast<std::size_t>(static_cast<std::uint32_t>(p.first)) << 32) ^
               static_cast<std::uint32_t>(p.second);
    }
};

// Serialize a word's symbol sequence to a string key for the unique-word map. Word symbols
// are byte values (0..255), so one byte per symbol is a lossless, compact key -- short enough
// to stay in std::string's small-buffer (no heap alloc) for typical words.
inline std::string seq_key(const std::vector<int>& s) {
    std::string k(s.size(), '\0');
    for (std::size_t i = 0; i < s.size(); ++i) k[i] = static_cast<char>(s[i] & 0xFF);
    return k;
}

constexpr std::size_t CHUNK_BYTES = 32u << 20;  // ~32 MB per streamed corpus chunk

// Stream the corpus to `fn` one newline-aligned chunk at a time, so it is never held in
// memory all at once -- the out-of-core path for a corpus larger than RAM. Newlines are
// safe split points: they never fall inside a UTF-8 glyph, a word unit, or the look-back
// the name detector and truecaser use (which already treats a newline as a line start),
// so processing the corpus chunk-by-chunk is byte-identical to processing it whole. The
// trailing partial line is carried into the next chunk; the final remainder is flushed at
// EOF. Returns false only if the file cannot be opened.
template <class Fn>
bool for_each_chunk(const std::string& path, Fn&& fn) {
    std::ifstream is(path, std::ios::binary);
    if (!is) return false;
    std::string buf, block(CHUNK_BYTES, '\0');
    for (;;) {
        is.read(block.data(), static_cast<std::streamsize>(block.size()));
        const std::streamsize got = is.gcount();
        if (got > 0) buf.append(block.data(), static_cast<std::size_t>(got));
        if (got <= 0 || is.eof()) {                       // last read: flush the remainder
            if (!buf.empty()) fn(std::string_view{buf});
            break;
        }
        const std::size_t nl = buf.rfind('\n');           // split at the last full line
        if (nl == std::string::npos) continue;            // no boundary yet: read more
        fn(std::string_view{buf.data(), nl + 1});
        buf.erase(0, nl + 1);
    }
    return true;
}

// --- Intermediate scan state (the cacheable output of the expensive corpus passes) -------
// A complete snapshot of what passes 1-2 produce: the bounded name-detection COUNTS, the
// unique-word frequency table, the alphabet usage and aggregate stats. This is the
// expensive-to-produce, cheap-to-store intermediate -- caching it lets a re-run with a
// different vocab size (or after a crash, or to re-tokenize for the JOIN scheme) SKIP the
// multi-pass corpus scan. It is the RAW, pre-BPE data so it is reusable across vocab targets;
// it keeps the name COUNTS (not just the derived `attested`) so the state is MERGEABLE -- a
// future step can sum two corpora's ScanStates to build a joint vocabulary or extend one
// incrementally across sessions. Binary: compact + fast for millions of words, validated by
// corpus size+mtime and a logic version. (A JSON/simdjson encoding would bloat the
// millions-of-words table ~4x and parse slower; binary stays while the format is internal and
// stable. Revisit only if the state must cross tools/languages.) Bump WCACHE_VERSION whenever
// casing.hpp's normalization/truecasing changes, since the cached tables encode that logic.
constexpr std::uint32_t WCACHE_MAGIC   = 0x43573053u;  // "S0WC"
constexpr std::uint32_t WCACHE_VERSION = 1u;

struct ScanState {
    std::size_t raw_bytes = 0, norm_bytes = 0;
    long long quote_repl = 0;
    sub0::casing::TokStats st;
    std::array<int, 256> byte_used{};                 // 0/1 usage flags
    bool used_cap = false, used_up = false;
    std::unordered_map<std::string, long long> lower_count, midcap_count;  // mergeable name stats (64-bit counts)
    std::vector<std::vector<int>> word_syms;          // unique word -> raw byte-symbol sequence
    std::vector<long long> word_freq;                 // 64-bit: a frequent word can exceed 2^31 on a large corpus
};

template <class T> void wr(std::ostream& os, const T& v) {
    os.write(reinterpret_cast<const char*>(&v), sizeof v);
}
template <class T> T rd(std::istream& is) {
    T v{}; is.read(reinterpret_cast<char*>(&v), sizeof v); return v;
}
void wr_counts(std::ostream& os, const std::unordered_map<std::string, long long>& m) {
    wr(os, static_cast<std::uint64_t>(m.size()));
    for (const auto& [w, c] : m) {
        wr(os, static_cast<std::uint16_t>(w.size()));
        os.write(w.data(), static_cast<std::streamsize>(w.size()));
        wr(os, static_cast<std::int64_t>(c));
    }
}
bool rd_counts(std::istream& is, std::unordered_map<std::string, long long>& m) {
    const std::uint64_t n = rd<std::uint64_t>(is);
    m.clear(); m.reserve(n);
    std::string w;
    for (std::uint64_t i = 0; i < n; ++i) {
        const std::uint16_t len = rd<std::uint16_t>(is);
        w.resize(len); is.read(w.data(), len);
        m.emplace(w, static_cast<long long>(rd<std::int64_t>(is)));
    }
    return static_cast<bool>(is);
}

// Save the scan state next to the corpus (<corpus>.words), keyed to the corpus file.
void save_scan_state(const std::string& path, const std::string& corpus, const ScanState& S) {
    std::ofstream os(path, std::ios::binary);
    if (!os) { std::println(stderr, "warning: cannot write scan cache '{}'", path); return; }
    std::error_code ec;
    wr(os, WCACHE_MAGIC); wr(os, WCACHE_VERSION);
    wr(os, static_cast<std::uint64_t>(std::filesystem::file_size(corpus, ec)));
    wr(os, static_cast<std::int64_t>(std::filesystem::last_write_time(corpus, ec).time_since_epoch().count()));
    wr(os, static_cast<std::uint64_t>(S.raw_bytes)); wr(os, static_cast<std::uint64_t>(S.norm_bytes));
    wr(os, static_cast<std::int64_t>(S.quote_repl));
    wr(os, static_cast<std::int64_t>(S.st.words)); wr(os, static_cast<std::int64_t>(S.st.cap));
    wr(os, static_cast<std::int64_t>(S.st.up));    wr(os, static_cast<std::int64_t>(S.st.names));
    for (int b = 0; b < 256; ++b) wr(os, static_cast<std::uint8_t>(S.byte_used[b] ? 1 : 0));
    wr(os, static_cast<std::uint8_t>(S.used_cap ? 1 : 0)); wr(os, static_cast<std::uint8_t>(S.used_up ? 1 : 0));
    wr_counts(os, S.lower_count); wr_counts(os, S.midcap_count);
    wr(os, static_cast<std::uint64_t>(S.word_syms.size()));
    std::vector<std::uint8_t> bb;
    for (std::size_t i = 0; i < S.word_syms.size(); ++i) {
        wr(os, static_cast<std::uint64_t>(S.word_freq[i]));
        const std::vector<int>& seq = S.word_syms[i];
        wr(os, static_cast<std::uint16_t>(seq.size()));
        bb.assign(seq.begin(), seq.end());                 // ints 0..255 -> bytes
        os.write(reinterpret_cast<const char*>(bb.data()), static_cast<std::streamsize>(bb.size()));
    }
}

// Load + validate against the corpus (size+mtime+version). Returns false (leaving S untouched-
// enough to fall back to a full scan) if absent, stale or malformed.
bool load_scan_state(const std::string& path, const std::string& corpus, ScanState& S) {
    std::ifstream is(path, std::ios::binary);
    if (!is) return false;
    if (rd<std::uint32_t>(is) != WCACHE_MAGIC || rd<std::uint32_t>(is) != WCACHE_VERSION) return false;
    std::error_code ec;
    if (rd<std::uint64_t>(is) != std::filesystem::file_size(corpus, ec)) return false;
    if (rd<std::int64_t>(is) !=
        static_cast<std::int64_t>(std::filesystem::last_write_time(corpus, ec).time_since_epoch().count()))
        return false;
    S.raw_bytes = static_cast<std::size_t>(rd<std::uint64_t>(is));
    S.norm_bytes = static_cast<std::size_t>(rd<std::uint64_t>(is));
    S.quote_repl = static_cast<long long>(rd<std::int64_t>(is));
    S.st.words = rd<std::int64_t>(is); S.st.cap = rd<std::int64_t>(is);
    S.st.up = rd<std::int64_t>(is);    S.st.names = rd<std::int64_t>(is);
    S.byte_used.fill(0);
    for (int b = 0; b < 256; ++b) S.byte_used[b] = rd<std::uint8_t>(is) ? 1 : 0;
    S.used_cap = rd<std::uint8_t>(is) != 0; S.used_up = rd<std::uint8_t>(is) != 0;
    if (!rd_counts(is, S.lower_count) || !rd_counts(is, S.midcap_count)) return false;
    const std::uint64_t nw = rd<std::uint64_t>(is);
    S.word_syms.clear(); S.word_syms.reserve(nw);
    S.word_freq.clear(); S.word_freq.reserve(nw);
    std::vector<std::uint8_t> bb;
    for (std::uint64_t i = 0; i < nw; ++i) {
        const long long f = static_cast<long long>(rd<std::uint64_t>(is));
        const std::uint16_t len = rd<std::uint16_t>(is);
        bb.resize(len); is.read(reinterpret_cast<char*>(bb.data()), len);
        S.word_syms.emplace_back(bb.begin(), bb.end());
        S.word_freq.push_back(f);
    }
    return static_cast<bool>(is);
}

}  // namespace

using namespace sub0::casing;

int main(int argc, char** argv) {
    CLI::App app{"sub0-configure: truecase + BPE a corpus, emit the tokenized corpus and a constexpr config header"};

    std::string corpus;
    std::string out;
    int d_model      = 96;
    int n_layers     = 3;
    int n_heads      = 4;
    int seq_len      = 64;
    int ternary      = 0;
    int vocab_target = 2048;
    int min_merge    = 2;
    int emit_tok     = 1;
    std::string tune_cache;

    app.add_option("--corpus", corpus,  "Training corpus path (drives vocabulary derivation)")
       ->required()->check(CLI::ExistingFile);
    app.add_option("-o",       out,     "Output generated header path")->required();
    app.add_option("--dmodel", d_model, "Embedding / residual width")->capture_default_str();
    app.add_option("--layers", n_layers,"Transformer block count")->capture_default_str();
    app.add_option("--heads",  n_heads, "Attention head count")->capture_default_str();
    app.add_option("--seq",    seq_len, "Context window length")->capture_default_str();
    app.add_option("--ternary",ternary, "1 = BitNet-style ternary block weights")
       ->capture_default_str()->check(CLI::Range(0, 1));
    app.add_option("--vocab",  vocab_target, "Target BPE vocabulary size (base symbols + markers + merges)")
       ->capture_default_str();
    app.add_option("--min-merge", min_merge, "Stop merging once the best pair occurs fewer than this many times")
       ->capture_default_str();
    app.add_option("--corpus-tok", emit_tok,
                   "1 = pre-tokenize the whole corpus to corpus.tok; 0 = skip it and let training tokenize "
                   "on demand (avoids the ~2x-on-disk token copy for a huge corpus)")
       ->capture_default_str()->check(CLI::Range(0, 1));
    app.add_option("--tune-cache", tune_cache,
                   "Persisted tuned-defaults cache; read to bake DEFAULT_THREADS / DEFAULT_WINDOWS_PER_THREAD")
       ->capture_default_str();

    CLI11_PARSE(app, argc, argv);

    if (d_model % n_heads != 0) {
        std::println(stderr, "configure error: dmodel ({}) not divisible by heads ({})",
                     d_model, n_heads);
        return 1;
    }

    const std::string abspath = std::filesystem::absolute(corpus).string();

    // ----------------------------------------------------------------------
    // Out-of-core tokenization: the corpus is STREAMED in newline-aligned chunks (never
    // held whole in memory) across three sequential passes that keep only bounded tables
    // resident -- the unique-word frequency table (saturates by Heaps' law), the base
    // alphabet, the case-marker stats and the BPE merge list. corpus.tok is written
    // incrementally. This is byte-identical to whole-corpus processing because a newline
    // never splits a glyph, a word unit, or the truecaser/name-detector look-back.
    // ----------------------------------------------------------------------

    // Phase timers: corpus ingest is the cost that grows with the corpus, so it is the
    // baseline to optimise against (caching the scan + parallelising the passes are levers).
    const auto _t0 = std::chrono::steady_clock::now();

    // The expensive corpus scan (passes 1-2) produces a bounded ScanState. Cache it next to
    // the corpus (<corpus>.words, validated by size+mtime+version) so a re-run with a
    // different vocab, after a crash, or to re-tokenize for a new scheme reloads it and SKIPS
    // the scan. The name COUNTS are kept (not just the derived `attested`) so the state is
    // mergeable for a future multi-corpus / incremental tokenizer.
    ScanState S;
    std::unordered_set<std::string> attested;
    long long names_withheld = 0;
    std::unordered_map<std::string, int> word_index;       // raw-key -> word id (also used by Pass 3)
    auto derive_attested = [&] {
        // Re-derivable from the (mergeable) name counts: a lowercase form is attested unless
        // its mid-sentence-capital ("name") uses dominate its lowercase uses.
        attested.clear(); names_withheld = 0;
        for (const auto& [w, lc] : S.lower_count) {
            const auto it = S.midcap_count.find(w);
            const long long mid = (it == S.midcap_count.end()) ? 0 : it->second;
            if (mid > lc) { ++names_withheld; continue; }
            attested.insert(w);
        }
    };

    std::chrono::steady_clock::time_point _t1{}, _t2{};
    const std::string cache_path = std::string(corpus) + ".words";
    if (load_scan_state(cache_path, corpus, S)) {
        derive_attested();
        for (std::size_t i = 0; i < S.word_syms.size(); ++i)
            word_index.emplace(seq_key(S.word_syms[i]), static_cast<int>(i));
        _t1 = _t2 = std::chrono::steady_clock::now();
        std::println(stderr, "scan cache: HIT '{}' ({} words, {} attested) -- corpus scan skipped",
                     cache_path, S.word_syms.size(), attested.size());
    } else {
        // --- Pass 1: decide which lowercase forms license a Capitalized/UPPER -> marker
        //     collapse. The tell is *position*: a capital following a lowercase word
        //     mid-sentence ("...to Spot") is a name; per lowercase form we count lowercase
        //     uses vs mid-sentence-capital uses, withholding name-dominant forms from `attested`.
        const bool open_ok = for_each_chunk(corpus, [&](std::string_view chunk) {
            S.raw_bytes += chunk.size();
            long qr = 0;
            const std::string norm = normalize_text(std::string(chunk), qr);
            S.quote_repl += qr;
            S.norm_bytes += norm.size();
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
                    S.lower_count[w] += 1;
                } else if (is_upper(static_cast<unsigned char>(w[0])) && rest_lower) {  // "Spot", "The"
                    if (preceded_by_lowercase(i)) {
                        std::string lw = w;
                        lw[0] = static_cast<char>(to_lower(static_cast<unsigned char>(lw[0])));
                        S.midcap_count[lw] += 1;
                    }
                }
                i = j;
            }
        });
        if (!open_ok)         { std::println(stderr, "configure error: cannot read corpus '{}'", corpus); return 1; }
        if (S.raw_bytes == 0) { std::println(stderr, "configure error: empty corpus"); return 1; }
        derive_attested();
        _t1 = std::chrono::steady_clock::now();

        // --- Pass 2: truecase each chunk + pre-tokenize into the unique-word table. A word
        //     unit (run of word bytes per word_unit_end) becomes one table entry keyed by its
        //     RAW byte-symbol sequence (independent of the not-yet-fixed base alphabet);
        //     punctuation, spaces and the markers are standalone symbols that never merge. We
        //     also record which byte values / markers actually occur.
        for_each_chunk(corpus, [&](std::string_view chunk) {
            long qr = 0;
            const std::string  norm   = normalize_text(std::string(chunk), qr);
            const std::vector<int> stream = truecase_tokenize(norm, attested, &S.st);
            for (std::size_t i = 0, n = stream.size(); i < n;) {
                const std::size_t end = word_unit_end(stream, i);
                if (end == i) {                                  // standalone symbol
                    const int s = stream[i];
                    if (s == TOK_CAP)      S.used_cap = true;
                    else if (s == TOK_UP)  S.used_up = true;
                    else                   S.byte_used[s] = 1;
                    ++i; continue;
                }
                // Build the lookup key directly from the byte run (1 byte/symbol, usually in
                // std::string's small buffer -> no heap), and only allocate the word's symbol
                // vector / mark its bytes on a CACHE MISS. Repeated words (the vast majority of
                // occurrences) then cost just a key build + a counter bump -- no per-occurrence
                // vector allocation.
                std::string key(end - i, '\0');
                for (std::size_t k = i; k < end; ++k) key[k - i] = static_cast<char>(stream[k] & 0xFF);
                auto it = word_index.find(key);
                if (it == word_index.end()) {
                    std::vector<int> seq(stream.begin() + static_cast<std::ptrdiff_t>(i),
                                         stream.begin() + static_cast<std::ptrdiff_t>(end));
                    for (int b : seq) S.byte_used[b] = 1;
                    word_index.emplace(std::move(key), static_cast<int>(S.word_syms.size()));
                    S.word_syms.push_back(std::move(seq));
                    S.word_freq.push_back(1);
                } else {
                    S.word_freq[static_cast<std::size_t>(it->second)] += 1;
                }
                i = end;
            }
        });
        _t2 = std::chrono::steady_clock::now();
        save_scan_state(cache_path, corpus, S);
        std::println(stderr, "scan cache: saved '{}' ({} words)", cache_path, S.word_syms.size());
    }

    // Aliases so the downstream base-alphabet / BPE / emit / reporting code is unchanged.
    auto& byte_used = S.byte_used;
    const bool used_cap = S.used_cap, used_up = S.used_up;
    auto& word_syms = S.word_syms;
    auto& word_freq = S.word_freq;
    const sub0::casing::TokStats& st = S.st;
    const std::size_t raw_bytes = S.raw_bytes, norm_bytes = S.norm_bytes;
    const long long quote_repl = S.quote_repl;
    S.lower_count = {}; S.midcap_count = {};   // attested derived; free the mergeable name counts

    // Fix the base alphabet: distinct byte values used (in byte order) then the markers --
    // the same ordering as before, so base ids match and BPE tie-breaking is unchanged.
    std::array<int, 256> byte_base; byte_base.fill(-1);
    std::vector<int> base_symbol;  // base id -> symbol code (0..255 byte, 256 cap, 257 up)
    for (int b = 0; b < 256; ++b)
        if (byte_used[b]) { byte_base[b] = static_cast<int>(base_symbol.size()); base_symbol.push_back(b); }
    int cap_id = -1, up_id = -1;
    if (used_cap) { cap_id = static_cast<int>(base_symbol.size()); base_symbol.push_back(TOK_CAP); }
    if (used_up)  { up_id  = static_cast<int>(base_symbol.size()); base_symbol.push_back(TOK_UP); }
    const int n_base = static_cast<int>(base_symbol.size());
    auto sym_to_base = [&](int s) { return s == TOK_CAP ? cap_id : s == TOK_UP ? up_id : byte_base[s]; };

    // Remap the word table from raw byte symbols to base ids. A bijection on used bytes, so
    // it preserves pair frequencies AND tie-breaking -> identical merges.
    for (std::vector<int>& w : word_syms)
        for (int& s : w) s = byte_base[s];

    // --- BPE (incremental): greedily merge the most frequent adjacent pair (weighted by word
    //     frequency) until the target vocab or the per-pair floor. The naive form rebuilt the
    //     ENTIRE pair-count map every merge (O(merges x total_symbols)) -- pathological on a
    //     large corpus (3h+ stuck on 43GB FineWeb). Here pair counts are maintained
    //     INCREMENTALLY: a pair->words index finds the few words a merge touches, each is
    //     re-counted in place, and a lazy-deletion max-heap yields the best pair without
    //     rescanning. The merge order is identical (same counts, same max-count / smallest-pair
    //     tie-break) so the output stays byte-identical -- just far faster.
    std::vector<std::vector<int>> expansion(static_cast<std::size_t>(n_base));
    for (int id = 0; id < n_base; ++id)
        expansion[static_cast<std::size_t>(id)] = {base_symbol[static_cast<std::size_t>(id)]};
    std::vector<std::pair<int, int>> merges;
    int vocab = n_base;

    std::unordered_map<std::pair<int, int>, long long, PairHash> pc;     // current pair -> count (64-bit)
    std::unordered_map<std::pair<int, int>, std::vector<int>, PairHash> pair_words;  // pair -> word ids
    // Heap top = MAX count, then SMALLEST pair -- matching the naive tie-break so the merge
    // sequence (hence the vocabulary) is identical. Lazy deletion: an entry is valid only while
    // its count still equals pc[pair]; stale ones are skipped on pop.
    struct HeapItem { long long count; std::pair<int, int> pr; };
    auto worse = [](const HeapItem& a, const HeapItem& b) {
        return a.count != b.count ? a.count < b.count : a.pr > b.pr;
    };
    std::priority_queue<HeapItem, std::vector<HeapItem>, decltype(worse)> heap(worse);

    // Seed counts + index from the word table, then push each DISTINCT pair once.
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

    // Per-merge scratch: the set of pairs whose count changed (pushed to the heap ONCE at the
    // end of the merge -- a merge touching millions of words changes only a few DISTINCT pairs,
    // so per-touch pushes were wasteful), and a small reused word-local pair-delta map.
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
        merges.push_back(best);
        std::vector<int> exp = expansion[static_cast<std::size_t>(best.first)];
        exp.insert(exp.end(), expansion[static_cast<std::size_t>(best.second)].begin(),
                   expansion[static_cast<std::size_t>(best.second)].end());
        expansion.push_back(std::move(exp));

        // Apply the merge only in the words that contain `best` (deduped; tolerate stale ids).
        std::vector<int> wl;
        if (auto pit = pair_words.find(best); pit != pair_words.end()) wl = std::move(pit->second);
        pair_words.erase(best);
        std::sort(wl.begin(), wl.end());
        wl.erase(std::unique(wl.begin(), wl.end()), wl.end());
        touched.clear();
        for (const int w : wl) {
            std::vector<int>& s = word_syms[static_cast<std::size_t>(w)];
            // Rebuild with best -> new_id (greedy, left-to-right), detecting whether it changed.
            std::vector<int> ns; ns.reserve(s.size());
            bool changed = false;
            for (std::size_t k = 0; k < s.size();) {
                if (k + 1 < s.size() && s[k] == best.first && s[k + 1] == best.second) { ns.push_back(new_id); k += 2; changed = true; }
                else { ns.push_back(s[k]); ++k; }
            }
            if (!changed) continue;                          // stale index entry
            const long long f = word_freq[static_cast<std::size_t>(w)];
            // Net pair-count change = (old pair multiset) - (new pair multiset). Only pairs with a
            // non-zero net actually changed, so we touch the big pc map / heap / index for those
            // alone -- the bulk of the bookkeeping stays in this small, cache-hot, word-local map.
            wd.clear();
            for (std::size_t k = 0; k + 1 < s.size();  ++k) wd[{s[k], s[k + 1]}]   += 1;
            for (std::size_t k = 0; k + 1 < ns.size(); ++k) wd[{ns[k], ns[k + 1]}] -= 1;
            for (const auto& [p, d] : wd) {
                if (d == 0) continue;
                pc[p] -= static_cast<long long>(d) * f;
                touched.insert(p);
                if (d < 0) pair_words[p].push_back(w);       // pair now present (more) in this word
            }
            s.swap(ns);
        }
        for (const auto& p : touched) {                      // one heap push per changed pair
            const auto it = pc.find(p);
            if (it != pc.end() && it->second > 0) heap.push({it->second, p});
        }
        pc.erase(best);
    }
    const auto _t3 = std::chrono::steady_clock::now();

    const std::filesystem::path gen_dir  = std::filesystem::path(out).parent_path();
    const std::filesystem::path tok_path = gen_dir / "corpus.tok";
    const std::filesystem::path tkz_path = gen_dir / "tokenizer.bin";

    // --- Pass 3 (optional): stream the corpus once more and emit the merged token stream to
    //     corpus.tok incrementally (never materialised in memory). Skipped when --corpus-tok
    //     0: training then tokenizes windows on demand from the raw corpus + tokenizer.bin,
    //     avoiding the ~2x-on-disk token copy for a huge corpus. Each word unit looks up its
    //     post-BPE id sequence in the table; standalone symbols map straight to base ids.
    //     Losslessness is verified per chunk by reconstructing the base symbols; the id
    //     array's length is back-patched after the count is known.
    std::size_t token_count = 0;
    bool tok_rt = true;
    if (emit_tok) {
        std::ofstream ts(tok_path, std::ios::binary);
        if (!ts) { std::println(stderr, "configure error: cannot write '{}'", tok_path.string()); return 1; }
        {
            const std::uint32_t magic = 0x4B543053u, vfield = static_cast<std::uint32_t>(vocab), nfield = 0u;
            ts.write(reinterpret_cast<const char*>(&magic),  sizeof magic);
            ts.write(reinterpret_cast<const char*>(&vfield), sizeof vfield);
            ts.write(reinterpret_cast<const char*>(&nfield), sizeof nfield);  // patched below
        }
        std::vector<std::int32_t> out_tokens;     // per-chunk emit buffer (bounded by the chunk)
        std::vector<int> recon;                   // per-chunk reconstruction (bounded)
        for_each_chunk(corpus, [&](std::string_view chunk) {
            long qr = 0;
            const std::string norm = normalize_text(std::string(chunk), qr);
            const std::vector<int> stream = truecase_tokenize(norm, attested, nullptr);
            if (detokenize(stream) != norm) tok_rt = false;     // round-trip 1: truecasing
            out_tokens.clear();
            for (std::size_t i = 0, n = stream.size(); i < n;) {
                const std::size_t end = word_unit_end(stream, i);
                if (end == i) { out_tokens.push_back(sym_to_base(stream[i])); ++i; continue; }
                std::vector<int> seq(stream.begin() + static_cast<std::ptrdiff_t>(i),
                                     stream.begin() + static_cast<std::ptrdiff_t>(end));
                auto it = word_index.find(seq_key(seq));
                if (it == word_index.end()) {                   // unreachable if passes agree
                    for (int b : seq) out_tokens.push_back(byte_base[b]);
                    tok_rt = false;
                } else {
                    for (int id : word_syms[static_cast<std::size_t>(it->second)]) out_tokens.push_back(id);
                }
                i = end;
            }
            recon.clear();                                       // round-trip 2: tokenization
            for (std::int32_t id : out_tokens)
                recon.insert(recon.end(), expansion[static_cast<std::size_t>(id)].begin(),
                             expansion[static_cast<std::size_t>(id)].end());
            if (recon != stream) tok_rt = false;
            ts.write(reinterpret_cast<const char*>(out_tokens.data()),
                     static_cast<std::streamsize>(out_tokens.size() * sizeof(std::int32_t)));
            token_count += out_tokens.size();
        });
        {
            const std::uint32_t nfield = static_cast<std::uint32_t>(token_count);  // back-patch ntok
            ts.seekp(8, std::ios::beg);
            ts.write(reinterpret_cast<const char*>(&nfield), sizeof nfield);
        }
        ts.close();
    } else {
        // Drop any stale corpus.tok from a previous emit so training reliably falls back to
        // on-demand tokenization instead of mmapping an out-of-date token copy.
        std::error_code ec;
        std::filesystem::remove(tok_path, ec);
    }
    const auto _t4 = std::chrono::steady_clock::now();

    // 9. Write the runtime tokenizer (base alphabet + merges + attested words).
    //    Generation uses this to encode prompts and detokenize output; training
    //    needs only corpus.tok.
    {
        std::ofstream tz(tkz_path, std::ios::binary);
        if (!tz) { std::println(stderr, "configure error: cannot write '{}'", tkz_path.string()); return 1; }
        auto wu32 = [&](std::uint32_t v) { tz.write(reinterpret_cast<const char*>(&v), sizeof v); };
        auto wu16 = [&](std::uint16_t v) { tz.write(reinterpret_cast<const char*>(&v), sizeof v); };
        wu32(0x5A543053);  // "S0TZ"
        wu32(static_cast<std::uint32_t>(vocab));
        wu32(static_cast<std::uint32_t>(n_base));
        for (int code : base_symbol) wu16(static_cast<std::uint16_t>(code));
        wu32(static_cast<std::uint32_t>(merges.size()));
        for (const auto& [a, b] : merges) { wu32(static_cast<std::uint32_t>(a)); wu32(static_cast<std::uint32_t>(b)); }
        // Sorted so tokenizer.bin is byte-deterministic regardless of set iteration order
        // (which the scan-cache load path reshuffles); membership is order-independent anyway.
        std::vector<std::string> att(attested.begin(), attested.end());
        std::sort(att.begin(), att.end());
        wu32(static_cast<std::uint32_t>(att.size()));
        for (const std::string& w : att) {
            wu16(static_cast<std::uint16_t>(w.size()));
            tz.write(w.data(), static_cast<std::streamsize>(w.size()));
        }
    }

    // 10. Emit the constexpr header (model dims + VOCAB + artifact paths).
    std::ofstream os(out, std::ios::binary);
    if (!os) { std::println(stderr, "configure error: cannot write '{}'", out); return 1; }
    auto emit_path = [&](const char* name, const std::string& p) {
        os << "inline constexpr char " << name << "[] = \"";
        for (char c : p) { if (c == '\\' || c == '"') os << '\\'; os << c; }
        os << "\";\n";
    };

    // Cache the machine's logical-core count so the engine sizes its statically
    // allocated worker pool to the hardware instead of a fixed guess. If a previous
    // `tune` run persisted better runtime defaults, fold those in here too; otherwise
    // fall back to the hardware core count and a conservative windows/thread of 4.
    const unsigned hw_concurrency = std::max(1u, std::thread::hardware_concurrency());
    const int      max_workers    = static_cast<int>(hw_concurrency);
    int            default_threads = static_cast<int>(hw_concurrency);
    int            default_wpt     = 4;
    std::string    tune_cache_abs;
    if (!tune_cache.empty()) {
        tune_cache_abs = std::filesystem::absolute(tune_cache).string();
        std::ifstream tc(tune_cache);
        for (std::string line; std::getline(tc, line); ) {
            const auto eq = line.find('=');
            if (eq == std::string::npos) continue;
            const std::string key = line.substr(0, eq);
            const int         val = std::atoi(line.substr(eq + 1).c_str());
            if      (key == "threads"            && val > 0) default_threads = val;
            else if (key == "windows_per_thread" && val > 0) default_wpt     = val;
        }
    }

    os << "// AUTO-GENERATED by sub0-configure. Do not edit.\n";
    os << "// Model dimensions and the BPE vocabulary size are baked in at build time.\n";
    os << "#pragma once\n\n";
    os << "constexpr int  D_MODEL     = " << d_model  << ";\n";
    os << "constexpr int  N_LAYERS    = " << n_layers << ";\n";
    os << "constexpr int  N_HEADS     = " << n_heads  << ";\n";
    os << "constexpr int  SEQ_LEN     = " << seq_len  << ";\n";
    os << "constexpr int  D_FF        = 4 * D_MODEL;\n";
    os << "constexpr int  D_HEAD      = D_MODEL / N_HEADS;\n";
    os << "constexpr bool USE_TERNARY = " << (ternary ? "true" : "false") << ";\n";
    os << "constexpr int  VOCAB       = " << vocab << ";\n\n";
    os << "// --- Cached hardware facts + persisted tuned runtime defaults -----------\n";
    os << "constexpr int  HW_CONCURRENCY            = " << hw_concurrency  << ";\n";
    os << "constexpr int  MAX_WORKERS               = " << max_workers     << ";\n";
    os << "constexpr int  DEFAULT_THREADS           = " << default_threads << ";\n";
    os << "constexpr int  DEFAULT_WINDOWS_PER_THREAD = " << default_wpt    << ";\n\n";
    emit_path("DEFAULT_CORPUS",     abspath);
    emit_path("DEFAULT_CORPUS_TOK", std::filesystem::absolute(tok_path).string());
    emit_path("DEFAULT_TOKENIZER",  std::filesystem::absolute(tkz_path).string());
    emit_path("DEFAULT_TUNE_CACHE", tune_cache_abs);

    // 11. Report the real numbers.
    const double bytes_per_tok =
        token_count == 0 ? 0.0 : static_cast<double>(norm_bytes) / static_cast<double>(token_count);
    std::size_t wordset_bytes = 0;
    for (const std::string& w : attested) wordset_bytes += w.size() + 1;

    std::println(stderr, "--- BPE truecasing tokenizer (real numbers) ---");
    std::println(stderr, "corpus bytes (raw / normalized): {} / {}", raw_bytes, norm_bytes);
    std::println(stderr, "quote glyphs collapsed:          {}", quote_repl);
    std::println(stderr, "alpha words (total / unique):    {} / {}", st.words, word_syms.size());
    std::println(stderr, "  collapsed <|cap|> / <|up|>:    {} / {}", st.cap, st.up);
    std::println(stderr, "  kept verbatim (names/mixed):   {}", st.names);
    std::println(stderr, "  names withheld (mid-sent cap): {}", names_withheld);
    std::println(stderr, "base symbols / merges / vocab:   {} / {} / {}", n_base, merges.size(), vocab);
    if (emit_tok) {
        std::println(stderr, "total tokens:                    {}", token_count);
        std::println(stderr, "compression (bytes/token):       {:.3f}", bytes_per_tok);
    } else {
        std::println(stderr, "corpus.tok:                      skipped (--corpus-tok 0; on-demand)");
    }
    std::println(stderr, "attested words / table size:     {} / {} bytes ({:.1f} KB)",
                 attested.size(), wordset_bytes, wordset_bytes / 1024.0);
    // Ingest throughput: the baseline to optimise (it scales with the corpus). "MB/s corpus"
    // is corpus bytes / wall time even though the corpus is read 2-3x -- the rate that matters
    // for "how long to ingest this corpus". The streaming passes are the parallelisation target.
    {
        const auto sec = [](auto a, auto b) { return std::chrono::duration<double>(b - a).count(); };
        const double total = sec(_t0, _t4);
        const double cmb   = static_cast<double>(raw_bytes) / 1e6;
        std::println(stderr,
            "ingest: {:.2f} GB | pass1 {:.1f}s | pass2 {:.1f}s | BPE {:.1f}s | pass3 {:.1f}s | total {:.1f}s ({:.0f} MB/s corpus)",
            static_cast<double>(raw_bytes) / 1e9, sec(_t0, _t1), sec(_t1, _t2), sec(_t2, _t3), sec(_t3, _t4),
            total, total > 0 ? cmb / total : 0.0);
    }
    std::println(stderr, "round-trip truecase / tokenize:  OK / {}", tok_rt ? "OK" : "FAIL");
    std::println(stderr, "corpus.tok / tokenizer.bin:      {} | {}",
                 emit_tok ? tok_path.string() : std::string("(on-demand)"), tkz_path.string());
    std::println(stderr, "-----------------------------------------------");
    if (!tok_rt) { std::println(stderr, "configure error: tokenization round-trip failed"); return 1; }

    std::println("sub0-configure: vocab={} (base {} + {} merges), d={} L={} H={} seq={}{} -> {}",
                 vocab, n_base, merges.size(), d_model, n_layers, n_heads, seq_len,
                 ternary ? " (ternary)" : "", out);
    return 0;
}
