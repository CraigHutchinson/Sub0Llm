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

// Like for_each_chunk but over the byte range [start, end) of the file. Both ends are
// expected to be newline-aligned (see segment_bounds), so the range is a whole number of
// lines and feeding it to `fn` is byte-identical to streaming the whole file -- this is the
// unit a worker thread owns when the passes run in parallel. Reads only its own range, in
// CHUNK_BYTES sub-chunks, so the resident set per worker stays bounded out-of-core.
template <class Fn>
bool for_each_chunk_range(const std::string& path, std::size_t start, std::size_t end, Fn&& fn) {
    std::ifstream is(path, std::ios::binary);
    if (!is) return false;
    if (start >= end) return true;
    is.seekg(static_cast<std::streamoff>(start));
    std::string buf, block(CHUNK_BYTES, '\0');
    std::size_t remaining = end - start;
    while (remaining > 0) {
        const std::size_t want = std::min(remaining, CHUNK_BYTES);
        is.read(block.data(), static_cast<std::streamsize>(want));
        const std::streamsize got = is.gcount();
        if (got <= 0) { if (!buf.empty()) fn(std::string_view{buf}); break; }
        buf.append(block.data(), static_cast<std::size_t>(got));
        remaining -= static_cast<std::size_t>(got);
        if (remaining == 0) { if (!buf.empty()) fn(std::string_view{buf}); break; }
        const std::size_t nl = buf.rfind('\n');
        if (nl == std::string::npos) continue;
        fn(std::string_view{buf.data(), nl + 1});
        buf.erase(0, nl + 1);
    }
    return true;
}

// Split [0, fsz) into `nseg` ranges whose interior boundaries land just AFTER a newline, so
// no line (and no word/glyph/look-back) is ever split -- the same invariant that makes
// chunking byte-identical to whole-file processing. Each interior cut starts at the even
// offset i*fsz/nseg and advances to the next newline. Returns nseg+1 monotonic offsets.
std::vector<std::size_t> segment_bounds(const std::string& path, std::size_t fsz, int nseg) {
    std::vector<std::size_t> b(static_cast<std::size_t>(nseg) + 1);
    b[0] = 0; b[static_cast<std::size_t>(nseg)] = fsz;
    std::ifstream is(path, std::ios::binary);
    std::string blk(1u << 20, '\0');
    for (int i = 1; i < nseg; ++i) {
        std::size_t pos = fsz * static_cast<std::size_t>(i) / static_cast<std::size_t>(nseg);
        if (pos <= b[static_cast<std::size_t>(i) - 1]) { b[static_cast<std::size_t>(i)] = b[static_cast<std::size_t>(i) - 1]; continue; }
        is.clear(); is.seekg(static_cast<std::streamoff>(pos));
        std::size_t adv = 0; bool found = false;                   // scan forward to the next '\n'
        while (!found) {
            is.read(blk.data(), static_cast<std::streamsize>(blk.size()));
            const std::streamsize got = is.gcount();
            if (got <= 0) break;
            for (std::streamsize k = 0; k < got; ++k) { ++adv; if (blk[static_cast<std::size_t>(k)] == '\n') { found = true; break; } }
        }
        std::size_t boundary = found ? std::min(pos + adv, fsz) : fsz;   // just past the newline
        if (boundary < b[static_cast<std::size_t>(i) - 1]) boundary = b[static_cast<std::size_t>(i) - 1];
        b[static_cast<std::size_t>(i)] = boundary;
    }
    for (int i = 1; i <= nseg; ++i)
        if (b[static_cast<std::size_t>(i)] < b[static_cast<std::size_t>(i) - 1]) b[static_cast<std::size_t>(i)] = b[static_cast<std::size_t>(i) - 1];
    return b;
}

// Run `body(t, start, end)` on each of `nseg` newline-aligned segments concurrently, one
// std::thread per segment (the configurator links no OpenMP -- it is a standalone host tool).
template <class Body>
void parallel_segments(const std::vector<std::size_t>& bnd, int nseg, Body&& body) {
    std::vector<std::thread> th;
    th.reserve(static_cast<std::size_t>(nseg));
    for (int t = 0; t < nseg; ++t)
        th.emplace_back([&, t] {
            body(t, bnd[static_cast<std::size_t>(t)], bnd[static_cast<std::size_t>(t) + 1]);
        });
    for (auto& x : th) x.join();
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
    int has_cuda    = 0;   // CUDA toolkit + device detected at configure time
    int cuda_arch   = 0;   // GPU compute capability as an int (e.g. 120 for sm_120)
    int gpu_vram_mb = 0;   // dedicated GPU VRAM in MB
    int gpu_shared_mb = 0; // shared/overflow system memory the GPU can address (WDDM), MB
    int compute     = 0;   // resolved backend: 0=CPU, 1=GPU, 2=HYBRID
    int cuda_tf32   = 0;   // bake TF32 tensor-core GEMM math on the GPU backend (tuned knob)

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
    app.add_option("--has-cuda", has_cuda, "1 = a CUDA toolkit + device was detected at configure time")
       ->capture_default_str()->check(CLI::Range(0, 1));
    app.add_option("--cuda-arch", cuda_arch,
                   "Detected GPU compute capability as an int (e.g. 120 for sm_120; 0 = none)")
       ->capture_default_str();
    app.add_option("--gpu-vram-mb", gpu_vram_mb, "Detected dedicated GPU VRAM in MB (0 = none)")
       ->capture_default_str();
    app.add_option("--gpu-shared-mb", gpu_shared_mb,
                   "Shared/overflow system memory the GPU can address in MB (WDDM; 0 = none)")
       ->capture_default_str();
    app.add_option("--compute", compute, "Resolved compute backend: 0=CPU, 1=GPU, 2=HYBRID")
       ->capture_default_str()->check(CLI::Range(0, 2));
    app.add_option("--cuda-tf32", cuda_tf32, "Bake TF32 tensor-core GEMM math on (1) or off (0)")
       ->capture_default_str()->check(CLI::Range(0, 1));

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
        // The corpus scan (passes 1-2) is the cost that scales with the corpus and is
        // CPU-bound (normalize + truecase + table build, far below disk bandwidth), so it is
        // run DATA-PARALLEL: the file is split into newline-aligned segments (one per worker),
        // each worker scans its own segment into THREAD-LOCAL tables, and the tables are merged
        // afterwards. Correctness rests on two facts: a newline never splits a unit/look-back
        // (so per-segment results equal whole-file results), and every downstream artifact is
        // INVARIANT to word-table ordering -- the base alphabet is byte-ordered, and BPE picks
        // each merge by (count, pair) where counts are order-independent sums and pair ids are
        // byte-derived. So the merged tables (hence tokenizer.bin / corpus.tok) are byte-
        // identical to the single-thread scan regardless of how the corpus is partitioned.
        std::error_code fec;
        const std::size_t fsz = static_cast<std::size_t>(std::filesystem::file_size(corpus, fec));
        if (fec || fsz == 0) { std::println(stderr, "configure error: empty corpus"); return 1; }
        const unsigned hw = std::max(1u, std::thread::hardware_concurrency());
        // >= ~16 MB per segment so the merge/thread overhead is amortised; capped at the cores.
        int nseg = static_cast<int>(std::min<std::size_t>(hw, std::max<std::size_t>(1, fsz / (16u << 20))));
        if (nseg < 1) nseg = 1;
        const std::vector<std::size_t> bnd = segment_bounds(corpus, fsz, nseg);

        // --- Pass 1 (parallel): decide which lowercase forms license a Capitalized/UPPER ->
        //     marker collapse. The tell is *position*: a capital following a lowercase word
        //     mid-sentence ("...to Spot") is a name; per lowercase form we count lowercase
        //     uses vs mid-sentence-capital uses, withholding name-dominant forms from `attested`.
        //     Each worker accumulates local name counts + byte tallies; we sum them after.
        struct P1Local {
            std::unordered_map<std::string, long long> lower_count, midcap_count;
            std::size_t raw = 0, norm = 0; long long qr = 0; bool ok = false;
        };
        std::vector<P1Local> p1(static_cast<std::size_t>(nseg));
        parallel_segments(bnd, nseg, [&](int t, std::size_t a, std::size_t b) {
            P1Local& L = p1[static_cast<std::size_t>(t)];
            L.ok = for_each_chunk_range(corpus, a, b, [&](std::string_view chunk) {
                L.raw += chunk.size();
                long qr = 0;
                const std::string norm = normalize_text(std::string(chunk), qr);
                L.qr += qr;
                L.norm += norm.size();
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
                        L.lower_count[w] += 1;
                    } else if (is_upper(static_cast<unsigned char>(w[0])) && rest_lower) {  // "Spot", "The"
                        if (preceded_by_lowercase(i)) {
                            std::string lw = w;
                            lw[0] = static_cast<char>(to_lower(static_cast<unsigned char>(lw[0])));
                            L.midcap_count[lw] += 1;
                        }
                    }
                    i = j;
                }
            });
        });
        for (P1Local& L : p1) {
            if (!L.ok) { std::println(stderr, "configure error: cannot read corpus '{}'", corpus); return 1; }
            S.raw_bytes += L.raw; S.norm_bytes += L.norm; S.quote_repl += L.qr;
            for (const auto& [w, c] : L.lower_count)  S.lower_count[w]  += c;   // sums commute -> order-free
            for (const auto& [w, c] : L.midcap_count) S.midcap_count[w] += c;
            L.lower_count = {}; L.midcap_count = {};                            // free as we merge
        }
        if (S.raw_bytes == 0) { std::println(stderr, "configure error: empty corpus"); return 1; }
        derive_attested();
        _t1 = std::chrono::steady_clock::now();

        // --- Pass 2 (parallel): truecase each chunk + pre-tokenize into the unique-word table.
        //     A word unit (run of word bytes per word_unit_end) becomes one table entry keyed by
        //     its RAW byte-symbol sequence (independent of the not-yet-fixed base alphabet);
        //     punctuation, spaces and the markers are standalone symbols that never merge. Each
        //     worker builds a local table; we fold the locals into the global one afterwards.
        struct P2Local {
            std::unordered_map<std::string, int> index;
            std::vector<std::vector<int>> word_syms;
            std::vector<long long> word_freq;
            std::array<int, 256> byte_used{};
            bool used_cap = false, used_up = false;
            sub0::casing::TokStats st;
        };
        std::vector<P2Local> p2(static_cast<std::size_t>(nseg));
        parallel_segments(bnd, nseg, [&](int t, std::size_t a, std::size_t b) {
            P2Local& L = p2[static_cast<std::size_t>(t)];
            for_each_chunk_range(corpus, a, b, [&](std::string_view chunk) {
                long qr = 0;
                const std::string  norm   = normalize_text(std::string(chunk), qr);
                const std::vector<int> stream = truecase_tokenize(norm, attested, &L.st);
                for (std::size_t i = 0, n = stream.size(); i < n;) {
                    const std::size_t end = word_unit_end(stream, i);
                    if (end == i) {                                  // standalone symbol
                        const int s = stream[i];
                        if (s == TOK_CAP)      L.used_cap = true;
                        else if (s == TOK_UP)  L.used_up = true;
                        else                   L.byte_used[s] = 1;
                        ++i; continue;
                    }
                    // Build the lookup key directly from the byte run (1 byte/symbol, usually in
                    // std::string's small buffer -> no heap), and only allocate the word's symbol
                    // vector / mark its bytes on a CACHE MISS. Repeated words (the vast majority of
                    // occurrences) then cost just a key build + a counter bump -- no per-occurrence
                    // vector allocation.
                    std::string key(end - i, '\0');
                    for (std::size_t k = i; k < end; ++k) key[k - i] = static_cast<char>(stream[k] & 0xFF);
                    auto it = L.index.find(key);
                    if (it == L.index.end()) {
                        std::vector<int> seq(stream.begin() + static_cast<std::ptrdiff_t>(i),
                                             stream.begin() + static_cast<std::ptrdiff_t>(end));
                        for (int bb : seq) L.byte_used[bb] = 1;
                        L.index.emplace(std::move(key), static_cast<int>(L.word_syms.size()));
                        L.word_syms.push_back(std::move(seq));
                        L.word_freq.push_back(1);
                    } else {
                        L.word_freq[static_cast<std::size_t>(it->second)] += 1;
                    }
                    i = end;
                }
            });
        });
        // Fold the worker tables into the global one. word_index ids are assigned in merge order
        // (immaterial to the outputs); per-word frequencies are summed across workers.
        for (P2Local& L : p2) {
            for (int bb = 0; bb < 256; ++bb) if (L.byte_used[bb]) S.byte_used[bb] = 1;
            S.used_cap = S.used_cap || L.used_cap;
            S.used_up  = S.used_up  || L.used_up;
            S.st.words += L.st.words; S.st.cap += L.st.cap; S.st.up += L.st.up; S.st.names += L.st.names;
            for (std::size_t i = 0; i < L.word_syms.size(); ++i) {
                const std::string k = seq_key(L.word_syms[i]);
                auto it = word_index.find(k);
                if (it == word_index.end()) {
                    word_index.emplace(k, static_cast<int>(S.word_syms.size()));
                    S.word_syms.push_back(std::move(L.word_syms[i]));
                    S.word_freq.push_back(L.word_freq[i]);
                } else {
                    S.word_freq[static_cast<std::size_t>(it->second)] += L.word_freq[i];
                }
            }
            L.index = {}; L.word_syms = {}; L.word_freq = {};                  // free as we merge
        }
        _t2 = std::chrono::steady_clock::now();
        save_scan_state(cache_path, corpus, S);
        std::println(stderr, "scan cache: saved '{}' ({} words, {} segments)", cache_path, S.word_syms.size(), nseg);
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
    int            default_gpu_batch    = 0;   // 0 -> derive from the CPU width below
    int            attn_bwd_per_query   = 0;   // GPU attention-backward strategy (0=per-head)
    bool           tf32_from_cache      = false;
    std::string    tune_cache_abs;
    if (!tune_cache.empty()) {
        tune_cache_abs = std::filesystem::absolute(tune_cache).string();
        std::ifstream tc(tune_cache);
        for (std::string line; std::getline(tc, line); ) {
            const auto eq = line.find('=');
            if (eq == std::string::npos) continue;
            const std::string key = line.substr(0, eq);
            const int         val = std::atoi(line.substr(eq + 1).c_str());
            if      (key == "threads"            && val > 0) default_threads   = val;
            else if (key == "windows_per_thread" && val > 0) default_wpt       = val;
            else if (key == "gpu_batch"          && val > 0) default_gpu_batch = val;
            else if (key == "attn_bwd_per_query")            attn_bwd_per_query = (val != 0);
            else if (key == "cuda_tf32") { cuda_tf32 = (val != 0); tf32_from_cache = true; }
        }
    }
    // The tuned GPU batch defaults to the CPU data-parallel width until `tune` measures one.
    if (default_gpu_batch <= 0) default_gpu_batch = default_threads * default_wpt;
    (void)tf32_from_cache;   // (cache value already folded into cuda_tf32 above)

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
    os << "constexpr int  DEFAULT_WINDOWS_PER_THREAD = " << default_wpt    << ";\n";
    os << "// DEFAULT_GPU_BATCH: the tuned device-training minibatch (throughput knob; `tune` sweeps\n";
    os << "// batch and persists the knee). Defaults to the CPU data-parallel width until tuned.\n";
    os << "constexpr int  DEFAULT_GPU_BATCH         = " << default_gpu_batch << ";\n\n";
    // --- Compute backend (resolved at configure time) ----------------------
    // The host build detects the CUDA toolkit + device and picks ONE backend; the
    // engine compiles only that path. HAS_CUDA records availability (so AUTO can flip
    // to GPU once the device backend lands); COMPUTE_MODE is the backend actually built.
    os << "// --- Compute backend (resolved at configure time) ----------------------\n";
    os << "enum class ComputeBackend { Cpu, Gpu, Hybrid };\n";
    os << "constexpr bool           HAS_CUDA     = " << (has_cuda ? "true" : "false") << ";\n";
    os << "constexpr ComputeBackend COMPUTE_MODE = ComputeBackend::"
       << (compute == 1 ? "Gpu" : compute == 2 ? "Hybrid" : "Cpu") << ";\n";
    os << "constexpr int            CUDA_ARCH    = " << cuda_arch   << ";\n";
    os << "constexpr int            GPU_VRAM_MB  = " << gpu_vram_mb << ";  // dedicated VRAM (fast budget)\n";
    os << "// GPU_SHARED_MEM_MB: system RAM the GPU may address as overflow (WDDM shared memory).\n";
    os << "// Exceeding GPU_VRAM_MB does NOT OOM on Windows -- it spills here and THRASHES over PCIe;\n";
    os << "// the same region can also back zero-copy / mapped host transfers. Treat VRAM as the hard\n";
    os << "// fast budget and (VRAM + shared) as the soft ceiling. 0 where there is no WDDM shared mem.\n";
    os << "constexpr int            GPU_SHARED_MEM_MB = " << gpu_shared_mb << ";\n";
    os << "// CUDA_TF32: bake TF32 tensor-core GEMM math (an autotuner knob; measured to give no\n";
    os << "// win at this model's small-K GEMMs, so default off). A SUB0_TUNING build overrides it\n";
    os << "// at runtime via the sub0::Knob mechanism; otherwise it is a baked compile-time constant.\n";
    os << "constexpr bool           CUDA_TF32    = " << (cuda_tf32 ? "true" : "false") << ";\n";
    os << "// ATTN_BWD_PER_QUERY: GPU attention-backward strategy (0=per-head, best at large batch;\n";
    os << "// 1=per-query, best at small batch). A tuned knob; `tune` persists the measured winner.\n";
    os << "constexpr bool           ATTN_BWD_PER_QUERY = " << (attn_bwd_per_query ? "true" : "false") << ";\n\n";
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
