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
//   - tokenizer.tok    the runtime tokenizer (base alphabet + merges + attested
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
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <filesystem>
#include <format>
#include <fstream>
#include <mutex>
#include <print>
#include <queue>
#include <map>
#include <string>
#include <string_view>
#include <system_error>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#if defined(_WIN32)
  #define NOMINMAX
  #define WIN32_LEAN_AND_MEAN
  #include <windows.h>   // GlobalMemoryStatusEx -- total_physical_ram_bytes() below
#else
  #include <unistd.h>    // sysconf -- total_physical_ram_bytes() below
#endif

#include <CLI/CLI.hpp>

#include "sub0/memplan.hpp"  // train_resident_mb: predict the GPU batch footprint to fail misconfig early

#include "sub0/casing.hpp"
#include "sub0/tokenizer.hpp"  // sub0::tok — the shared truecasing + BPE tokenizer (scan/learn/serialize)
#include "sub0/unigram.hpp"    // sub0::tok::learn_unigram — the Unigram LM vocabulariser (A/B vs BPE)
#include "sub0/tokmap.hpp"     // sub0::TokWriter — v2 corpus.tok (u64 counts + byte-aligned token width)
#include "sub0/config_util.hpp"// sub0::config — pure, unit-tested config decisions (autosize / tune-cache / precision)
#include "sub0/registry.hpp"  // registry::describe_config — the X-macro-driven config summary shared
                            // with config.json/state.json, so this banner cannot omit an axis
#include "sub0_build_facts.hpp" // CMake-baked build facts (device caps + output paths) -> the CLI defaults

namespace tok = sub0::tok;

namespace {

// Total physical RAM in bytes (0 if it cannot be determined -- should_pretokenize treats that as
// "don't block on it"). Only consumer is the --corpus-pretok AUTO resolution below.
std::uintmax_t total_physical_ram_bytes() {
#if defined(_WIN32)
    MEMORYSTATUSEX st{}; st.dwLength = sizeof(st);
    return GlobalMemoryStatusEx(&st) ? static_cast<std::uintmax_t>(st.ullTotalPhys) : 0;
#else
    const long pages = sysconf(_SC_PHYS_PAGES), page_size = sysconf(_SC_PAGE_SIZE);
    return (pages > 0 && page_size > 0)
        ? static_cast<std::uintmax_t>(pages) * static_cast<std::uintmax_t>(page_size) : 0;
#endif
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

// Bounded single-producer/single-consumer queue of ready-to-encode batches (Pass 3's IO/compute
// overlap): a dedicated reader thread fills it straight off disk while the main thread drains it
// (parallel encode + sequential write), so reading batch N+1 runs concurrently with encoding +
// writing batch N -- previously strictly sequential (read all of N, THEN encode+write it, THEN read
// N+1), so the disk sat idle during every encode/write phase and the CPU sat idle during every read.
// Capacity 2 double-buffers (one batch draining, one prefetched) without unbounded memory growth.
template <class T>
class BoundedQueue {
public:
    explicit BoundedQueue(std::size_t cap) : cap_(cap) {}
    void push(T item) {
        std::unique_lock<std::mutex> lk(m_);
        not_full_.wait(lk, [&] { return q_.size() < cap_; });
        q_.push_back(std::move(item));
        lk.unlock();
        not_empty_.notify_one();
    }
    // Blocks for the next item; returns false once the producer has closed() and the queue is
    // drained (the "no more work" signal -- distinct from a momentarily-empty-but-still-open queue).
    bool pop(T& out) {
        std::unique_lock<std::mutex> lk(m_);
        not_empty_.wait(lk, [&] { return !q_.empty() || done_; });
        if (q_.empty()) return false;
        out = std::move(q_.front());
        q_.pop_front();
        lk.unlock();
        not_full_.notify_one();
        return true;
    }
    void close() {
        { std::lock_guard<std::mutex> lk(m_); done_ = true; }
        not_empty_.notify_all();
    }
private:
    std::mutex m_;
    std::condition_variable not_full_, not_empty_;
    std::deque<T> q_;
    std::size_t cap_;
    bool done_ = false;
};

// Rename tmp -> dst, replacing any existing file at dst. So a crash/kill mid-write never leaves a
// truncated corpus.tok / tokenizer.tok behind (the readable file at `dst` is always either the old
// complete one or the new complete one, never a partial write) -- mirrors train_stage.cpp's
// atomic_replace. This is also what makes a model dir's HARDLINKED pin of these files safe: a rename
// repoints ONLY the `dst` directory entry to the new data, leaving any other hardlink (e.g. the one
// train_stage.cpp pins into a model directory before it starts reading) pointing at the OLD data,
// untouched -- an in-place truncate+rewrite at a fixed path would not have that property.
void atomic_replace(const std::filesystem::path& tmp, const std::filesystem::path& dst) {
    std::error_code ec;
    std::filesystem::rename(tmp, dst, ec);
    if (ec) {  // some platforms (Windows) won't clobber an existing target
        std::filesystem::remove(dst, ec);
        std::filesystem::rename(tmp, dst, ec);
    }
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
constexpr std::uint32_t WCACHE_VERSION = 2u;

// The cacheable scan state is sub0::tok::Scan (the same struct the parallel passes
// accumulate into); these helpers persist its fields next to the corpus. The derived
// `index` is not stored — it is rebuilt from word_syms on load.
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
void save_scan_state(const std::string& path, const std::string& corpus, const sub0::tok::Scan& S) {
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
    // modality ledger (schemeV4, D2): per-codepoint SS/SG/GS/GG -> the corpus-derived glue table. Cached
    // WITH the scan so a cache HIT bakes the SAME tokenizer.tok as a full scan (deterministic).
    wr(os, static_cast<std::uint64_t>(S.modality.scanned_bytes));
    wr(os, static_cast<std::uint64_t>(S.modality.chars.size()));
    for (const auto& [cp, cm] : S.modality.chars) {
        wr(os, static_cast<std::uint32_t>(cp));
        for (int k = 0; k < 4; ++k) wr(os, static_cast<std::uint64_t>(cm.n[static_cast<std::size_t>(k)]));
    }
}

// Load + validate against the corpus (size+mtime+version). Returns false (leaving S untouched-
// enough to fall back to a full scan) if absent, stale or malformed. The `index` is rebuilt
// from word_syms by the caller (Scan::rebuild_index).
bool load_scan_state(const std::string& path, const std::string& corpus, sub0::tok::Scan& S) {
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
    // modality ledger (schemeV4, D2), cached with the scan (see save_scan_state).
    S.modality = sub0::modality::ModalityStats{};
    S.modality.scanned_bytes = rd<std::uint64_t>(is);
    const std::uint64_t nch = rd<std::uint64_t>(is);
    for (std::uint64_t i = 0; i < nch; ++i) {
        const std::uint32_t cp = rd<std::uint32_t>(is);
        sub0::modality::CharModality cm{};
        for (int k = 0; k < 4; ++k) cm.n[static_cast<std::size_t>(k)] = rd<std::uint64_t>(is);
        S.modality.chars[cp] = cm;
    }
    return static_cast<bool>(is);
}

// --- Tokenizer/corpus.tok reuse stamp -------------------------------------------------------
// The scan cache above (.words) only skips passes 1-2; the vocab LEARN (EM/prune) and the full
// corpus.tok tokenize pass (Pass 3) are far more expensive for a large corpus, and both are pure
// functions of exactly four things beyond the corpus bytes themselves: vocab_target, min_merge,
// whether corpus.tok was requested at all, and casing::kSchemeVersion (the encode_join/
// detokenize_join TRANSITION RULES -- e.g. schemeV3's 2-piece-word SPELL-wrap change actually
// changes corpus.tok's bytes for the SAME vocab; an earlier version of this stamp assumed "the
// tokenizer scheme is always JOIN, so it is not a variable", which was only ever true of the
// MARKER FAMILY, not its transition rules, and would have silently kept serving a stale
// wrong-scheme corpus.tok across a scheme bump with no warning). When those and the corpus are
// unchanged since the last configure run, and tokenizer.tok/corpus.tok already exist and parse,
// re-deriving the SAME tokenizer and re-encoding the SAME corpus into the SAME bytes is pure
// waste -- for a FineWeb-scale corpus that waste dominates a `sub0llm-configure` re-run made only
// to change an unrelated knob (precision, dims, tune-cache). Stamped next to tokenizer.tok,
// validated the same way as the scan cache (corpus size+mtime+version) plus the learn parameters.
constexpr std::uint32_t TOKSTAMP_MAGIC   = 0x53543053u;  // "S0TS"
constexpr std::uint32_t TOKSTAMP_VERSION = 2u;   // 1->2: added casing::kSchemeVersion (see above)

void save_tok_stamp(const std::string& path, const std::string& corpus,
                     int vocab_target, int min_merge, int emit_tok) {
    std::ofstream os(path, std::ios::binary);
    if (!os) { std::println(stderr, "warning: cannot write tokenizer stamp '{}'", path); return; }
    std::error_code ec;
    wr(os, TOKSTAMP_MAGIC); wr(os, TOKSTAMP_VERSION);
    wr(os, static_cast<std::uint64_t>(std::filesystem::file_size(corpus, ec)));
    wr(os, static_cast<std::int64_t>(std::filesystem::last_write_time(corpus, ec).time_since_epoch().count()));
    wr(os, static_cast<std::int32_t>(vocab_target));
    wr(os, static_cast<std::int32_t>(min_merge));
    wr(os, static_cast<std::int32_t>(emit_tok));
    wr(os, sub0::casing::kSchemeVersion);
}

// True iff the stamp at `path` was written for this exact (corpus size+mtime, vocab_target,
// min_merge, emit_tok, kSchemeVersion) combination. Absent, stale or malformed -> false (fall back
// to a full run). TOKSTAMP_VERSION itself bumped alongside adding the kSchemeVersion field, so a
// stamp written by an older configurator binary (which never wrote that field) fails the very
// first check here rather than being misread.
bool tok_stamp_matches(const std::string& path, const std::string& corpus,
                        int vocab_target, int min_merge, int emit_tok) {
    std::ifstream is(path, std::ios::binary);
    if (!is) return false;
    if (rd<std::uint32_t>(is) != TOKSTAMP_MAGIC || rd<std::uint32_t>(is) != TOKSTAMP_VERSION) return false;
    std::error_code ec;
    if (rd<std::uint64_t>(is) != static_cast<std::uint64_t>(std::filesystem::file_size(corpus, ec))) return false;
    if (rd<std::int64_t>(is) !=
        static_cast<std::int64_t>(std::filesystem::last_write_time(corpus, ec).time_since_epoch().count()))
        return false;
    if (rd<std::int32_t>(is) != static_cast<std::int32_t>(vocab_target)) return false;
    if (rd<std::int32_t>(is) != static_cast<std::int32_t>(min_merge)) return false;
    if (rd<std::int32_t>(is) != static_cast<std::int32_t>(emit_tok)) return false;
    if (rd<std::uint32_t>(is) != sub0::casing::kSchemeVersion) return false;
    return static_cast<bool>(is);
}

// Readable vocabulary-analysis dumps (--dump-vocab). Three files for reviewing how the tokenizer
// compacts the corpus and whether the vocab size / BPE merges are well spent:
//   <prefix>.corpus_vocab.txt — every unique word-unit (truecased) + occurrence count, freq-desc.
//   <prefix>.token_vocab.txt  — every learned token (base byte / marker / merge), its text, and the
//                               corpus occurrences it covers (so over/under-used tokens + tail waste
//                               are visible at a glance).
//   <prefix>.ngrams.txt       — k-char substring frequencies (k=2..16) over the corpus vocab, weighted
//                               by word count: the common "tion"/"ing"/"ience" chunks a sub-token
//                               scheme would target, independent of the BPE merge ORDER. The basis for
//                               a ground-up "longest common sub-section" vocabulariser.
void dump_vocab_files(const tok::Scan& S, const tok::BpeAnalysisVocab& tkz, const std::string& prefix) {
    auto vis = [](const std::string& k) {
        std::string o;
        for (unsigned char c : k) {
            if (c >= 32 && c < 127) o += static_cast<char>(c);
            else { char b[8]; std::snprintf(b, sizeof b, "\\x%02x", c); o += b; }
        }
        return o;
    };
    auto by_count_desc = [](const auto& a, const auto& b) { return a.first > b.first; };

    // 1. Corpus vocabulary: (count, word) freq-desc.
    std::vector<std::pair<long long, std::string>> cv;
    cv.reserve(S.index.size());
    long long total_occ = 0;
    for (const auto& [key, id] : S.index) {
        const long long f = S.word_freq[static_cast<std::size_t>(id)];
        cv.push_back({f, key});
        total_occ += f;
    }
    std::sort(cv.begin(), cv.end(), by_count_desc);
    {
        std::ofstream os(prefix + ".corpus_vocab.txt");
        std::println(os, "# corpus vocabulary: {} unique word-units, {} total occurrences (truecased; count<TAB>word, freq-desc)",
                     cv.size(), total_occ);
        for (const auto& [c, w] : cv) std::println(os, "{}\t{}", c, vis(w));
    }

    // 2. Token vocabulary: per learned token, the corpus occurrences it covers (sum of word counts
    //    over every word whose final id-sequence includes it -- S.word_syms holds those final ids).
    {
        std::vector<long long> occ(static_cast<std::size_t>(tkz.vocab), 0);
        for (std::size_t w = 0; w < S.word_syms.size(); ++w) {
            const long long f = S.word_freq[w];
            for (int tid : S.word_syms[w]) if (tid >= 0 && tid < tkz.vocab) occ[static_cast<std::size_t>(tid)] += f;
        }
        std::ofstream os(prefix + ".token_vocab.txt");
        std::println(os, "# token vocabulary: {} tokens (base {} + {} merges); id<TAB>kind<TAB>occurrences<TAB>text",
                     tkz.vocab, tkz.n_base, tkz.merges.size());
        for (int id = 0; id < tkz.vocab; ++id) {
            std::string text;
            for (int code : tkz.expansion[static_cast<std::size_t>(id)])
                text += (code >= 0 && code < 256) ? vis(std::string(1, static_cast<char>(code)))
                                                  : std::format("<{}>", code);
            const char* kind = id >= tkz.n_base ? "merge"
                             : (tkz.base_symbol[static_cast<std::size_t>(id)] < 256 ? "byte" : "marker");
            std::println(os, "{}\t{}\t{}\t{}", id, kind, occ[static_cast<std::size_t>(id)], text);
        }
    }

    // 3. k-char substring (n-gram) frequencies over the corpus vocab, weighted by word count. This is
    //    the order-independent view of the common sub-chunks (BPE's merges depend on greedy order;
    //    this does not), the input a "minimise tokens by occurrence" search would consume.
    {
        std::unordered_map<std::string, long long> ng;
        for (const auto& [key, id] : S.index) {
            const long long f = S.word_freq[static_cast<std::size_t>(id)];
            const int n = static_cast<int>(key.size());
            const int kmax = std::min(n, 16);
            for (int k = 2; k <= kmax; ++k)
                for (int i = 0; i + k <= n; ++i) ng[key.substr(static_cast<std::size_t>(i), static_cast<std::size_t>(k))] += f;
        }
        std::vector<std::pair<long long, std::string>> v;
        v.reserve(ng.size());
        for (const auto& [s, c] : ng) v.push_back({c, s});
        std::sort(v.begin(), v.end(), by_count_desc);
        std::ofstream os(prefix + ".ngrams.txt");
        std::println(os, "# {} distinct k-char substrings (k=2..16) over the corpus vocab, freq-weighted; count<TAB>len<TAB>substring, desc (top 30000)",
                     v.size());
        for (std::size_t i = 0; i < v.size() && i < 30000; ++i)
            std::println(os, "{}\t{}\t{}", v[i].first, v[i].second.size(), vis(v[i].second));
    }

    // 4. Vocab-size curve + ideal-size detection. Each BPE merge removes merge_count[i] corpus tokens,
    //    so total_word_tokens(n_base+k) = total_word_bytes - sum_{i<k} merge_count[i] gives the WHOLE
    //    curve from ONE learn. As vocab collapses toward the base alphabet, bytes/token -> 1 (character
    //    encoding -- the "devolves to char" floor); as it grows the curve flattens (diminishing
    //    returns). The "ideal" vocab is the knee, reported as the size capturing X% of the total
    //    achievable token reduction. Run --dump-vocab with a generous --vocab to extend the curve past
    //    the knee.
    {
        long long total_bytes = 0;
        for (const auto& [key, id] : S.index) total_bytes += S.word_freq[static_cast<std::size_t>(id)] * static_cast<long long>(key.size());
        const std::vector<long long>& mc = tkz.merge_count;
        const int nm = static_cast<int>(mc.size());
        std::vector<long long> cum(static_cast<std::size_t>(nm) + 1, 0);
        for (int i = 0; i < nm; ++i) cum[static_cast<std::size_t>(i) + 1] = cum[static_cast<std::size_t>(i)] + mc[static_cast<std::size_t>(i)];
        const long long total_reduction = cum[static_cast<std::size_t>(nm)];
        auto tokens_at = [&](int k) { return total_bytes - cum[static_cast<std::size_t>(std::min(k, nm))]; };
        auto bpt_at    = [&](int k) { return sub0::config::bytes_per_token_at(total_bytes, mc, k); };
        auto vocab_at_frac = [&](double frac) { return sub0::config::vocab_at_fraction(total_reduction, mc, tkz.n_base, frac); };
        {
            std::ofstream os(prefix + ".vocab_curve.txt");
            std::println(os, "# vocab-size curve (word encoding): total_word_bytes={}, base_vocab={}, max_vocab={}",
                         total_bytes, tkz.n_base, tkz.n_base + nm);
            std::println(os, "# vocab<TAB>total_word_tokens<TAB>bytes_per_token<TAB>tokens_saved_by_this_merge");
            std::vector<int> ks{0};
            for (int k = 1; k <= nm; k *= 2) ks.push_back(k);
            if (ks.back() != nm) ks.push_back(nm);
            for (int k : ks)
                std::println(os, "{}\t{}\t{:.3f}\t{}", tkz.n_base + k, tokens_at(k), bpt_at(k),
                             (k > 0 && k <= nm) ? tkz.merge_count[static_cast<std::size_t>(k - 1)] : 0);
        }
        std::println(stderr,
            "vocab curve: char-level bytes/token=1.00 -> full vocab {} bytes/token={:.2f}. "
            "Ideal vocab (knee): 90% of compression @ ~{}, 95% @ ~{}, 99% @ ~{} "
            "(diminishing returns beyond -- raise --vocab to extend the curve).",
            tkz.n_base + nm, bpt_at(nm), vocab_at_frac(0.90), vocab_at_frac(0.95), vocab_at_frac(0.99));
    }

    // 5. Unigram-vs-BPE A/B: learn a Unigram LM vocab at the SAME size and compare word compression.
    //    BPE word-tokens = sum freq*len(final id-seq); Unigram = sum freq*Viterbi-seg length. Fewer =
    //    better. Unigram also has NO dead tokens (every kept piece has count>0), unlike BPE.
    {
        std::vector<std::pair<std::string, long long>> words;
        words.reserve(S.index.size());
        for (const auto& [key, id] : S.index) words.push_back({key, S.word_freq[static_cast<std::size_t>(id)]});

        // BPE word-token total (S.word_syms holds the final id sequences after learn) and corpus bytes.
        long long bpe_tok = 0, total_bytes = 0;
        for (std::size_t w = 0; w < S.word_syms.size(); ++w)
            bpe_tok += static_cast<long long>(S.word_syms[w].size()) * S.word_freq[w];
        for (const auto& [k, f] : words) total_bytes += static_cast<long long>(k.size()) * f;
        // BPE dead merges (never appear in any final word encoding).
        std::vector<char> used(static_cast<std::size_t>(tkz.vocab), 0);
        for (const auto& seq : S.word_syms) for (int id : seq) if (id >= 0 && id < tkz.vocab) used[static_cast<std::size_t>(id)] = 1;
        long long bpe_dead = 0;
        for (int id = tkz.n_base; id < tkz.vocab; ++id) if (!used[static_cast<std::size_t>(id)]) ++bpe_dead;

        sub0::tok::UnigramOptions uo;
        uo.target = tkz.vocab;
        uo.em_iters = 8;          // more EM passes + a gentler prune -> a tighter fit before integration
        uo.drop_frac = 0.1;
        const sub0::tok::Unigram u = sub0::tok::learn_unigram(words, uo);
        long long ub = 0;
        const long long uni_tok = sub0::tok::corpus_tokens(u, words, &ub);

        std::println(stderr,
            "Unigram A/B @ vocab~{}: BPE bytes/token={:.3f} ({} word-tokens, {} dead merges) | "
            "Unigram {} tokens bytes/token={:.3f} ({} word-tokens, 0 dead) -> {:+.1f}% tokens vs BPE",
            tkz.vocab, static_cast<double>(total_bytes) / static_cast<double>(bpe_tok), bpe_tok, bpe_dead,
            u.size(), static_cast<double>(total_bytes) / static_cast<double>(uni_tok), uni_tok,
            100.0 * (static_cast<double>(uni_tok) / static_cast<double>(bpe_tok) - 1.0));

        std::ofstream os(prefix + ".unigram_vocab.txt");
        std::println(os, "# Unigram vocabulary: {} tokens; id<TAB>len<TAB>logp<TAB>text", u.size());
        for (int id = 0; id < u.size(); ++id)
            std::println(os, "{}\t{}\t{:.4f}\t{}", id, u.token[static_cast<std::size_t>(id)].size(),
                         u.logp[static_cast<std::size_t>(id)], vis(u.token[static_cast<std::size_t>(id)]));
    }
}

}  // namespace

using namespace sub0::casing;

int main(int argc, char** argv) {
    CLI::App app{"sub0-configure: truecase + BPE a corpus, emit the tokenized corpus and a constexpr config header"};

    std::string corpus;
    std::string out = sub0::build_facts::GEN_HEADER;   // default: the build's generated umbrella header
    int d_model      = 0;       // 0 = auto-size from corpus scale (autosize_dims); any nonzero pins it
    int n_layers     = 0;
    int n_heads      = 0;
    int n_kv_heads   = 0;        // 0 = same as n_heads (plain MHA); < n_heads selects GQA
    int loop_middle  = 0;        // LoopSplit: middle-block size (0 = no looping)
    int depth_attn_stride = 0;   // Depth attention: cache-append stride (0 = off); see docs/DEPTH_ATTENTION.md
    int loop_repeats = 1;        // LoopSplit: how many times the middle block runs
    // Gated DeltaNet: Stage 1 -- CPU forward exists (op_gdn / gdn_forward_one, backend_cpu.cpp),
    // correctness-gated against the real Qwen4-preview fixtures (tests/gdn_qwen4_fixture_tests.cpp).
    // No backward pass yet (backward_node aborts loudly on an Op::GDN node -- see its comment) and no
    // CUDA (backend_cuda.cu keeps its own static_assert) -- so a GDN build works for gen/eval/report but
    // NOT for train/tune. See docs/GATED_DELTANET.md.
    int gdn_full_attn_stride = 0;
    int ngram_max_n        = 0;  // N-gram embeddings: highest n-gram order (0 = off, else >= 2); see docs/NGRAM_EMBEDDING.md
    int ngram_tables       = 1;  // N-gram embeddings: hash tables per order (k)
    int ngram_table_size   = 0;  // N-gram embeddings: per-table vocab size m (0 = auto: VOCAB)
    // Gated Residual (hyper-connections): Stage 1 -- CPU forward only (docs/GATED_RESIDUAL.md).
    // HC_COUNT parallel residual streams read/write-gated per sub-block. 0 = off, the default.
    int hc_count           = 0;
    int hc_lowrank         = 0;
    // Mixture of Experts (docs/MOE.md): Stage 0 -- config skeleton, hard-clamped to 0 (off) until Stage 1
    // relaxes the range. NUM_EXPERTS routed experts + 1 always-on shared expert per layer,
    // EXPERTS_PER_TOK selected per token, replacing the FFN block for EVERY layer when on.
    int num_experts        = 0;
    int experts_per_tok    = 0;
    int    rope_scaling      = 0;    // 0 = none, 1 = linear position scaling
    double rope_scale_factor = 1.0;  // linear scaling divisor (context-extension factor)
    int seq_len      = 0;
    int ternary      = 0;
    // The proven architecture stack defaults ON (each is a measured win, verified at production d448 --
    // gated FFN, tied embeddings, QK-norm; project memory arch-features-off-by-default). Pass `--gated-ffn 0`
    // etc. to opt out (e.g. importing a GGUF of a different shape).
    int gated_ffn    = 1;       // 1 = SwiGLU-gated FFN (Wgate/Wup/Wdown, no FFN bias) instead of the
                                 // plain 2-matrix GELU+bias FFN -- matches the GGUF/Llama-family convention
    int tie_embeddings = 1;    // 1 = the LM head reuses tok_emb (transposed) instead of its own
                                 // matrix+bias -- a real param-count win at this project's small-model scale
    int qk_norm      = 1;       // 1 = RMSNorm applied per-head to Q/K right after their projection,
                                 // before RoPE (Gemma2-style) -- stabilizes attention-logit magnitude
    int pos_encoding = 1;       // 0 = absolute learned, 1 = RoPE (default)
    double rope_theta = 10000.0;// RoPE frequency base
    int bf16         = 2;       // float16 capability: 0=off, 1=on, 2=AUTO (on if GPU >= sm_80)
    int prec_gemm    = 9;       // GEMM input precision: 0=F32,1=BF16,2=F16; 9=AUTO (16b if capable)
    int prec_act     = 9;       // saved-activation storage precision: same codes; 9=AUTO
    int vocab_target = 0;       // 0 = auto-size from corpus scale (autosize_dims.vocab); nonzero pins it
    double size_scale = 1.0;    // multiplier on the auto-sizer's target-parameter budget: <1 = smaller/
                                 // faster/safer starting point, >1 = more generous -- same formula
                                 // throughout, just a different point on it (see config_util.hpp)
    int min_merge    = 2;
    int emit_tok     = 2;       // pretokenize corpus.tok: 0=off, 1=on, 2=AUTO (by corpus size vs half RAM)
    std::string tune_cache = sub0::build_facts::GEN_TUNE_CACHE;  // default: the build's tune cache
    int has_cuda    = sub0::build_facts::HAS_CUDA;      // 1 = the CUDA device-training backend was BUILT
    int cuda_arch   = sub0::build_facts::CUDA_ARCH;     // GPU compute capability as an int (e.g. 120 for sm_120)
    int gpu_vram_mb = sub0::build_facts::GPU_VRAM_MB;   // dedicated GPU VRAM in MB
    int gpu_shared_mb = sub0::build_facts::GPU_SHARED_MB; // shared/overflow system memory (WDDM), MB
    int compute     = -1;  // backend to USE: 0=CPU, 1=GPU, 2=HYBRID; -1 = decide from has_cuda
    int cuda_tf32   = 0;   // bake TF32 tensor-core GEMM math on the GPU backend (tuned knob)
    std::string dump_vocab; // prefix for the readable vocabulary-analysis dumps (empty = off)

    app.add_option("--corpus", corpus,  "Training corpus path (drives vocabulary derivation)")
       ->required()->check(CLI::ExistingFile);
    app.add_option("-o",       out,     "Output generated umbrella header path (default: the build's generated dir)")
       ->capture_default_str();
    app.add_option("--dmodel", d_model, "Embedding / residual width (0 = auto-size from corpus)")->capture_default_str();
    app.add_option("--layers", n_layers,"Transformer block count (0 = auto)")->capture_default_str();
    app.add_option("--heads",  n_heads, "Attention head count (0 = auto)")->capture_default_str();
    app.add_option("--kv-heads", n_kv_heads,
                   "Key/value head count for grouped-query attention (0 = AUTO: half of --heads, i.e. "
                   "GQA-2 -- measured +8.4% throughput and -50% KV cache at no measurable quality cost. "
                   "Pass --kv-heads equal to --heads for plain MHA; must divide --heads exactly)")
       ->capture_default_str();
    app.add_option("--loop-middle-layers", loop_middle,
                   "LoopSplit: size of the weight-shared middle block re-executed each pass (0 = off)")
       ->capture_default_str();
    app.add_option("--loop-repeats", loop_repeats,
                   "LoopSplit: how many times the middle block runs (1 = off)")
       ->capture_default_str();
    app.add_option("--depth-attn-stride", depth_attn_stride,
                   "Depth attention: every Nth EXECUTION contributes its K/V to a depth cache that "
                   "later executions mix their value vector over (0 = off). Adds no parameters. "
                   "Memory scales as executions/N, so this is the knob that makes it fit -- see "
                   "docs/DEPTH_ATTENTION.md")
       ->capture_default_str();
    app.add_option("--gdn-full-attn-stride", gdn_full_attn_stride,
                   "Gated DeltaNet (docs/GATED_DELTANET.md): every Nth layer keeps softmax attention, "
                   "the rest become Gated DeltaNet linear-attention layers (0 = off, all layers softmax "
                   "attention). Stage 1: CPU forward only -- gen/eval/report work, train/tune do not "
                   "(no backward pass yet, and no CUDA build)")
       ->capture_default_str()->check(CLI::Range(0, 1024));
    app.add_option("--ngram-max-n", ngram_max_n,
                   "N-gram embeddings: highest n-gram order to hash (bigrams..N-grams added into the "
                   "input embedding); 0 = off, else must be >= 2. See docs/NGRAM_EMBEDDING.md")
       ->capture_default_str();
    app.add_option("--ngram-tables", ngram_tables,
                   "N-gram embeddings: independent hash tables per n-gram order (k)")
       ->capture_default_str()->check(CLI::Range(1, 64));
    app.add_option("--ngram-table-size", ngram_table_size,
                   "N-gram embeddings: per-table small hashed vocab size (0 = auto: same as --vocab)")
       ->capture_default_str();
    app.add_option("--hc-count", hc_count,
                   "Gated Residual (docs/GATED_RESIDUAL.md): number of parallel hyper-connection "
                   "residual streams (0 = off, else >= 2). Stage 1: CPU forward only -- gen/eval/report "
                   "work, train/tune do not (no backward pass yet)")
       ->capture_default_str()->check(CLI::Range(0, 64));
    app.add_option("--hc-lowrank", hc_lowrank,
                   "Gated Residual: low-rank bottleneck width of the input-mixer gate (0 = off, must "
                   "match --hc-count's on/off state, i.e. >= 1 iff --hc-count >= 2)")
       ->capture_default_str()->check(CLI::Range(0, 4096));
    app.add_option("--num-experts", num_experts,
                   "Mixture of Experts (docs/MOE.md): number of routed experts per layer, replacing the "
                   "FFN block for EVERY layer when on (0 = off, else >= 2). Stage 1: CPU forward only -- "
                   "gen/eval/report work, train/tune do not (no backward pass yet, and no CUDA build)")
       ->capture_default_str()->check(CLI::Range(0, 512));
    app.add_option("--experts-per-tok", experts_per_tok,
                   "Mixture of Experts: top-k routed experts selected per token (0 = off, must match "
                   "--num-experts' on/off state, and must not exceed it)")
       ->capture_default_str()->check(CLI::Range(0, 16));
    app.add_option("--rope-scaling", rope_scaling,
                   "RoPE position scaling for context extension: none (default) | linear")
       ->transform(CLI::CheckedTransformer(std::map<std::string, int>{{"none", 0}, {"linear", 1}},
                                           CLI::ignore_case))
       ->default_str("none");
    app.add_option("--rope-scale-factor", rope_scale_factor,
                   "Linear RoPE scaling divisor, e.g. 4 to run a 4x longer window")
       ->capture_default_str();
    app.add_option("--seq",    seq_len, "Context window length (0 = auto)")->capture_default_str();
    app.add_option("--ternary",ternary, "1 = BitNet-style ternary block weights")
       ->capture_default_str()->check(CLI::Range(0, 1));
    app.add_option("--gated-ffn", gated_ffn,
                   "1 = SwiGLU-gated FFN (Wgate/Wup/Wdown, no FFN bias) instead of the plain "
                   "2-matrix GELU+bias FFN -- matches the GGUF/Llama-family convention, for "
                   "importing open weights of that shape (CPU and GPU)")
       ->capture_default_str()->check(CLI::Range(0, 1));
    app.add_option("--tie-embeddings", tie_embeddings,
                   "1 = the LM head reuses the token embedding matrix (transposed) instead of its own "
                   "matrix+bias -- fewer params, no separate head weight")
       ->capture_default_str()->check(CLI::Range(0, 1));
    app.add_option("--qk-norm", qk_norm,
                   "1 = RMSNorm applied per-head to Q/K right after their projection, before RoPE "
                   "(Gemma2-style) -- stabilizes attention-logit magnitude")
       ->capture_default_str()->check(CLI::Range(0, 1));
    app.add_option("--pos-encoding", pos_encoding,
                   "Positional encoding scheme: absolute (learned) | rope (rotary)")
       ->transform(CLI::CheckedTransformer(std::map<std::string, int>{{"absolute", 0}, {"rope", 1}},
                                           CLI::ignore_case))
       ->default_str("rope");
    app.add_option("--rope-theta", rope_theta, "RoPE frequency base (theta)")
       ->capture_default_str();
    app.add_option("--bf16", bf16, "Float16 capability: 0=off, 1=on, 2=AUTO (on if GPU >= sm_80)")
       ->capture_default_str()->check(CLI::Range(0, 2));
    app.add_option("--prec-gemm", prec_gemm, "GEMM input precision: 0=F32,1=BF16,2=F16,9=AUTO")
       ->capture_default_str()->check(CLI::Range(0, 9));
    app.add_option("--prec-act", prec_act, "Saved-activation storage precision: 0=F32,1=BF16,2=F16,9=AUTO")
       ->capture_default_str()->check(CLI::Range(0, 9));
    app.add_option("--dump-vocab", dump_vocab,
                   "Write readable vocabulary-analysis dumps to <prefix>.{corpus_vocab,token_vocab,ngrams}.txt and exit");
    app.add_option("--vocab",  vocab_target, "Target vocabulary size (base + markers + word pieces; 0 = auto from corpus)")
       ->capture_default_str();
    app.add_option("--size-scale", size_scale,
                   "Multiplier on the auto-sizer's target-parameter budget: <1 = a smaller/faster/safer "
                   "starting point, >1 = more generous. Only affects auto-sized dims (--dmodel etc. or a "
                   "pinned <corpus>.model sidecar override this entirely, same as always).")
       ->capture_default_str()->check(CLI::PositiveNumber);
    app.add_option("--min-merge", min_merge, "Stop merging once the best pair occurs fewer than this many times")
       ->capture_default_str();
    app.add_option("--corpus-pretok", emit_tok,
                   "1 = pre-tokenize the whole corpus to corpus.tok; 0 = skip it and let training tokenize "
                   "on demand (avoids the on-disk token copy -- corpus.tok is itself ~0.6x the source size "
                   "-- for a corpus too large to keep resident); 2 = AUTO (by corpus size vs half of "
                   "physical RAM)")
       ->capture_default_str()->check(CLI::Range(0, 2));
    app.add_option("--tune-cache", tune_cache,
                   "Persisted tuned-defaults cache; read to bake DEFAULT_THREADS / DEFAULT_WINDOWS_PER_THREAD")
       ->capture_default_str();
    app.add_option("--has-cuda", has_cuda, "1 = the CUDA device-training backend was built (CMake's CUDA existence check)")
       ->capture_default_str()->check(CLI::Range(0, 1));
    app.add_option("--cuda-arch", cuda_arch,
                   "Detected GPU compute capability as an int (e.g. 120 for sm_120; 0 = none)")
       ->capture_default_str();
    app.add_option("--gpu-vram-mb", gpu_vram_mb, "Detected dedicated GPU VRAM in MB (0 = none)")
       ->capture_default_str();
    app.add_option("--gpu-shared-mb", gpu_shared_mb,
                   "Shared/overflow system memory the GPU can address in MB (WDDM; 0 = none)")
       ->capture_default_str();
    app.add_option("--compute", compute, "Backend to use: 0=CPU, 1=GPU, 2=HYBRID; -1 = auto (GPU when the CUDA backend is built)")
       ->capture_default_str()->check(CLI::Range(-1, 2));
    app.add_option("--cuda-tf32", cuda_tf32, "Bake TF32 tensor-core GEMM math on (1) or off (0)")
       ->capture_default_str()->check(CLI::Range(0, 1));

    CLI11_PARSE(app, argc, argv);

    // The configurator DECIDES whether to USE CUDA: default the compute backend to GPU when the CUDA
    // device backend was built (has_cuda, from CMake's existence check), else CPU. An explicit
    // --compute overrides (e.g. CPU on a GPU build). CMake checks existence; the configurator picks.
    if (compute < 0) compute = has_cuda ? 1 : 0;
    if (compute != 0 && !has_cuda) {
        std::println(stderr, "configure warning: --compute={} requested but no CUDA backend is built; using CPU", compute);
        compute = 0;
    }
    // Tied embeddings: GPU support landed (launch_tied_head/launch_tied_head_bwd in backend_cuda.cu,
    // gated purely by `if constexpr (USE_TIED_EMBEDDINGS)` at each forward/backward call site -- no
    // static_assert restriction left for this axis), so --tie-embeddings=1 is now valid with any
    // --compute backend. QK-norm: GPU support landed the same way (qknorm_act_kernel/
    // qknorm_backward_act_kernel in backend_cuda.cu, `if constexpr (USE_QK_NORM)`-gated at each
    // forward/backward call site) -- --qk-norm=1 is now valid with any --compute backend too. Gated
    // (SwiGLU) FFN: GPU support landed the same way too (swiglu_kernel/swiglu_act_kernel/
    // swiglu_backward_act_kernel in backend_cuda.cu, `if constexpr (USE_GATED_FFN)`-gated at each
    // forward/backward call site) -- --gated-ffn=1 is now valid with any --compute backend too.
    // (Ternary remains CPU-only -- see cmake/Backends.cmake's FATAL_ERROR guard and the backend_cuda.cu
    // static_assert for that axis.)

    // Resolve the model dims with precedence CLI (nonzero) > <corpus>.model sidecar > auto-size, so a
    // pinned size PERSISTS across re-runs (a build-time auto-regen keeps it). Seed the sidecar if
    // absent so it is there to edit. The corpus file exists (required + ExistingFile).
    std::error_code corpus_size_ec;
    const std::uintmax_t corpus_bytes = std::filesystem::file_size(corpus, corpus_size_ec);
    {
        const std::string sidecar = corpus + ".model";
        sub0::config::ModelDims dims{d_model, n_layers, n_heads, seq_len, vocab_target};   // CLI (0 = auto)
        bool have_sidecar = false;
        if (std::ifstream sf(sidecar); sf) { dims = sub0::config::fill_defaults(dims, sub0::config::parse_model_sidecar(sf)); have_sidecar = true; }
        // VRAM budget only applies when GPU is the RESOLVED backend (compute, already resolved above) --
        // an explicit --compute 0 on a CUDA-capable build trains on CPU for this run, so clamping to the
        // (irrelevant here) detected VRAM would shrink the auto-sized shape for no reason.
        const int vram_budget = (compute == 0) ? 0 : gpu_vram_mb;
        // Cross-corpus bytes/token calibration (see config_util.hpp's TokenCalibration): CWD-relative,
        // matching how --corpus itself is conventionally invoked (repo root) rather than corpus-relative
        // like the .model sidecar above -- this is a project-wide prior, not a per-corpus one.
        double bytes_per_token = 4.0;
        if (std::ifstream cf("data/tokenizer_calibration.txt"); cf) {
            bytes_per_token = sub0::config::bytes_per_token_calibrated(sub0::config::parse_token_calibration(cf));
        }
        dims = sub0::config::apply_autosize(dims, corpus_size_ec ? 0 : corpus_bytes, vram_budget, size_scale,
                                             bytes_per_token);
        d_model = dims.d_model; n_layers = dims.n_layers; n_heads = dims.n_heads; seq_len = dims.seq_len; vocab_target = dims.vocab;
        if (!have_sidecar) { if (std::ofstream sf(sidecar); sf) sf << sub0::config::format_model_sidecar(dims); }
        std::println(stderr, "model dims: corpus {:.0f} MB -> d={} L={} H={} seq={} vocab={} ({}; CLI pins)",
                     (corpus_size_ec ? 0.0 : static_cast<double>(corpus_bytes) / 1e6), d_model, n_layers, n_heads, seq_len, vocab_target,
                     have_sidecar ? "from " + std::filesystem::path(sidecar).filename().string() : "auto-sized + seeded sidecar");
    }

    // --corpus-pretok AUTO (2): resolve by corpus size vs half of physical RAM, mirroring the old
    // CMake-level heuristic (now removed -- the corpus path used to have to be a CMake cache variable
    // for this decision to see it; the tool has --corpus directly, so the decision belongs here).
    if (emit_tok == 2) {
        const std::uintmax_t ram_bytes = total_physical_ram_bytes();
        const bool pretok = sub0::config::should_pretokenize(corpus_size_ec ? 0 : corpus_bytes, ram_bytes);
        std::println(stderr, "corpus.tok: --corpus-pretok=AUTO -> {} (corpus {:.0f} MB est-tok {:.0f} MB vs {:.0f} MB half-RAM)",
                     pretok ? "emit" : "skip (on-demand)",
                     (corpus_size_ec ? 0.0 : static_cast<double>(corpus_bytes) / 1e6),
                     (corpus_size_ec ? 0.0 : static_cast<double>(corpus_bytes) * 2 / 1e6),
                     static_cast<double>(ram_bytes) / 2e6);
        emit_tok = pretok ? 1 : 0;
    }

    if (d_model % n_heads != 0) {
        std::println(stderr, "configure error: dmodel ({}) not divisible by heads ({})",
                     d_model, n_heads);
        return 1;
    }
    // GQA: n_kv_heads must divide n_heads so every KV head serves an equal-sized query group.
    // 0 means "not requested" -> plain MHA, the default that leaves every existing build unchanged.
    // GQA-2 (half as many KV heads as query heads) is the DEFAULT, not plain MHA. This is an
    // evidence-backed default change, not a guess -- the three-pillar A/B (d384 L6 H8, TinyStories,
    // matched batch 256, TWO seeds; project memory gqa-ab-three-pillar-result) measured against MHA:
    //     throughput  +8.4%        (monotonic, gaps 2-3x the within-variant stdev)
    //     KV cache    -50%         (exact, and the reason GQA exists)
    //     quality     NOT RESOLVABLE -- seed noise exceeded the between-variant signal 1.73x and the
    //                 ranking FLIPPED between seeds, so "no measurable cost" means exactly that
    // Faster and half the KV cache for a quality difference this setup cannot detect is the right
    // default. `--kv-heads <heads>` restores MHA; the A/B could not separate them on quality, so that
    // remains a legitimate choice rather than a worse one.
    //
    // Note this deliberately departs from AGENTS.md 4's "new capability = off by default". That rule
    // protects against perturbing existing builds with unvalidated changes; here the capability IS
    // validated, on both backends, and there are no production models to perturb. Odd head counts fall
    // back to MHA rather than picking a non-divisor -- N_HEADS % N_KV_HEADS == 0 is a hard static_assert.
    if (n_kv_heads == 0) n_kv_heads = (n_heads % 2 == 0) ? n_heads / 2 : n_heads;
    if (n_kv_heads < 1 || n_heads % n_kv_heads != 0) {
        std::println(stderr, "configure error: kv-heads ({}) must be >= 1 and divide heads ({}) evenly",
                     n_kv_heads, n_heads);
        return 1;
    }
    // LoopSplit: the head and tail blocks split what's left of the stack evenly around the middle,
    // so (n_layers - loop_middle) must be even. Mirrors layout.hpp's own static_asserts, but as a
    // configure-time diagnostic naming the flags rather than a compile error in the engine.
    if (loop_middle < 0 || loop_middle > n_layers) {
        std::println(stderr, "configure error: loop-middle-layers ({}) must be within [0, layers ({})]",
                     loop_middle, n_layers);
        return 1;
    }
    if (loop_repeats < 1) {
        std::println(stderr, "configure error: loop-repeats ({}) must be at least 1", loop_repeats);
        return 1;
    }
    // Only when looping is actually requested -- an odd-layer model with the feature OFF is fine.
    if (loop_middle > 0 && loop_repeats > 1 && (n_layers - loop_middle) % 2 != 0) {
        std::println(stderr,
                     "configure error: layers ({}) - loop-middle-layers ({}) must be even so the "
                     "head and tail blocks are symmetric", n_layers, loop_middle);
        return 1;
    }
    // N-gram embeddings: mirrors layout.hpp's own static_asserts, as a configure-time diagnostic naming
    // the flags rather than a compile error in the engine. D_MODEL must already be resolved here (the
    // autosize/CLI-pin above finalizes it) since the divisibility check needs the real width.
    if (ngram_max_n != 0 && ngram_max_n < 2) {
        std::println(stderr, "configure error: ngram-max-n ({}) must be 0 (off) or >= 2 (bigrams)", ngram_max_n);
        return 1;
    }
    if (ngram_max_n >= 2) {
        // Inlined rather than calling layout.hpp's ngram_num_embedders(): this tool deliberately does
        // not include layout.hpp/sub0_config.hpp (it is GENERATING the next config, see memplan.hpp's
        // own "dependency-free" doc comment for the same reasoning) -- so the formula is duplicated
        // here, exactly like the LoopSplit symmetry check just above.
        const int num_embedders = ngram_tables * (ngram_max_n - 1);
        if (d_model % num_embedders != 0) {
            std::println(stderr,
                         "configure error: d-model ({}) must be divisible by ngram-tables*({}-1)={} "
                         "so each n-gram table's embedding slice is an integer width",
                         d_model, ngram_max_n, num_embedders);
            return 1;
        }
    }
    // Gated Residual: mirrors layout.hpp's own static_asserts, as a configure-time diagnostic naming the
    // flags rather than a compile error in the engine (same pattern as the ngram check just above).
    if (hc_count != 0 && hc_count < 2) {
        std::println(stderr, "configure error: hc-count ({}) must be 0 (off) or >= 2", hc_count);
        return 1;
    }
    if ((hc_count >= 2) != (hc_lowrank >= 1)) {
        std::println(stderr,
                     "configure error: hc-count ({}) and hc-lowrank ({}) must be on or off together "
                     "(hc-count >= 2 iff hc-lowrank >= 1)", hc_count, hc_lowrank);
        return 1;
    }
    // Mixture of Experts: mirrors layout.hpp's own static_asserts, as a configure-time diagnostic naming
    // the flags rather than a compile error in the engine (same pattern as the hc-count/hc-lowrank check
    // just above).
    if (num_experts != 0 && num_experts < 2) {
        std::println(stderr, "configure error: num-experts ({}) must be 0 (off) or >= 2", num_experts);
        return 1;
    }
    if ((num_experts >= 2) != (experts_per_tok >= 1)) {
        std::println(stderr,
                     "configure error: num-experts ({}) and experts-per-tok ({}) must be on or off "
                     "together (num-experts >= 2 iff experts-per-tok >= 1)", num_experts, experts_per_tok);
        return 1;
    }
    if (experts_per_tok > num_experts) {
        std::println(stderr, "configure error: experts-per-tok ({}) cannot exceed num-experts ({})",
                     experts_per_tok, num_experts);
        return 1;
    }
    // Plain FFN keeps the long-standing 4*D_MODEL width; gated (SwiGLU) uses a narrower width chosen
    // so the two styles land at roughly the same total FFN param count -- see d_ff_for's doc comment.
    const int d_ff = sub0::config::d_ff_for(d_model, gated_ffn != 0);

    const std::string abspath = std::filesystem::absolute(corpus).string();

    const std::filesystem::path gen_dir        = std::filesystem::path(out).parent_path();
    const std::filesystem::path tok_path       = gen_dir / "corpus.tok";
    const std::filesystem::path tkz_path       = gen_dir / "tokenizer.tok";
    const std::filesystem::path tok_stamp_path = gen_dir / "tokenizer.stamp";

    // ----------------------------------------------------------------------
    // Reuse fast-path: tokenizer.tok/corpus.tok are pure functions of the corpus bytes +
    // vocab_target + min_merge + emit_tok (see tok_stamp_matches above). When the stamp says
    // none of those changed since the last configure run, and the files still parse, skip the
    // scan/learn/tokenize entirely -- a re-run made only to change an unrelated knob (a dims
    // pin, precision, tune-cache) would otherwise re-tokenize a FineWeb-scale corpus for
    // nothing. Never applies in --dump-vocab analysis mode, which needs the full scan
    // regardless of what is already on disk.
    // ----------------------------------------------------------------------
    tok::Tokenizer tkz;
    bool reused = false;
    if (dump_vocab.empty() &&
        tok_stamp_matches(tok_stamp_path.string(), corpus, vocab_target, min_merge, emit_tok)) {
        std::ifstream tzin(tkz_path, std::ios::binary);
        if (tzin && tok::deserialize(tkz, tzin)) {
            reused = true;
            if (emit_tok) {
                const sub0::TokMap tm(tok_path.string());   // corpus.tok must also exist and match
                reused = tm.ok() && tm.vocab() == tkz.vocab;
            }
        }
    }

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
    std::chrono::steady_clock::time_point _t1 = _t0, _t2 = _t0, _t3 = _t0, _t4 = _t0;

    tok::Scan S;
    std::unordered_set<std::string> attested;
    long long names_withheld = 0;
    std::size_t token_count = 0, doc_count = 0;
    bool tok_rt = true;

    if (reused) {
        if (emit_tok) {
            const sub0::TokMap tm(tok_path.string());
            token_count = tm.tokens().size();
            doc_count   = tm.doc_starts().size();
        }
        std::println(stderr, "reuse: tokenizer.tok{} unchanged since the last configure (corpus + vocab {} + "
                     "min-merge {} match) -- scan/learn/tokenize skipped",
                     emit_tok ? " + corpus.tok" : "", vocab_target, min_merge);
    } else {

    // The expensive corpus scan (passes 1-2) produces a bounded ScanState. Cache it next to
    // the corpus (<corpus>.words, validated by size+mtime+version) so a re-run with a
    // different vocab, after a crash, or to re-tokenize for a new scheme reloads it and SKIPS
    // the scan. The name COUNTS are kept (not just the derived `attested`) so the state is
    // mergeable for a future multi-corpus / incremental tokenizer.
    const std::string cache_path = std::string(corpus) + ".words";
    if (load_scan_state(cache_path, corpus, S)) {
        attested = tok::derive_attested(S, &names_withheld);
        S.rebuild_index();                                 // raw-key -> word id, for Pass 3
        _t1 = _t2 = std::chrono::steady_clock::now();
        std::println(stderr, "scan cache: HIT '{}' ({} words, {} attested) -- corpus scan skipped",
                     cache_path, S.word_syms.size(), attested.size());
    } else {
        // The corpus scan (passes 1-2) is the cost that scales with the corpus and is
        // CPU-bound (normalize + truecase + table build, far below disk bandwidth), so it is
        // run DATA-PARALLEL: the file is split into newline-aligned segments (one per worker),
        // each worker scans its own segment into a THREAD-LOCAL tok::Scan, and the scans are
        // merged afterwards. Correctness rests on two facts: a newline never splits a unit/
        // look-back (so per-segment results equal whole-file results), and every downstream
        // artifact is INVARIANT to word-table ordering -- the base alphabet is byte-ordered, and
        // BPE picks each merge by (count, pair) where counts are order-independent sums and pair
        // ids are byte-derived. So the merged scan (hence tokenizer.tok / corpus.tok) is byte-
        // identical to the single-thread scan regardless of how the corpus is partitioned.
        std::error_code fec;
        const std::size_t fsz = static_cast<std::size_t>(std::filesystem::file_size(corpus, fec));
        if (fec || fsz == 0) { std::println(stderr, "configure error: empty corpus"); return 1; }
        const unsigned hw = std::max(1u, std::thread::hardware_concurrency());
        // >= ~16 MB per segment so the merge/thread overhead is amortised; capped at the cores.
        int nseg = static_cast<int>(std::min<std::size_t>(hw, std::max<std::size_t>(1, fsz / (16u << 20))));
        if (nseg < 1) nseg = 1;
        const std::vector<std::size_t> bnd = segment_bounds(corpus, fsz, nseg);

        // --- Pass 1 (parallel): name detection (tok::Scan::add_names). Decides which lowercase
        //     forms license a Capitalized/UPPER -> marker collapse; a capital following a
        //     lowercase word mid-sentence ("...to Spot") is a name use. Each worker accumulates
        //     a local Scan; we fold them after.
        std::vector<tok::Scan> p1(static_cast<std::size_t>(nseg));
        std::vector<char>      p1_ok(static_cast<std::size_t>(nseg), 0);
        parallel_segments(bnd, nseg, [&](int t, std::size_t a, std::size_t b) {
            tok::Scan& L = p1[static_cast<std::size_t>(t)];
            p1_ok[static_cast<std::size_t>(t)] =
                for_each_chunk_range(corpus, a, b, [&](std::string_view chunk) { L.add_names(chunk); }) ? 1 : 0;
        });
        for (int t = 0; t < nseg; ++t) {
            if (!p1_ok[static_cast<std::size_t>(t)]) { std::println(stderr, "configure error: cannot read corpus '{}'", corpus); return 1; }
            S.merge_names(p1[static_cast<std::size_t>(t)]);
        }
        if (S.raw_bytes == 0) { std::println(stderr, "configure error: empty corpus"); return 1; }
        attested = tok::derive_attested(S, &names_withheld);
        _t1 = std::chrono::steady_clock::now();

        // --- Pass 2 (parallel): truecase + pre-tokenize into the unique-word table
        //     (tok::Scan::add_words). Each worker builds a local Scan; we fold the locals into S.
        std::vector<tok::Scan> p2(static_cast<std::size_t>(nseg));
        parallel_segments(bnd, nseg, [&](int t, std::size_t a, std::size_t b) {
            tok::Scan& L = p2[static_cast<std::size_t>(t)];
            for_each_chunk_range(corpus, a, b, [&](std::string_view chunk) { L.add_words(chunk, attested); });
        });
        for (int t = 0; t < nseg; ++t) S.merge_words(p2[static_cast<std::size_t>(t)]);
        _t2 = std::chrono::steady_clock::now();
        save_scan_state(cache_path, corpus, S);
        std::println(stderr, "scan cache: saved '{}' ({} words, {} segments)", cache_path, S.word_syms.size(), nseg);
    }

    // Learn the base alphabet + word pieces from the scan (sub0::tok::learn). This REMAPS S.word_syms
    // in place from raw byte symbols to final token ids, so Pass 3 can emit the tokenized corpus from
    // S.index + S.word_syms. The Unigram EM/prune runs multi-threaded with per-round progress, and for
    // a huge corpus it learns from the frequent-word HEAD only: the Zipf tail of rare/hapax words is
    // near-lossless to drop (single bytes keep every word encodable) but would otherwise dominate the
    // O(unique-words) iteration count. The cap self-scales -- it is a no-op below ~2M unique words.
    tok::LearnOptions lopts;
    lopts.vocab_target    = vocab_target;
    lopts.min_merge       = min_merge;
    lopts.verbose         = true;                 // the configurator is a CLI tool -- show learn progress
    lopts.max_learn_words = 2'000'000;            // frequent-word head for the EM/prune (near-lossless)
    const auto _tl0 = std::chrono::steady_clock::now();
    std::println(stderr, "learning {} vocab from {} unique words (unigram, multi-threaded)...", vocab_target, S.word_syms.size());
    tkz = tok::learn(S, attested, lopts);   // always JOIN scheme (fills the outer tkz declared above)
    {   // Post-learn summary: the FINAL vocabulary + the compression it achieves over the word corpus
        // (word bytes / word tokens, from the now-remapped S.word_syms). This is the headline "how many
        // tokens did we settle on" the long tokenize pass below is about to encode the corpus into.
        long long tot_tok = 0, tot_by = 0;
        for (const auto& [key, id] : S.index) {
            const long long f = S.word_freq[static_cast<std::size_t>(id)];
            tot_tok += f * static_cast<long long>(S.word_syms[static_cast<std::size_t>(id)].size());
            tot_by  += f * static_cast<long long>(key.size());
        }
        std::println(stderr, "vocab learned in {:.1f}s: {} tokens = {} base + {} word pieces (target {}) | "
                     "{:.3f} bytes/token over the word corpus",
                     std::chrono::duration<double>(std::chrono::steady_clock::now() - _tl0).count(),
                     tkz.vocab, tkz.n_base, tkz.vocab - tkz.n_base, vocab_target,
                     tot_tok > 0 ? static_cast<double>(tot_by) / static_cast<double>(tot_tok) : 0.0);
    }

    // Analysis mode: write the readable vocabulary dumps and exit (no corpus.tok / config header). The
    // vocab curve + the BPE side of the A/B need the greedy merges, but `tkz` above is now Unigram (the
    // default), so learn a BPE tokenizer for the comparison here (it remaps S.word_syms to BPE ids,
    // which the A/B reads; the A/B re-learns Unigram internally).
    if (!dump_vocab.empty()) {
        const tok::BpeAnalysisVocab tkz_bpe = tok::learn_bpe_analysis(S, vocab_target, min_merge);
        dump_vocab_files(S, tkz_bpe, dump_vocab);
        std::println(stderr, "vocab dumps written: {}.{{corpus_vocab,token_vocab,ngrams,vocab_curve,unigram_vocab}}.txt", dump_vocab);
        return 0;
    }

    _t3 = std::chrono::steady_clock::now();

    // --- Pass 3 (optional): stream the corpus once more and emit the merged token stream to
    //     corpus.tok incrementally (never materialised in memory). Skipped when --corpus-pretok
    //     0: training then tokenizes windows on demand from the raw corpus + tokenizer.tok,
    //     avoiding the on-disk token copy (corpus.tok is itself ~0.6x the source size, see
    //     should_pretokenize in config_util.hpp) for a corpus too large to keep resident. Each word
    //     unit looks up its post-BPE id sequence in the table; standalone symbols map straight to base ids.
    //     Losslessness is verified per chunk by reconstructing the base symbols; the id
    //     array's length is back-patched after the count is known.
    const std::filesystem::path tok_tmp_path = tok_path.string() + ".tmp";
    if (emit_tok) {
        std::ofstream ts(tok_tmp_path, std::ios::binary);
        if (!ts) { std::println(stderr, "configure error: cannot write '{}'", tok_tmp_path.string()); return 1; }
        // v2 corpus.tok: u64 counts (a 46 GB corpus is ~14B tokens, past the legacy u32) + the minimal
        // byte-aligned token width for this vocab (16/24/32 bits) instead of a fixed int32.
        sub0::TokWriter tw(ts, tkz.vocab);
        std::vector<std::uint8_t> pack_scratch;
        std::println(stderr, "corpus.tok: {} bits/token (vocab {})", tw.bytes_per_token() * 8, tkz.vocab);
        // Document boundaries come from the explicit eos_id token ONLY -- the literal `<|endoftext|>`
        // marker the extraction scripts insert between documents (see casing.hpp's TOK_EOS comment).
        // The old blank-line fallback is gone: a blank line also occurs mid-document as an ordinary
        // paragraph break, so it manufactures spurious boundaries on any corpus with paragraphs, and
        // every corpus this project ships now carries the marker. A corpus without markers is now a
        // hard error below rather than silently becoming one giant document.
        // Record each document's start token index so training keeps windows inside one document
        // (the scan logic itself -- eos_id-preferred, \n\n-fallback -- lives in sub0::tok::
        // scan_doc_boundaries so tests/engine_tests.cpp can exercise it without duplicating it).
        std::vector<std::uint64_t> doc_starts{0u};                  // document 0 begins at token 0 (u64: ~14B tokens)
        int nl_run = 0;
        // Tokenize Pass 3, PARALLEL + IO-OVERLAPPED: a dedicated reader thread streams newline-aligned
        // chunks off disk into batches on a bounded queue (BoundedQueue above), while THIS thread pops
        // a ready batch, fans it out to `nth` encode workers (each is independent -- tok::encode reads
        // the const tokenizer), then writes the results in FILE ORDER and folds the doc-boundary +
        // count state sequentially. Reading batch N+1 now runs concurrently with encoding/writing batch
        // N (previously strictly sequential: read all of N, THEN encode+write it, THEN read N+1 -- disk
        // idle during encode, CPU idle during read). The per-word Viterbi encode is the cost, so this
        // lifts the pass off one core (~8 MB/s) toward the disk/memory ceiling.
        const int nth = std::max(1, static_cast<int>(std::thread::hardware_concurrency()));
        std::vector<std::vector<std::int32_t>> outs(static_cast<std::size_t>(nth));
        std::vector<char> rtok(static_cast<std::size_t>(nth), 1);
        bool rt_reported = false;
        std::size_t bytes_done = 0;
        const double tot_gb = static_cast<double>(S.raw_bytes) / 1e9;
        const auto _p3t0 = std::chrono::steady_clock::now();
        auto _p3last = _p3t0;
        std::println(stderr, "tokenizing corpus -> corpus.tok ({:.1f} GB, {} threads)...", tot_gb, nth);

        BoundedQueue<std::vector<std::string>> bq(2);   // 2 batches: one draining, one prefetched
        std::thread reader([&] {
            std::vector<std::string> batch; batch.reserve(static_cast<std::size_t>(nth));
            for_each_chunk(corpus, [&](std::string_view chunk) {
                batch.emplace_back(chunk);
                if (static_cast<int>(batch.size()) >= nth) {
                    bq.push(std::move(batch));
                    batch = std::vector<std::string>(); batch.reserve(static_cast<std::size_t>(nth));
                }
            });
            if (!batch.empty()) bq.push(std::move(batch));
            bq.close();
        });

        std::vector<std::string> batch;
        while (bq.pop(batch)) {
            const int k = static_cast<int>(batch.size());
            std::vector<std::thread> th;                            // parallel encode over the batch
            const int per = (k + nth - 1) / nth;
            for (int t = 0; t < nth; ++t) {
                const int lo = std::min(k, t * per), hi = std::min(k, lo + per);
                if (lo >= hi) break;
                th.emplace_back([&, lo, hi] {
                    for (int i = lo; i < hi; ++i) {
                        long qr = 0;
                        const std::string norm = normalize_text(batch[static_cast<std::size_t>(i)], qr);
                        const std::vector<int> ids = tok::encode(tkz, batch[static_cast<std::size_t>(i)]);
                        rtok[static_cast<std::size_t>(i)] = (tok::detokenize(tkz, ids) == norm) ? 1 : 0;
                        outs[static_cast<std::size_t>(i)].assign(ids.begin(), ids.end());
                    }
                });
            }
            for (auto& x : th) x.join();
            for (int i = 0; i < k; ++i) {                           // write in file order; sequential state
                const std::vector<std::int32_t>& toks = outs[static_cast<std::size_t>(i)];
                if (!rtok[static_cast<std::size_t>(i)]) {
                    tok_rt = false;
                    if (!rt_reported) {                            // re-encode this chunk to show the first divergence (rare)
                        rt_reported = true;
                        long qr = 0; const std::string norm = normalize_text(batch[static_cast<std::size_t>(i)], qr);
                        const std::string dec = tok::detokenize(tkz, tok::encode(tkz, batch[static_cast<std::size_t>(i)]));
                        std::size_t p = 0; while (p < dec.size() && p < norm.size() && dec[p] == norm[p]) ++p;
                        const std::size_t lc = p > 48 ? p - 48 : 0;
                        auto vis = [](std::string s) { std::string o; for (char c : s) { if (c=='\n') o+="\\n"; else if (c=='\t') o+="\\t"; else o+=c; } return o; };
                        std::println(stderr, "ROUND-TRIP FAIL at byte {} (norm {}B / dec {}B):", p, norm.size(), dec.size());
                        std::println(stderr, "  expected: ...{}...", vis(norm.substr(lc, 96)));
                        std::println(stderr, "  actual:   ...{}...", vis(dec.substr(lc, 96)));
                    }
                }
                // doc boundaries (global index = token_count + li)
                sub0::tok::scan_doc_boundaries(
                    std::span<const std::int32_t>(toks.data(), toks.size()),
                    static_cast<std::uint64_t>(token_count), tkz, nl_run, doc_starts);
                tw.append(toks.data(), toks.size(), pack_scratch);   // pack to the v2 width + write
                token_count += toks.size();
                bytes_done  += batch[static_cast<std::size_t>(i)].size();
            }
            const auto _now = std::chrono::steady_clock::now();     // throughput heartbeat every ~2s
            if (std::chrono::duration<double>(_now - _p3last).count() >= 2.0) {
                _p3last = _now;
                const double gb = static_cast<double>(bytes_done) / 1e9;
                const double secs = std::chrono::duration<double>(_now - _p3t0).count();
                std::println(stderr, "  tokenizing: {:.1f} / {:.1f} GB ({:.0f}%)  {:.0f} MB/s  {} tokens",
                             gb, tot_gb, tot_gb > 0 ? 100.0 * gb / tot_gb : 0.0,
                             secs > 0 ? static_cast<double>(bytes_done) / 1e6 / secs : 0.0, token_count);
            }
        }
        reader.join();
        tw.finish(doc_starts);                                      // u64 doc index + back-patch ntok/ndoc
        doc_count = doc_starts.size();
        ts.close();
        // Atomic publish: a reader (training) that already has corpus.tok open, or a HARDLINKED pin
        // of it in a model directory, is unaffected by this rename -- it keeps seeing the old complete
        // file. Only a fresh open of `tok_path` after this point sees the new one.
        atomic_replace(tok_tmp_path, tok_path);
    } else {
        // Drop any stale corpus.tok from a previous emit so training reliably falls back to
        // on-demand tokenization instead of mmapping an out-of-date token copy.
        std::error_code ec;
        std::filesystem::remove(tok_path, ec);
    }
    _t4 = std::chrono::steady_clock::now();

    // 9. Write the runtime tokenizer (base alphabet + merges + attested words).
    //    Generation uses this to encode prompts and detokenize output; training
    //    needs only corpus.tok. Same atomic tmp+rename publish as corpus.tok above.
    {
        const std::filesystem::path tkz_tmp_path = tkz_path.string() + ".tmp";
        std::ofstream tz(tkz_tmp_path, std::ios::binary);
        if (!tz) { std::println(stderr, "configure error: cannot write '{}'", tkz_tmp_path.string()); return 1; }
        tok::serialize(tkz, tz);
        tz.close();
        atomic_replace(tkz_tmp_path, tkz_path);
    }

    // Stamp tokenizer.tok/corpus.tok with the inputs that produced them, so a future configure
    // run against the same corpus + vocab_target + min_merge + emit_tok hits the reuse fast-path
    // above instead of re-deriving them.
    save_tok_stamp(tok_stamp_path.string(), corpus, vocab_target, min_merge, emit_tok);
    }  // end fresh (!reused) branch

    const int n_base = tkz.n_base, vocab = tkz.vocab;

    // N-gram embeddings: table-size 0 means "auto", resolved now that the real vocab is known (the
    // tokenizer learning above is what finalizes it) -- same auto-then-resolve shape as d_model/vocab_target.
    // Only when the feature is actually on -- otherwise a resolved nonzero table-size would show up in
    // describe_config()'s "differs from default" banner for a build that never enabled n-gram at all.
    if (ngram_max_n >= 2 && ngram_table_size <= 0) ngram_table_size = vocab;

    // 10. Emit the generated headers, SPLIT so the corpus and the machine vary independently:
    //   sub0_corpus.hpp  -- model dims + vocab + tokenizer scheme + artifact paths (frozen per corpus)
    //   sub0_system.hpp  -- precision + cached hardware facts + tuned runtime defaults (per machine)
    //   <out> (sub0_config.hpp) -- a thin umbrella that #includes both (what the engine includes).
    // See docs/CONFIGURE_ARCHITECTURE.md.
    std::ofstream cos(gen_dir / "sub0_corpus.hpp", std::ios::binary);   // corpus-specific
    std::ofstream sos(gen_dir / "sub0_system.hpp", std::ios::binary);   // system-specific
    std::ofstream os(out, std::ios::binary);                            // umbrella
    if (!cos || !sos || !os) { std::println(stderr, "configure error: cannot write the generated headers in '{}'", gen_dir.string()); return 1; }
    auto emit_path = [](std::ostream& o, const char* name, const std::string& p) {
        o << "inline constexpr char " << name << "[] = \"";
        for (char c : p) { if (c == '\\' || c == '"') o << '\\'; o << c; }
        o << "\";\n";
    };

    // Cache the machine's logical-core count so the engine sizes its statically
    // allocated worker pool to the hardware instead of a fixed guess. If a previous
    // `tune` run persisted better runtime defaults, fold those in here too; otherwise
    // fall back to the hardware core count and a conservative windows/thread of 4.
    const unsigned hw_concurrency = std::max(1u, std::thread::hardware_concurrency());
    const int      max_workers    = static_cast<int>(hw_concurrency);
    std::string    tune_cache_abs;
    std::ifstream  tc;                                       // a non-open stream yields no lines -> defaults
    if (!tune_cache.empty()) { tune_cache_abs = std::filesystem::absolute(tune_cache).string(); tc.open(tune_cache); }
    const sub0::config::TuneDefaults td = sub0::config::parse_tune_cache(tc, static_cast<int>(hw_concurrency));
    const int default_threads    = td.threads;
    const int default_wpt        = td.windows_per_thread;
    int       default_gpu_batch  = td.gpu_batch;            // derived from the width if untuned; clamped to VRAM below
    if (td.tf32_from_cache) cuda_tf32 = td.cuda_tf32 ? 1 : 0;

    // Resolve precision FIRST (it can error, so no header is half-written on a bad --prec choice; it
    // is ALSO needed by the VRAM clamp just below -- activation byte width changes the resident
    // footprint by 2x, so this must run before that clamp, not after it as it used to). Float16
    // capability is a detected hardware fact (BF16 needs sm_80+); F16/Q8/Q4 are reserved.
    const bool f16_ok = sub0::config::f16_capable(bf16, cuda_arch);
    const auto resolve_dtype = [&](int code, const char* section) -> std::string {
        const sub0::config::Precision p = sub0::config::resolve_precision(code, f16_ok);
        if (p.status == sub0::config::PrecStatus::NeedsF16Hardware) {
            std::fprintf(stderr, "configure error: BF16 %s requested but no BF16-capable GPU\n", section); std::exit(2);
        }
        if (p.status == sub0::config::PrecStatus::F16Unsupported) {
            std::fprintf(stderr, "configure error: F16 %s not yet supported (BF16 only)\n", section); std::exit(2);
        }
        return p.dtype;
    };
    const std::string gemm_dt = resolve_dtype(prec_gemm, "GEMM");
    const std::string act_dt  = resolve_dtype(prec_act, "activations");

    // Keep DEFAULT_GPU_BATCH within the VRAM budget. The batch sizes the resident device training
    // scratch; on Windows an over-budget cudaMalloc does not OOM, it silently spills to WDDM shared
    // memory and thrashes over PCIe (~10x slower) -- a silent performance cliff. A cached batch tuned
    // for a SMALLER model goes stale when the corpus auto-sizes UP, so rather than hard-blocking the
    // build we CLAMP to the largest batch that fits -- the SAME memplan primitive the tuner bounds its
    // sweep with, so the two agree -- and warn to re-tune for the optimum. Only a model too big for
    // even batch 1 is a hard error.
    //
    // act_bytes MUST match the activation precision actually resolved above: this clamp used to
    // default to memplan::FLOAT (4 bytes) unconditionally, silently modeling the F32 footprint on a
    // BF16 build (2 bytes) -- a real ~2x sizing error that happened to roughly cancel against a
    // separate missing-headroom gap at this project's exact current config, so it wasn't visibly
    // broken, but is not a bet worth relying on for other dims/cards.
    if (has_cuda && gpu_vram_mb > 0) {
        const sub0::memplan::u64 act_bytes = (act_dt == "BF16") ? 2 : sub0::memplan::FLOAT;
        // tied=true drops param_floats()'s lm_head/lm_bias term (see memplan.hpp); must match
        // USE_TIED_EMBEDDINGS or this VRAM prediction over-estimates a tied model's footprint.
        // qk_norm=true adds the q_norm/k_norm gamma floats + the qk_pre training scratch term;
        // must match USE_QK_NORM the same way. gated=true replaces the plain FFN's b1/b2 bias floats
        // with a third Wg weight matrix (and a third bf16 GEMM-weight mirror); must match
        // USE_GATED_FFN the same way.
        const sub0::memplan::Dims dims{ d_model, n_layers, n_heads, d_ff, seq_len, vocab,
                                         tie_embeddings != 0, qk_norm != 0, gated_ffn != 0,
                                         pos_encoding == 0, n_kv_heads,
                                         // LoopSplit: per-EXECUTION activation checkpoints, not
                                         // per-layer. Omitting this under-predicts activation memory
                                         // (measured 1.43x at 10 executions from 6 layers), and this
                                         // estimate feeds gpu_batch_estimate() -- i.e. the batch we
                                         // RECOMMEND. Under-predicting pushes the batch UP, which is
                                         // the dangerous direction (see the open large-batch fault).
                                         n_layers + loop_middle * (loop_repeats - 1) };
        // No real `tune --backend gpu` result yet: the CPU-width fallback (threads*windows_per_thread)
        // has nothing to do with GPU VRAM, so start from a VRAM-scaled estimate instead of leaving a
        // big card mostly idle until someone remembers to tune (see gpu_batch_estimate()'s own doc
        // comment -- this is exactly the gap a real production run was caught hitting: 31% VRAM used
        // at batch 96 because that build had never been tuned for GPU).
        if (!td.gpu_batch_from_cache) {
            constexpr int kTuneMaxBatch = 4096;   // sanity ceiling, matches sub0_tune_stage's own cap
            const int est = sub0::config::gpu_batch_estimate(dims, gpu_vram_mb, kTuneMaxBatch, act_bytes);
            if (est > default_gpu_batch) default_gpu_batch = est;
        }
        const int need = sub0::memplan::train_resident_mb(dims, default_gpu_batch, act_bytes);
        if (need > gpu_vram_mb) {
            const int fit = sub0::memplan::max_batch_for_vram(dims, gpu_vram_mb, default_gpu_batch, act_bytes);
            if (fit >= 1) {
                std::println(stderr,
                             "configure warning: cached DEFAULT_GPU_BATCH={} needs ~{} MiB but only {} MiB VRAM "
                             "is available -- clamping to {}. Re-tune (`sub0llm-tune --backend gpu`) for the optimum.",
                             default_gpu_batch, need, gpu_vram_mb, fit);
                default_gpu_batch = fit;
            } else {
                std::println(stderr,
                             "configure error: the model needs ~{} MiB for even batch 1 but only {} MiB VRAM is "
                             "available. Reduce the model size (or build CPU-only) before rebuilding.",
                             sub0::memplan::train_resident_mb(dims, 1, act_bytes), gpu_vram_mb);
                return 1;
            }
        }
    }

    // --- corpus header: the model identity (dims + vocab + tokenizer scheme + paths), per corpus ---
    cos << "// AUTO-GENERATED by sub0-configure (corpus-specific). Do not edit.\n";
    cos << "// Model dimensions, vocabulary and tokenizer scheme baked from the corpus.\n";
    cos << "#pragma once\n\n";
    cos << "constexpr int  D_MODEL     = " << d_model  << ";\n";
    cos << "constexpr int  N_LAYERS    = " << n_layers << ";\n";
    cos << "constexpr int  N_HEADS     = " << n_heads  << ";\n";
    // Grouped-query attention: N_KV_HEADS key/value heads shared across N_HEADS query heads
    // (N_KV_HEADS == N_HEADS is plain MHA, the default). Shrinks Wk/Wv and the KV cache by
    // N_HEADS/N_KV_HEADS -- see op_attn in backend_cpu.cpp and layout.hpp's Wk/Wv entries.
    cos << "constexpr int  N_KV_HEADS  = " << n_kv_heads << ";\n";
    // LoopSplit: an un-looped head block, a middle block of LOOP_MIDDLE_LAYERS re-executed
    // LOOP_REPEATS times reusing the SAME weights, then an un-looped tail -- effective depth at a
    // fixed parameter count. 0/1 is off. See layout.hpp's make_layer_execution_order().
    cos << "constexpr int  LOOP_MIDDLE_LAYERS = " << loop_middle  << ";\n";
    cos << "constexpr int  LOOP_REPEATS       = " << loop_repeats << ";\n";
    // Depth attention: every DEPTH_ATTN_STRIDE-th execution appends its (K, V) to a per-window depth
    // cache, and every execution replaces its value vector with a softmax-weighted mixture over that
    // cache plus its own -- a distribution across DEPTH, per (batch, kv-head, position). K is passed
    // through unchanged; only V is rewritten. Adds NO parameters, so PARAM_FLOATS and the checkpoint
    // shape are untouched -- which is exactly why it must enter ARCH_FINGERPRINT instead (it changes
    // computation invisibly to nfloat). 0 = off. See docs/DEPTH_ATTENTION.md for the derivation.
    cos << "constexpr int  DEPTH_ATTN_STRIDE  = " << depth_attn_stride << ";\n";
    // Gated DeltaNet Stage 1: every Nth layer (0-indexed, N = stride) stays softmax attention; the rest
    // become GDN layers (layout.hpp's GDN_SCHEDULE, op_gdn/gdn_math.hpp's CPU forward). 0 = off, every
    // layer softmax attention. See docs/GATED_DELTANET.md.
    cos << "constexpr int  GDN_FULL_ATTN_STRIDE = " << gdn_full_attn_stride << ";\n";
    // N-gram embeddings: additional per-position features from rolling polynomial-hash token n-grams,
    // added into the input embedding ("concat" fusion mode, Nanbeige's NanbeigeNgramEmbedding). 0 = off.
    // Unlike depth attention this DOES add parameters (see layout.hpp's NGRAM_EMBED section), so it is
    // discriminated by PARAM_FLOATS rather than ARCH_FINGERPRINT. See docs/NGRAM_EMBEDDING.md.
    cos << "constexpr int  NGRAM_MAX_N         = " << ngram_max_n << ";\n";
    cos << "constexpr int  NGRAM_TABLES_PER_ORDER = " << ngram_tables << ";\n";
    cos << "constexpr int  NGRAM_TABLE_SIZE    = " << ngram_table_size << ";\n";
    // Gated Residual (hyper-connections) Stage 1: HC_COUNT parallel residual streams read/write-gated
    // per sub-block (layout.hpp's USE_GATED_RESIDUAL/GR_DIMS). 0 = off, the default. See
    // docs/GATED_RESIDUAL.md.
    cos << "constexpr int  HC_COUNT   = " << hc_count << ";\n";
    cos << "constexpr int  HC_LOWRANK = " << hc_lowrank << ";\n";
    // Mixture of Experts Stage 0/1 (layout.hpp's USE_MOE/MOE_DIMS). 0 = off, the default. See docs/MOE.md.
    cos << "constexpr int  NUM_EXPERTS     = " << num_experts << ";\n";
    cos << "constexpr int  EXPERTS_PER_TOK = " << experts_per_tok << ";\n";
    // RoPE position scaling (context extension): 0 = none, 1 = linear (divide the position by
    // ROPE_SCALE_FACTOR before forming the angle). See layout.hpp's ROPE_POS_SCALE.
    cos << "constexpr int   ROPE_SCALING      = " << rope_scaling << ";\n";
    cos << "constexpr float ROPE_SCALE_FACTOR = " << std::format("{:.4f}", rope_scale_factor) << "f;\n";
    cos << "constexpr int  SEQ_LEN     = " << seq_len  << ";\n";
    // D_FF: 4*D_MODEL for the plain FFN; a narrower, param-matched width for the gated (SwiGLU) FFN
    // -- see sub0::config::d_ff_for's doc comment (config_util.hpp) for the derivation.
    cos << "constexpr int  D_FF        = " << d_ff << ";\n";
    cos << "constexpr int  D_HEAD      = D_MODEL / N_HEADS;\n";
    cos << "constexpr bool USE_TERNARY = " << (ternary ? "true" : "false") << ";\n";
    // SwiGLU-gated FFN (Wgate/Wup/Wdown, no FFN bias) vs the plain 2-matrix GELU+bias FFN -- see
    // include/sub0/layout.hpp. Valid with any --compute backend (swiglu_kernel/swiglu_act_kernel/
    // swiglu_backward_act_kernel in backend_cuda.cu).
    cos << "constexpr bool USE_GATED_FFN = " << (gated_ffn ? "true" : "false") << ";\n";
    // Tied embeddings: the LM head reuses tok_emb (transposed) instead of its own matrix+bias --
    // see op_tied_head in backend_cpu.cpp. No dims/checkpoint-shape interaction with USE_GATED_FFN,
    // the two are independent axes.
    cos << "constexpr bool USE_TIED_EMBEDDINGS = " << (tie_embeddings ? "true" : "false") << ";\n";
    // QK-norm: RMSNorm applied per-head to Q/K right after their projection, before RoPE -- see
    // op_qknorm in backend_cpu.cpp (CPU) / qknorm_act_kernel in backend_cuda.cu (GPU). Valid with
    // any --compute backend; independent axis from USE_GATED_FFN/USE_TIED_EMBEDDINGS.
    cos << "constexpr bool USE_QK_NORM = " << (qk_norm ? "true" : "false") << ";\n";
    cos << "constexpr int  VOCAB       = " << vocab << ";\n";
    cos << "// --- Positional encoding (compile-time) --------------------------------\n";
    cos << "enum class PosEncoding { Absolute, Rope };\n";
    cos << "constexpr PosEncoding POS_ENCODING = PosEncoding::" << (pos_encoding == 0 ? "Absolute" : "Rope") << ";\n";
    cos << "// ROPE_THETA: RoPE frequency base; angle(pos, pair m) = pos * THETA^(-2m/D_HEAD).\n";
    cos << "constexpr float       ROPE_THETA   = " << std::format("{:.1f}", rope_theta) << "f;\n\n";
    emit_path(cos, "DEFAULT_CORPUS",     abspath);
    emit_path(cos, "DEFAULT_CORPUS_TOK", std::filesystem::absolute(tok_path).string());
    emit_path(cos, "DEFAULT_TOKENIZER",  std::filesystem::absolute(tkz_path).string());

    // --- system header: precision + cached hardware facts + tuned runtime defaults, per machine ---
    sos << "// AUTO-GENERATED by sub0-configure (system-specific). Do not edit.\n";
    sos << "// Precision + hardware facts + tuned runtime defaults; regenerate per machine / after `tune`.\n";
    sos << "#pragma once\n\n";
    sos << "// --- Reduced precision: per-section Dtype (FP32 accumulate + FP32 master weights) ---\n";
    sos << "enum class Dtype { F32, BF16, F16, Q8, Q4 };  // F16/Q8/Q4 reserved; only F32/BF16 wired\n";
    sos << "constexpr bool  FLOAT16_OK    = " << (f16_ok ? "true" : "false") << ";\n";
    sos << "constexpr Dtype FLOAT16_KIND  = Dtype::BF16;\n";
    sos << "constexpr bool  BF16_OK       = " << (f16_ok ? "true" : "false") << ";\n";
    sos << "constexpr Dtype GEMM_DTYPE    = Dtype::" << gemm_dt << ";  // block GEMM inputs\n";
    sos << "constexpr Dtype ACT_DTYPE     = Dtype::" << act_dt  << ";  // saved activations (VRAM)\n";
    sos << "constexpr Dtype MASTER_DTYPE  = Dtype::F32;\n";
    sos << "constexpr Dtype HEAD_DTYPE    = Dtype::F32;\n\n";
    sos << "// --- Cached hardware facts + persisted tuned runtime defaults -----------\n";
    sos << "constexpr int  HW_CONCURRENCY            = " << hw_concurrency  << ";\n";
    sos << "constexpr int  MAX_WORKERS               = " << max_workers     << ";\n";
    sos << "constexpr int  DEFAULT_THREADS           = " << default_threads << ";\n";
    sos << "constexpr int  DEFAULT_WINDOWS_PER_THREAD = " << default_wpt    << ";\n";
    sos << "// DEFAULT_GPU_BATCH: tuned device-training minibatch, or (until `tune --backend gpu` runs) a\n";
    sos << "// VRAM-scaled estimate (gpu_batch_estimate) -- see its own doc comment in config_util.hpp.\n";
    sos << "constexpr int  DEFAULT_GPU_BATCH         = " << default_gpu_batch << ";\n\n";
    sos << "// --- Compute backend ---------------------------------------------------\n";
    sos << "// HAS_CUDA: the CUDA device backend was built (CMake's existence check). COMPUTE_MODE: the\n";
    sos << "// backend the configurator chose to USE (GPU by default when HAS_CUDA, else CPU; --compute pins).\n";
    sos << "enum class ComputeBackend { Cpu, Gpu, Hybrid };\n";
    sos << "constexpr bool           HAS_CUDA     = " << (has_cuda ? "true" : "false") << ";\n";
    sos << "constexpr ComputeBackend COMPUTE_MODE = ComputeBackend::" << (compute == 1 ? "Gpu" : compute == 2 ? "Hybrid" : "Cpu") << ";\n";
    sos << "constexpr int            CUDA_ARCH    = " << cuda_arch   << ";\n";
    sos << "constexpr int            GPU_VRAM_MB  = " << gpu_vram_mb << ";  // dedicated VRAM (fast budget)\n";
    sos << "// GPU_SHARED_MEM_MB: WDDM shared memory the GPU overflows into (no OOM, but THRASHES over PCIe).\n";
    sos << "constexpr int            GPU_SHARED_MEM_MB = " << gpu_shared_mb << ";\n";
    sos << "// CUDA_TF32: tuned GPU knob (`tune` persists the measured winner).\n";
    sos << "constexpr bool           CUDA_TF32    = " << (cuda_tf32 ? "true" : "false") << ";\n\n";
    emit_path(sos, "DEFAULT_TUNE_CACHE", tune_cache_abs);

    // --- umbrella: the single header the engine includes ---
    os << "// AUTO-GENERATED by sub0-configure. Do not edit. Umbrella over the split corpus/system headers.\n";
    os << "#pragma once\n";
    os << "#include \"sub0_corpus.hpp\"\n";
    os << "#include \"sub0_system.hpp\"\n";

    // 11. Report the real numbers (the full scan/learn diagnostics only apply when this run
    //     actually derived them; a reused tokenizer gets a short confirmation instead).
    if (!reused) {
        const auto& word_syms   = S.word_syms;
        const sub0::casing::TokStats& st = S.st;
        const std::size_t raw_bytes = S.raw_bytes, norm_bytes = S.norm_bytes;
        const long long quote_repl = S.quote_repl;
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
        std::println(stderr, "base symbols / word pieces / vocab: {} / {} / {}", n_base, vocab - n_base, vocab);
        // Word sub-token count distribution (N = BPE pieces per word). In the JOIN scheme (schemeV3+,
        // casing.hpp's kSchemeVersion) this is the word-encoding lever: N=1 bare, N>=2 SPELL-encapsulated
        // -- so N2+N>=3 together is the real SPELL rate (a bare JOIN between two piece ids used to mark
        // N==2 instead, but that shape is indistinguishable from an ordinary word glued to trailing
        // punctuation via the SAME general-glue JOIN -- see docs/TOKENIZER_DESIGN.md §4). Frequency-
        // weighted, so it reflects the real token stream (common words dominate); the N2 vs N>=3 split is
        // kept separate purely as informational detail, not because they encode differently anymore.
        {
            long long occ[4] = {0, 0, 0, 0};   // by occurrence: N==1, ==2, ==3, >=4
            long long tot = 0;
            for (std::size_t w = 0; w < word_syms.size(); ++w) {
                const std::size_t N = word_syms[w].size();
                const long long f = S.word_freq[w];
                occ[N <= 1 ? 0 : N == 2 ? 1 : N == 3 ? 2 : 3] += f;
                tot += f;
            }
            const double d = static_cast<double>(std::max<long long>(1, tot));
            std::println(stderr, "word sub-token N (by occ):       N1 {:.1f}% / N2 {:.1f}% / N>=3 {:.1f}% (N2+N>=3 = SPELL rate)",
                         100.0 * static_cast<double>(occ[0]) / d, 100.0 * static_cast<double>(occ[1]) / d,
                         100.0 * static_cast<double>(occ[2] + occ[3]) / d);
        }
        if (emit_tok) {
            std::println(stderr, "total tokens:                    {}", token_count);
            std::println(stderr, "documents (<|endoftext|>-marked): {}", doc_count);
            // A corpus with no EOS markers scans to a single document, and training would then draw
            // windows spanning unrelated documents with nothing to show for it. The blank-line fallback
            // used to mask that; with the fallback gone it must fail loudly instead. doc_count counts
            // document STARTS and is seeded with 1 for "document 0 begins at token 0", so <= 1 means
            // not one marker was seen in the entire corpus.
            if (doc_count <= 1 && token_count > 0) {
                std::println(stderr,
                    "error: no <|endoftext|> markers found in '{}' -- the whole corpus scans as ONE "
                    "document, so training windows would span unrelated documents. Re-extract it with "
                    "the marker between documents (see the extraction scripts); the blank-line fallback "
                    "that used to paper over this has been removed as unsound.", corpus);
                return 1;
            }
            std::println(stderr, "compression (bytes/token):       {:.3f}", bytes_per_tok);
            // Feed this REAL measurement back into the cross-corpus calibration (config_util.hpp's
            // TokenCalibration) so the NEXT corpus's auto-sizing starts from a slightly better prior.
            // Only reachable here (a fresh, non-reused, fully-pretokenized run), never on a cache hit.
            // Keyed by corpus path and UPSERTED (not accumulated): this same run can be reached again
            // from a different build dir that forces a fresh re-tokenize of the SAME corpus (a new
            // --dmodel/--layers/--heads pin, a fresh checkout, ...), and a flat running total would
            // double-count it every time -- see upsert_token_calibration()'s own doc comment.
            if (token_count > 0) {
                sub0::config::TokenCalibration cal;
                if (std::ifstream cf("data/tokenizer_calibration.txt"); cf) cal = sub0::config::parse_token_calibration(cf);
                sub0::config::upsert_token_calibration(
                    cal, std::filesystem::weakly_canonical(corpus).string(), norm_bytes, token_count);
                if (std::ofstream cf("data/tokenizer_calibration.txt"); cf) cf << sub0::config::format_token_calibration(cal);
            }
        } else {
            std::println(stderr, "corpus.tok:                      skipped (--corpus-pretok 0; on-demand)");
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
        std::println(stderr, "corpus.tok / tokenizer.tok:      {} | {}",
                     emit_tok ? tok_path.string() : std::string("(on-demand)"), tkz_path.string());
        std::println(stderr, "-----------------------------------------------");
        if (!tok_rt) { std::println(stderr, "configure error: tokenization round-trip failed"); return 1; }
    } else {
        std::println(stderr, "--- BPE truecasing tokenizer (reused, unchanged since the last configure) ---");
        std::println(stderr, "base symbols / vocab:            {} / {}", n_base, vocab);
        if (emit_tok) std::println(stderr, "total tokens / documents:        {} / {}", token_count, doc_count);
        std::println(stderr, "corpus.tok / tokenizer.tok:      {} | {}",
                     emit_tok ? tok_path.string() : std::string("(on-demand)"), tkz_path.string());
        std::println(stderr, "-----------------------------------------------");
    }

    // Summary comes from the SAME X-macro that drives config.json (registry::describe_config), NOT a
    // hand-listed subset: this banner used to omit every axis added after it was written, so three
    // variants of a --kv-heads A/B printed identical text. A new SUB0_RUN_CONFIG_FIELDS row now shows
    // up here automatically.
    sub0::registry::RunConfig shown;
    shown.d_model     = d_model;      shown.n_layers    = n_layers;
    shown.n_heads     = n_heads;      shown.n_kv_heads  = n_kv_heads;
    shown.loop_middle = loop_middle;  shown.loop_repeats = loop_repeats;
    // depth_attn_stride was missing here (pre-existing omission predating this change -- fixed while
    // touching this exact block, per the boy-scout rule: the banner is meant to show every axis a run
    // can differ on, and this one silently never did).
    shown.depth_attn_stride = depth_attn_stride;
    // Same omission, same fix: GDN's own Stage 0 pass added the field to RunConfig but not to this
    // banner (it is always 0 today anyway -- see the static_assert -- so nothing has SHOWN it missing
    // yet, but the banner's whole job is to show every axis a run can differ on).
    shown.gdn_full_attn_stride = gdn_full_attn_stride;
    shown.ngram_max_n = ngram_max_n; shown.ngram_tables = ngram_tables;
    shown.ngram_table_size = ngram_table_size;
    shown.hc_count = hc_count; shown.hc_lowrank = hc_lowrank;
    shown.num_experts = num_experts; shown.experts_per_tok = experts_per_tok;
    shown.rope_scaling = rope_scaling; shown.rope_scale_fac = rope_scale_factor;
    shown.rope_theta  = rope_theta;
    shown.seq_len     = seq_len;      shown.vocab       = vocab;
    shown.ternary     = ternary;      shown.pos_encoding = pos_encoding;
    shown.gated_ffn   = gated_ffn;    shown.tied_embeddings = tie_embeddings;
    shown.qk_norm     = qk_norm;
    std::println("sub0-configure: vocab={} (base {} + {} {}) | {} -> {}",
                 vocab, n_base, vocab - n_base, tkz.max_piece > 0 ? "unigram pieces" : "BPE merges",
                 sub0::registry::describe_config(shown), out);
    return 0;
}
