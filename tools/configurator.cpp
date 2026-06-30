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
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <format>
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

#include "sub0/memplan.hpp"  // train_resident_mb: predict the GPU batch footprint to fail misconfig early

#include "sub0/casing.hpp"
#include "sub0/tokenizer.hpp"  // sub0::tok — the shared truecasing + BPE tokenizer (scan/learn/serialize)
#include "sub0/unigram.hpp"    // sub0::tok::learn_unigram — the Unigram LM vocabulariser (A/B vs BPE)

namespace tok = sub0::tok;

namespace {

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
void dump_vocab_files(const tok::Scan& S, const tok::Tokenizer& tkz, const std::string& prefix) {
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
        const int nm = static_cast<int>(tkz.merge_count.size());
        std::vector<long long> cum(static_cast<std::size_t>(nm) + 1, 0);
        for (int i = 0; i < nm; ++i) cum[static_cast<std::size_t>(i) + 1] = cum[static_cast<std::size_t>(i)] + tkz.merge_count[static_cast<std::size_t>(i)];
        const long long total_reduction = cum[static_cast<std::size_t>(nm)];
        auto tokens_at = [&](int k) { return total_bytes - cum[static_cast<std::size_t>(std::min(k, nm))]; };
        auto bpt_at    = [&](int k) { const long long tk = tokens_at(k); return tk > 0 ? static_cast<double>(total_bytes) / static_cast<double>(tk) : 0.0; };
        auto vocab_at_frac = [&](double frac) {
            const long long target = static_cast<long long>(frac * static_cast<double>(total_reduction));
            int k = 0; while (k < nm && cum[static_cast<std::size_t>(k)] < target) ++k;
            return tkz.n_base + k;
        };
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
    std::string out;
    int d_model      = 96;
    int n_layers     = 3;
    int n_heads      = 4;
    int seq_len      = 64;
    int ternary      = 0;
    int pos_encoding = 1;       // 0 = absolute learned, 1 = RoPE (default)
    double rope_theta = 10000.0;// RoPE frequency base
    int bf16         = 2;       // float16 capability: 0=off, 1=on, 2=AUTO (on if GPU >= sm_80)
    int prec_gemm    = 9;       // GEMM input precision: 0=F32,1=BF16,2=F16; 9=AUTO (16b if capable)
    int prec_act     = 9;       // saved-activation storage precision: same codes; 9=AUTO
    int vocab_target = 2048;
    int min_merge    = 2;
    int emit_tok     = 1;
    int join_scheme  = 0;       // 1 = JOIN/implicit-space tokenizer (complete 256 base + spacing markers)
    std::string tune_cache;
    int has_cuda    = 0;   // CUDA toolkit + device detected at configure time
    int cuda_arch   = 0;   // GPU compute capability as an int (e.g. 120 for sm_120)
    int gpu_vram_mb = 0;   // dedicated GPU VRAM in MB
    int gpu_shared_mb = 0; // shared/overflow system memory the GPU can address (WDDM), MB
    int compute     = 0;   // resolved backend: 0=CPU, 1=GPU, 2=HYBRID
    int cuda_tf32   = 0;   // bake TF32 tensor-core GEMM math on the GPU backend (tuned knob)
    std::string dump_vocab; // prefix for the readable vocabulary-analysis dumps (empty = off)

    app.add_option("--corpus", corpus,  "Training corpus path (drives vocabulary derivation)")
       ->required()->check(CLI::ExistingFile);
    app.add_option("-o",       out,     "Output generated header path")->required();
    app.add_option("--dmodel", d_model, "Embedding / residual width")->capture_default_str();
    app.add_option("--layers", n_layers,"Transformer block count")->capture_default_str();
    app.add_option("--heads",  n_heads, "Attention head count")->capture_default_str();
    app.add_option("--seq",    seq_len, "Context window length")->capture_default_str();
    app.add_option("--ternary",ternary, "1 = BitNet-style ternary block weights")
       ->capture_default_str()->check(CLI::Range(0, 1));
    app.add_option("--pos-encoding", pos_encoding,
                   "Positional encoding scheme: 0 = absolute learned, 1 = RoPE (rotary)")
       ->capture_default_str()->check(CLI::Range(0, 1));
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
    app.add_option("--vocab",  vocab_target, "Target BPE vocabulary size (base symbols + markers + merges)")
       ->capture_default_str();
    app.add_option("--min-merge", min_merge, "Stop merging once the best pair occurs fewer than this many times")
       ->capture_default_str();
    app.add_option("--corpus-pretok", emit_tok,
                   "1 = pre-tokenize the whole corpus to corpus.tok; 0 = skip it and let training tokenize "
                   "on demand (avoids the ~2x-on-disk token copy for a huge corpus)")
       ->capture_default_str()->check(CLI::Range(0, 1));
    app.add_option("--join", join_scheme,
                   "1 = JOIN/implicit-space tokenizer (complete 256-byte base + JOIN/NEWLINE/PARA markers; "
                   "single inter-word space is free). 0 = legacy space-as-token scheme.")
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
    tok::Scan S;
    std::unordered_set<std::string> attested;
    long long names_withheld = 0;

    std::chrono::steady_clock::time_point _t1{}, _t2{};
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
        // ids are byte-derived. So the merged scan (hence tokenizer.bin / corpus.tok) is byte-
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

    // Learn the base alphabet + BPE merges from the scan (sub0::tok::learn). This REMAPS
    // S.word_syms in place from raw byte symbols to final token ids, so Pass 3 can emit the
    // tokenized corpus from S.index + S.word_syms. The learned Tokenizer carries the base
    // alphabet, the ordered merges, the per-id expansion and the attested set.
    const tok::Tokenizer tkz = tok::learn(S, attested, {vocab_target, min_merge, join_scheme != 0});

    // Analysis mode: write the readable vocabulary dumps and exit (no corpus.tok / config header).
    if (!dump_vocab.empty()) {
        dump_vocab_files(S, tkz, dump_vocab);
        std::println(stderr, "vocab dumps written: {}.{{corpus_vocab,token_vocab,ngrams}}.txt", dump_vocab);
        return 0;
    }

    // Aliases so the downstream emit / reporting code reads the learned tokenizer + scan stats.
    const auto& word_syms   = S.word_syms;   // now final token-id sequences
    const auto& word_index  = S.index;       // raw-key -> word id
    const auto& expansion   = tkz.expansion;
    const int   n_base      = tkz.n_base;
    const int   vocab       = tkz.vocab;
    const auto& merges      = tkz.merges;
    auto sym_to_base = [&](int s) { return tkz.sym_to_base(s); };
    const sub0::casing::TokStats& st = S.st;
    const std::size_t raw_bytes = S.raw_bytes, norm_bytes = S.norm_bytes;
    const long long quote_repl = S.quote_repl;

    const auto _t3 = std::chrono::steady_clock::now();

    const std::filesystem::path gen_dir  = std::filesystem::path(out).parent_path();
    const std::filesystem::path tok_path = gen_dir / "corpus.tok";
    const std::filesystem::path tkz_path = gen_dir / "tokenizer.bin";

    // --- Pass 3 (optional): stream the corpus once more and emit the merged token stream to
    //     corpus.tok incrementally (never materialised in memory). Skipped when --corpus-pretok
    //     0: training then tokenizes windows on demand from the raw corpus + tokenizer.bin,
    //     avoiding the ~2x-on-disk token copy for a huge corpus. Each word unit looks up its
    //     post-BPE id sequence in the table; standalone symbols map straight to base ids.
    //     Losslessness is verified per chunk by reconstructing the base symbols; the id
    //     array's length is back-patched after the count is known.
    std::size_t token_count = 0;
    std::size_t doc_count   = 0;
    bool tok_rt = true;
    if (emit_tok) {
        std::ofstream ts(tok_path, std::ios::binary);
        if (!ts) { std::println(stderr, "configure error: cannot write '{}'", tok_path.string()); return 1; }
        {
            const std::uint32_t magic = 0x44543053u,                // "S0TD" (doc-aware corpus.tok)
                                vfield = static_cast<std::uint32_t>(vocab), zero = 0u;
            ts.write(reinterpret_cast<const char*>(&magic),  sizeof magic);
            ts.write(reinterpret_cast<const char*>(&vfield), sizeof vfield);
            ts.write(reinterpret_cast<const char*>(&zero),   sizeof zero);  // ntok, patched below
            ts.write(reinterpret_cast<const char*>(&zero),   sizeof zero);  // ndoc, patched below
        }
        // Document boundaries: get_fineweb.py separates documents with a blank line ("\n\n"), and a
        // newline is always emitted as its own base token (non-alpha bytes never BPE-merge), so a
        // run of >=2 newline tokens marks a document break -- the next token starts a new document.
        // Record each document's start token index so training keeps windows inside one document.
        const bool join = (join_scheme != 0);
        const std::int32_t nl_id      = static_cast<std::int32_t>(sym_to_base(10));      // base id of '\n'
        const std::int32_t para_id    = join ? static_cast<std::int32_t>(tkz.para_id)    : -1;
        const std::int32_t newline_id = join ? static_cast<std::int32_t>(tkz.newline_id) : -1;
        std::vector<std::uint32_t> doc_starts{0u};                  // document 0 begins at token 0
        int nl_run = 0;
        bool rt_reported = false;                                   // print the first round-trip divergence once
        std::vector<std::int32_t> out_tokens;     // per-chunk emit buffer (bounded by the chunk)
        std::vector<int> recon;                   // per-chunk reconstruction (legacy round-trip 2)
        for_each_chunk(corpus, [&](std::string_view chunk) {
            long qr = 0;
            const std::string norm = normalize_text(std::string(chunk), qr);
            out_tokens.clear();
            if (join) {
                // JOIN scheme: encode the chunk with the learned tokenizer (implicit single space +
                // JOIN/NEWLINE/PARA + verbatim fallback). The single contract is the round-trip
                // detokenize(encode(chunk)) == normalize_text(chunk).
                const std::vector<int> ids = tok::encode(tkz, std::string(chunk));
                const std::string dec = tok::detokenize(tkz, ids);
                if (dec != norm) {
                    tok_rt = false;
                    if (!rt_reported) {                            // show the first divergence (expected vs actual)
                        rt_reported = true;
                        std::size_t p = 0;
                        while (p < dec.size() && p < norm.size() && dec[p] == norm[p]) ++p;
                        const std::size_t lo = p > 48 ? p - 48 : 0;
                        auto vis = [](std::string s) {             // make newlines/tabs visible
                            std::string o; for (char c : s) {
                                if (c == '\n') o += "\\n"; else if (c == '\t') o += "\\t"; else o += c;
                            } return o;
                        };
                        std::println(stderr, "ROUND-TRIP FAIL at byte {} (norm {}B / dec {}B):", p, norm.size(), dec.size());
                        std::println(stderr, "  expected: ...{}...", vis(norm.substr(lo, 96)));
                        std::println(stderr, "  actual:   ...{}...", vis(dec.substr(lo, 96)));
                    }
                }
                out_tokens.assign(ids.begin(), ids.end());
            } else {
                const std::vector<int> stream = truecase_tokenize(norm, attested, nullptr);
                if (detokenize(stream) != norm) tok_rt = false;     // round-trip 1: truecasing
                for (std::size_t i = 0, n = stream.size(); i < n;) {
                    const std::size_t end = word_unit_end(stream, i);
                    if (end == i) { out_tokens.push_back(sym_to_base(stream[i])); ++i; continue; }
                    std::vector<int> seq(stream.begin() + static_cast<std::ptrdiff_t>(i),
                                         stream.begin() + static_cast<std::ptrdiff_t>(end));
                    auto it = word_index.find(seq_key(seq));
                    if (it == word_index.end()) {                   // unreachable if passes agree
                        for (int b : seq) out_tokens.push_back(sym_to_base(b));
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
            }
            // Document boundaries (global token index = chunk base + local index). A blank line
            // separates documents: legacy counts >=2 consecutive '\n' tokens; the JOIN scheme emits
            // "\n\n" as one PARA token (weight 2) and a lone "\n" as NEWLINE / '\n' (weight 1), so a
            // cumulative newline weight >= 2 before the next content token marks the break.
            for (std::size_t li = 0; li < out_tokens.size(); ++li) {
                const std::int32_t tk = out_tokens[li];
                const int w = join ? (tk == para_id ? 2 : (tk == newline_id || tk == nl_id ? 1 : 0))
                                   : (tk == nl_id ? 1 : 0);
                if (w > 0) { nl_run += w; continue; }
                if (nl_run >= 2) doc_starts.push_back(static_cast<std::uint32_t>(token_count + li));
                nl_run = 0;
            }
            ts.write(reinterpret_cast<const char*>(out_tokens.data()),
                     static_cast<std::streamsize>(out_tokens.size() * sizeof(std::int32_t)));
            token_count += out_tokens.size();
        });
        // Append the document-start index after the token stream, then back-patch ntok + ndoc.
        ts.write(reinterpret_cast<const char*>(doc_starts.data()),
                 static_cast<std::streamsize>(doc_starts.size() * sizeof(std::uint32_t)));
        doc_count = doc_starts.size();
        {
            const std::uint32_t nfield = static_cast<std::uint32_t>(token_count);
            const std::uint32_t dfield = static_cast<std::uint32_t>(doc_starts.size());
            ts.seekp(8, std::ios::beg);
            ts.write(reinterpret_cast<const char*>(&nfield), sizeof nfield);
            ts.seekp(12, std::ios::beg);
            ts.write(reinterpret_cast<const char*>(&dfield), sizeof dfield);
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
        tok::serialize(tkz, tz);
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

    // Fail the build on a GPU batch that cannot fit. DEFAULT_GPU_BATCH sizes the resident device
    // training scratch; on Windows an over-budget cudaMalloc does not OOM, it silently spills to WDDM
    // shared memory and thrashes over PCIe (~10x slower) -- a silent performance cliff, not a crash.
    // We know the model dims and the VRAM budget right here, so pre-compute the footprint (the SAME
    // pure model the backend is validated against) and turn the misconfiguration into a hard configure
    // error instead of baking a batch that will spill at the first training step.
    if (has_cuda && gpu_vram_mb > 0) {
        const sub0::memplan::Dims dims{ d_model, n_layers, n_heads, 4 * d_model, seq_len, vocab };
        const int need = sub0::memplan::train_resident_mb(dims, default_gpu_batch);
        if (need > gpu_vram_mb) {
            std::println(stderr,
                         "configure error: DEFAULT_GPU_BATCH={} needs ~{} MiB of resident VRAM but only "
                         "{} MiB is available.\n"
                         "       Re-tune (`sub0llm tune --backend gpu`) so a fitting batch is persisted, or "
                         "reduce the model size, before rebuilding.",
                         default_gpu_batch, need, gpu_vram_mb);
            return 1;
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
    os << "constexpr int  VOCAB       = " << vocab << ";\n";
    // JOIN_TOKENIZER: the implicit-space tokenizer scheme is baked in (it changes corpus.tok +
    // tokenizer.bin meaning, so models built under it are incompatible with legacy ones). The
    // registry tags the model dir with it; the engine uses it only for provenance/compatibility.
    os << "constexpr bool JOIN_TOKENIZER = " << (join_scheme ? "true" : "false") << ";\n\n";
    // --- Positional encoding (compile-time, an enum for future schemes) -----
    // Absolute = a learned pos_emb[SEQ_LEN, D_MODEL] added to the token embedding (cannot
    // extrapolate past SEQ_LEN). Rope = rotary embeddings applied to Q/K inside attention,
    // encoding RELATIVE position with no learned table and far better length behaviour.
    os << "// --- Positional encoding (compile-time) --------------------------------\n";
    os << "enum class PosEncoding { Absolute, Rope };\n";
    os << "constexpr PosEncoding POS_ENCODING = PosEncoding::"
       << (pos_encoding == 0 ? "Absolute" : "Rope") << ";\n";
    os << "// ROPE_THETA: RoPE frequency base; angle(pos, pair m) = pos * THETA^(-2m/D_HEAD).\n";
    os << "constexpr float       ROPE_THETA   = " << std::format("{:.1f}", rope_theta) << "f;\n\n";
    // --- Reduced precision (per-section, baked) -----------------------------
    // Float16 capability is a detected hardware fact: BF16 needs sm_80+. We currently support BF16
    // only; FP16 (and future integer quant) are reserved enum values so per-section knobs stay
    // forward-compatible, but selecting them errors here until the backends implement them. Each
    // pipeline section (GEMM inputs, saved activations) picks a Dtype independently; numerically
    // sensitive sections (master weights, head/logits/softmax) stay FP32.
    const bool f16_ok = (bf16 == 1) || (bf16 == 2 && cuda_arch >= 80);
    // Resolve a per-section code (0=F32,1=BF16,2=F16,9=AUTO) to a Dtype name, erroring on the
    // not-yet-supported choices. AUTO -> the capable 16-bit float (BF16) else F32.
    const auto resolve_dtype = [&](int code, const char* section) -> std::string {
        switch (code) {
            case 0: return "F32";
            case 1: if (!f16_ok) { std::fprintf(stderr, "configure error: BF16 %s requested but no BF16-capable GPU\n", section); std::exit(2); } return "BF16";
            case 2: std::fprintf(stderr, "configure error: F16 %s not yet supported (BF16 only)\n", section); std::exit(2);
            default: return f16_ok ? "BF16" : "F32";   // AUTO
        }
    };
    const std::string gemm_dt = resolve_dtype(prec_gemm, "GEMM");
    // BF16 activation storage (FFN) cuts train scratch; checked by direction/finiteness, not raw f32
    // parity. AUTO enables it when float16-capable; --prec-act F32 forces the tight-parity baseline.
    const std::string act_dt  = resolve_dtype(prec_act, "activations");
    os << "// --- Reduced precision: per-section Dtype (FP32 accumulate + FP32 master weights) ---\n";
    os << "// F16/Q8/Q4 are reserved for future backends; only F32/BF16 are wired today.\n";
    os << "enum class Dtype { F32, BF16, F16, Q8, Q4 };\n";
    os << "constexpr bool  FLOAT16_OK    = " << (f16_ok ? "true" : "false") << ";  // GPU supports a 16-bit float\n";
    os << "constexpr Dtype FLOAT16_KIND  = Dtype::BF16;  // which 16-bit float this platform uses\n";
    os << "constexpr bool  BF16_OK       = " << (f16_ok ? "true" : "false") << ";  // alias: BF16 is the supported f16\n";
    os << "constexpr Dtype GEMM_DTYPE    = Dtype::" << gemm_dt << ";  // block GEMM inputs\n";
    os << "constexpr Dtype ACT_DTYPE     = Dtype::" << act_dt  << ";  // saved activations (VRAM)\n";
    os << "constexpr Dtype MASTER_DTYPE  = Dtype::F32;   // params + AdamW moments stay FP32\n";
    os << "constexpr Dtype HEAD_DTYPE    = Dtype::F32;   // lm_head + logits + softmax stay FP32\n\n";
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
    // Word sub-token count distribution (N = BPE pieces per word). In the JOIN scheme this is the
    // word-encoding lever: N=1 bare, N=2 one JOIN, N>=3 SPELL-encapsulated -- so N>=3 is the SPELL
    // rate. Frequency-weighted, so it reflects the real token stream (common words dominate).
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
        std::println(stderr, "word sub-token N (by occ):       N1 {:.1f}% / N2 {:.1f}% / N>=3 {:.1f}% (SPELL in join mode)",
                     100.0 * static_cast<double>(occ[0]) / d, 100.0 * static_cast<double>(occ[1]) / d,
                     100.0 * static_cast<double>(occ[2] + occ[3]) / d);
    }
    if (emit_tok) {
        std::println(stderr, "total tokens:                    {}", token_count);
        std::println(stderr, "documents (\\n\\n-separated):       {}", doc_count);
        std::println(stderr, "compression (bytes/token):       {:.3f}", bytes_per_tok);
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
    std::println(stderr, "corpus.tok / tokenizer.bin:      {} | {}",
                 emit_tok ? tok_path.string() : std::string("(on-demand)"), tkz_path.string());
    std::println(stderr, "-----------------------------------------------");
    if (!tok_rt) { std::println(stderr, "configure error: tokenization round-trip failed"); return 1; }

    std::println("sub0-configure: vocab={} (base {} + {} merges), d={} L={} H={} seq={}{} -> {}",
                 vocab, n_base, merges.size(), d_model, n_layers, n_heads, seq_len,
                 ternary ? " (ternary)" : "", out);
    return 0;
}
