// backbone_quant_dot_bench.cpp -- O5 phase 2a microbenchmark: sub0::bbqd's fused int8-activation x
// native-quant-weight GEMV (include/sub0/backbone_quant_dot.hpp), at REAL backbone tensor shapes from
// the real Qwen3.8-Flash-Next UD-IQ1_S GGUF shards, threaded at 1/2/4/8/16 threads, DRAM-streamed,
// against the real bf16 `gemv::axpy` primitive (include/sub0/gemv.hpp) at the SAME logical (in, out)
// shape -- the actual kernel decode.cpp's own backbone path uses, not a naive scalar stand-in.
//
// WHAT PHASE 1's VERSION OF THIS FILE GOT WRONG, FIXED HERE (docs/BACKBONE_NATIVE_QUANT.md S8a/S12):
//   1. The bf16 baseline was a hand-rolled scalar accumulate loop, not `gemv::axpy` -- the real kernel
//      every other backbone GEMV in this project goes through (O2, `include/sub0/gemv.hpp`). Fixed: this
//      file links `gemv.hpp` and times `sub0::gemv::axpy<Threads>` directly, at a `[row_elems, out_dim]`
//      bf16 matrix -- the SAME logical (in, out) shape as the native-quant plane's own (row_elems,
//      n_rows), even though the physical layout differs (AXPY's [in,out] vs this header's own
//      DOT-over-[out,in] -- the design doc's own S3a/S3d already name this as a real, unresolved
//      reconciliation question for phase 2's engine wiring, not something a microbenchmark can paper
//      over).
//   2. Every arm spawned fresh `std::thread`s per call and reused ONE cache-resident buffer across every
//      repetition, so neither the threaded numbers nor the single-thread ones said anything about DRAM
//      bandwidth. Fixed: both arms use OpenMP (linked the way `sub0llm-bench-gemv` does, in the root
//      `CMakeLists.txt`) with a persistent team (`#pragma omp parallel for num_threads(Threads)` per
//      call, relying on libomp's own idle-thread pool rather than spawning), and both arms cycle through
//      a POOL of several DISTINCT real byte ranges (native-quant) or distinct random bf16 matrices,
//      sized to exceed 3x this host's 36 MiB L3 combined, so every call in the timed loop streams from
//      DRAM rather than replaying the same cached bytes -- the identical technique `sub0llm-bench-gemv`
//      already established for the bf16 primitive, applied here to the native-quant one too.
//
// REAL SHAPES, REAL BYTES (AGENTS.md S9): the pool for each format is built from REAL tensors (or real
// DISJOINT row-ranges of one real tensor, for Q4_K's two enormous 248,320-row planes) found by scanning
// every shard's own header/tensor table, filtered to the row_elems value most real tensors of that format
// actually share (K-quant sub-block scale/qh layouts are per-format, not per-tensor, so any real tensor
// of the SAME format and row_elems produces directly comparable bytes).
//
// PLAIN main(), NOT Catch2 -- same reasoning as moe_expert_bench.cpp/backbone_dequant_bench.cpp: this
// times real wall-clock across explicit thread counts, which a benchmark harness' own scheduling would
// obscure.
//
// ENGINE-FREE: gguf.hpp + backbone_quant_dot.hpp + bf16.hpp + gemv.hpp only. No sub0_config.hpp, no
// layout.hpp.
//
// Usage: sub0_backbone_quant_dot_bench --gguf <dir> [--seconds N]

#include "sub0/backbone_quant_dot.hpp"
#include "sub0/bf16.hpp"
#include "sub0/gemv.hpp"
#include "sub0/gguf.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <map>
#include <numeric>
#include <random>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using namespace sub0;
using Clock = std::chrono::steady_clock;

namespace {

// Three times this host's 36 MiB L3 (docs' own host-cpu-arrow-lake-hx note) -- the same target
// `sub0llm-bench-gemv` uses, so every timed call in the pool below streams from DRAM.
constexpr double kPoolTargetBytes = 3.0 * 36.0 * 1024.0 * 1024.0;
// Bound how many rows any ONE real tensor contributes, so a single 248,320-row plane (Q4_K's
// token_embd/output.weight) does not dominate this tool's own read time; the pool still reaches its DRAM
// target by drawing several DISJOINT row-ranges from such a tensor instead.
//
// AGENTS.md S13 pass 3: a first cut at 3072 rows/entry under-measured EVERY format's own multi-thread
// scaling and, worse, under-measured the bf16 `gemv::axpy` BASELINE too (its own 8-thread number came in
// well under this host's documented ~50-65 GB/s range) -- both arms were paying a per-call OpenMP
// parallel-region spin-up on every pool entry, not a property of either kernel. 16384 rows/entry brought
// the bf16 baseline into its documented range and is what the numbers in
// docs/BACKBONE_NATIVE_QUANT.md S12 were measured at; a real decode call is smaller than this per layer,
// so amortizing OpenMP region overhead over MORE than one call (persistent-team reuse across layers, not
// just within one) is a real question for phase 2b's wiring, not resolved by this constant alone.
constexpr int kMaxRowsPerEntry = 16384;

double secs(Clock::time_point a, Clock::time_point b) { return std::chrono::duration<double>(b - a).count(); }

const char* type_name(std::uint32_t t) {
    switch (static_cast<gguf::TensorType>(t)) {
        case gguf::TensorType::Q8_0: return "Q8_0";
        case gguf::TensorType::Q4_K: return "Q4_K";
        case gguf::TensorType::Q5_K: return "Q5_K";
        case gguf::TensorType::Q6_K: return "Q6_K";
        default:                     return "?";
    }
}

struct Shard { fs::path path; std::uint64_t data_offset = 0; std::vector<gguf::TensorInfo> tensors; };

/// One real, DRAM-resident byte range: `n_rows` rows of `row_elems` (shared across the whole pool for a
/// given format), read from one real tensor starting at row `row_off`.
struct PoolEntry {
    std::vector<std::uint8_t> bytes;
    int n_rows = 0;
};

std::vector<Shard> load_shards(const std::string& dir) {
    std::vector<Shard> shards;
    std::vector<fs::path> files;
    for (const auto& e : fs::directory_iterator(dir))
        if (e.is_regular_file() && e.path().extension() == ".gguf") files.push_back(e.path());
    std::sort(files.begin(), files.end());
    for (const auto& p : files) {
        std::ifstream f(p, std::ios::binary);
        if (!f) { std::fprintf(stderr, "error: cannot open %s\n", p.string().c_str()); std::exit(2); }
        std::vector<std::uint8_t> head(64ull * 1024 * 1024);
        f.read(reinterpret_cast<char*>(head.data()), static_cast<std::streamsize>(head.size()));
        head.resize(static_cast<std::size_t>(f.gcount()));
        gguf::Reader r(head);
        if (!r.ok()) { std::fprintf(stderr, "error: %s unreadable\n", p.filename().string().c_str()); std::exit(2); }
        Shard sh;
        sh.path = p;
        sh.data_offset = r.data_offset();
        sh.tensors.assign(r.tensors().begin(), r.tensors().end());
        shards.push_back(std::move(sh));
    }
    return shards;
}

/// Every real, eligible (2-D, row-aligned, non-expert, non-PLE) tensor of `type` across every shard, with
/// its own absolute byte offset -- the same selection idiom tests/backbone_quant_dot_tests.cpp's own
/// find_real_picks() uses.
struct Candidate {
    const Shard* shard;   // non-owning; into main()'s own `shards` vector, which outlives every use
    gguf::TensorInfo info;
};
std::vector<Candidate> find_candidates(const std::vector<Shard>& shards, gguf::TensorType type) {
    std::vector<Candidate> out;
    for (const auto& sh : shards)
        for (const auto& t : sh.tensors) {
            if (t.type_raw != static_cast<std::uint32_t>(type)) continue;
            if (t.name.find("_exps") != std::string::npos) continue;
            if (t.name.find("per_layer_token_embd") != std::string::npos) continue;
            if (t.name.find("ple_") != std::string::npos) continue;
            if (t.dims.size() != 2 || t.dims[0] % bbqd::GROUP != 0) continue;
            out.push_back({&sh, t});
        }
    return out;
}

/// Builds a DRAM-sized pool for `type`: real tensors of the row_elems shared by the MOST candidates
/// (K-quant formats mix a few different row_elems across their real consumers -- picking the modal one
/// keeps the whole pool shape-homogeneous, so one ActBlocks activation row serves every entry), read in
/// <= kMaxRowsPerEntry row chunks (possibly several disjoint chunks of the SAME huge tensor) until the
/// pool's total byte size clears kPoolTargetBytes or candidates run out.
struct Pool { int row_elems = 0; std::vector<PoolEntry> entries; double total_bytes = 0.0; };
Pool build_pool(const std::vector<Shard>& shards, gguf::TensorType type) {
    Pool pool;
    auto cands = find_candidates(shards, type);
    if (cands.empty()) return pool;

    std::map<std::uint64_t, int> row_elems_votes;
    for (auto& c : cands) row_elems_votes[c.info.dims[0]]++;
    std::uint64_t modal_row_elems = cands.front().info.dims[0];
    int best_votes = 0;
    for (auto& [re, votes] : row_elems_votes)
        if (votes > best_votes) { best_votes = votes; modal_row_elems = re; }
    pool.row_elems = static_cast<int>(modal_row_elems);

    const gguf::BlockSpec spec = gguf::block_spec(static_cast<std::uint32_t>(type));
    for (auto& c : cands) {
        if (pool.total_bytes >= kPoolTargetBytes) break;
        if (c.info.dims[0] != modal_row_elems) continue;
        const std::uint64_t rows_total = c.info.dims[1];
        for (std::uint64_t row_off = 0; row_off < rows_total && pool.total_bytes < kPoolTargetBytes;
             row_off += static_cast<std::uint64_t>(kMaxRowsPerEntry)) {
            const int n_rows = static_cast<int>(
                std::min<std::uint64_t>(kMaxRowsPerEntry, rows_total - row_off));
            const std::uint64_t elems = static_cast<std::uint64_t>(n_rows) * pool.row_elems;
            const std::uint64_t byte_len = elems / spec.elems * spec.bytes;
            const std::uint64_t row_byte_off = row_off * pool.row_elems / spec.elems * spec.bytes;

            std::ifstream f(c.shard->path, std::ios::binary);
            f.seekg(static_cast<std::streamoff>(c.shard->data_offset + c.info.offset + row_byte_off));
            PoolEntry entry;
            entry.bytes.resize(static_cast<std::size_t>(byte_len));
            f.read(reinterpret_cast<char*>(entry.bytes.data()), static_cast<std::streamsize>(byte_len));
            if (static_cast<std::uint64_t>(f.gcount()) != byte_len) {
                std::fprintf(stderr, "error: short read on %s\n", c.info.name.c_str());
                std::exit(4);
            }
            entry.n_rows = n_rows;
            pool.total_bytes += static_cast<double>(byte_len);
            pool.entries.push_back(std::move(entry));
        }
    }
    return pool;
}

/// One rep of the native-quant pool: every entry's whole GEMV, `Threads`-way OpenMP-parallel per call
/// (a persistent team -- libomp keeps its worker threads alive between `#pragma omp parallel` regions
/// rather than spawning fresh ones, the same assumption `sub0llm-bench-gemv` relies on).
template <int Threads>
double time_native_pool(const Pool& pool, std::uint32_t type_raw, const bbqd::ActBlocks& x,
                        std::vector<float>& out, double min_seconds) {
    double total_bytes_read = 0.0;
    const auto t0 = Clock::now();
    double el = 0.0;
    do {
        for (const auto& entry : pool.entries) {
            const bool ok = bbqd::gemv_plane<Threads>(type_raw, entry.bytes, entry.n_rows, pool.row_elems,
                                                       x, out.data());
            if (!ok) { std::fprintf(stderr, "error: gemv_plane refused a real pool entry\n"); std::exit(5); }
            total_bytes_read += static_cast<double>(entry.bytes.size());
        }
        el = secs(t0, Clock::now());
    } while (el < min_seconds);
    return el / total_bytes_read;   // seconds per byte -- caller turns this into GB/s and us/row-call
}

/// bf16 baseline: `gemv::axpy<Threads>` over a pool of distinct [row_elems, out_dim] bf16 matrices sized
/// the same way (identical technique to sub0llm-bench-gemv's own time_shape), at the SAME logical (in,
/// out) shape as the native-quant pool's own (row_elems, out_dim) -- see this file's own header comment
/// on what "same shape" does and does not claim.
template <int Threads>
double time_axpy_pool(const std::vector<std::vector<bf16>>& pool, int in, int out_dim, const float* x,
                      float* y, double min_seconds) {
    std::size_t calls = 0;
    const auto t0 = Clock::now();
    double el = 0.0;
    do {
        for (const auto& w : pool) gemv::axpy<Threads>(x, Bf16CPtr{w.data()}, in, out_dim, y);
        calls += pool.size();
        el = secs(t0, Clock::now());
    } while (el < min_seconds);
    const double bytes_per_call = static_cast<double>(in) * out_dim * sizeof(bf16);
    return el / (static_cast<double>(calls) * bytes_per_call);   // seconds per byte
}

}  // namespace

int main(int argc, char** argv) {
    std::string gguf_dir;
    double min_seconds = 0.4;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() { return i + 1 < argc ? argv[++i] : ""; };
        if (a == "--gguf") gguf_dir = next();
        else if (a == "--seconds") min_seconds = std::atof(next());
        else { std::fprintf(stderr, "unknown argument '%s'\n", a.c_str()); return 2; }
    }
    if (gguf_dir.empty()) {
        std::fprintf(stderr, "usage: sub0_backbone_quant_dot_bench --gguf <dir> [--seconds N]\n");
        return 2;
    }

    const auto shards = load_shards(gguf_dir);
    constexpr gguf::TensorType kFormats[4] = {gguf::TensorType::Q8_0, gguf::TensorType::Q4_K,
                                               gguf::TensorType::Q5_K, gguf::TensorType::Q6_K};
    constexpr int kThreadCounts[5] = {1, 2, 4, 8, 16};

    for (auto type : kFormats) {
        const Pool pool = build_pool(shards, type);
        if (pool.entries.empty()) {
            std::fprintf(stderr, "error: no real backbone tensor of type %s found\n", type_name(static_cast<std::uint32_t>(type)));
            return 3;
        }
        // Representative out_dim for the bf16 baseline: the pool's own mean entry row count.
        double mean_rows = 0.0;
        for (auto& e : pool.entries) mean_rows += e.n_rows;
        mean_rows /= static_cast<double>(pool.entries.size());
        const int out_dim = std::max(1, static_cast<int>(mean_rows));

        std::printf("\n=== %-6s row_elems=%d out_dim(bf16 baseline)~%d  pool: %zu entries, %.1f MiB ===\n",
                    type_name(static_cast<std::uint32_t>(type)), pool.row_elems, out_dim, pool.entries.size(),
                    pool.total_bytes / (1024.0 * 1024.0));

        std::vector<float> x(static_cast<std::size_t>(pool.row_elems));
        for (int i = 0; i < pool.row_elems; ++i) x[static_cast<std::size_t>(i)] = 0.01f * static_cast<float>((i % 17) - 8);
        bbqd::ActBlocks xq;
        xq.quantize(x.data(), pool.row_elems);
        std::vector<float> native_out(static_cast<std::size_t>(kMaxRowsPerEntry));

        // Distinct bf16 matrices for the axpy baseline, sized to the SAME DRAM target.
        const std::size_t bf16_bytes_per = static_cast<std::size_t>(pool.row_elems) * out_dim * sizeof(bf16);
        const std::size_t bf16_pool_n =
            std::max<std::size_t>(1, static_cast<std::size_t>(kPoolTargetBytes / static_cast<double>(bf16_bytes_per)) + 1);
        std::vector<std::vector<bf16>> bf16_pool(bf16_pool_n,
                                                  std::vector<bf16>(static_cast<std::size_t>(pool.row_elems) * out_dim));
        std::mt19937 rng(11);
        std::uniform_int_distribution<int> bits(0x3c00, 0x3f80);
        for (auto& m : bf16_pool)
            for (auto& v : m) v.bits = static_cast<std::uint16_t>(bits(rng));
        std::vector<float> bf16_y(static_cast<std::size_t>(out_dim));

        std::printf("%-8s %14s %14s %10s\n", "threads", "native us/row", "native GB/s", "bf16 GB/s");
        for (int threads : kThreadCounts) {
            double sec_per_byte = 0.0;
            switch (threads) {
                case 1:  sec_per_byte = time_native_pool<1>(pool, static_cast<std::uint32_t>(type), xq, native_out, min_seconds); break;
                case 2:  sec_per_byte = time_native_pool<2>(pool, static_cast<std::uint32_t>(type), xq, native_out, min_seconds); break;
                case 4:  sec_per_byte = time_native_pool<4>(pool, static_cast<std::uint32_t>(type), xq, native_out, min_seconds); break;
                case 8:  sec_per_byte = time_native_pool<8>(pool, static_cast<std::uint32_t>(type), xq, native_out, min_seconds); break;
                case 16: sec_per_byte = time_native_pool<16>(pool, static_cast<std::uint32_t>(type), xq, native_out, min_seconds); break;
                default: break;
            }
            const double native_gbs = 1.0 / sec_per_byte / 1e9;
            const double bytes_per_row = pool.total_bytes / std::accumulate(pool.entries.begin(), pool.entries.end(), 0.0,
                                                                             [](double s, const PoolEntry& e) { return s + e.n_rows; });
            const double native_us_row = sec_per_byte * bytes_per_row * 1e6;

            double axpy_sec_per_byte = 0.0;
            switch (threads) {
                case 1:  axpy_sec_per_byte = time_axpy_pool<1>(bf16_pool, pool.row_elems, out_dim, x.data(), bf16_y.data(), min_seconds); break;
                case 2:  axpy_sec_per_byte = time_axpy_pool<2>(bf16_pool, pool.row_elems, out_dim, x.data(), bf16_y.data(), min_seconds); break;
                case 4:  axpy_sec_per_byte = time_axpy_pool<4>(bf16_pool, pool.row_elems, out_dim, x.data(), bf16_y.data(), min_seconds); break;
                case 8:  axpy_sec_per_byte = time_axpy_pool<8>(bf16_pool, pool.row_elems, out_dim, x.data(), bf16_y.data(), min_seconds); break;
                case 16: axpy_sec_per_byte = time_axpy_pool<16>(bf16_pool, pool.row_elems, out_dim, x.data(), bf16_y.data(), min_seconds); break;
                default: break;
            }
            const double axpy_gbs = 1.0 / axpy_sec_per_byte / 1e9;

            std::printf("%-8d %14.3f %14.2f %10.2f\n", threads, native_us_row, native_gbs, axpy_gbs);
        }
    }
    return 0;
}
