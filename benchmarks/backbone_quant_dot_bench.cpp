// backbone_quant_dot_bench.cpp -- O5 phase 1 microbenchmark: sub0::bbqd's fused int8-activation x
// native-quant-weight GEMV (include/sub0/backbone_quant_dot.hpp), at REAL backbone tensor shapes from
// the real Qwen3.8-Flash-Next UD-IQ1_S GGUF shards, threaded by output row at 1/4/8 threads, portable
// vs AVX2, and compared against a bf16-promote DOT baseline at the identical shape.
//
// WHAT "compared against bf16" MEANS HERE, PRECISELY (stated because it is a real simplification, not
// hidden). The engine's actual resident bf16 backbone path (docs/BACKBONE_PRECISION.md) is an AXPY
// kernel over a [in,out] layout (`Node::pdata`'s `Bf16CPtr`, one activation scalar broadcast down a
// contiguous output row) -- a different access pattern from this header's own DOT-over-[out,in] shape.
// KNOWN WEAKNESSES, to fix before this tool gates phase 2 (docs/BACKBONE_NATIVE_QUANT.md S8a): the bf16
// arm below is a naive scalar loop, NOT the engine's real kernel (include/sub0/gemv.hpp's axpy, which
// sub0llm-bench-gemv measures); the 1024-row tensors stay cache-resident across reps; and the threaded
// arms spawn fresh std::threads on every ~250 us call, so they measure spawn cost, not scaling.
// What IS measured here is the isolated PER-ELEMENT PROMOTE cost:
// `sub0::bf16_widen` (a branchless 16-bit shift, the exact function `Bf16CPtr::operator[]` calls) run
// over the SAME row-dot access shape as the native-quant kernels, so the comparison answers "does the
// native-quant block-decode cost more CPU per element than bf16's promote, at the same row length" --
// the decode-cost question -- without claiming to reproduce the AXPY-vs-DOT layout difference the design
// doc's own LAYOUT section discusses qualitatively instead. See docs/BACKBONE_NATIVE_QUANT.md.
//
// REAL SHAPES, REAL BYTES (AGENTS.md S9): picks the largest real backbone tensor of each format
// (Q8_0/Q4_K/Q5_K/Q6_K) from the real shards, same selection idiom as backbone_dequant_bench.cpp. Two of
// the four (Q4_K: token_embd/lm_head, ~635M elements) are large enough that benchmarking every output
// row would dominate this tool's own runtime budget (this session shares the host with another agent's
// real-model measurements) -- ROW COUNT is capped per format (kMaxRows below) while ROW_ELEMS (the
// contraction dim) and every byte read stay exactly the real tensor's own, so the per-row/per-byte
// figures are still real, only the TOTAL row count benchmarked is bounded.
//
// PLAIN main(), NOT Catch2 -- same reasoning as moe_expert_bench.cpp/backbone_dequant_bench.cpp: this
// times real wall-clock across explicit thread counts, which a benchmark harness' own scheduling would
// obscure.
//
// ENGINE-FREE: gguf.hpp + backbone_quant_dot.hpp + bf16.hpp only. No sub0_config.hpp, no layout.hpp.
//
// Usage: sub0_backbone_quant_dot_bench --gguf <dir> [--reps N] [--max-rows N]

#include "sub0/backbone_quant_dot.hpp"
#include "sub0/bf16.hpp"
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
#include <string>
#include <thread>
#include <vector>

namespace fs = std::filesystem;
using namespace sub0;

namespace {

using Clock = std::chrono::steady_clock;
double secs(Clock::time_point a, Clock::time_point b) { return std::chrono::duration<double>(b - a).count(); }
volatile double g_sink = 0.0;

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

struct Picked {
    std::string name;
    std::uint64_t out_rows = 0, in_elems = 0;
    const Shard* shard = nullptr;
    gguf::TensorInfo info;
};

// Runs `gemv_plane<UseAvx2>` over [0, n_rows) split into `threads` contiguous, non-overlapping row
// ranges, one std::thread per range (threads==1 runs inline, no thread spawned). Returns wall-clock
// seconds for one full pass.
template <bool UseAvx2>
double run_threaded(std::uint32_t type_raw, std::span<const std::uint8_t> raw, int n_rows, int row_elems,
                     const bbqd::ActBlocks& x, float* out, int threads) {
    const auto t0 = Clock::now();
    bool ok = true;
    if (threads <= 1) {
        ok = bbqd::gemv_plane<UseAvx2>(type_raw, raw, n_rows, row_elems, x, out, 0, n_rows);
    } else {
        std::vector<std::thread> pool;
        std::vector<bool> arm_ok(static_cast<std::size_t>(threads), true);
        const int per = (n_rows + threads - 1) / threads;
        for (int t = 0; t < threads; ++t) {
            const int lo = std::min(n_rows, t * per), hi = std::min(n_rows, (t + 1) * per);
            if (lo >= hi) continue;
            pool.emplace_back([&, lo, hi, t]() {
                arm_ok[static_cast<std::size_t>(t)] =
                    bbqd::gemv_plane<UseAvx2>(type_raw, raw, n_rows, row_elems, x, out + lo, lo, hi);
            });
        }
        for (auto& th : pool) th.join();
        for (bool v : arm_ok) ok = ok && v;
    }
    if (!ok) { std::fprintf(stderr, "error: gemv_plane refused a real tensor's own bytes\n"); std::exit(5); }
    return secs(t0, Clock::now());
}

double bf16_promote_dot_baseline(int n_rows, int row_elems, const std::vector<std::uint16_t>& bf16_bits,
                                  const std::vector<float>& x, float* out, int threads) {
    auto one_row = [&](int r) {
        const std::uint16_t* row = bf16_bits.data() + static_cast<std::size_t>(r) * static_cast<std::size_t>(row_elems);
        float acc = 0.f;
        for (int i = 0; i < row_elems; ++i) acc += x[static_cast<std::size_t>(i)] * bf16_widen(row[i]);
        out[r] = acc;
    };
    const auto t0 = Clock::now();
    if (threads <= 1) {
        for (int r = 0; r < n_rows; ++r) one_row(r);
    } else {
        std::vector<std::thread> pool;
        const int per = (n_rows + threads - 1) / threads;
        for (int t = 0; t < threads; ++t) {
            const int lo = std::min(n_rows, t * per), hi = std::min(n_rows, (t + 1) * per);
            if (lo >= hi) continue;
            pool.emplace_back([&, lo, hi]() { for (int r = lo; r < hi; ++r) one_row(r); });
        }
        for (auto& th : pool) th.join();
    }
    return secs(t0, Clock::now());
}

}  // namespace

int main(int argc, char** argv) {
    std::string gguf_dir;
    int reps = 5, max_rows = 1024;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() { return i + 1 < argc ? argv[++i] : ""; };
        if (a == "--gguf") gguf_dir = next();
        else if (a == "--reps") reps = std::atoi(next());
        else if (a == "--max-rows") max_rows = std::atoi(next());
        else { std::fprintf(stderr, "unknown argument '%s'\n", a.c_str()); return 2; }
    }
    if (gguf_dir.empty()) {
        std::fprintf(stderr, "usage: sub0_backbone_quant_dot_bench --gguf <dir> [--reps N] [--max-rows N]\n");
        return 2;
    }

    std::vector<Shard> shards;
    std::vector<fs::path> files;
    for (const auto& e : fs::directory_iterator(gguf_dir))
        if (e.is_regular_file() && e.path().extension() == ".gguf") files.push_back(e.path());
    std::sort(files.begin(), files.end());
    if (files.empty()) { std::fprintf(stderr, "error: no .gguf under %s\n", gguf_dir.c_str()); return 2; }
    for (const auto& p : files) {
        std::ifstream f(p, std::ios::binary);
        if (!f) { std::fprintf(stderr, "error: cannot open %s\n", p.string().c_str()); return 2; }
        std::vector<std::uint8_t> head(64ull * 1024 * 1024);
        f.read(reinterpret_cast<char*>(head.data()), static_cast<std::streamsize>(head.size()));
        head.resize(static_cast<std::size_t>(f.gcount()));
        gguf::Reader r(head);
        if (!r.ok()) { std::fprintf(stderr, "error: %s unreadable\n", p.filename().string().c_str()); return 2; }
        Shard sh;
        sh.path = p;
        sh.data_offset = r.data_offset();
        sh.tensors.assign(r.tensors().begin(), r.tensors().end());
        shards.push_back(std::move(sh));
    }

    constexpr gguf::TensorType kFormats[4] = {gguf::TensorType::Q8_0, gguf::TensorType::Q4_K,
                                               gguf::TensorType::Q5_K, gguf::TensorType::Q6_K};
    std::map<std::uint32_t, Picked> picks;
    for (const Shard& sh : shards)
        for (const auto& t : sh.tensors) {
            bool wanted = false;
            for (auto f : kFormats) if (t.type_raw == static_cast<std::uint32_t>(f)) wanted = true;
            if (!wanted) continue;
            if (t.name.find("_exps") != std::string::npos) continue;
            if (t.name.find("per_layer_token_embd") != std::string::npos) continue;
            if (t.name.find("ple_") != std::string::npos) continue;
            if (t.dims.size() != 2 || t.dims[0] % bbqd::GROUP != 0) continue;
            const std::uint64_t n = t.element_count();
            auto& cur = picks[t.type_raw];
            if (n > cur.out_rows * cur.in_elems)
                cur = Picked{t.name, t.dims[1], t.dims[0], &sh, t};
        }
    for (auto f : kFormats)
        if (!picks.count(static_cast<std::uint32_t>(f))) {
            std::fprintf(stderr, "error: no real backbone tensor of type %s found\n", type_name(static_cast<std::uint32_t>(f)));
            return 3;
        }

    std::printf("picked real backbone tensors (largest per format, row_elems = contraction dim):\n");
    for (auto& [type, pk] : picks)
        std::printf("  %-6s %-28s out_rows=%-8llu row_elems=%-6llu\n", type_name(type), pk.name.c_str(),
                    static_cast<unsigned long long>(pk.out_rows), static_cast<unsigned long long>(pk.in_elems));

    const int threads_arms[3] = {1, 4, 8};
    for (auto& [type, pk] : picks) {
        const int row_elems = static_cast<int>(pk.in_elems);
        const int n_rows = static_cast<int>(std::min<std::uint64_t>(pk.out_rows, static_cast<std::uint64_t>(max_rows)));
        const gguf::BlockSpec spec = gguf::block_spec(type);
        const std::uint64_t byte_len = ((static_cast<std::uint64_t>(n_rows) * row_elems + spec.elems - 1) / spec.elems) * spec.bytes;

        std::ifstream f(pk.shard->path, std::ios::binary);
        f.seekg(static_cast<std::streamoff>(pk.shard->data_offset + pk.info.offset));
        std::vector<std::uint8_t> raw(static_cast<std::size_t>(byte_len));
        f.read(reinterpret_cast<char*>(raw.data()), static_cast<std::streamsize>(byte_len));
        if (static_cast<std::uint64_t>(f.gcount()) != byte_len) {
            std::fprintf(stderr, "error: short read on %s\n", pk.name.c_str());
            return 4;
        }

        std::vector<float> x(static_cast<std::size_t>(row_elems));
        for (int i = 0; i < row_elems; ++i) x[static_cast<std::size_t>(i)] = 0.01f * static_cast<float>((i % 17) - 8);
        bbqd::ActBlocks xq;
        xq.quantize(x.data(), row_elems);
        std::vector<float> out(static_cast<std::size_t>(n_rows));

        const double native_bytes = static_cast<double>(byte_len);
        std::printf("\n=== %-6s %s  n_rows=%d (capped from %llu)  row_elems=%d  bytes=%.2f MiB ===\n",
                    type_name(type), pk.name.c_str(), n_rows, static_cast<unsigned long long>(pk.out_rows),
                    row_elems, native_bytes / (1024.0 * 1024.0));
        std::printf("%-8s %-10s %10s %10s %10s\n", "threads", "arm", "us/call", "GB/s", "vs bf16");

        double bf16_us_1t = 0.0;
        for (int threads : threads_arms) {
            for (int arm = 0; arm < 2; ++arm) {
                std::vector<double> timings;
                for (int r = 0; r < reps; ++r) {
                    const double s = (arm == 0)
                        ? run_threaded<false>(type, std::span<const std::uint8_t>(raw), n_rows, row_elems, xq, out.data(), threads)
                        : run_threaded<true>(type, std::span<const std::uint8_t>(raw), n_rows, row_elems, xq, out.data(), threads);
                    timings.push_back(s);
                    g_sink = out.empty() ? 0.0 : out[0];
                }
                const double mean = std::accumulate(timings.begin(), timings.end(), 0.0) / timings.size();
                const double gbs = native_bytes / mean / 1e9;
                std::printf("%-8d %-10s %10.1f %10.2f\n", threads, arm == 0 ? "portable" : "avx2",
                            mean * 1e6, gbs);
            }

            // bf16-promote baseline at the identical shape (see this file's own header comment on scope).
            std::vector<std::uint16_t> bf16_bits(static_cast<std::size_t>(n_rows) * static_cast<std::size_t>(row_elems));
            for (std::size_t i = 0; i < bf16_bits.size(); ++i) bf16_bits[i] = static_cast<std::uint16_t>(0x3F00u + (i * 37u) % 256u);
            std::vector<double> bf16_timings;
            for (int r = 0; r < reps; ++r) {
                bf16_timings.push_back(bf16_promote_dot_baseline(n_rows, row_elems, bf16_bits, x, out.data(), threads));
                g_sink = out.empty() ? 0.0 : out[0];
            }
            const double bf16_mean = std::accumulate(bf16_timings.begin(), bf16_timings.end(), 0.0) / bf16_timings.size();
            const double bf16_gbs = static_cast<double>(n_rows) * row_elems * 2.0 / bf16_mean / 1e9;
            if (threads == 1) bf16_us_1t = bf16_mean * 1e6;
            std::printf("%-8d %-10s %10.1f %10.2f\n", threads, "bf16-dot", bf16_mean * 1e6, bf16_gbs);
        }
        std::printf("(bf16-promote-dot baseline at 1 thread: %.1f us/call -- see this file's own header\n"
                    " comment for exactly what this baseline does and does not model)\n", bf16_us_1t);
    }
    return 0;
}
