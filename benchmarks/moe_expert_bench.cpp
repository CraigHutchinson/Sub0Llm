// moe_expert_bench.cpp -- decompose ONE routed MoE expert's decode-time cost into its four real arms.
//
// WHY THIS EXISTS. Every performance number this project has for the real 48-layer decode path is
// END-TO-END: a whole 30-token generation run, or a 60-second VTune window across the entire decode
// loop. Those are real, and they are what found both of B20's results -- but they conflate four costs
// that respond to completely different fixes:
//
//   1. faulting an expert's encoded bytes in from the S0Q1 sidecar (disk / OS page cache)
//   2. unpacking those bytes -- the format-specific IQ1_S / IQ2_XXS / IQ4_NL decoders
//   3. transposing the decoded plane into this project's [in, out] convention
//   4. the FFN itself -- moe::expert_ffn_row over the resulting f32 planes
//
// A single conflated figure cannot answer the question docs/WP4_SCOPE.md's WP4e addendum actually left
// open ("is a Sub0Llm-native re-encode of the sidecar worth it?"), because that question is entirely
// about the ratio between (1)+(2) and (4). This binary measures each arm on its own, on the REAL
// sidecar at the REAL axes, so that ratio is a measurement rather than an estimate.
//
// WHY A PLAIN main() AND NOT CATCH2'S BENCHMARK MACROS, unlike its two neighbours here. Catch2's
// benchmark support warms up, then runs a chosen sample count and reports a mean -- exactly the wrong
// instrument for arm 1, whose entire subject is the FIRST touch of a page. A warm-up pass destroys the
// measurement it is trying to take. Arms 2-4 would be a fine Catch2 fit, but splitting the file in two
// so that half of it can use a macro would trade the one property that makes this useful: all four arms
// measured in one process, in a controlled order, over disjoint experts of the same real file.
//
// WHAT MAKES "COLD" COLD, AND WHAT IT CANNOT PROMISE. The cold arms run FIRST, before anything else in
// this process has touched the mapping, over (layer, expert) pairs drawn from a fixed-seed shuffle and
// kept DISJOINT from every warm arm's pairs. What this binary cannot do without administrative rights is
// evict the OS standby list, so a page some earlier run left cached will fault soft rather than hard.
// That is not papered over: the achieved MB/s is reported for every arm, and it is the discriminator --
// a hard fault from an NVMe device lands in the tens of MB/s per thread, a soft fault from the standby
// list in the GB/s. Read the throughput, not the label.
//
// ENGINE-FREE, deliberately, exactly like the three headers it measures: no sub0_config.hpp, no
// layout.hpp, no sub0_core. Every dimension comes from the sidecar's OWN header (d_model, d_ff,
// n_layers, num_experts), so this binary characterizes whatever real file it is pointed at rather than
// whatever the surrounding build happens to be configured for.
//
// Usage:
//   sub0_moe_expert_bench --sidecar <model.bin.moeq> [--experts N] [--reps N] [--seed N]

#include "sub0/gguf.hpp"
#include "sub0/moe_math.hpp"
#include "sub0/moe_quant.hpp"
#include "sub0/transplant.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <random>
#include <span>
#include <string>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;
using sub0::moeq::Desc;
using sub0::moeq::Store;

double secs(Clock::time_point a, Clock::time_point b) {
    return std::chrono::duration<double>(b - a).count();
}

// The sink every timed loop writes into. `volatile` rather than a std::atomic or an inline-asm barrier
// because it is the cheapest construct that stops -O3 deleting a loop whose result is otherwise unused,
// and it is written ONCE per timed region, not per element, so it cannot itself distort the measurement.
volatile double g_sink = 0.0;

struct Stat {
    double mean = 0, lo = 0, hi = 0;
    int n = 0;
};

// Mean plus the observed range. The range is reported rather than a standard deviation because at these
// sample counts (a handful of reps, deliberately -- see the file header on warm-up) the spread IS the
// uncertainty, and this project's own measurement discipline (memory: thermal-confounds-ab-wallclock-
// testing) is about not hiding it behind a single number.
Stat summarize(std::vector<double>& v) {
    Stat s;
    if (v.empty()) return s;
    s.n = static_cast<int>(v.size());
    s.mean = std::accumulate(v.begin(), v.end(), 0.0) / v.size();
    s.lo = *std::min_element(v.begin(), v.end());
    s.hi = *std::max_element(v.begin(), v.end());
    return s;
}

// Touch one byte per 4 KiB page of `raw`, which is what actually forces the fault; reading every byte
// would measure memcpy bandwidth on top of it. Returns a value derived from the bytes so the loop
// survives optimization.
double touch_pages(std::span<const std::uint8_t> raw) {
    double acc = 0;
    for (std::size_t i = 0; i < raw.size(); i += 4096) acc += raw[i];
    if (!raw.empty()) acc += raw.back();
    return acc;
}

const char* type_name(std::uint32_t t) {
    switch (static_cast<sub0::gguf::TensorType>(t)) {
        case sub0::gguf::TensorType::F32:     return "F32";
        case sub0::gguf::TensorType::F16:     return "F16";
        case sub0::gguf::TensorType::BF16:    return "BF16";
        case sub0::gguf::TensorType::Q8_0:    return "Q8_0";
        case sub0::gguf::TensorType::Q4_K:    return "Q4_K";
        case sub0::gguf::TensorType::Q5_K:    return "Q5_K";
        case sub0::gguf::TensorType::Q6_K:    return "Q6_K";
        case sub0::gguf::TensorType::IQ2_XXS: return "IQ2_XXS";
        case sub0::gguf::TensorType::IQ1_S:   return "IQ1_S";
        case sub0::gguf::TensorType::IQ4_NL:  return "IQ4_NL";
        default:                              return "?";
    }
}

struct Pair { int layer, expert; };

// The three planes an expert is (gate/up/down). `moeq::PerExpert` is an enumerator of `enum Which`,
// so it cannot be multiplied by a double directly; naming the count once here keeps every per-expert
// figure below derived from the same constant the format itself is defined in terms of.
constexpr int kPlanes = static_cast<int>(sub0::moeq::PerExpert);

}  // namespace

int main(int argc, char** argv) {
    std::string sidecar;
    int n_experts_sample = 24, reps = 5;
    unsigned seed = 1234;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() { return i + 1 < argc ? argv[++i] : ""; };
        if (a == "--sidecar") sidecar = next();
        else if (a == "--experts") n_experts_sample = std::atoi(next());
        else if (a == "--reps") reps = std::atoi(next());
        else if (a == "--seed") seed = static_cast<unsigned>(std::strtoul(next(), nullptr, 10));
        else { std::fprintf(stderr, "unknown argument '%s'\n", a.c_str()); return 2; }
    }
    if (sidecar.empty()) {
        std::fprintf(stderr,
                     "usage: sub0_moe_expert_bench --sidecar <model.bin.moeq> [--experts N] [--reps N]"
                     " [--seed N]\n\nThis benchmark measures a REAL S0Q1 sidecar; it has nothing to\n"
                     "measure without one, and deliberately does not fall back to synthetic bytes --\n"
                     "the whole point is the real formats at the real axes (AGENTS.md S9).\n");
        return 2;
    }

    Store store;
    std::string err;
    if (!store.open(sidecar, err)) {
        std::fprintf(stderr, "FAIL: %s\n", err.c_str());
        return 3;
    }
    const auto& h = store.header();
    const int L = h.n_layers, E = h.num_experts, D = h.d_model, F = h.d_ff;
    const std::size_t plane_floats = static_cast<std::size_t>(D) * F;

    std::printf("sidecar: %s\n", sidecar.c_str());
    std::printf("  n_layers %d  num_experts %d  d_model %d  d_ff %d  payload %.2f GiB\n", L, E, D, F,
                static_cast<double>(store.resident_bytes()) / (1024.0 * 1024.0 * 1024.0));
    std::printf("  one expert = 3 planes x %zu floats = %.2f MiB of f32\n", plane_floats,
                3.0 * static_cast<double>(plane_floats) * 4.0 / (1024.0 * 1024.0));

    // Disjoint (layer, expert) pools, so no arm can be warmed by another. Fixed seed: the SAME experts
    // are measured on every run of this binary, which is what makes two runs comparable at all.
    std::vector<Pair> all;
    all.reserve(static_cast<std::size_t>(L) * E);
    for (int l = 0; l < L; ++l)
        for (int e = 0; e < E; ++e) all.push_back({l, e});
    std::mt19937 rng(seed);
    std::shuffle(all.begin(), all.end(), rng);
    const int n = std::min<int>(n_experts_sample, static_cast<int>(all.size()) / 3);
    const std::span<Pair> cold_read(all.data(), static_cast<std::size_t>(n));
    const std::span<Pair> cold_pipe(all.data() + n, static_cast<std::size_t>(n));
    const std::span<Pair> warm(all.data() + 2 * n, static_cast<std::size_t>(n));
    std::printf("  sampling %d experts per arm (disjoint pools), %d reps for the warm arms, seed %u\n\n",
                n, reps, seed);

    // Report the encoded formats actually present, which is a property of the file and not an
    // assumption: the sidecar carries each plane's OWN type_raw (moe_quant.hpp's own S3a-bis note).
    {
        std::vector<std::pair<std::uint32_t, long long>> hist;
        for (int l = 0; l < L; ++l)
            for (int e = 0; e < E; ++e)
                for (int w = 0; w < sub0::moeq::PerExpert; ++w) {
                    const std::uint32_t t = store.desc(l, e, w).type_raw;
                    auto it = std::find_if(hist.begin(), hist.end(),
                                           [&](auto& p) { return p.first == t; });
                    if (it == hist.end()) hist.push_back({t, 1});
                    else ++it->second;
                }
        std::sort(hist.begin(), hist.end(), [](auto& a, auto& b) { return a.second > b.second; });
        std::printf("encoded plane census (this file, not an assumption):\n");
        for (auto& [t, c] : hist)
            std::printf("  %-8s %8lld planes\n", type_name(t), c);
        std::printf("\n");
    }

    // ============================================================================================
    //  ARM 1 -- cold read: the encoded bytes only. FIRST, before anything else touches the mapping.
    // ============================================================================================
    double cold_read_s = 0;
    std::uint64_t cold_read_bytes = 0;
    {
        const auto t0 = Clock::now();
        double acc = 0;
        for (const Pair& p : cold_read)
            for (int w = 0; w < sub0::moeq::PerExpert; ++w) {
                const Desc& d = store.desc(p.layer, p.expert, w);
                cold_read_bytes += d.bytes;
                acc += touch_pages(store.raw(d));
            }
        g_sink = acc;
        cold_read_s = secs(t0, Clock::now());
    }

    // ============================================================================================
    //  ARM 5a -- full cold pipeline: resolve (fault + dequant + transpose) + FFN, per expert.
    //  Still ahead of every warm arm, over its own disjoint pool.
    // ============================================================================================
    // One-slot pool, exactly the shape decode uses (internal.hpp's MOE_DECODE_SLOTS): a decode row's
    // hit rate against a bigger pool is provably zero, so a bigger one here would only flatter the
    // numbers.
    using Pool = sub0::moeq::ExpertCache<1, 2560u * 640u>;
    if (plane_floats != Pool::kFloats) {
        std::fprintf(stderr,
                     "FAIL: this sidecar's d_model*d_ff is %zu, but the resolve pool in this binary is\n"
                     "compiled for %zu (the real Qwen4-preview axes). The pool's slot width is a\n"
                     "COMPILE-TIME constant in the engine too (internal.hpp's MOE_EXPERT_SLOT_FLOATS),\n"
                     "so this is the same constraint the engine has, not a benchmark shortcut.\n",
                     plane_floats, Pool::kFloats);
        return 4;
    }
    auto pool = std::make_unique<Pool>();
    pool->allocate();

    const sub0::moe::Dims md{.hidden_size = D, .d_ff = F, .num_experts = E, .experts_per_tok = 10};
    std::vector<float> x(static_cast<std::size_t>(D)), out(static_cast<std::size_t>(D)),
        pre(static_cast<std::size_t>(F)), gsc(static_cast<std::size_t>(F));
    {   // a plausible non-sparse activation row; expert_ffn_row skips exact zeros, so an all-zero x
        // would measure a branch instead of a GEMV.
        std::mt19937 r2(7);
        std::uniform_real_distribution<float> u(-1.f, 1.f);
        for (float& v : x) v = u(r2);
    }

    double cold_pipe_s = 0;
    {
        const auto t0 = Clock::now();
        double acc = 0;
        for (const Pair& p : cold_pipe) {
            const auto r = pool->resolve(store, p.layer, p.expert);
            if (!r.gate) { std::fprintf(stderr, "FAIL: resolve failed (unsupported type?)\n"); return 5; }
            sub0::moe::expert_ffn_row(md, x.data(), r.gate, r.up, r.down, out.data(), pre.data(),
                                      gsc.data());
            acc += out[0];
        }
        g_sink = acc;
        cold_pipe_s = secs(t0, Clock::now());
    }

    // ============================================================================================
    //  From here on: WARM. Fault the whole warm pool in once, untimed.
    // ============================================================================================
    {
        double acc = 0;
        for (const Pair& p : warm)
            for (int w = 0; w < sub0::moeq::PerExpert; ++w)
                acc += touch_pages(store.raw(store.desc(p.layer, p.expert, w)));
        g_sink = acc;
    }

    // --- ARM 2: the format-specific decoders alone, on real encoded planes ------------------------
    // Grouped by the plane's OWN type_raw, so a format that appears on only some layers still gets its
    // own number instead of being averaged into a neighbour's.
    struct FmtAcc { std::uint32_t type; std::vector<double> per_plane_s; std::uint64_t elems = 0,
                    enc_bytes = 0; };
    std::vector<FmtAcc> fmts;
    {
        std::vector<float> scratch;
        for (int rep = 0; rep < reps; ++rep)
            for (const Pair& p : warm)
                for (int w = 0; w < sub0::moeq::PerExpert; ++w) {
                    const Desc& d = store.desc(p.layer, p.expert, w);
                    sub0::gguf::TensorInfo t;
                    t.type_raw = d.type_raw;
                    t.dims = {static_cast<std::uint64_t>(d.in_f) * d.out_f};
                    const auto t0 = Clock::now();
                    const bool ok = sub0::gguf::to_f32(t, store.raw(d), scratch);
                    const double s = secs(t0, Clock::now());
                    if (!ok) { std::fprintf(stderr, "FAIL: to_f32 failed\n"); return 6; }
                    g_sink = scratch[0];
                    auto it = std::find_if(fmts.begin(), fmts.end(),
                                           [&](auto& f) { return f.type == d.type_raw; });
                    if (it == fmts.end()) { fmts.push_back({d.type_raw, {}, 0, 0}); it = fmts.end() - 1; }
                    it->per_plane_s.push_back(s);
                    it->elems += static_cast<std::uint64_t>(d.in_f) * d.out_f;
                    it->enc_bytes += d.bytes;
                }
    }

    // --- ARM 3: the cache-blocked transpose alone, at both real plane shapes ----------------------
    struct ShapeAcc { int out_f, in_f; std::vector<double> s; };
    std::vector<ShapeAcc> shapes;
    {
        std::vector<float> src(plane_floats), dst(plane_floats);
        std::mt19937 r3(11);
        std::uniform_real_distribution<float> u(-1.f, 1.f);
        for (float& v : src) v = u(r3);
        // Both real shapes, taken from the file's own descriptors rather than hard-coded: gate/up and
        // down transpose in opposite directions and are NOT the same measurement.
        for (int w = 0; w < sub0::moeq::PerExpert; ++w) {
            const Desc& d0 = store.desc(warm[0].layer, warm[0].expert, w);
            auto it = std::find_if(shapes.begin(), shapes.end(), [&](auto& sh) {
                return sh.out_f == static_cast<int>(d0.out_f) && sh.in_f == static_cast<int>(d0.in_f);
            });
            if (it == shapes.end())
                shapes.push_back({static_cast<int>(d0.out_f), static_cast<int>(d0.in_f), {}});
        }
        for (ShapeAcc& sh : shapes)
            for (int rep = 0; rep < reps * static_cast<int>(warm.size()); ++rep) {
                const auto t0 = Clock::now();
                sub0::transplant::transpose_out_in(src.data(), sh.out_f, sh.in_f, dst.data());
                sh.s.push_back(secs(t0, Clock::now()));
                g_sink = dst[0];
            }
    }

    // --- ARM 4: moe::expert_ffn_row alone, on already-dequantized planes --------------------------
    Stat ffn;
    {
        // Resolve one warm expert and keep it; every rep runs the same real weights, so this measures
        // the FFN and nothing else. Deliberately the LAST warm pair, not the first: the pool has one
        // slot and arm 5b below walks the same pairs in order starting at warm[0], so leaving warm[0]
        // resident here would hand 5b a free cache hit on its very first timing and drag its mean down
        // by an amount nothing in the report would explain.
        const auto r = pool->resolve(store, warm.back().layer, warm.back().expert);
        if (!r.gate) { std::fprintf(stderr, "FAIL: resolve failed\n"); return 7; }
        std::vector<double> v;
        for (int rep = 0; rep < reps * static_cast<int>(warm.size()); ++rep) {
            const auto t0 = Clock::now();
            sub0::moe::expert_ffn_row(md, x.data(), r.gate, r.up, r.down, out.data(), pre.data(),
                                      gsc.data());
            v.push_back(secs(t0, Clock::now()));
            g_sink = out[0];
        }
        ffn = summarize(v);
    }

    // --- ARM 5b: the full pipeline again, WARM ----------------------------------------------------
    Stat warm_pipe;
    const std::uint64_t hits_before = pool->hits();
    {
        std::vector<double> v;
        for (int rep = 0; rep < reps; ++rep)
            for (const Pair& p : warm) {
                const auto t0 = Clock::now();
                const auto r = pool->resolve(store, p.layer, p.expert);
                if (!r.gate) { std::fprintf(stderr, "FAIL: resolve failed\n"); return 8; }
                sub0::moe::expert_ffn_row(md, x.data(), r.gate, r.up, r.down, out.data(), pre.data(),
                                          gsc.data());
                v.push_back(secs(t0, Clock::now()));
                g_sink = out[0];
            }
        warm_pipe = summarize(v);
    }

    // ============================================================================================
    //  Report
    // ============================================================================================
    const double cold_read_ms = cold_read_s * 1000.0 / n;
    const double cold_pipe_ms = cold_pipe_s * 1000.0 / n;
    const double enc_bytes_per_expert = static_cast<double>(cold_read_bytes) / n;

    std::printf("=== per-arm, one expert (3 planes) ===\n");
    std::printf("%-34s %12s %12s %12s   %s\n", "arm", "mean ms", "min ms", "max ms", "throughput");
    std::printf("1  cold read (encoded bytes only)   %12.3f %12s %12s   %.1f MB/s over %.2f MiB/expert\n",
                cold_read_ms, "-", "-",
                static_cast<double>(cold_read_bytes) / cold_read_s / 1e6,
                enc_bytes_per_expert / (1024.0 * 1024.0));

    for (const FmtAcc& f : fmts) {
        std::vector<double> v = f.per_plane_s;
        const Stat s = summarize(v);
        const double per_expert_ms = s.mean * 1000.0 * kPlanes;
        std::printf("2  warm dequant %-8s (x3 planes) %11.3f %12.3f %12.3f   %.1f Melem/s (%d planes timed)\n",
                    type_name(f.type), per_expert_ms, s.lo * 1000.0 * kPlanes,
                    s.hi * 1000.0 * kPlanes,
                    static_cast<double>(f.elems) / static_cast<double>(f.per_plane_s.size())
                        / s.mean / 1e6,
                    s.n);
    }
    for (const ShapeAcc& sh : shapes) {
        std::vector<double> v = sh.s;
        const Stat s = summarize(v);
        char label[64];
        std::snprintf(label, sizeof label, "3  transpose %dx%d (x3 planes)", sh.out_f, sh.in_f);
        std::printf("%-34s %12.3f %12.3f %12.3f   %.1f MB/s of f32\n", label,
                    s.mean * 1000.0 * kPlanes, s.lo * 1000.0 * kPlanes,
                    s.hi * 1000.0 * kPlanes,
                    static_cast<double>(plane_floats) * 4.0 / s.mean / 1e6);
    }
    std::printf("%-34s %12.3f %12.3f %12.3f   %.2f GFLOP/s\n", "4  expert_ffn_row (f32 planes)",
                ffn.mean * 1000.0, ffn.lo * 1000.0, ffn.hi * 1000.0,
                4.0 * static_cast<double>(plane_floats) / ffn.mean / 1e9);
    std::printf("%-34s %12.3f %12s %12s   %d experts, one pass\n", "5a full pipeline, COLD",
                cold_pipe_ms, "-", "-", n);
    // The hit count is printed, not assumed: with a ONE-slot pool over distinct pairs it must be zero,
    // and a non-zero value would mean some of 5b's timings skipped the resolve entirely.
    std::printf("%-34s %12.3f %12.3f %12.3f   %d timings, %llu pool hits\n", "5b full pipeline, WARM",
                warm_pipe.mean * 1000.0, warm_pipe.lo * 1000.0, warm_pipe.hi * 1000.0, warm_pipe.n,
                static_cast<unsigned long long>(pool->hits() - hits_before));

    // The split the whole exercise exists to make legible.
    double dequant_ms = 0;
    for (const FmtAcc& f : fmts) {
        std::vector<double> v = f.per_plane_s;
        dequant_ms += summarize(v).mean * 1000.0 * static_cast<double>(f.per_plane_s.size());
    }
    {
        std::size_t total = 0;
        for (const FmtAcc& f : fmts) total += f.per_plane_s.size();
        if (total) dequant_ms = dequant_ms / static_cast<double>(total) * kPlanes;
    }
    double transpose_ms = 0;
    for (const ShapeAcc& sh : shapes) {
        std::vector<double> v = sh.s;
        transpose_ms += summarize(v).mean * 1000.0;
    }
    transpose_ms = shapes.empty() ? 0.0
                                  : transpose_ms / static_cast<double>(shapes.size())
                                        * kPlanes;

    std::printf("\n=== where a COLD single-expert resolve's time goes ===\n");
    const double io_ms = cold_pipe_ms - warm_pipe.mean * 1000.0;
    auto pct = [&](double ms) { return cold_pipe_ms > 0 ? 100.0 * ms / cold_pipe_ms : 0.0; };
    std::printf("  page-in (cold minus warm)      %9.3f ms  %5.1f%%   (%.1f MB/s of encoded bytes)\n",
                io_ms, pct(io_ms), io_ms > 0 ? enc_bytes_per_expert / (io_ms / 1000.0) / 1e6 : 0.0);
    std::printf("  dequantize (all 3 planes)      %9.3f ms  %5.1f%%\n", dequant_ms, pct(dequant_ms));
    std::printf("  transpose (all 3 planes)       %9.3f ms  %5.1f%%\n", transpose_ms, pct(transpose_ms));
    std::printf("  expert_ffn_row                 %9.3f ms  %5.1f%%\n", ffn.mean * 1000.0,
                pct(ffn.mean * 1000.0));
    std::printf("  ---------------------------------------------------\n");
    std::printf("  cold pipeline, measured        %9.3f ms  100.0%%\n", cold_pipe_ms);
    std::printf("  warm pipeline, measured        %9.3f ms  %5.1f%%\n", warm_pipe.mean * 1000.0,
                pct(warm_pipe.mean * 1000.0));
    std::printf("\nNote: arm 2 measures gguf::to_f32, which for these types IS the dequantize_iq* call\n"
                "it dispatches to; arm 3's transpose and arm 2's dequant together are what a resolve\n"
                "does, so 5b should land near their sum plus arm 4. Any gap is allocator/cache effects\n"
                "between the isolated and composed forms, and is reported rather than reconciled away.\n");
    return 0;
}
