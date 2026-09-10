// backbone_dequant_bench.cpp -- B24: does inline per-read dequant of the BACKBONE (dense, 100% of it,
// every token) pay for itself against the DRAM bandwidth it saves, the same question WP6b's
// moe_expert_bench.cpp already answered for the SPARSE (~2%-density) MoE sidecar?
//
// WHY THIS EXISTS. docs/BACKBONE_PRECISION.md S2a names the fork and explicitly leaves it open: "2a"
// (dequantize once into a resident buffer, same shape as the Phase-1 BF16 work) vs "2b" (keep the
// quantized bytes resident, dequantize inline on every read -- the shape that actually delivers S0's
// ~23-48 ms/token estimate, IF the dequant cost doesn't eat the saving). WP6b's own single-expert
// decomposition found dequant is ~46% of a cold resolve's cost at ~2% density; this measures the same
// two costs (format-specific block dequant, and achieved DRAM bandwidth per byte-width) at the
// BACKBONE's own real tensor shapes, and computes the actual crossover rather than assuming either
// side wins.
//
// WHAT "CROSSOVER" MEANS HERE, PRECISELY (worked out before writing the report, not left to the
// reader). Decode is SEQ_LEN batch size 1 -- every backbone weight is read exactly ONCE per token, so
// this is not an amortized-over-many-reads question the way the sidecar's own resolve-pool hit rate is
// (S3b of the memory map: decode's hit rate against ANY pool size is provably zero). Per element:
//   2a (resident F32, today's shape): cost = R / BW            (R = 4 bytes, BW = achieved GB/s)
//   2b (resident quantized bytes):    cost = Q / BW + D          (Q = format's bytes/element, D = this
//                                                                  format's measured dequant ns/element)
// 2b wins iff Q/BW + D < R/BW, i.e. iff D < (R-Q)/BW -- a single per-format threshold, D*, computed
// below from measured Q, R and BW and compared directly against the measured D.
//
// REAL FILE, REAL SHAPES, REAL FORMATS (AGENTS.md S9 -- import/interop validated against a real file,
// not a fixture). Reads the actual downloaded Qwen3.8-Flash-Next UD-IQ1_S shards via --gguf <dir> (the
// same multi-shard indexing tools/sub0llm-transplant.cpp uses) and picks, per quant format, the
// largest real backbone tensor of that type it finds (a body projection, not a tiny norm vector) --
// dims come from the file's own tensor table, never guessed. Bandwidth is MEASURED, not estimated: a
// replica-array streaming touch (this session's warm_sidecar.cpp precedent), sized well past this
// host's L3 (36 MiB, docs/host-cpu-arrow-lake-hx.md) so the result is a genuine DRAM figure, not an L3
// hit rate wearing DRAM's name -- the achieved GB/s is printed either way, which is the discriminator.
//
// PLAIN main(), NOT Catch2, exactly like moe_expert_bench.cpp and for the same reason: arm-by-arm cold/
// warm control that a benchmark harness' own warm-up would destroy.
//
// ENGINE-FREE: only gguf.hpp. No sub0_config.hpp, no layout.hpp -- this characterizes whatever real
// GGUF directory it's pointed at, independent of what the surrounding build happens to be configured
// for.
//
// Usage:
//   sub0_backbone_dequant_bench --gguf <dir holding the model's .gguf shards> [--reps N]

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
#include <vector>

namespace fs = std::filesystem;

namespace {

using Clock = std::chrono::steady_clock;
double secs(Clock::time_point a, Clock::time_point b) {
    return std::chrono::duration<double>(b - a).count();
}
volatile double g_sink = 0.0;   // see moe_expert_bench.cpp's own comment on why volatile, not atomic

struct Stat {
    double mean = 0, lo = 0, hi = 0;
    int n = 0;
};
Stat summarize(std::vector<double>& v) {
    Stat s;
    if (v.empty()) return s;
    s.n = static_cast<int>(v.size());
    s.mean = std::accumulate(v.begin(), v.end(), 0.0) / v.size();
    s.lo = *std::min_element(v.begin(), v.end());
    s.hi = *std::max_element(v.begin(), v.end());
    return s;
}

const char* type_name(std::uint32_t t) {
    switch (static_cast<sub0::gguf::TensorType>(t)) {
        case sub0::gguf::TensorType::F32:  return "F32";
        case sub0::gguf::TensorType::BF16: return "BF16";
        case sub0::gguf::TensorType::Q8_0: return "Q8_0";
        case sub0::gguf::TensorType::Q5_K: return "Q5_K";
        case sub0::gguf::TensorType::Q6_K: return "Q6_K";
        default:                           return "?";
    }
}

struct Shard {
    fs::path path;
    std::uint64_t data_offset = 0;
    std::vector<sub0::gguf::TensorInfo> tensors;
};

// One representative real backbone tensor of a given quant format: the largest body-projection tensor
// (>1e6 elements, so a norm/bias vector can never be picked) of that type_raw found anywhere across the
// indexed shards -- picking the LARGEST rather than the first also means a format present on only a
// few layers (Q6_K is out_proj-only, per QWEN4_MEMORY_MAP.md S7d) still gets its biggest real instance.
struct Picked {
    std::string name;
    std::uint64_t elements = 0;
    const Shard* shard = nullptr;
    sub0::gguf::TensorInfo info;
};

// Streams `reps` full passes over a REPLICATED buffer of `unit_bytes` laid out back-to-back until the
// total footprint clears `min_total_bytes` (well past L3), touching one byte per cache line (64B) --
// what actually forces the DRAM transaction; a full memcpy-style read would additionally measure
// per-byte load-store throughput on top of the pure streaming-bandwidth question this probe asks.
// Returns achieved bytes/sec, measured directly (this session's warm_sidecar.cpp precedent), not
// estimated from a spec sheet.
double measure_stream_bandwidth(std::size_t unit_bytes, std::size_t min_total_bytes, int reps) {
    const std::size_t replicas = std::max<std::size_t>(1, (min_total_bytes + unit_bytes - 1) / unit_bytes);
    const std::size_t total = replicas * unit_bytes;
    std::vector<std::uint8_t> buf(total);
    // Fill once (not timed): touching every byte here is what actually commits the pages, so the
    // TIMED passes below never pay a first-touch page-fault cost that isn't representative of a
    // steady-state stream.
    for (std::size_t i = 0; i < total; i += 4096) buf[i] = static_cast<std::uint8_t>(i);
    if (!buf.empty()) buf.back() = 1;

    double acc = 0;
    const auto t0 = Clock::now();
    for (int r = 0; r < reps; ++r)
        for (std::size_t i = 0; i < total; i += 64) acc += buf[i];
    const double s = secs(t0, Clock::now());
    g_sink = acc;
    return static_cast<double>(total) * reps / s;   // bytes/sec
}

}  // namespace

int main(int argc, char** argv) {
    std::string gguf_dir;
    int reps = 8, bw_reps = 4;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() { return i + 1 < argc ? argv[++i] : ""; };
        if (a == "--gguf") gguf_dir = next();
        else if (a == "--reps") reps = std::atoi(next());
        else if (a == "--bw-reps") bw_reps = std::atoi(next());
        else { std::fprintf(stderr, "unknown argument '%s'\n", a.c_str()); return 2; }
    }
    if (gguf_dir.empty()) {
        std::fprintf(stderr,
                     "usage: sub0_backbone_dequant_bench --gguf <dir> [--reps N] [--bw-reps N]\n\n"
                     "Measures REAL Q5_K/Q6_K/Q8_0 backbone tensors from the real Qwen3.8-Flash-Next\n"
                     "GGUF shards; no synthetic fallback (AGENTS.md S9 -- real file, not a fixture).\n");
        return 2;
    }

    // --- index every shard's header, exactly tools/sub0llm-transplant.cpp's own pattern -------------
    std::vector<Shard> shards;
    std::vector<fs::path> files;
    for (const auto& e : fs::directory_iterator(gguf_dir))
        if (e.is_regular_file() && e.path().extension() == ".gguf") files.push_back(e.path());
    std::sort(files.begin(), files.end());
    if (files.empty()) { std::fprintf(stderr, "error: no .gguf files under %s\n", gguf_dir.c_str()); return 2; }
    for (const auto& p : files) {
        std::ifstream f(p, std::ios::binary);
        if (!f) { std::fprintf(stderr, "error: cannot open %s\n", p.string().c_str()); return 2; }
        std::vector<std::uint8_t> head(64ull * 1024 * 1024);
        f.read(reinterpret_cast<char*>(head.data()), static_cast<std::streamsize>(head.size()));
        head.resize(static_cast<std::size_t>(f.gcount()));
        sub0::gguf::Reader r(head);
        if (!r.ok()) {
            std::fprintf(stderr, "error: %s not a readable GGUF (err %d)\n", p.filename().string().c_str(),
                         static_cast<int>(r.error()));
            return 2;
        }
        Shard sh;
        sh.path = p;
        sh.data_offset = r.data_offset();
        sh.tensors.assign(r.tensors().begin(), r.tensors().end());
        std::printf("shard %-48s tensors %5zu  data_offset %llu\n", p.filename().string().c_str(),
                    sh.tensors.size(), static_cast<unsigned long long>(sh.data_offset));
        shards.push_back(std::move(sh));
    }

    // --- pick the largest real backbone tensor per target quant format ------------------------------
    const std::uint32_t kQ5K = static_cast<std::uint32_t>(sub0::gguf::TensorType::Q5_K);
    const std::uint32_t kQ6K = static_cast<std::uint32_t>(sub0::gguf::TensorType::Q6_K);
    const std::uint32_t kQ8_0 = static_cast<std::uint32_t>(sub0::gguf::TensorType::Q8_0);
    std::map<std::uint32_t, Picked> picks;
    for (const Shard& sh : shards)
        for (const auto& t : sh.tensors) {
            if (t.type_raw != kQ5K && t.type_raw != kQ6K && t.type_raw != kQ8_0) continue;
            // Only backbone body-projection tensors this project's engine actually reads every token:
            // routed-expert tensors ("_exps") are WP6b's own sparse-density subject, not this one, and
            // the n-gram/PLE table's own per-layer tensors ("ple_") are explicitly OUT of transplant's
            // scope (docs/WP4_SCOPE.md S5 -- "the n-gram/PLE table ... NOT transplanted"), so a pick
            // from either would silently measure something S2's dense-backbone question isn't about.
            if (t.name.find("_exps") != std::string::npos) continue;
            if (t.name.find("ple_") != std::string::npos) continue;
            const std::uint64_t n = t.element_count();
            if (n < 1'000'000) continue;
            auto& cur = picks[t.type_raw];
            if (n > cur.elements) cur = Picked{t.name, n, &sh, t};
        }
    for (auto type : {kQ5K, kQ6K, kQ8_0})
        if (!picks.count(type)) {
            std::fprintf(stderr, "error: no real backbone tensor of type %s found under %s\n",
                        type_name(type), gguf_dir.c_str());
            return 3;
        }
    std::printf("\npicked real backbone tensors (largest body projection per format, non-expert):\n");
    for (auto& [type, pk] : picks)
        std::printf("  %-6s %-32s %10llu elements (%.2f MiB f32-equivalent)\n", type_name(type),
                    pk.name.c_str(), static_cast<unsigned long long>(pk.elements),
                    static_cast<double>(pk.elements) * 4.0 / (1024.0 * 1024.0));

    // --- arm A: cold read + dequant each picked tensor's REAL bytes, `reps` warm passes over the -----
    // SAME decoded bytes (arm 2 of moe_expert_bench.cpp's own pattern) ------------------------------
    std::printf("\n=== arm A: format-specific dequant cost (gguf::to_f32 on real block bytes) ===\n");
    struct DequantResult { std::uint32_t type; double ns_per_elem = 0, ms_per_tensor = 0; std::uint64_t n = 0; };
    std::vector<DequantResult> dequant_results;
    for (auto& [type, pk] : picks) {
        const sub0::gguf::BlockSpec b = sub0::gguf::block_spec(type);
        const std::uint64_t byte_len = ((pk.elements + b.elems - 1) / b.elems) * b.bytes;
        std::vector<std::uint8_t> raw(static_cast<std::size_t>(byte_len));
        std::ifstream f(pk.shard->path, std::ios::binary);
        f.seekg(static_cast<std::streamoff>(pk.shard->data_offset + pk.info.offset));
        f.read(reinterpret_cast<char*>(raw.data()), static_cast<std::streamsize>(byte_len));
        if (static_cast<std::uint64_t>(f.gcount()) != byte_len) {
            std::fprintf(stderr, "error: short read on %s\n", pk.name.c_str());
            return 4;
        }
        sub0::gguf::TensorInfo slice = pk.info;
        slice.dims = {pk.elements};
        std::vector<float> out;
        std::vector<double> timings;
        for (int r = 0; r < reps; ++r) {
            const auto t0 = Clock::now();
            const bool ok = sub0::gguf::to_f32(slice, raw, out);
            timings.push_back(secs(t0, Clock::now()));
            if (!ok) { std::fprintf(stderr, "error: to_f32 failed on %s\n", pk.name.c_str()); return 5; }
            g_sink = out.empty() ? 0.0 : out[0];
        }
        const Stat s = summarize(timings);
        const double ns_per_elem = s.mean * 1e9 / static_cast<double>(pk.elements);
        dequant_results.push_back({type, ns_per_elem, s.mean * 1000.0, pk.elements});
        std::printf("  %-6s %-32s %8.3f ns/elem  %9.3f ms/tensor  (%.1f Melem/s, %d reps, min %.3f max %.3f ms)\n",
                    type_name(type), pk.name.c_str(), ns_per_elem, s.mean * 1000.0,
                    static_cast<double>(pk.elements) / s.mean / 1e6, s.n, s.lo * 1000.0, s.hi * 1000.0);
    }

    // --- BF16 promote cost, synthetic (single shift, no real block structure to validate) -----------
    // gguf::bf16_to_f32 is a pure function of a 16-bit pattern; timing it against real bytes vs
    // synthetic bytes cannot differ, so a real file is not needed for THIS number the way it is for
    // the block-quant decoders above (AGENTS.md S9's "validate against real file" is about DECODER
    // correctness on real byte layouts, not about a stateless per-value promote already proven correct
    // elsewhere in this codebase).
    double bf16_ns_per_elem = 0;
    {
        const std::uint64_t n = picks[kQ5K].elements;   // same element count as a real picked tensor
        std::vector<std::uint16_t> src(static_cast<std::size_t>(n));
        for (std::size_t i = 0; i < src.size(); ++i) src[i] = static_cast<std::uint16_t>(i * 2654435761u);
        std::vector<double> timings;
        std::vector<float> out(static_cast<std::size_t>(n));
        for (int r = 0; r < reps; ++r) {
            const auto t0 = Clock::now();
            for (std::size_t i = 0; i < src.size(); ++i) out[i] = sub0::gguf::bf16_to_f32(src[i]);
            timings.push_back(secs(t0, Clock::now()));
            g_sink = out[0];
        }
        const Stat s = summarize(timings);
        bf16_ns_per_elem = s.mean * 1e9 / static_cast<double>(n);
        std::printf("  %-6s %-32s %8.3f ns/elem  %9.3f ms/tensor  (%.1f Melem/s, synthetic bit patterns,\n"
                    "                                                                   correctness proven"
                    " elsewhere -- see comment)\n",
                    "BF16", "(synthetic, same n as Q5_K pick)", bf16_ns_per_elem, s.mean * 1000.0,
                    static_cast<double>(n) / s.mean / 1e6);
    }

    // --- arm B: achieved DRAM bandwidth at each byte-width, measured directly -----------------------
    std::printf("\n=== arm B: achieved streaming bandwidth per byte-width (replica-array, past L3) ===\n");
    constexpr std::size_t kMinFootprint = 512ull * 1024 * 1024;   // well past this host's 36 MiB L3
    struct BwResult { std::string label; double bytes_per_sec = 0; double bytes_per_elem = 0; };
    std::vector<BwResult> bw_results;
    for (auto& [type, pk] : picks) {
        const sub0::gguf::BlockSpec b = sub0::gguf::block_spec(type);
        const double bpe = static_cast<double>(b.bytes) / static_cast<double>(b.elems);
        const std::size_t unit_bytes = static_cast<std::size_t>(bpe * static_cast<double>(pk.elements));
        const double bw = measure_stream_bandwidth(unit_bytes, kMinFootprint, bw_reps);
        bw_results.push_back({std::string(type_name(type)) + " bytes", bw, bpe});
        std::printf("  %-6s bytes (%.4f B/elem)  %10.2f GB/s achieved over %.1f MiB touched\n",
                    type_name(type), bpe, bw / 1e9,
                    static_cast<double>(unit_bytes) / (1024.0 * 1024.0));
    }
    double f32_bw = 0, bf16_bw = 0;
    {
        const std::uint64_t n = picks[kQ5K].elements;
        f32_bw = measure_stream_bandwidth(static_cast<std::size_t>(n) * 4, kMinFootprint, bw_reps);
        bf16_bw = measure_stream_bandwidth(static_cast<std::size_t>(n) * 2, kMinFootprint, bw_reps);
        std::printf("  %-6s bytes (4.0000 B/elem)  %10.2f GB/s achieved (baseline: today's resident format)\n",
                    "F32", f32_bw / 1e9);
        std::printf("  %-6s bytes (2.0000 B/elem)  %10.2f GB/s achieved (Phase-1's resident format)\n",
                    "BF16", bf16_bw / 1e9);
    }

    // --- the crossover, computed, not left to the reader ---------------------------------------------
    std::printf("\n=== crossover: 2b (inline dequant) vs 2a (resident, promote-once) ===\n");
    std::printf("decode reads every backbone weight EXACTLY ONCE per token (SEQ_LEN=1, no reuse --\n"
                "QWEN4_MEMORY_MAP.md S3b's own zero-hit-rate finding applies here too), so this is a\n"
                "single-read-per-token comparison: 2b wins iff its per-element cost (Q/BW + D) beats\n"
                "2a's (R/BW). Solved for D: 2b wins iff  D  <  D* = (R - Q) / BW.\n\n");
    std::printf("%-8s %10s %10s %14s %14s %14s %8s\n", "format", "Q B/elem", "R B/elem", "measured D",
                "D* (break-even)", "margin", "verdict");
    for (auto& dr : dequant_results) {
        auto bw_it = std::find_if(bw_results.begin(), bw_results.end(), [&](auto& b) {
            return b.label == std::string(type_name(dr.type)) + " bytes";
        });
        const double Q = bw_it->bytes_per_elem;
        const double BW = bw_it->bytes_per_sec;          // bytes/sec, quantized stream
        const double R = 4.0;                              // resident F32, bytes/elem
        const double D_star_ns = (R - Q) / BW * 1e9;        // ns/elem, using the QUANTIZED stream's own
                                                            // achieved BW as the shared BW term (S2's
                                                            // own framing: same memory bus either way)
        const double D_ns = dr.ns_per_elem;
        const bool inline_wins = D_ns < D_star_ns;
        std::printf("%-8s %10.4f %10.4f %11.3fns %11.3fns %13.3fns %8s\n", type_name(dr.type), Q, R, D_ns,
                    D_star_ns, D_star_ns - D_ns, inline_wins ? "2b (inline)" : "2a (resident)");
    }
    // Same comparison against BF16 as the resident baseline (Phase 1's own end state), so the fork is
    // readable against EITHER phase's resident format, not just today's F32.
    std::printf("\n...and against a BF16-resident baseline (R=2 B/elem, Phase 1's own end state):\n");
    for (auto& dr : dequant_results) {
        auto bw_it = std::find_if(bw_results.begin(), bw_results.end(), [&](auto& b) {
            return b.label == std::string(type_name(dr.type)) + " bytes";
        });
        const double Q = bw_it->bytes_per_elem;
        const double BW = bw_it->bytes_per_sec;
        const double R = 2.0;
        const double D_star_ns = (R - Q) / BW * 1e9;
        const double D_ns = dr.ns_per_elem;
        const bool inline_wins = D_ns < D_star_ns;
        std::printf("%-8s %10.4f %10.4f %11.3fns %11.3fns %13.3fns %8s\n", type_name(dr.type), Q, R, D_ns,
                    D_star_ns, D_star_ns - D_ns, inline_wins ? "2b (inline)" : "2a (resident)");
    }

    // --- full-backbone projection, so the per-tensor numbers land in a token-level context -----------
    std::printf("\n=== per-token projection (informational, not a substitute for the per-format verdict) ===\n");
    std::printf("F32 achieved bandwidth:   %.2f GB/s -> 18.31 GiB backbone would stream in %.1f ms\n",
               f32_bw / 1e9, 18.31 * 1024 * 1024 * 1024 / f32_bw * 1000.0);
    std::printf("BF16 achieved bandwidth:  %.2f GB/s -> 9.16 GiB backbone would stream in %.1f ms\n",
               bf16_bw / 1e9, 9.16 * 1024 * 1024 * 1024 / bf16_bw * 1000.0);
    std::printf("(native-quant per-token figure is format-blended across the real per-tensor mix; see\n"
               " the per-format verdict table above -- a single blended number would hide exactly the\n"
               " Q8_0-vs-Q5_K/Q6_K divergence this bench exists to surface.)\n");
    return 0;
}
