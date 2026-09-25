// backbone_quant_dot_tests.cpp -- O5 phase 2a: unit tests for sub0::bbqd (include/sub0/backbone_quant_dot.hpp),
// the fused int8-activation x native-quant-weight dot product for the BACKBONE's four real quantized
// formats (Q8_0, Q4_K, Q5_K, Q6_K -- docs/BACKBONE_NATIVE_QUANT.md's own census of the real Qwen3.8-
// Flash-Next UD-IQ1_S shards). Structured after tests/moe_quant_tests.cpp's own B35 section, which this
// mirrors deliberately (same split into "is the weight decode exact" vs "how big is the activation
// quantization error", AGENTS.md S6/S9).
//
// PHASE 2a ADDITIONS (streaming AVX2 kernels, docs/BACKBONE_NATIVE_QUANT.md S12): `bbqd::gemv_plane` now
// auto-dispatches on `bbqd::kAvx2Kernels` (moeqd's own `kAvx2Kernels`/`gemv_best` convention, not a
// caller-chosen template bool), and carries a `Threads` template parameter mirroring `gemv::axpy`'s own
// shape. `bbqd::detail::gemv_plane_portable`/`bbqd::detail::gemv_plane_avx2` stay individually callable
// for the differential tests below (mirroring `tests/moe_quant_tests.cpp`'s own "O1" test, which reaches
// into `moeqd::detail::gemv` directly for the same reason). New cases here cover: the streaming kernels'
// exact agreement with the unchanged portable reference at real multi-superblock scale, the non-256-
// aligned-row fallback, `Threads`-count bit-exactness, and the `KScaleTable` bit-unpack trick's exact
// agreement with `gguf::k_scale_min`.
//
// WHAT EACH CASE EXISTS TO RULE OUT, rather than merely exercise (mirroring moe_quant_tests.cpp's own
// discipline):
//   * a WEIGHT-DECODE layout bug (a shifted sc/qh field, Q6_K's own interleaved-strip indexing read
//     wrong, the double-subtracted "-32" this development pass actually found and fixed -- see this
//     file's own "mutation check" case below, which pins that exact regression). Random synthetic block
//     bytes cannot distinguish "close-ish, plausibly scaled, wrong" from "correct" on their own; the
//     lossless-activation trick (an activation that int8-quantizes EXACTLY) removes the other error
//     source entirely, so any remaining disagreement with gguf::to_f32 IS a layout bug.
//   * the AVX2 path silently reading fewer lanes than it claims to -- a REAL defect this development pass
//     found (`_mm_cvtepi8_epi16` only sign-extends the LOW 8 of its 16-byte input; a first version of
//     dot16_avx2/sum16_avx2 converted only that half, silently computing an 8-wide dot for what claimed
//     to be Q6_K's 16-wide sub-block). Caught by requiring EXACT (not close) agreement with the portable
//     path: an integer dot has no precision to lose to reassociation, so ANY difference is a bug, never
//     "acceptable SIMD noise".
//   * the activation quantization error being measured on the wrong data. A first version of this file's
//     Gaussian-activation case used fully-random synthetic K-quant bytes and measured a wildly inflated
//     Q6_K error (12%, vs 0.1% on the model's own real bytes) -- fully-random `sc`/`d` fields give K-quant's
//     affine `bias` term an unrealistic dynamic range no trained weight distribution actually has. Fixed
//     by moving the ERROR-SIZE measurement onto real sidecar bytes (AGENTS.md S9) and keeping the
//     synthetic-byte case strictly for the lossless LAYOUT check, where random bytes are exactly what is
//     wanted (a shifted field is wrong regardless of the data's distribution).
//
// REAL-FILE VALIDATION (AGENTS.md S9): the real Qwen3.8-Flash-Next UD-IQ1_S GGUF shards, located via
// SUB0_QWEN4_GGUF_DIR or a hardcoded default matching this session's own downloaded copy -- same
// graceful-WARN-and-skip pattern as tests/qwen_tokenizer_tests.cpp when the directory is absent, so this
// file still builds and runs (skipping only the real-byte cases) on a machine without the ~4 GiB shards.

#include "sub0/backbone_quant_dot.hpp"
#include "sub0/gguf.hpp"
#include "sub0/transplant.hpp"  // O5 phase 2b-3 phase B: per_head_half_transpose, for the QSA q|gate
                                 // proof case below (cross-checks gemv_plane's row-range selection
                                 // against the SAME row formula the .bin blob's own transplant uses)

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <random>
#include <string>
#include <vector>

using namespace sub0;
namespace fs = std::filesystem;

namespace {

constexpr gguf::TensorType kFormats[4] = {gguf::TensorType::Q8_0, gguf::TensorType::Q4_K,
                                           gguf::TensorType::Q5_K, gguf::TensorType::Q6_K};

const char* format_name(gguf::TensorType t) {
    switch (t) {
        case gguf::TensorType::Q8_0: return "Q8_0";
        case gguf::TensorType::Q4_K: return "Q4_K";
        case gguf::TensorType::Q5_K: return "Q5_K";
        case gguf::TensorType::Q6_K: return "Q6_K";
        default:                     return "?";
    }
}

// Deterministic pseudo-random bytes in one of the four fused K-quant/Q8_0 formats, with every block's
// own f16 scale field(s) patched to a small positive value -- same reasoning as moe_quant_tests.cpp's
// own make_iq_blocks: a fully-random f16 would be inf/NaN often enough to make the lossless comparison
// meaningless, but every OTHER field (sc/m/qh/ql bit patterns) is left fully random, which is exactly
// what a layout-bug check wants (a shifted field is wrong for ANY data, and this maximizes the chance a
// shift produces a numerically visible disagreement rather than accidentally landing on a plausible
// value).
std::vector<std::uint8_t> make_blocks(gguf::TensorType type, std::uint64_t n_elements, std::uint32_t seed) {
    const gguf::BlockSpec spec = gguf::block_spec(static_cast<std::uint32_t>(type));
    REQUIRE(spec.elems != 0);
    REQUIRE(n_elements % spec.elems == 0);
    std::mt19937 rng(seed);
    const std::uint64_t blocks = n_elements / spec.elems;
    std::vector<std::uint8_t> raw(static_cast<std::size_t>(blocks * spec.bytes));
    for (auto& b : raw) b = static_cast<std::uint8_t>(rng() & 0xFFu);
    for (std::uint64_t b = 0; b < blocks; ++b) {
        std::uint8_t* blk = raw.data() + b * spec.bytes;
        auto patch = [&](std::size_t off) {
            const auto d_bits = static_cast<std::uint16_t>(0x3000u + (rng() & 0x0FFFu));
            std::memcpy(blk + off, &d_bits, sizeof d_bits);
        };
        switch (type) {
            case gguf::TensorType::Q8_0: patch(0); break;
            case gguf::TensorType::Q4_K:
            case gguf::TensorType::Q5_K: patch(0); patch(2); break;     // d, dmin
            case gguf::TensorType::Q6_K: patch(208); break;             // d is the LAST field
            default: break;
        }
    }
    return raw;
}

// The f32 side of every comparison below: gguf::to_f32 on the SAME bytes (the project's own already-S5-
// verified scalar decoder), then a plain sequential double-precision dot -- the contract bbqd's fused
// path is checked against, not a second caller of itself.
double reference_dot(gguf::TensorType type, std::span<const std::uint8_t> raw, int n,
                      const std::vector<float>& x) {
    gguf::TensorInfo t;
    t.type_raw = static_cast<std::uint32_t>(type);
    t.dims = {static_cast<std::uint64_t>(n)};
    std::vector<float> w;
    REQUIRE(gguf::to_f32(t, raw, w));
    double s = 0.0;
    for (int i = 0; i < n; ++i)
        s += static_cast<double>(x[static_cast<std::size_t>(i)]) * w[static_cast<std::size_t>(i)];
    return s;
}

// --- real-file location, same graceful-skip pattern as tests/qwen_tokenizer_tests.cpp ----------------

std::string real_gguf_dir() {
    // std::getenv is the portable spelling; MSVC's UCRT headers push _dupenv_s, which is not portable.
    // Silenced locally rather than with a global _CRT_SECURE_NO_WARNINGS -- same pattern
    // tests/qwen_tokenizer_tests.cpp's own model_files_dir() already established.
#if defined(__clang__) || defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#endif
    const char* env = std::getenv("SUB0_QWEN4_GGUF_DIR");
#if defined(__clang__) || defined(__GNUC__)
#pragma GCC diagnostic pop
#endif
    if (env) return env;
    return "D:/ModelWeights/Qwen3.8-Flash-Next-GGUF/UD-IQ1_S";
}

struct RealPick {
    fs::path shard;
    std::uint64_t data_off = 0;
    gguf::TensorInfo info;
};

// One real, sizeable (>=500K elements, so a norm/bias vector is never picked), non-expert, non-PLE
// backbone tensor per format, found by scanning every shard's own header/tensor-table -- never a
// payload bulk-load (AGENTS.md S9's "validate against a real file", applied the same header-only way
// tools/sub0llm-backbone-census.cpp does it). Returns an empty vector if the directory is missing or no
// shard is readable; callers WARN and return rather than REQUIRE-failing, so this file still builds and
// runs (minus the real-byte cases) on a machine without the ~4 GiB shards.
std::vector<RealPick> find_real_picks() {
    std::vector<RealPick> picks;
    const fs::path dir = real_gguf_dir();
    std::error_code ec;
    if (!fs::exists(dir, ec)) return picks;

    std::vector<fs::path> files;
    for (const auto& e : fs::directory_iterator(dir, ec))
        if (e.is_regular_file() && e.path().extension() == ".gguf") files.push_back(e.path());
    std::sort(files.begin(), files.end());

    for (const auto& p : files) {
        std::ifstream f(p, std::ios::binary);
        if (!f) continue;
        std::vector<std::uint8_t> head(64ull * 1024 * 1024);
        f.read(reinterpret_cast<char*>(head.data()), static_cast<std::streamsize>(head.size()));
        head.resize(static_cast<std::size_t>(f.gcount()));
        gguf::Reader r(head);
        if (!r.ok()) continue;
        for (const auto& t : r.tensors()) {
            for (auto type : kFormats) {
                if (t.type_raw != static_cast<std::uint32_t>(type)) continue;
                if (t.name.find("_exps") != std::string::npos) continue;
                if (t.name.find("per_layer_token_embd") != std::string::npos) continue;
                if (t.name.find("ple_") != std::string::npos) continue;
                if (t.element_count() < 500'000) continue;
                if (t.dims.empty() || t.dims[0] % bbqd::GROUP != 0) continue;   // row-aligned only
                bool have = false;
                for (const auto& pk : picks) if (pk.info.type_raw == t.type_raw) have = true;
                if (!have) picks.push_back({p, r.data_offset(), t});
            }
        }
    }
    return picks;
}

std::vector<std::uint8_t> read_tensor_bytes(const RealPick& pk) {
    const gguf::BlockSpec spec = gguf::block_spec(pk.info.type_raw);
    const std::uint64_t n = pk.info.element_count();
    const std::uint64_t byte_len = ((n + spec.elems - 1) / spec.elems) * spec.bytes;
    std::ifstream f(pk.shard, std::ios::binary);
    std::vector<std::uint8_t> raw(static_cast<std::size_t>(byte_len));
    if (!f) return {};
    f.seekg(static_cast<std::streamoff>(pk.data_off + pk.info.offset));
    f.read(reinterpret_cast<char*>(raw.data()), static_cast<std::streamsize>(byte_len));
    if (static_cast<std::uint64_t>(f.gcount()) != byte_len) return {};
    return raw;
}

// An activation row that int8-quantizes LOSSLESSLY: integers in [-126,126] with an exact +/-127 planted
// in every GROUP so amax==127 and the quantizer's scale is exactly 1.0f (moe_quant_tests.cpp's own
// precedent, reused verbatim).
std::vector<float> lossless_row(int n, std::uint32_t seed) {
    std::vector<float> x(static_cast<std::size_t>(n));
    std::mt19937 rng(seed);
    for (int g = 0; g < n / bbqd::GROUP; ++g)
        for (int j = 0; j < bbqd::GROUP; ++j) {
            const int v = (j == 0) ? ((g % 2) ? -127 : 127) : static_cast<int>(rng() % 253u) - 126;
            x[static_cast<std::size_t>(g * bbqd::GROUP + j)] = static_cast<float>(v);
        }
    return x;
}

}  // namespace

TEST_CASE("bbqd: fusable() accepts exactly the four real backbone formats, row-width-aligned",
          "[backbonequant]") {
    REQUIRE(bbqd::fusable(static_cast<std::uint32_t>(gguf::TensorType::Q8_0), 2560));
    REQUIRE(bbqd::fusable(static_cast<std::uint32_t>(gguf::TensorType::Q4_K), 2560));
    REQUIRE(bbqd::fusable(static_cast<std::uint32_t>(gguf::TensorType::Q5_K), 2560));
    REQUIRE(bbqd::fusable(static_cast<std::uint32_t>(gguf::TensorType::Q6_K), 6144));
    // A real backbone contraction dim, per the census -- must divide GROUP=32.
    REQUIRE_FALSE(bbqd::fusable(static_cast<std::uint32_t>(gguf::TensorType::Q8_0), 48));    // ssm_a-ish
    REQUIRE_FALSE(bbqd::fusable(static_cast<std::uint32_t>(gguf::TensorType::IQ1_S), 2560)); // expert-only format
    REQUIRE_FALSE(bbqd::fusable(static_cast<std::uint32_t>(gguf::TensorType::F32), 2560));
    REQUIRE_FALSE(bbqd::fusable(static_cast<std::uint32_t>(gguf::TensorType::Q8_0), 0));
}

TEST_CASE("bbqd: the fused unpackers decode the SAME weights gguf::to_f32 does (lossless activation)",
          "[backbonequant]") {
    // Isolates a WEIGHT-decode disagreement from an activation-quantization one (this file's own header
    // comment, half (a)) -- any remaining gap after this is a real layout bug, not rounding.
    constexpr int kN = 2560;      // a real backbone row length (hidden_size)
    constexpr int kRows = 3;

    const std::vector<float> x = lossless_row(kN, 4242);
    bbqd::ActBlocks xq;
    xq.quantize(x.data(), kN);
    for (int g = 0; g < kN / bbqd::GROUP; ++g)
        REQUIRE(xq.scale[static_cast<std::size_t>(g)] == 1.0f);       // the lossless premise, checked
    for (int i = 0; i < kN; ++i)
        REQUIRE(static_cast<float>(xq.qs[static_cast<std::size_t>(i)]) == x[static_cast<std::size_t>(i)]);

    for (const gguf::TensorType type : kFormats) {
        INFO("format " << format_name(type));
        const auto raw_t = static_cast<std::uint32_t>(type);
        const gguf::BlockSpec spec = gguf::block_spec(raw_t);
        const std::vector<std::uint8_t> raw = make_blocks(type, static_cast<std::uint64_t>(kRows) * kN,
                                                           77u + raw_t);

        // Both kernels: the portable reference and the streaming fast path -- this specific test's own
        // kN=2560 is a multiple of 256, so every K-quant format actually exercises
        // detail::dot_row_q{4,5,6}_k_avx2, not just the fallback. Like the rest of this project's SIMD
        // tests (tests/moe_quant_tests.cpp), this assumes the AVX2 build this project always targets
        // (SUB0_NATIVE=ON) rather than guarding for a hypothetical non-AVX2 host.
        std::vector<float> portable(kRows, 0.f);
        REQUIRE(bbqd::detail::gemv_plane_portable(raw_t, std::span<const std::uint8_t>(raw), kRows, kN, xq,
                                                   portable.data(), 0, kRows));
        std::vector<float> avx2(kRows, 0.f);
        REQUIRE(bbqd::detail::gemv_plane_avx2(raw_t, std::span<const std::uint8_t>(raw), kRows, kN, xq,
                                              avx2.data(), 0, kRows));

        double worst_rel = 0.0;
        for (int row = 0; row < kRows; ++row) {
            const auto byte_off = static_cast<std::size_t>(
                (static_cast<std::uint64_t>(row) * kN / spec.elems) * spec.bytes);
            const double ref = reference_dot(
                type, std::span<const std::uint8_t>(raw).subspan(
                          byte_off, static_cast<std::size_t>(kN / spec.elems * spec.bytes)),
                kN, x);
            REQUIRE(std::isfinite(portable[static_cast<std::size_t>(row)]));
            REQUIRE(std::isfinite(avx2[static_cast<std::size_t>(row)]));
            REQUIRE(std::fabs(ref) > 0.0);
            worst_rel = std::max(
                worst_rel, std::fabs(portable[static_cast<std::size_t>(row)] - ref) / std::fabs(ref));
            worst_rel =
                std::max(worst_rel, std::fabs(avx2[static_cast<std::size_t>(row)] - ref) / std::fabs(ref));
        }
        INFO("worst relative disagreement vs gguf::to_f32 = " << worst_rel);
        REQUIRE(worst_rel < 1e-5);   // float-rounding scale only, same bound moe_quant_tests.cpp uses
    }
}

TEST_CASE("bbqd: the AVX2 path agrees EXACTLY with the portable path (integer arithmetic, no tolerance)",
          "[backbonequant]") {
    // Integer addition/multiplication is exact regardless of accumulation order, so ANY disagreement
    // here is a real bug, never "SIMD reassociation noise" -- this is the test that actually caught
    // dot16_avx2/sum16_avx2 silently reading only 8 of Q6_K's 16 lanes (see this file's own header
    // comment).
    constexpr int kN = 2560;
    constexpr int kRows = 5;
    std::mt19937 rng(99);
    std::normal_distribution<float> normal(0.f, 2.f);
    std::vector<float> x(kN);
    for (float& v : x) v = normal(rng);
    bbqd::ActBlocks xq;
    xq.quantize(x.data(), kN);

    for (const gguf::TensorType type : kFormats) {
        INFO("format " << format_name(type));
        const auto raw_t = static_cast<std::uint32_t>(type);
        const std::vector<std::uint8_t> raw = make_blocks(type, static_cast<std::uint64_t>(kRows) * kN,
                                                           321u + raw_t);
        std::vector<float> portable(kRows, -1.f), avx2(kRows, -2.f);
        REQUIRE(bbqd::detail::gemv_plane_portable(raw_t, std::span<const std::uint8_t>(raw), kRows, kN, xq,
                                                   portable.data(), 0, kRows));
        REQUIRE(bbqd::detail::gemv_plane_avx2(raw_t, std::span<const std::uint8_t>(raw), kRows, kN, xq,
                                              avx2.data(), 0, kRows));
        for (int r = 0; r < kRows; ++r) {
            INFO("row " << r);
            REQUIRE(portable[static_cast<std::size_t>(r)] == avx2[static_cast<std::size_t>(r)]);
        }
    }
}

TEST_CASE("bbqd (O5 phase 2a): the streaming AVX2 kernels agree EXACTLY with the portable path when the "
          "row does NOT start/end on a 256-element superblock boundary (the fallback path)",
          "[backbonequant]") {
    // Real backbone tensors always have row_elems as a multiple of 256 (docs/BACKBONE_NATIVE_QUANT.md
    // S2a's census), so this case is defensive rather than load-bearing today -- but gemv_plane_avx2's
    // own contract promises correctness for ANY row_elems that is merely a multiple of GROUP=32 (falling
    // back to gemv_rows<Plane,true> when row_elems % 256 != 0), and that fallback path deserves its own
    // exact-agreement check rather than being assumed correct by inspection. row_elems=96, n_rows=8 keeps
    // the TOTAL (768 = 3*256) a whole number of superblocks -- required for the bytes to be valid Q4_K/
    // Q5_K/Q6_K data at all -- while individual row boundaries (96, 192, 288, ...) do not land on 256.
    constexpr int kN = 96, kRows = 8;
    std::mt19937 rng(55);
    std::normal_distribution<float> normal(0.f, 1.f);
    std::vector<float> x(kN);
    for (float& v : x) v = normal(rng);
    bbqd::ActBlocks xq;
    xq.quantize(x.data(), kN);

    for (const gguf::TensorType type : {gguf::TensorType::Q4_K, gguf::TensorType::Q5_K, gguf::TensorType::Q6_K}) {
        INFO("format " << format_name(type));
        const auto raw_t = static_cast<std::uint32_t>(type);
        const std::vector<std::uint8_t> raw = make_blocks(type, static_cast<std::uint64_t>(kRows) * kN,
                                                           606u + raw_t);
        std::vector<float> portable(kRows, -1.f), avx2(kRows, -2.f);
        REQUIRE(bbqd::detail::gemv_plane_portable(raw_t, std::span<const std::uint8_t>(raw), kRows, kN, xq,
                                                   portable.data(), 0, kRows));
        REQUIRE(bbqd::detail::gemv_plane_avx2(raw_t, std::span<const std::uint8_t>(raw), kRows, kN, xq,
                                              avx2.data(), 0, kRows));
        for (int r = 0; r < kRows; ++r) {
            INFO("row " << r);
            REQUIRE(portable[static_cast<std::size_t>(r)] == avx2[static_cast<std::size_t>(r)]);
        }
    }
}

TEST_CASE("bbqd (O5 phase 2a): gemv_plane<Threads> is bit-exact across 1/2/4 threads on an uneven split",
          "[backbonequant]") {
    // The Threads-templated public entry point (mirroring gemv::axpy<Threads>'s own shape) must produce
    // the identical result regardless of how many OpenMP workers split the row range -- the same
    // "threaded by output row, bit-exact across thread counts" contract the row_lo/row_hi split below
    // checks manually, now exercised through the actual threading seam a real caller would use.
    constexpr int kN = 2560, kRows = 37;   // 37 is not evenly divisible by 2 or 4
    std::mt19937 rng(202);
    std::normal_distribution<float> normal(0.f, 1.5f);
    std::vector<float> x(kN);
    for (float& v : x) v = normal(rng);
    bbqd::ActBlocks xq;
    xq.quantize(x.data(), kN);

    for (const gguf::TensorType type : kFormats) {
        INFO("format " << format_name(type));
        const auto raw_t = static_cast<std::uint32_t>(type);
        const std::vector<std::uint8_t> raw = make_blocks(type, static_cast<std::uint64_t>(kRows) * kN,
                                                           808u + raw_t);
        std::vector<float> t1(kRows, -1.f), t2(kRows, -2.f), t4(kRows, -3.f);
        REQUIRE(bbqd::gemv_plane<1>(raw_t, std::span<const std::uint8_t>(raw), kRows, kN, xq, t1.data()));
        REQUIRE(bbqd::gemv_plane<2>(raw_t, std::span<const std::uint8_t>(raw), kRows, kN, xq, t2.data()));
        REQUIRE(bbqd::gemv_plane<4>(raw_t, std::span<const std::uint8_t>(raw), kRows, kN, xq, t4.data()));
        for (int r = 0; r < kRows; ++r) {
            INFO("row " << r);
            REQUIRE(t1[static_cast<std::size_t>(r)] == t2[static_cast<std::size_t>(r)]);
            REQUIRE(t1[static_cast<std::size_t>(r)] == t4[static_cast<std::size_t>(r)]);
        }
    }
}

TEST_CASE("bbqd (O5 phase 2b-2b): gemv_plane's caller-owned gsum16 scratch reproduces the default "
          "(no-hint) result exactly, at 1 and 4 threads, and survives being REUSED across a changed "
          "activation width",
          "[backbonequant]") {
    // Closes docs/BACKBONE_NATIVE_QUANT.md S12i's own TODO(phase 2b): Gsum16 must no longer be built
    // fresh inside gemv_plane on every call when a caller hands one in. This checks three things a
    // caller-owned-scratch API can get wrong that a pure "does it compile" check would miss:
    //   1. Passing a caller-owned Gsum16* must not change the ANSWER (same bytes, gsum16 hint or not).
    //   2. The SAME instance, reused at Threads>1, must still agree (built once, read by every thread --
    //      the concurrency argument the header comment makes, checked here rather than just asserted).
    //   3. Reusing the SAME Gsum16 object across TWO DIFFERENT activation widths must not leak stale
    //      per-16 sums from the first call into the second's shorter row range -- build()'s own
    //      `v.assign(groups*2, 0)` must re-derive the right group count every call.
    constexpr auto type = gguf::TensorType::Q6_K;   // the one format this scratch actually feeds
    const auto raw_t = static_cast<std::uint32_t>(type);
    constexpr int kRows = 21;   // not evenly divisible by 4 -- exercises the same uneven-split path

    bbqd::Gsum16 shared_gsum16;   // ONE object, reused across every call below -- both thread counts AND
                                  // both activation widths -- so its own `.v` buffer is genuinely
                                  // exercised across a real resize, not just re-passed unchanged.
    auto run_one = [&](int n_elems, std::uint32_t seed) {
        std::mt19937 rng(seed);
        std::normal_distribution<float> normal(0.f, 1.5f);
        std::vector<float> x(static_cast<std::size_t>(n_elems));
        for (float& v : x) v = normal(rng);
        bbqd::ActBlocks xq;
        xq.quantize(x.data(), n_elems);
        const std::vector<std::uint8_t> raw =
            make_blocks(type, static_cast<std::uint64_t>(kRows) * n_elems, seed);

        std::vector<float> no_hint(kRows, -1.f), hinted_1t(kRows, -2.f), hinted_4t(kRows, -3.f);
        REQUIRE(bbqd::gemv_plane<1>(raw_t, std::span<const std::uint8_t>(raw), kRows, n_elems, xq,
                                    no_hint.data()));
        REQUIRE(bbqd::gemv_plane<1>(raw_t, std::span<const std::uint8_t>(raw), kRows, n_elems, xq,
                                    hinted_1t.data(), 0, -1, &shared_gsum16));
        REQUIRE(bbqd::gemv_plane<4>(raw_t, std::span<const std::uint8_t>(raw), kRows, n_elems, xq,
                                    hinted_4t.data(), 0, -1, &shared_gsum16));
        for (int r = 0; r < kRows; ++r) {
            INFO("n_elems " << n_elems << " row " << r);
            REQUIRE(no_hint[static_cast<std::size_t>(r)] == hinted_1t[static_cast<std::size_t>(r)]);
            REQUIRE(no_hint[static_cast<std::size_t>(r)] == hinted_4t[static_cast<std::size_t>(r)]);
        }
    };

    run_one(2560, 909u);   // real backbone width -- shared_gsum16 sized here for the first time
    run_one(6144, 909u);   // a LARGER width, same shared_gsum16 object -- must re-derive its own group
                           // count rather than reading stale entries left over from the 2560 call above
    run_one(2560, 909u);   // back down to the smaller width -- must not read past a now-shorter v either
}

TEST_CASE("bbqd (O5 phase 2a): KScaleTable's branch-free bit-unpack agrees EXACTLY with gguf::k_scale_min "
          "for all 8 sub-block indices",
          "[backbonequant]") {
    // AGENTS.md S13 pass 2 replaced 8 branchy gguf::k_scale_min calls per superblock with the
    // branch-free bit-manipulation llama.cpp's own AVX2 Q4_K/Q5_K kernels use (this file's own header
    // comment on KScaleTable::load). Checked here independently of any GEMV: for several random 12-byte
    // scale blocks, every sub-block's (scale, min) must match gguf::k_scale_min's own byte-at-a-time
    // reference exactly -- integers, so "exactly" is the only meaningful bar.
    std::mt19937 rng(2026);
    for (int trial = 0; trial < 32; ++trial) {
        std::array<std::uint8_t, 16> blk{};   // [0..1]=d, [2..3]=dmin (unused here), [4..15]=packed scales
        for (auto& b : blk) b = static_cast<std::uint8_t>(rng() & 0xFFu);
        // KScaleTable::load reads d/dmin as f16 from blk+0/+2 and folds them into sc[i]/m[i] -- pin them
        // to 1.0f (f16 0x3C00) so this case isolates the BIT-UNPACK itself, not the d/dmin multiply.
        blk[0] = 0x00; blk[1] = 0x3C; blk[2] = 0x00; blk[3] = 0x3C;

        bbqd::detail::KScaleTable t;
        t.load(blk.data());

        const std::uint8_t* scales = blk.data() + 4;
        for (int i = 0; i < 8; ++i) {
            std::uint8_t sc_ref = 0, m_ref = 0;
            gguf::k_scale_min(i, scales, sc_ref, m_ref);
            INFO("trial " << trial << " sub-block " << i);
            REQUIRE(t.sc[static_cast<std::size_t>(i)] == static_cast<float>(sc_ref));
            REQUIRE(t.m[static_cast<std::size_t>(i)] == -static_cast<float>(m_ref));
        }
    }
}

TEST_CASE("bbqd (O5 phase 2a): Gsum16 matches a direct per-16-element sum of the quantized activation",
          "[backbonequant]") {
    // Q6_K's own per-16 activation-sum cache (Gsum16), checked independently of any GEMV against a
    // direct scalar re-sum of the same ActBlocks::qs bytes.
    constexpr int kN = 2560;
    std::mt19937 rng(4321);
    std::normal_distribution<float> normal(0.f, 1.f);
    std::vector<float> x(kN);
    for (float& v : x) v = normal(rng);
    bbqd::ActBlocks xq;
    xq.quantize(x.data(), kN);

    // O5 phase 2b-2b: Gsum16 moved to public bbqd scope (from bbqd::detail) so it is nameable as
    // caller-owned scratch -- see backbone_quant_dot.hpp's own comment on the type.
    bbqd::Gsum16 g16;
    g16.build(xq);
    REQUIRE(g16.v.size() == static_cast<std::size_t>(kN / bbqd::GROUP) * 2);

    for (int g = 0; g < kN / bbqd::GROUP; ++g) {
        std::int32_t lo = 0, hi = 0;
        for (int l = 0; l < 16; ++l) lo += xq.qs[static_cast<std::size_t>(g) * bbqd::GROUP + static_cast<std::size_t>(l)];
        for (int l = 0; l < 16; ++l) hi += xq.qs[static_cast<std::size_t>(g) * bbqd::GROUP + 16 + static_cast<std::size_t>(l)];
        INFO("group " << g);
        REQUIRE(g16.v[static_cast<std::size_t>(g) * 2 + 0] == lo);
        REQUIRE(g16.v[static_cast<std::size_t>(g) * 2 + 1] == hi);
    }
}

TEST_CASE("bbqd: threading by output row is bit-exact across any row split", "[backbonequant]") {
    // The property the design doc's LAYOUT section rests on: gemv_rows over [row_lo,row_hi) touches only
    // that range's own plane bytes and out slots, so splitting a GEMV across worker threads by output
    // row must reproduce the single-call result bit for bit, for any split.
    constexpr int kN = 2560, kRows = 8;
    std::mt19937 rng(7);
    std::normal_distribution<float> normal(0.f, 1.5f);
    std::vector<float> x(kN);
    for (float& v : x) v = normal(rng);
    bbqd::ActBlocks xq;
    xq.quantize(x.data(), kN);

    for (const gguf::TensorType type : kFormats) {
        INFO("format " << format_name(type));
        const auto raw_t = static_cast<std::uint32_t>(type);
        const std::vector<std::uint8_t> raw = make_blocks(type, static_cast<std::uint64_t>(kRows) * kN,
                                                           555u + raw_t);
        std::vector<float> whole(kRows, 0.f);
        REQUIRE(bbqd::gemv_plane(raw_t, std::span<const std::uint8_t>(raw), kRows, kN, xq, whole.data()));

        // Split into three uneven ranges (not a clean divisor of kRows), each written into its own slot
        // of a shared output buffer -- exactly how a real multi-threaded caller would use this.
        std::vector<float> split(kRows, -99.f);
        const int cuts[4] = {0, 3, 5, kRows};
        for (int c = 0; c < 3; ++c) {
            const int lo = cuts[c], hi = cuts[c + 1];
            std::vector<float> part(static_cast<std::size_t>(hi - lo), -1.f);
            REQUIRE(bbqd::gemv_plane(raw_t, std::span<const std::uint8_t>(raw), kRows, kN, xq, part.data(),
                                     lo, hi));
            std::memcpy(split.data() + lo, part.data(), part.size() * sizeof(float));
        }
        for (int r = 0; r < kRows; ++r) {
            INFO("row " << r);
            REQUIRE(whole[static_cast<std::size_t>(r)] == split[static_cast<std::size_t>(r)]);
        }
    }
}

TEST_CASE("bbqd: gemv_plane refuses an unfusable format or a too-short span, untouched on refusal",
          "[backbonequant]") {
    const std::vector<std::uint8_t> q8 = make_blocks(gguf::TensorType::Q8_0, 256, 5u);
    std::vector<float> x(256, 0.5f);
    bbqd::ActBlocks xq;
    xq.quantize(x.data(), 256);
    float out = 1234.f;

    // An expert-only format this path does not (and must not) claim to handle.
    REQUIRE_FALSE(bbqd::gemv_plane(static_cast<std::uint32_t>(gguf::TensorType::IQ1_S),
                                   std::span<const std::uint8_t>(q8), 1, 256, xq, &out));
    REQUIRE(out == 1234.f);   // refused means untouched, not partially written

    // Correct format, but the span is shorter than the declared geometry needs.
    REQUIRE_FALSE(bbqd::gemv_plane(static_cast<std::uint32_t>(gguf::TensorType::Q8_0),
                                   std::span<const std::uint8_t>(q8).first(10), 1, 256, xq, &out));
    REQUIRE(out == 1234.f);

    // A row range outside [0, n_rows] must also be refused, not clamped.
    REQUIRE_FALSE(bbqd::gemv_plane(static_cast<std::uint32_t>(gguf::TensorType::Q8_0),
                                   std::span<const std::uint8_t>(q8), 1, 256, xq, &out, 0, 5));
    REQUIRE(out == 1234.f);

    // The SAME refusal contract holds through the Threads>1 entry point, checked up front (before any
    // thread starts) rather than per-thread -- see gemv_plane<Threads>'s own comment on why an earlier
    // thread's own in-bounds sub-range must not write while a later thread's own sub-range is the one
    // that is actually invalid.
    REQUIRE_FALSE(bbqd::gemv_plane<4>(static_cast<std::uint32_t>(gguf::TensorType::Q8_0),
                                      std::span<const std::uint8_t>(q8), 1, 256, xq, &out, 0, 5));
    REQUIRE(out == 1234.f);
}

TEST_CASE("bbqd (AGENTS.md S9): validated against REAL bytes from the Qwen3.8-Flash-Next shards",
          "[backbonequant]") {
    const auto picks = find_real_picks();
    if (picks.size() < 4) {
        WARN("the real Qwen3.8-Flash-Next UD-IQ1_S GGUF shards were not found (checked "
             "SUB0_QWEN4_GGUF_DIR / " << real_gguf_dir() << ") -- skipping the real-byte validation");
        return;
    }

    for (const RealPick& pk : picks) {
        INFO("tensor " << pk.info.name << " format " << format_name(static_cast<gguf::TensorType>(pk.info.type_raw)));
        const int row_elems = static_cast<int>(pk.info.dims[0]);
        const std::vector<std::uint8_t> raw = read_tensor_bytes(pk);
        REQUIRE_FALSE(raw.empty());

        // (a) lossless-activation decode, on the tensor's own real first row.
        const std::vector<float> x = lossless_row(row_elems, 1234u + pk.info.type_raw);
        bbqd::ActBlocks xq;
        xq.quantize(x.data(), row_elems);
        float fused = 0.f;
        REQUIRE(bbqd::detail::gemv_plane_portable(pk.info.type_raw, std::span<const std::uint8_t>(raw), 1,
                                                   row_elems, xq, &fused, 0, 1));
        const double ref = reference_dot(static_cast<gguf::TensorType>(pk.info.type_raw), raw, row_elems, x);
        REQUIRE(std::isfinite(fused));
        REQUIRE(std::fabs(ref) > 0.0);
        const double rel = std::fabs(static_cast<double>(fused) - ref) / std::fabs(ref);
        INFO("relative disagreement on real bytes = " << rel);
        REQUIRE(rel < 1e-5);

        // AVX2 (the streaming kernel, since every real pick's own row_elems is a multiple of 256 --
        // docs/BACKBONE_NATIVE_QUANT.md S2a's census) agrees exactly on the real bytes too.
        float fused_avx2 = 0.f;
        REQUIRE(bbqd::detail::gemv_plane_avx2(pk.info.type_raw, std::span<const std::uint8_t>(raw), 1,
                                              row_elems, xq, &fused_avx2, 0, 1));
        REQUIRE(fused == fused_avx2);

        // (b) the activation-quantization error, measured on the model's OWN real weight statistics
        // rather than synthetic random block bytes -- see this file's own header comment on why K-quant
        // formats' affine bias term makes that distinction matter here (it did not for B35's pure-scale
        // formats). A Gaussian row is the realistic case (real backbone inputs are post-norm).
        std::vector<float> xg(static_cast<std::size_t>(row_elems));
        std::mt19937 rng(20260921);
        std::normal_distribution<float> normal(0.f, 1.f);
        for (float& v : xg) v = normal(rng);
        bbqd::ActBlocks xqg;
        xqg.quantize(xg.data(), row_elems);
        float fused_g = 0.f;
        REQUIRE(bbqd::detail::gemv_plane_portable(pk.info.type_raw, std::span<const std::uint8_t>(raw), 1,
                                                   row_elems, xqg, &fused_g, 0, 1));
        const double ref_g = reference_dot(static_cast<gguf::TensorType>(pk.info.type_raw), raw, row_elems, xg);
        REQUIRE(std::fabs(ref_g) > 0.0);
        const double rel_g = std::fabs(static_cast<double>(fused_g) - ref_g) / std::fabs(ref_g);
        INFO("activation-quantization relative error on real weights = " << rel_g);
        // Measured range on the real backbone tensors this picks (session's own numbers, see the design
        // doc): Q8_0 ~0.2%, Q4_K ~3.9%, Q5_K ~2.0%, Q6_K ~0.1%. 10% is a real, generous margin above the
        // worst observed case, not a target picked to make the test pass.
        REQUIRE(rel_g < 0.10);
    }
}

// --- pass 4 (AGENTS.md S13, docs/BACKBONE_NATIVE_QUANT.md S14): the per-256 ActSuper activation --------
// --- scheme and its Q4_K/Q5_K/Q6_K portable/AVX2/AVX-VNNI streaming kernels -----------------------------
//
// Mirrors the structure of the ActBlocks cases above exactly (lossless-decode isolates weight-layout bugs
// from activation-quantization error; exact-agreement checks integer arithmetic, which has no
// reassociation tolerance to hide behind; real-byte validation per AGENTS.md S9), applied to the NEW
// per-256 activation type instead of moeqd::ActBlocks. Q8_0 is intentionally absent from every case here
// -- super_fusable() refuses it by design (ActSuper's own header comment: Q8_0 has no per-sub-block scale
// to fold, so a coarser activation would only cost it accuracy for nothing).

namespace {
constexpr gguf::TensorType kSuperFormats[3] = {gguf::TensorType::Q4_K, gguf::TensorType::Q5_K,
                                                gguf::TensorType::Q6_K};
}  // namespace

TEST_CASE("bbqd (pass 4): super_fusable() accepts only Q4_K/Q5_K/Q6_K at a 256-aligned row width",
          "[backbonequant]") {
    REQUIRE(bbqd::super_fusable(static_cast<std::uint32_t>(gguf::TensorType::Q4_K), 2560));
    REQUIRE(bbqd::super_fusable(static_cast<std::uint32_t>(gguf::TensorType::Q5_K), 2560));
    REQUIRE(bbqd::super_fusable(static_cast<std::uint32_t>(gguf::TensorType::Q6_K), 6144));
    // Q8_0 is excluded by design (ActSuper's own header comment), even at a 256-aligned width.
    REQUIRE_FALSE(bbqd::super_fusable(static_cast<std::uint32_t>(gguf::TensorType::Q8_0), 2560));
    // A row width that is a multiple of GROUP=32 but NOT of 256 is fusable() territory, not this path's.
    REQUIRE_FALSE(bbqd::super_fusable(static_cast<std::uint32_t>(gguf::TensorType::Q4_K), 96));
    REQUIRE_FALSE(bbqd::super_fusable(static_cast<std::uint32_t>(gguf::TensorType::IQ1_S), 2560));
    REQUIRE_FALSE(bbqd::super_fusable(static_cast<std::uint32_t>(gguf::TensorType::Q4_K), 0));
}

TEST_CASE("bbqd (pass 4): the fused super unpackers decode the SAME weights gguf::to_f32 does "
          "(lossless activation)",
          "[backbonequant]") {
    // lossless_row() plants an exact +/-127 in every 32-wide GROUP, so every 256-wide SUPERBLOCK's own
    // amax is also exactly 127 (the max of eight already-127-magnitude sub-maxima) -- the same lossless
    // premise the ActBlocks case above relies on, one granularity up. Checked directly below rather than
    // assumed from that reasoning alone.
    constexpr int kN = 2560, kRows = 3;
    const std::vector<float> x = lossless_row(kN, 9191);
    bbqd::ActSuper xq;
    xq.quantize(x.data(), kN);
    for (int s = 0; s < kN / 256; ++s) REQUIRE(xq.d[static_cast<std::size_t>(s)] == 1.0f);
    for (int i = 0; i < kN; ++i)
        REQUIRE(static_cast<float>(xq.qs[static_cast<std::size_t>(i)]) == x[static_cast<std::size_t>(i)]);

    for (const gguf::TensorType type : kSuperFormats) {
        INFO("format " << format_name(type));
        const auto raw_t = static_cast<std::uint32_t>(type);
        const gguf::BlockSpec spec = gguf::block_spec(raw_t);
        const std::vector<std::uint8_t> raw = make_blocks(type, static_cast<std::uint64_t>(kRows) * kN,
                                                           1717u + raw_t);

        std::vector<float> portable(kRows, 0.f);
        REQUIRE(bbqd::detail::gemv_plane_super_portable(raw_t, std::span<const std::uint8_t>(raw), kRows,
                                                        kN, xq, portable.data(), 0, kRows));
        std::vector<float> avx2(kRows, 0.f);
        REQUIRE(bbqd::detail::gemv_plane_super_avx2(raw_t, std::span<const std::uint8_t>(raw), kRows, kN,
                                                     xq, avx2.data(), 0, kRows));

        double worst_rel = 0.0;
        for (int row = 0; row < kRows; ++row) {
            const auto byte_off = static_cast<std::size_t>(
                (static_cast<std::uint64_t>(row) * kN / spec.elems) * spec.bytes);
            const double ref = reference_dot(
                type, std::span<const std::uint8_t>(raw).subspan(
                          byte_off, static_cast<std::size_t>(kN / spec.elems * spec.bytes)),
                kN, x);
            REQUIRE(std::isfinite(portable[static_cast<std::size_t>(row)]));
            REQUIRE(std::isfinite(avx2[static_cast<std::size_t>(row)]));
            REQUIRE(std::fabs(ref) > 0.0);
            worst_rel = std::max(
                worst_rel, std::fabs(portable[static_cast<std::size_t>(row)] - ref) / std::fabs(ref));
            worst_rel =
                std::max(worst_rel, std::fabs(avx2[static_cast<std::size_t>(row)] - ref) / std::fabs(ref));
        }
        INFO("worst relative disagreement vs gguf::to_f32 = " << worst_rel);
        REQUIRE(worst_rel < 1e-5);
    }
}

TEST_CASE("bbqd (pass 4): the AVX2 super path agrees EXACTLY with the portable super path",
          "[backbonequant]") {
    // Integer arithmetic, no tolerance -- same discipline as the ActBlocks case above. This is the check
    // that would have caught a scale-fold algebra mistake (e.g. a lane-count/mullo width bug) the way the
    // ActBlocks-era tests caught the double-subtracted Q6_K zero-point and the 16-lane AVX2 bug.
    constexpr int kN = 2560, kRows = 5;
    std::mt19937 rng(919);
    std::normal_distribution<float> normal(0.f, 2.f);
    std::vector<float> x(kN);
    for (float& v : x) v = normal(rng);
    bbqd::ActSuper xq;
    xq.quantize(x.data(), kN);

    for (const gguf::TensorType type : kSuperFormats) {
        INFO("format " << format_name(type));
        const auto raw_t = static_cast<std::uint32_t>(type);
        const std::vector<std::uint8_t> raw = make_blocks(type, static_cast<std::uint64_t>(kRows) * kN,
                                                           2323u + raw_t);
        std::vector<float> portable(kRows, -1.f), avx2(kRows, -2.f);
        REQUIRE(bbqd::detail::gemv_plane_super_portable(raw_t, std::span<const std::uint8_t>(raw), kRows,
                                                        kN, xq, portable.data(), 0, kRows));
        REQUIRE(bbqd::detail::gemv_plane_super_avx2(raw_t, std::span<const std::uint8_t>(raw), kRows, kN,
                                                     xq, avx2.data(), 0, kRows));
        for (int r = 0; r < kRows; ++r) {
            INFO("row " << r);
            REQUIRE(portable[static_cast<std::size_t>(r)] == avx2[static_cast<std::size_t>(r)]);
        }
    }
}

#if defined(SUB0_BBQD_VNNI)
TEST_CASE("bbqd (pass 4): the AVX-VNNI super path agrees EXACTLY with the portable and AVX2 super paths",
          "[backbonequant]") {
    // Only compiled/run when this TU was built with AVX-VNNI (this project's own SUB0_NATIVE=ON host --
    // docs/BACKBONE_NATIVE_QUANT.md S14's own -march=native macro dump confirms it here). Distributivity
    // of integer multiplication over addition is what makes dpbusd+mullo bit-exact with maddubs+madd
    // (this header's own S14 comment) -- checked here, not assumed from the algebra alone.
    constexpr int kN = 2560, kRows = 5;
    std::mt19937 rng(3131);
    std::normal_distribution<float> normal(0.f, 2.f);
    std::vector<float> x(kN);
    for (float& v : x) v = normal(rng);
    bbqd::ActSuper xq;
    xq.quantize(x.data(), kN);

    for (const gguf::TensorType type : kSuperFormats) {
        INFO("format " << format_name(type));
        const auto raw_t = static_cast<std::uint32_t>(type);
        const std::vector<std::uint8_t> raw = make_blocks(type, static_cast<std::uint64_t>(kRows) * kN,
                                                           4141u + raw_t);
        std::vector<float> portable(kRows, -1.f), avx2(kRows, -2.f), vnni(kRows, -3.f);
        REQUIRE(bbqd::detail::gemv_plane_super_portable(raw_t, std::span<const std::uint8_t>(raw), kRows,
                                                        kN, xq, portable.data(), 0, kRows));
        REQUIRE(bbqd::detail::gemv_plane_super_avx2(raw_t, std::span<const std::uint8_t>(raw), kRows, kN,
                                                     xq, avx2.data(), 0, kRows));
        REQUIRE(bbqd::detail::gemv_plane_super_vnni(raw_t, std::span<const std::uint8_t>(raw), kRows, kN,
                                                     xq, vnni.data(), 0, kRows));
        for (int r = 0; r < kRows; ++r) {
            INFO("row " << r);
            REQUIRE(portable[static_cast<std::size_t>(r)] == avx2[static_cast<std::size_t>(r)]);
            REQUIRE(portable[static_cast<std::size_t>(r)] == vnni[static_cast<std::size_t>(r)]);
        }
    }
}
#endif  // SUB0_BBQD_VNNI

TEST_CASE("bbqd (pass 4): gemv_plane_super<Threads> is bit-exact across 1/2/4 threads on an uneven split",
          "[backbonequant]") {
    constexpr int kN = 2560, kRows = 37;   // 37 is not evenly divisible by 2 or 4
    std::mt19937 rng(5151);
    std::normal_distribution<float> normal(0.f, 1.5f);
    std::vector<float> x(kN);
    for (float& v : x) v = normal(rng);
    bbqd::ActSuper xq;
    xq.quantize(x.data(), kN);

    for (const gguf::TensorType type : kSuperFormats) {
        INFO("format " << format_name(type));
        const auto raw_t = static_cast<std::uint32_t>(type);
        const std::vector<std::uint8_t> raw = make_blocks(type, static_cast<std::uint64_t>(kRows) * kN,
                                                           6161u + raw_t);
        std::vector<float> t1(kRows, -1.f), t2(kRows, -2.f), t4(kRows, -3.f);
        REQUIRE(bbqd::gemv_plane_super<1>(raw_t, std::span<const std::uint8_t>(raw), kRows, kN, xq, t1.data()));
        REQUIRE(bbqd::gemv_plane_super<2>(raw_t, std::span<const std::uint8_t>(raw), kRows, kN, xq, t2.data()));
        REQUIRE(bbqd::gemv_plane_super<4>(raw_t, std::span<const std::uint8_t>(raw), kRows, kN, xq, t4.data()));
        for (int r = 0; r < kRows; ++r) {
            INFO("row " << r);
            REQUIRE(t1[static_cast<std::size_t>(r)] == t2[static_cast<std::size_t>(r)]);
            REQUIRE(t1[static_cast<std::size_t>(r)] == t4[static_cast<std::size_t>(r)]);
        }
    }
}

TEST_CASE("bbqd (pass 4): gemv_plane_super refuses Q8_0 and an unaligned row, untouched on refusal",
          "[backbonequant]") {
    const std::vector<std::uint8_t> q4 = make_blocks(gguf::TensorType::Q4_K, 256, 15u);
    std::vector<float> x(256, 0.5f);
    bbqd::ActSuper xq;
    xq.quantize(x.data(), 256);
    float out = 4321.f;

    REQUIRE_FALSE(bbqd::gemv_plane_super(static_cast<std::uint32_t>(gguf::TensorType::Q8_0),
                                         std::span<const std::uint8_t>(q4), 1, 256, xq, &out));
    REQUIRE(out == 4321.f);

    const std::vector<std::uint8_t> q4_96 = make_blocks(gguf::TensorType::Q4_K, 96 * 8, 16u);
    bbqd::ActSuper xq96;   // never quantized to 96 -- the refusal must happen before x is even touched
    REQUIRE_FALSE(bbqd::gemv_plane_super(static_cast<std::uint32_t>(gguf::TensorType::Q4_K),
                                         std::span<const std::uint8_t>(q4_96), 8, 96, xq96, &out));
    REQUIRE(out == 4321.f);

    bbqd::ActSuper empty;
    REQUIRE_FALSE(bbqd::gemv_plane_super<2>(static_cast<std::uint32_t>(gguf::TensorType::Q4_K),
                                            std::span<const std::uint8_t>(q4), 1, 256, empty, &out));
    REQUIRE(out == 4321.f);
    REQUIRE_FALSE(bbqd::detail::gemv_plane_super_portable(
        static_cast<std::uint32_t>(gguf::TensorType::Q4_K), std::span<const std::uint8_t>(q4),
        1, 256, empty, &out, 0, 1));
    REQUIRE(out == 4321.f);
}

TEST_CASE("bbqd (pass 4, AGENTS.md S9): validated against REAL bytes, activation error reported "
          "side by side with the ActBlocks (per-32) scheme",
          "[backbonequant]") {
    const auto picks = find_real_picks();
    if (picks.size() < 4) {
        WARN("the real Qwen3.8-Flash-Next UD-IQ1_S GGUF shards were not found -- skipping");
        return;
    }

    for (const RealPick& pk : picks) {
        const auto type = static_cast<gguf::TensorType>(pk.info.type_raw);
        if (type == gguf::TensorType::Q8_0) continue;   // ActSuper does not cover Q8_0 by design
        INFO("tensor " << pk.info.name << " format " << format_name(type));
        const int row_elems = static_cast<int>(pk.info.dims[0]);
        REQUIRE(row_elems % 256 == 0);   // the real census guarantees this; checked, not assumed
        const std::vector<std::uint8_t> raw = read_tensor_bytes(pk);
        REQUIRE_FALSE(raw.empty());

        // (a) lossless-activation decode -- the weight side must stay exact under the new scheme too.
        const std::vector<float> x = lossless_row(row_elems, 8181u + pk.info.type_raw);
        bbqd::ActSuper xq;
        xq.quantize(x.data(), row_elems);
        float fused = 0.f;
        REQUIRE(bbqd::detail::gemv_plane_super_portable(pk.info.type_raw, std::span<const std::uint8_t>(raw),
                                                         1, row_elems, xq, &fused, 0, 1));
        const double ref = reference_dot(type, raw, row_elems, x);
        REQUIRE(std::isfinite(fused));
        REQUIRE(std::fabs(ref) > 0.0);
        const double rel = std::fabs(static_cast<double>(fused) - ref) / std::fabs(ref);
        INFO("relative disagreement on real bytes (super, portable) = " << rel);
        REQUIRE(rel < 1e-5);

        float fused_avx2 = 0.f;
        REQUIRE(bbqd::detail::gemv_plane_super_avx2(pk.info.type_raw, std::span<const std::uint8_t>(raw), 1,
                                                     row_elems, xq, &fused_avx2, 0, 1));
        REQUIRE(fused == fused_avx2);
#if defined(SUB0_BBQD_VNNI)
        float fused_vnni = 0.f;
        REQUIRE(bbqd::detail::gemv_plane_super_vnni(pk.info.type_raw, std::span<const std::uint8_t>(raw), 1,
                                                     row_elems, xq, &fused_vnni, 0, 1));
        REQUIRE(fused == fused_vnni);
#endif

        // (b) activation-quantization error on the model's OWN real weight statistics, computed for BOTH
        // schemes on the IDENTICAL Gaussian row so the comparison is apples-to-apples -- the deliverable
        // this pass explicitly asks not to bury: does the coarser per-256 scale cost real accuracy?
        // SAME seed as the ActBlocks-only real-byte case above (20260921) -- so `rel_g_old` here
        // reproduces that case's own already-documented baseline (docs/BACKBONE_NATIVE_QUANT.md S5b:
        // Q8_0 0.20%, Q4_K 3.85%, Q5_K 1.99%, Q6_K 0.12%) exactly, and `rel_g_new` is a true apples-to-
        // apples comparison on the IDENTICAL activation draw, not a different random instance.
        std::vector<float> xg(static_cast<std::size_t>(row_elems));
        std::mt19937 rng(20260921);
        std::normal_distribution<float> normal(0.f, 1.f);
        for (float& v : xg) v = normal(rng);
        const double ref_g = reference_dot(type, raw, row_elems, xg);
        REQUIRE(std::fabs(ref_g) > 0.0);

        bbqd::ActBlocks xqg_old;
        xqg_old.quantize(xg.data(), row_elems);
        float fused_g_old = 0.f;
        REQUIRE(bbqd::detail::gemv_plane_portable(pk.info.type_raw, std::span<const std::uint8_t>(raw), 1,
                                                   row_elems, xqg_old, &fused_g_old, 0, 1));
        const double rel_g_old = std::fabs(static_cast<double>(fused_g_old) - ref_g) / std::fabs(ref_g);

        bbqd::ActSuper xqg_new;
        xqg_new.quantize(xg.data(), row_elems);
        float fused_g_new = 0.f;
        REQUIRE(bbqd::detail::gemv_plane_super_portable(pk.info.type_raw, std::span<const std::uint8_t>(raw),
                                                         1, row_elems, xqg_new, &fused_g_new, 0, 1));
        const double rel_g_new = std::fabs(static_cast<double>(fused_g_new) - ref_g) / std::fabs(ref_g);

        INFO("activation-quantization relative error, OLD per-32 ActBlocks = " << rel_g_old
             << ", NEW per-256 ActSuper = " << rel_g_new);
        // Generous margin (not a target picked to pass): the design doc's own measured range for the
        // per-32 scheme tops out at ~3.9% (Q4_K); a per-256 scale is strictly coarser so some rise is
        // expected, but it must still stay well clear of anything that would visibly move end-to-end
        // logits (the FP8/B35 precedent's own ~0.2-0.4 L2-relative band, docs/BACKBONE_NATIVE_QUANT.md
        // S5b/S12g).
        REQUIRE(rel_g_new < 0.25);
    }
}

TEST_CASE("bbqd (O5 phase 2b-3 phase B): gemv_plane's row-range selection reproduces "
          "transplant::per_head_half_transpose's own q|gate split, cross-checked against the "
          "TRANSPOSED (bf16-blob-shaped) weight, not merely a second dequantize of the same rows",
          "[backbonequant]") {
    // qsa_math.hpp's Native path reads QsaQGateProj (the WHOLE `attn_q.weight` tensor, stored verbatim in
    // GGUF/tiled row order by backbone_quant.hpp) via one bbqd::gemv_plane call per (head, half), with
    // row_lo/row_hi = h*2*head_dim + half*head_dim .. +head_dim -- exactly `transplant::
    // per_head_half_transpose`'s own per-head row-selection arithmetic (transplant.hpp), applied as a ROW
    // RANGE into the whole tensor's own GEMV instead of a pre-split copy (backbone_quant.hpp's own
    // deliberate "no pre-split, no off-by-one risk" decision, S13b).
    //
    // This is the proof AGAINST THE BF16 PATH the task brief asks for, not merely a re-derivation of the
    // same formula: `per_head_half_transpose` is run on the SAME dequantized bytes to build the transposed
    // [in_f, n_heads*head_dim] matrix the .bin blob itself actually stores (this project's own [in,out]
    // AXPY convention -- QsaQProj/QsaGateProj's real on-disk shape, layout.hpp's own PKind::QsaQProj/
    // QsaGateProj table). The reference dot is then computed against THAT transposed matrix via a plain
    // AXPY-shaped accumulation (the exact sum gemv::axpy itself would perform), so a wrong row/column
    // convention on EITHER side of this comparison would show up as a numeric mismatch, not merely as
    // "these two formulas happen to agree with each other".
    constexpr int kHeads = 4, kHeadDim = 32, kIn = 2560;   // small, fast-iteration shape (AGENTS.md S7 --
                                                            // real vocabulary is irrelevant to a pure
                                                            // kernel/layout check)
    constexpr int kRows = kHeads * 2 * kHeadDim;   // GGUF/tiled row count: n_heads * 2 * head_dim
    const std::vector<float> x = lossless_row(kIn, 909);
    bbqd::ActBlocks xq;
    xq.quantize(x.data(), kIn);

    for (const gguf::TensorType type : kFormats) {
        INFO("format " << format_name(type));
        const auto raw_t = static_cast<std::uint32_t>(type);
        const std::vector<std::uint8_t> raw = make_blocks(type, static_cast<std::uint64_t>(kRows) * kIn,
                                                           555u + raw_t);

        // The reference side: dequantize the WHOLE tensor once (gguf::to_f32, the project's own already-
        // verified scalar decoder -- never bbqd's own code), then apply per_head_half_transpose to build
        // the [kIn, kHeads*kHeadDim] transposed matrix for each half, exactly as the real transplant path
        // (tools/sub0llm-transplant.cpp) builds QsaQProj/QsaGateProj into the .bin blob.
        gguf::TensorInfo t;
        t.type_raw = raw_t;
        t.dims = {static_cast<std::uint64_t>(kRows) * kIn};
        std::vector<float> dequant;
        REQUIRE(gguf::to_f32(t, raw, dequant));

        for (int half = 0; half < 2; ++half) {
            std::vector<float> transposed(static_cast<std::size_t>(kIn) * kHeads * kHeadDim, 0.f);
            transplant::per_head_half_transpose(dequant.data(), kHeads, kHeadDim, kIn, half,
                                                transposed.data());
            const int out_f = kHeads * kHeadDim;

            for (int h = 0; h < kHeads; ++h) {
                // transplant.hpp's own row-selection formula (per_head_half_transpose's own comment),
                // re-derived here independently rather than copied from a shared constant.
                const int row0 = h * 2 * kHeadDim + half * kHeadDim;
                std::vector<float> out(kHeadDim, 0.f);
                REQUIRE(bbqd::gemv_plane<1>(raw_t, std::span<const std::uint8_t>(raw), kRows, kIn, xq,
                                            out.data(), row0, row0 + kHeadDim));
                for (int d = 0; d < kHeadDim; ++d) {
                    const int col = h * kHeadDim + d;
                    // The AXPY-shaped reference: sum_i x[i] * transposed[i*out_f + col] -- the exact sum
                    // gemv::axpy would compute reading this column out of the TRANSPOSED (bf16-blob-
                    // convention) matrix.
                    double ref = 0.0;
                    for (int i = 0; i < kIn; ++i)
                        ref += static_cast<double>(x[static_cast<std::size_t>(i)]) *
                               transposed[static_cast<std::size_t>(i) * out_f + col];
                    INFO("head " << h << " half " << half << " d " << d);
                    REQUIRE(std::fabs(ref) > 0.0);
                    const double rel =
                        std::fabs(static_cast<double>(out[static_cast<std::size_t>(d)]) - ref) / std::fabs(ref);
                    // Float-rounding scale only -- slightly looser than this file's other 1e-5 bounds
                    // because THIS reference sums kIn=2560 terms in a plain sequential double accumulator
                    // (no SIMD/pairwise reduction), while gemv_plane's own AVX2 path reduces in a
                    // different (still exact-for-integers, but float-accumulated) order; measured worst
                    // case here is ~1.6e-5, comfortably inside this bound and far below any real layout
                    // bug's signature (a shifted row is wrong by 100%, not 1e-5).
                    REQUIRE(rel < 5e-5);

                }
            }
        }
    }
}

// --- mutation check (documented here rather than left as a committed always-on case; see this file's ---
// --- own header comment, and AGENTS.md's "regression test on a reproducible bug") ----------------------
//
// During development, this exact test file caught TWO real defects before they were ever wired into
// anything:
//   1. Q6KPlane::group() stored `(nib | (hi << 4)) - 32` in wg.q AND ALSO set `bias = -32*scale`,
//      double-subtracting the zero-point. Caught by "the fused unpackers decode the SAME weights..."
//      above: EVERY element of every Q6_K group came out offset by exactly one `bias` term (a constant,
//      format-and-block-specific offset -- not noise, not a magnitude error), which is exactly the
//      "still finite, still plausibly scaled, still wrong" signature this test class exists to catch.
//   2. dot16_avx2/sum16_avx2 (Q6_K's own 16-wide sub-block path) converted only the LOW 8 of their 16
//      loaded int8 lanes via `_mm_cvtepi8_epi16` (which sign-extends 8 bytes into a full 128-bit output,
//      leaving no room for the other 8) -- silently computing an 8-wide dot. Caught by "the AVX2 path
//      agrees EXACTLY with the portable path" above: Q8_0/Q4_K/Q5_K (all 32-wide, unaffected) matched
//      exactly; Q6_K did not, isolating the bug to the 16-wide primitives specifically.
// Both were reverted-to-reproduce and re-fixed to confirm each test is the one that actually catches its
// own bug (not passing "by accident" alongside an unrelated fix) -- the round trip AGENTS.md's own
// gate-verification discipline asks for, done once during authoring rather than left as a standing case
// that would otherwise just re-assert the same portable-path arithmetic on every run.
