// backbone_quant_dot_tests.cpp -- O5 phase 1: unit tests for sub0::bbqd (include/sub0/backbone_quant_dot.hpp),
// the fused int8-activation x native-quant-weight dot product for the BACKBONE's four real quantized
// formats (Q8_0, Q4_K, Q5_K, Q6_K -- docs/BACKBONE_NATIVE_QUANT.md's own census of the real Qwen3.8-
// Flash-Next UD-IQ1_S shards). Structured after tests/moe_quant_tests.cpp's own B35 section, which this
// mirrors deliberately (same split into "is the weight decode exact" vs "how big is the activation
// quantization error", AGENTS.md S6/S9).
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
        std::vector<float> fused(kRows, 0.f);
        REQUIRE(bbqd::gemv_plane<false>(raw_t, std::span<const std::uint8_t>(raw), kRows, kN, xq,
                                        fused.data()));

        double worst_rel = 0.0;
        for (int row = 0; row < kRows; ++row) {
            const auto byte_off = static_cast<std::size_t>(
                (static_cast<std::uint64_t>(row) * kN / spec.elems) * spec.bytes);
            const double ref = reference_dot(
                type, std::span<const std::uint8_t>(raw).subspan(
                          byte_off, static_cast<std::size_t>(kN / spec.elems * spec.bytes)),
                kN, x);
            REQUIRE(std::isfinite(fused[static_cast<std::size_t>(row)]));
            REQUIRE(std::fabs(ref) > 0.0);
            worst_rel = std::max(worst_rel,
                                  std::fabs(fused[static_cast<std::size_t>(row)] - ref) / std::fabs(ref));
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
        REQUIRE(bbqd::gemv_plane<false>(raw_t, std::span<const std::uint8_t>(raw), kRows, kN, xq,
                                        portable.data()));
        REQUIRE(bbqd::gemv_plane<true>(raw_t, std::span<const std::uint8_t>(raw), kRows, kN, xq,
                                       avx2.data()));
        for (int r = 0; r < kRows; ++r) {
            INFO("row " << r);
            REQUIRE(portable[static_cast<std::size_t>(r)] == avx2[static_cast<std::size_t>(r)]);
        }
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
        REQUIRE(bbqd::gemv_plane<false>(raw_t, std::span<const std::uint8_t>(raw), kRows, kN, xq,
                                        whole.data()));

        // Split into three uneven ranges (not a clean divisor of kRows), each written into its own slot
        // of a shared output buffer -- exactly how a real multi-threaded caller would use this.
        std::vector<float> split(kRows, -99.f);
        const int cuts[4] = {0, 3, 5, kRows};
        for (int c = 0; c < 3; ++c) {
            const int lo = cuts[c], hi = cuts[c + 1];
            std::vector<float> part(static_cast<std::size_t>(hi - lo), -1.f);
            REQUIRE(bbqd::gemv_plane<false>(raw_t, std::span<const std::uint8_t>(raw), kRows, kN, xq,
                                            part.data(), lo, hi));
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
    REQUIRE_FALSE(bbqd::gemv_plane<false>(static_cast<std::uint32_t>(gguf::TensorType::IQ1_S),
                                          std::span<const std::uint8_t>(q8), 1, 256, xq, &out));
    REQUIRE(out == 1234.f);   // refused means untouched, not partially written

    // Correct format, but the span is shorter than the declared geometry needs.
    REQUIRE_FALSE(bbqd::gemv_plane<false>(static_cast<std::uint32_t>(gguf::TensorType::Q8_0),
                                          std::span<const std::uint8_t>(q8).first(10), 1, 256, xq, &out));
    REQUIRE(out == 1234.f);

    // A row range outside [0, n_rows] must also be refused, not clamped.
    REQUIRE_FALSE(bbqd::gemv_plane<false>(static_cast<std::uint32_t>(gguf::TensorType::Q8_0),
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
        REQUIRE(bbqd::gemv_plane<false>(pk.info.type_raw, std::span<const std::uint8_t>(raw), 1,
                                        row_elems, xq, &fused));
        const double ref = reference_dot(static_cast<gguf::TensorType>(pk.info.type_raw), raw, row_elems, x);
        REQUIRE(std::isfinite(fused));
        REQUIRE(std::fabs(ref) > 0.0);
        const double rel = std::fabs(static_cast<double>(fused) - ref) / std::fabs(ref);
        INFO("relative disagreement on real bytes = " << rel);
        REQUIRE(rel < 1e-5);

        // AVX2 agrees exactly on the real bytes too.
        float fused_avx2 = 0.f;
        REQUIRE(bbqd::gemv_plane<true>(pk.info.type_raw, std::span<const std::uint8_t>(raw), 1,
                                       row_elems, xq, &fused_avx2));
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
        REQUIRE(bbqd::gemv_plane<false>(pk.info.type_raw, std::span<const std::uint8_t>(raw), 1,
                                        row_elems, xqg, &fused_g));
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
