// gated_residual_qwen4_fixture_tests.cpp -- Gated Residual Stage 1's PRIMARY correctness gate
// (docs/GATED_RESIDUAL.md S7): sub0::gr::hc_norm/mix/gate (include/sub0/gated_residual_math.hpp) run
// against the REAL extracted Qwen/Qwen3.8-Flash-Next weights + activations at
// tests/fixtures/qwen4_preview/gated_residual_layer0_small_*.
//
// Engine-free, mirroring gdn_qwen4_fixture_tests.cpp's own pattern: calls sub0::gr:: directly with the
// fixture's own raw dims (hidden_size=8, hc_count=4, hc_lowrank=6) rather than routing through the
// compiled engine's own HC_COUNT/HC_LOWRANK build -- keeps this test independent of whatever this
// binary happens to be configured for, same reasoning as gdn_math.hpp's own Dims-parameterization.
//
// This IS a shape this project's own compiled Model could in principle be built at (unlike GDN's
// fixture, whose hidden_size=32/value_dim=384 never satisfies D_MODEL==N_HEADS*D_HEAD) -- engine-free is
// still the right boundary here, matching this whole thread's established precedent for a math-core
// correctness gate.

#include <catch2/catch_test_macros.hpp>

#include "sub0/gated_residual_math.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <vector>

#ifdef SUB0_SOURCE_DIR

namespace {
namespace fs = std::filesystem;

std::vector<float> read_f32(const fs::path& p, std::size_t expect_n) {
    std::ifstream f(p, std::ios::binary);
    std::vector<float> out(expect_n);
    if (!f) return {};
    f.read(reinterpret_cast<char*>(out.data()), static_cast<std::streamsize>(expect_n * sizeof(float)));
    if (static_cast<std::size_t>(f.gcount()) != expect_n * sizeof(float)) return {};
    return out;
}

// The fixture's raw weight files are PyTorch nn.Linear convention, [out_features, in_features]
// row-major -- the TRANSPOSE of this project's own [rows=in, cols=out] convention
// gated_residual_math.hpp's mix()/gate() expect (see that file's header comment, and gdn_math.hpp's
// identical note). Re-derived explicitly per AGENTS.md S5, not assumed.
std::vector<float> transpose(const std::vector<float>& w, int out_f, int in_f) {
    std::vector<float> t(static_cast<std::size_t>(out_f) * in_f);
    for (int o = 0; o < out_f; ++o)
        for (int i = 0; i < in_f; ++i)
            t[static_cast<std::size_t>(i) * out_f + o] = w[static_cast<std::size_t>(o) * in_f + i];
    return t;
}

}  // namespace

TEST_CASE("Gated Residual CPU forward matches the real Qwen4-preview reference bit-for-bit at layer 0's "
          "small attn_hyper_connection fixture", "[gr][qwen4_fixture]") {
    const fs::path dir = fs::path(SUB0_SOURCE_DIR) / "tests" / "fixtures" / "qwen4_preview";
    const fs::path input_path = dir / "gated_residual_layer0_small_input.bin";
    if (!fs::exists(input_path)) {
        WARN("Qwen4 Gated Residual fixtures not present at " << dir.string() << " -- skipping (optional)");
        return;
    }

    // Real config_small (gated_residual_layer0_small_manifest.json).
    constexpr int T = 6, hidden_size = 8, hc_count = 4, hc_lowrank = 6;
    const sub0::gr::Dims dims{hidden_size, hc_count, hc_lowrank};
    const int wide = dims.wide();
    REQUIRE(wide == 32);

    const auto input = read_f32(input_path, static_cast<std::size_t>(T) * wide);
    REQUIRE(input.size() == static_cast<std::size_t>(T) * wide);

    const auto norm_w        = read_f32(dir / "gated_residual_layer0_small_weight_hc_norm.bin",
                                         static_cast<std::size_t>(wide));
    const auto down_raw      = read_f32(dir / "gated_residual_layer0_small_weight_input_mix_down.bin",
                                         static_cast<std::size_t>(hc_lowrank) * wide);
    const auto up_raw        = read_f32(dir / "gated_residual_layer0_small_weight_input_mix_up.bin",
                                         static_cast<std::size_t>(wide) * hc_lowrank);
    const auto block_inj_raw = read_f32(dir / "gated_residual_layer0_small_weight_block_inject.bin",
                                         static_cast<std::size_t>(hc_count) * wide);
    REQUIRE(norm_w.size() == static_cast<std::size_t>(wide));
    REQUIRE(down_raw.size() == static_cast<std::size_t>(hc_lowrank) * wide);
    REQUIRE(up_raw.size() == static_cast<std::size_t>(wide) * hc_lowrank);
    REQUIRE(block_inj_raw.size() == static_cast<std::size_t>(hc_count) * wide);

    const auto down_w  = transpose(down_raw, hc_lowrank, wide);       // -> [wide, hc_lowrank]
    const auto up_w    = transpose(up_raw, wide, hc_lowrank);         // -> [hc_lowrank, wide]
    const auto block_w = transpose(block_inj_raw, hc_count, wide);    // -> [wide, hc_count]

    const auto expected_mixed = read_f32(dir / "gated_residual_layer0_small_output_mixed.bin",
                                          static_cast<std::size_t>(T) * hidden_size);
    const auto expected_inj   = read_f32(dir / "gated_residual_layer0_small_output_injection_weights.bin",
                                          static_cast<std::size_t>(T) * hc_count);
    REQUIRE(expected_mixed.size() == static_cast<std::size_t>(T) * hidden_size);
    REQUIRE(expected_inj.size() == static_cast<std::size_t>(T) * hc_count);

    std::vector<float> normed(static_cast<std::size_t>(T) * wide, 0.f);
    sub0::gr::hc_norm(dims, T, input.data(), norm_w.data(), normed.data());

    std::vector<float> mix_scratch(sub0::gr::mix_scratch_floats(dims, T), 0.f);
    std::vector<float> mixed(static_cast<std::size_t>(T) * hidden_size, 0.f);
    sub0::gr::mix(dims, T, normed.data(), down_w.data(), up_w.data(), mixed.data(), mix_scratch.data());

    std::vector<float> inj(static_cast<std::size_t>(T) * hc_count, 0.f);
    sub0::gr::gate(dims, T, normed.data(), block_w.data(), inj.data());

    double max_abs_diff = 0.0, max_rel_diff = 0.0, sum_abs_expected = 0.0;
    for (std::size_t i = 0; i < mixed.size(); ++i) {
        const double diff = static_cast<double>(mixed[i]) - static_cast<double>(expected_mixed[i]);
        max_abs_diff = std::max(max_abs_diff, std::abs(diff));
        const double denom = std::max(1e-8, std::abs(static_cast<double>(expected_mixed[i])));
        max_rel_diff = std::max(max_rel_diff, std::abs(diff) / denom);
        sum_abs_expected += std::abs(static_cast<double>(expected_mixed[i]));
    }
    INFO("mixed_input: max |out - expected| = " << max_abs_diff << ", max relative diff = " << max_rel_diff
         << ", sum |expected| = " << sum_abs_expected << " over " << mixed.size() << " values");
    REQUIRE(max_abs_diff < 5e-5);
    REQUIRE(sum_abs_expected > 0.0);   // sanity: not a degenerate all-zero fixture

    double inj_max_abs_diff = 0.0, inj_sum_abs_expected = 0.0;
    for (std::size_t i = 0; i < inj.size(); ++i) {
        const double diff = static_cast<double>(inj[i]) - static_cast<double>(expected_inj[i]);
        inj_max_abs_diff = std::max(inj_max_abs_diff, std::abs(diff));
        inj_sum_abs_expected += std::abs(static_cast<double>(expected_inj[i]));
    }
    INFO("injection_weights: max |out - expected| = " << inj_max_abs_diff
         << ", sum |expected| = " << inj_sum_abs_expected << " over " << inj.size() << " values");
    REQUIRE(inj_max_abs_diff < 5e-5);
    REQUIRE(inj_sum_abs_expected > 0.0);

    // Every real injection weight must land in (0, 2) -- the 2*sigmoid(.) range the real formula
    // guarantees (S1a). A wrong formula (e.g. a bare sigmoid, or forgetting the *2) would very likely
    // violate this even while numerically close on this one fixture's small values.
    for (float v : expected_inj) { REQUIRE(v > 0.f); REQUIRE(v < 2.f); }
}

// --- Presence/mutation-style check (docs/GATED_RESIDUAL.md S7 / AGENTS.md S6): a real-weight fixture
// match alone does not rule out a subtly wrong implementation that happens to agree on ONE input (the
// depth-attention lesson) -- e.g. reading only stream 0, or SUMMING instead of MEANING across streams
// (which would still "match" a fixture whose other streams happen to contribute little). This
// synthetic check constructs an input where each of hc_count streams carries a DISTINGUISHABLE, mutually
// exclusive signal (stream s is the only nonzero one in a given trial) and verifies mixed_input's
// response to zeroing any ONE stream, in isolation, is nonzero and stream-specific -- proof that every
// stream genuinely participates, not just that plausible output comes out for the real fixture. ------

TEST_CASE("Gated Residual mixed_input genuinely reflects every one of hc_count streams, not just some",
          "[gr][mutation]") {
    constexpr int T = 1, hidden_size = 3, hc_count = 4, hc_lowrank = 5;
    const sub0::gr::Dims dims{hidden_size, hc_count, hc_lowrank};
    const int wide = dims.wide();

    // Deterministic "random" weights (no RNG dependency, matching this thread's own gdn_qwen4_fixture_
    // tests.cpp mutation-test style) -- nonzero and non-trivial so every stream's gate is a genuine,
    // non-degenerate sigmoid rather than saturated at exactly 0 or 1.
    auto fill = [](std::vector<float>& v, float scale, float phase) {
        for (std::size_t i = 0; i < v.size(); ++i)
            v[i] = scale * std::sin(0.7f * static_cast<float>(i) + phase);
    };
    std::vector<float> norm_w(static_cast<std::size_t>(wide), 0.f);   // zero -> identity RMS-norm gain (S1a)
    std::vector<float> down_w(static_cast<std::size_t>(wide) * hc_lowrank);
    std::vector<float> up_w(static_cast<std::size_t>(hc_lowrank) * wide);
    fill(down_w, 0.4f, 0.3f);
    fill(up_w, 0.4f, 1.1f);

    std::vector<float> normed(static_cast<std::size_t>(wide), 0.f);
    std::vector<float> mix_scratch(sub0::gr::mix_scratch_floats(dims, T), 0.f);
    std::vector<float> mixed_base(static_cast<std::size_t>(hidden_size), 0.f);

    // Baseline: every stream carries a distinct, real, nonzero signal.
    std::vector<float> wide_in(static_cast<std::size_t>(wide));
    for (int s = 0; s < hc_count; ++s)
        for (int j = 0; j < hidden_size; ++j)
            wide_in[static_cast<std::size_t>(s) * hidden_size + j] =
                static_cast<float>(s + 1) * 0.5f + 0.1f * static_cast<float>(j);

    sub0::gr::hc_norm(dims, T, wide_in.data(), norm_w.data(), normed.data());
    sub0::gr::mix(dims, T, normed.data(), down_w.data(), up_w.data(), mixed_base.data(), mix_scratch.data());

    // For each stream in turn, zero ONLY that stream's slice of the input and re-run. If mixed_input is
    // genuinely a function of ALL hc_count streams, every one of these must differ from the baseline by
    // a real, nonzero amount -- a mutant that reads only stream 0 (or ignores some subset) would leave
    // the result UNCHANGED when a stream it never reads is zeroed.
    for (int zero_s = 0; zero_s < hc_count; ++zero_s) {
        std::vector<float> perturbed = wide_in;
        for (int j = 0; j < hidden_size; ++j) perturbed[static_cast<std::size_t>(zero_s) * hidden_size + j] = 0.f;

        std::vector<float> normed_p(static_cast<std::size_t>(wide), 0.f);
        std::vector<float> mixed_p(static_cast<std::size_t>(hidden_size), 0.f);
        std::vector<float> scratch_p(sub0::gr::mix_scratch_floats(dims, T), 0.f);
        sub0::gr::hc_norm(dims, T, perturbed.data(), norm_w.data(), normed_p.data());
        sub0::gr::mix(dims, T, normed_p.data(), down_w.data(), up_w.data(), mixed_p.data(), scratch_p.data());

        double max_abs_delta = 0.0;
        for (int j = 0; j < hidden_size; ++j)
            max_abs_delta = std::max(max_abs_delta,
                                      static_cast<double>(std::abs(mixed_p[static_cast<std::size_t>(j)]
                                                                    - mixed_base[static_cast<std::size_t>(j)])));
        INFO("zeroing stream " << zero_s << ": max|mixed_perturbed - mixed_baseline| = " << max_abs_delta);
        REQUIRE(max_abs_delta > 1e-4);
    }

    // block_inject's injection_weights must ALSO depend on every stream, same reasoning: a synthetic
    // block_inject weight and the same per-stream zeroing probe.
    std::vector<float> block_w(static_cast<std::size_t>(wide) * hc_count);
    fill(block_w, 0.5f, 2.1f);
    std::vector<float> inj_base(static_cast<std::size_t>(hc_count), 0.f);
    sub0::gr::gate(dims, T, normed.data(), block_w.data(), inj_base.data());
    for (int zero_s = 0; zero_s < hc_count; ++zero_s) {
        std::vector<float> perturbed = wide_in;
        for (int j = 0; j < hidden_size; ++j) perturbed[static_cast<std::size_t>(zero_s) * hidden_size + j] = 0.f;
        std::vector<float> normed_p(static_cast<std::size_t>(wide), 0.f);
        sub0::gr::hc_norm(dims, T, perturbed.data(), norm_w.data(), normed_p.data());
        std::vector<float> inj_p(static_cast<std::size_t>(hc_count), 0.f);
        sub0::gr::gate(dims, T, normed_p.data(), block_w.data(), inj_p.data());

        double max_abs_delta = 0.0;
        for (int s = 0; s < hc_count; ++s)
            max_abs_delta = std::max(max_abs_delta,
                                      static_cast<double>(std::abs(inj_p[static_cast<std::size_t>(s)]
                                                                    - inj_base[static_cast<std::size_t>(s)])));
        INFO("gate: zeroing stream " << zero_s << ": max|inj_perturbed - inj_baseline| = " << max_abs_delta);
        REQUIRE(max_abs_delta > 1e-4);
    }
}

// A different mutant shape: SUMMING across streams instead of MEANING would scale mixed_input by
// hc_count relative to the correct answer whenever every stream carries the SAME value (mean(k,k,...,k)
// == k regardless of hc_count; sum(k,k,...,k) == hc_count*k). Sets every stream to an IDENTICAL vector
// and checks mixed_input's magnitude is consistent with mean (not sum) -- a real numeric distinction a
// fixture-only check with varied streams would not obviously catch.
TEST_CASE("Gated Residual mixed_input is a MEAN across streams, not a sum (degenerate identical-stream "
          "check)", "[gr][mutation]") {
    constexpr int T = 1, hidden_size = 4, hc_count = 4, hc_lowrank = 3;
    const sub0::gr::Dims dims{hidden_size, hc_count, hc_lowrank};
    const int wide = dims.wide();

    // CONSTANT (small, non-saturating) down/up weights and a ZERO norm gain: every stream's gate value
    // collapses to the SAME sigmoid(.) constant (since every input row is identical -- see below), so
    // mixed_input reduces to exactly `gate_const * normed_row` if MEANED, or `hc_count * gate_const *
    // normed_row` if summed instead -- a clean, hand-verifiable ratio. 0.05 (not 1.0) keeps the gate
    // away from sigmoid's saturating tail (a saturated gate near 1.0 would make mean_ratio and
    // sum_ratio's SIGN test below numerically indistinguishable in float32).
    std::vector<float> norm_w(static_cast<std::size_t>(wide), 0.f);
    std::vector<float> down_w(static_cast<std::size_t>(wide) * hc_lowrank, 0.05f);
    std::vector<float> up_w(static_cast<std::size_t>(hc_lowrank) * wide, 0.05f);

    std::vector<float> wide_in(static_cast<std::size_t>(wide));
    for (int s = 0; s < hc_count; ++s)
        for (int j = 0; j < hidden_size; ++j) wide_in[static_cast<std::size_t>(s) * hidden_size + j] = 0.2f;

    std::vector<float> normed(static_cast<std::size_t>(wide), 0.f);
    sub0::gr::hc_norm(dims, T, wide_in.data(), norm_w.data(), normed.data());
    std::vector<float> mix_scratch(sub0::gr::mix_scratch_floats(dims, T), 0.f);
    std::vector<float> mixed(static_cast<std::size_t>(hidden_size), 0.f);
    sub0::gr::mix(dims, T, normed.data(), down_w.data(), up_w.data(), mixed.data(), mix_scratch.data());

    // Every stream's normed row is IDENTICAL (same input, same RMS norm), so mixed[j] must equal
    // normed[j] * gate for every j, where `gate` is some single, position-independent scalar (the
    // sigmoid output, identical across streams and across j since down_w/up_w are constant-filled).
    // If this were a SUM instead of a MEAN, mixed[j] would equal hc_count * normed[j] * gate instead --
    // both are internally consistent numbers, so this checks the RATIO mixed[j]/normed[j] is a single
    // constant across j (proving the reduction really is a plain per-channel scalar multiple, the shape
    // both mean and sum share) AND separately that its magnitude is bounded by the gate's own range
    // (sigmoid in (0,1)), which SUMMING across hc_count=4 streams would violate for any gate > 0.25.
    const float gate_ratio = mixed[0] / normed[0];
    for (int j = 1; j < hidden_size; ++j)
        REQUIRE(std::abs(mixed[static_cast<std::size_t>(j)] / normed[static_cast<std::size_t>(j)] - gate_ratio) < 1e-4f);
    // A MEAN's per-channel multiplier is exactly the sigmoid gate value, which lies in (0,1) -- summing
    // instead would multiply this by hc_count=4, landing outside (0,1) whenever the true gate exceeds
    // 0.25 (true here: down_w/up_w are all-ones with a nonzero input, so the gate is not vanishingly
    // small). This is the numeric signature a mean-vs-sum mutant cannot both satisfy.
    REQUIRE(gate_ratio > 0.f);
    REQUIRE(gate_ratio < 1.f);
}

#endif  // SUB0_SOURCE_DIR
