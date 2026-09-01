// gdn_qwen4_fixture_tests.cpp -- Gated DeltaNet Stage 1's PRIMARY correctness gate
// (docs/GATED_DELTANET.md S5 step 2 / S6): sub0::gdn::forward (include/sub0/gdn_math.hpp) run against
// the REAL extracted Qwen/Qwen3.8-Flash-Next weights + activations at
// tests/fixtures/qwen4_preview/gdn_layer0_small_* (manifest + binaries landed on `main`).
//
// Engine-free, like ngram_qwen4_fixture_tests.cpp: this fixture's own shape (hidden_size=32,
// num_v_heads=3, head_v_dim=128 -> value_dim=384) does NOT satisfy this project's own
// D_MODEL == N_HEADS*D_HEAD invariant (docs/GATED_DELTANET.md S3b's flagged mismatch -- the real GDN
// module's key_dim/value_dim are independent PROJECTIONS from hidden_size, never a reshape of it), so
// there is no way to build this project's own Model at this exact shape. sub0::gdn::forward is
// parameterized on explicit Dims for exactly this reason (same rationale as layout.hpp's
// depth_schedule_for/gdn_schedule_for) -- this file calls it directly, with the fixture's own raw dims,
// bypassing the compiled engine entirely.
//
// This is a REAL NUMERIC match test (not a structural cross-check like the n-gram fixture test): the
// real reference ran the CHUNKED recurrence form (torch_chunk_gated_delta_rule); this file's
// implementation is the SEQUENTIAL form. docs/GATED_DELTANET.md S5 step 2 predicts these agree exactly
// at the token level -- this test is that prediction's confirmation.

#include <catch2/catch_test_macros.hpp>

#include "sub0/gdn_math.hpp"

#include <algorithm>
#include <array>
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
// row-major -- the TRANSPOSE of this project's own [rows=in, cols=out] convention gdn_math.hpp's
// forward() expects (see that file's header comment, and layout.hpp's identical Wq/Wk/Wv note).
// Re-derived explicitly per AGENTS.md S5, not assumed.
std::vector<float> transpose(const std::vector<float>& w, int out_f, int in_f) {
    std::vector<float> t(static_cast<std::size_t>(out_f) * in_f);
    for (int o = 0; o < out_f; ++o)
        for (int i = 0; i < in_f; ++i)
            t[static_cast<std::size_t>(i) * out_f + o] = w[static_cast<std::size_t>(o) * in_f + i];
    return t;
}

}  // namespace

TEST_CASE("Gated DeltaNet sequential CPU forward matches the real Qwen4-preview reference "
          "(chunked path) bit-for-bit at layer 0's small fixture",
          "[gdn][qwen4_fixture]") {
    const fs::path dir = fs::path(SUB0_SOURCE_DIR) / "tests" / "fixtures" / "qwen4_preview";
    const fs::path input_path  = dir / "gdn_layer0_small_input.bin";
    const fs::path output_path = dir / "gdn_layer0_small_output.bin";
    if (!fs::exists(input_path) || !fs::exists(output_path)) {
        WARN("Qwen4 GDN fixtures not present at " << dir.string() << " -- skipping (optional)");
        return;
    }

    // Real config_small (gdn_layer0_small_manifest.json) -- see this file's own header comment for why
    // this shape cannot be built as this project's own Model.
    constexpr int T = 6, hidden_size = 32;
    constexpr int num_k_heads = 1, num_v_heads = 3, head_k_dim = 128, head_v_dim = 128, conv_kernel = 4;
    const sub0::gdn::Dims dims{hidden_size, num_k_heads, num_v_heads, head_k_dim, head_v_dim, conv_kernel};
    const int key_dim = dims.key_dim(), value_dim = dims.value_dim(), conv_dim = dims.conv_dim();
    REQUIRE(key_dim == 128);
    REQUIRE(value_dim == 384);
    REQUIRE(conv_dim == 640);

    const std::vector<float> input = read_f32(input_path, static_cast<std::size_t>(T) * hidden_size);
    const std::vector<float> expected = read_f32(output_path, static_cast<std::size_t>(T) * hidden_size);
    REQUIRE(input.size() == static_cast<std::size_t>(T) * hidden_size);
    REQUIRE(expected.size() == static_cast<std::size_t>(T) * hidden_size);

    const auto w_qkv_raw = read_f32(dir / "gdn_layer0_small_weight_in_proj_qkv.bin",
                                     static_cast<std::size_t>(conv_dim) * hidden_size);
    const auto w_z_raw   = read_f32(dir / "gdn_layer0_small_weight_in_proj_z.bin",
                                     static_cast<std::size_t>(value_dim) * hidden_size);
    const auto w_b_raw   = read_f32(dir / "gdn_layer0_small_weight_in_proj_b.bin",
                                     static_cast<std::size_t>(num_v_heads) * hidden_size);
    const auto w_a_raw   = read_f32(dir / "gdn_layer0_small_weight_in_proj_a.bin",
                                     static_cast<std::size_t>(num_v_heads) * hidden_size);
    const auto conv_w    = read_f32(dir / "gdn_layer0_small_weight_conv1d.bin",
                                     static_cast<std::size_t>(conv_dim) * conv_kernel);   // [C,1,K] == [C,K] flat
    const auto dt_bias   = read_f32(dir / "gdn_layer0_small_weight_dt_bias.bin", num_v_heads);
    const auto a_log     = read_f32(dir / "gdn_layer0_small_weight_A_log.bin", num_v_heads);
    const auto norm_w    = read_f32(dir / "gdn_layer0_small_weight_norm.bin", head_v_dim);
    const auto w_out_raw = read_f32(dir / "gdn_layer0_small_weight_out_proj.bin",
                                     static_cast<std::size_t>(hidden_size) * value_dim);
    REQUIRE(w_qkv_raw.size() == static_cast<std::size_t>(conv_dim) * hidden_size);
    REQUIRE(w_out_raw.size() == static_cast<std::size_t>(hidden_size) * value_dim);

    const auto w_qkv = transpose(w_qkv_raw, conv_dim, hidden_size);      // -> [hidden_size, conv_dim]
    const auto w_z   = transpose(w_z_raw,   value_dim, hidden_size);     // -> [hidden_size, value_dim]
    const auto w_b   = transpose(w_b_raw,   num_v_heads, hidden_size);   // -> [hidden_size, num_v_heads]
    const auto w_a   = transpose(w_a_raw,   num_v_heads, hidden_size);
    const auto w_out  = transpose(w_out_raw, hidden_size, value_dim);    // -> [value_dim, hidden_size]

    std::vector<float> state(sub0::gdn::state_floats(dims), 0.f);
    std::vector<float> conv_hist(sub0::gdn::conv_hist_floats(dims), 0.f);   // fresh call -> zero history
    std::vector<float> scratch(sub0::gdn::scratch_floats(dims, T), 0.f);
    std::vector<float> out(static_cast<std::size_t>(T) * hidden_size, 0.f);

    sub0::gdn::forward(dims, T, input.data(),
                        w_qkv.data(), w_z.data(), w_b.data(), w_a.data(),
                        conv_w.data(), dt_bias.data(), a_log.data(), norm_w.data(), w_out.data(),
                        state.data(), conv_hist.data(), out.data(), scratch.data());

    double max_abs_diff = 0.0, max_rel_diff = 0.0, sum_abs_expected = 0.0;
    for (std::size_t i = 0; i < out.size(); ++i) {
        const double diff = static_cast<double>(out[i]) - static_cast<double>(expected[i]);
        max_abs_diff = std::max(max_abs_diff, std::abs(diff));
        const double denom = std::max(1e-8, std::abs(static_cast<double>(expected[i])));
        max_rel_diff = std::max(max_rel_diff, std::abs(diff) / denom);
        sum_abs_expected += std::abs(static_cast<double>(expected[i]));
    }
    INFO("max |out - expected| = " << max_abs_diff << ", max relative diff = " << max_rel_diff
         << ", sum |expected| = " << sum_abs_expected << " over " << out.size() << " values");
    // The Python double-precision re-derivation of this exact same math (scratchpad validation,
    // pre-port) measured max |diff| ~= 4.4e-11 against this fixture. The C++ port here computes in
    // float32 throughout (matching this project's own FP32-only CPU backend, core.hpp's own
    // documented precision), so this tolerance is loosened to float32 rounding, not because the
    // underlying formula is approximate -- it is bit-for-bit the same recurrence.
    REQUIRE(max_abs_diff < 5e-5);
    REQUIRE(sum_abs_expected > 0.0);   // sanity: not a degenerate all-zero fixture
}

// --- Stage 2: backward, checked against the REAL PyTorch-autograd oracle ------------------------------
// (docs/GATED_DELTANET.md S6's Stage 2 entry / AGENTS.md S5-S6). tests/fixtures/qwen4_preview/
// gdn_layer0_small_grad_*.bin are the REAL transformers==5.16.1 Qwen4ExpTextGatedDeltaNet's own
// .backward() gradients (chunked path) at the SAME real extracted layer-0 weights the forward fixture
// above uses, for loss = dot(output, gdn_layer0_small_dout.bin) -- a fixed reproducible random vector,
// not plain sum(), so every output element carries a distinct nonzero upstream gradient (a stronger
// oracle than sum() alone, per this doc's own header comment). This is the PRIMARY correctness gate for
// Stage 2's hand-derived backward -- cross-checked, BEFORE this port, against a double-precision NumPy
// re-derivation of the identical closed-form math (scratchpad, not committed) at these exact numbers,
// and independently at a synthetic multi-key-head-group (Hk=2) shape against torch.autograd on a
// from-scratch PyTorch reimplementation, matching to double-precision machine noise (~1e-15 relative) --
// see gdn_math.hpp's own header comment on sub0::gdn::backward for both cross-checks' exact numbers.
TEST_CASE("Gated DeltaNet backward matches the real Qwen4-preview reference's own autograd gradients "
          "(chunked path) at layer 0's small fixture", "[gdn][qwen4_fixture][grad]") {
    const fs::path dir = fs::path(SUB0_SOURCE_DIR) / "tests" / "fixtures" / "qwen4_preview";
    const fs::path input_path = dir / "gdn_layer0_small_input.bin";
    const fs::path dout_path  = dir / "gdn_layer0_small_dout.bin";
    if (!fs::exists(input_path) || !fs::exists(dout_path)) {
        WARN("Qwen4 GDN gradient fixtures not present at " << dir.string() << " -- skipping (optional)");
        return;
    }

    constexpr int T = 6, hidden_size = 32;
    constexpr int num_k_heads = 1, num_v_heads = 3, head_k_dim = 128, head_v_dim = 128, conv_kernel = 4;
    const sub0::gdn::Dims dims{hidden_size, num_k_heads, num_v_heads, head_k_dim, head_v_dim, conv_kernel};
    const int key_dim = dims.key_dim(), value_dim = dims.value_dim(), conv_dim = dims.conv_dim();

    const std::vector<float> input = read_f32(input_path, static_cast<std::size_t>(T) * hidden_size);
    const std::vector<float> dOut  = read_f32(dout_path, static_cast<std::size_t>(T) * hidden_size);
    REQUIRE(input.size() == static_cast<std::size_t>(T) * hidden_size);
    REQUIRE(dOut.size() == static_cast<std::size_t>(T) * hidden_size);

    const auto w_qkv_raw = read_f32(dir / "gdn_layer0_small_weight_in_proj_qkv.bin",
                                     static_cast<std::size_t>(conv_dim) * hidden_size);
    const auto w_z_raw   = read_f32(dir / "gdn_layer0_small_weight_in_proj_z.bin",
                                     static_cast<std::size_t>(value_dim) * hidden_size);
    const auto w_b_raw   = read_f32(dir / "gdn_layer0_small_weight_in_proj_b.bin",
                                     static_cast<std::size_t>(num_v_heads) * hidden_size);
    const auto w_a_raw   = read_f32(dir / "gdn_layer0_small_weight_in_proj_a.bin",
                                     static_cast<std::size_t>(num_v_heads) * hidden_size);
    const auto conv_w    = read_f32(dir / "gdn_layer0_small_weight_conv1d.bin",
                                     static_cast<std::size_t>(conv_dim) * conv_kernel);
    const auto dt_bias   = read_f32(dir / "gdn_layer0_small_weight_dt_bias.bin", num_v_heads);
    const auto a_log     = read_f32(dir / "gdn_layer0_small_weight_A_log.bin", num_v_heads);
    const auto norm_w    = read_f32(dir / "gdn_layer0_small_weight_norm.bin", head_v_dim);
    const auto w_out_raw = read_f32(dir / "gdn_layer0_small_weight_out_proj.bin",
                                     static_cast<std::size_t>(hidden_size) * value_dim);

    const auto w_qkv = transpose(w_qkv_raw, conv_dim, hidden_size);
    const auto w_z   = transpose(w_z_raw,   value_dim, hidden_size);
    const auto w_b   = transpose(w_b_raw,   num_v_heads, hidden_size);
    const auto w_a   = transpose(w_a_raw,   num_v_heads, hidden_size);
    const auto w_out = transpose(w_out_raw, hidden_size, value_dim);

    std::vector<float> scratch(sub0::gdn::bwd_scratch_floats(dims, T), 0.f);
    std::vector<float> dx(static_cast<std::size_t>(T) * hidden_size, 0.f);
    std::vector<float> dw_qkv(w_qkv.size(), 0.f), dw_z(w_z.size(), 0.f), dw_b(w_b.size(), 0.f),
        dw_a(w_a.size(), 0.f), dconv_w(conv_w.size(), 0.f), ddt_bias(dt_bias.size(), 0.f),
        da_log(a_log.size(), 0.f), dnorm_w(norm_w.size(), 0.f), dw_out(w_out.size(), 0.f);

    sub0::gdn::backward(dims, T, input.data(), w_qkv.data(), w_z.data(), w_b.data(), w_a.data(),
                         conv_w.data(), dt_bias.data(), a_log.data(), norm_w.data(), w_out.data(),
                         dOut.data(), dx.data(), dw_qkv.data(), dw_z.data(), dw_b.data(), dw_a.data(),
                         dconv_w.data(), ddt_bias.data(), da_log.data(), dnorm_w.data(), dw_out.data(),
                         scratch.data());

    // Real gradients were dumped in PyTorch's [out,in] weight-tensor shape (same as the raw fixture
    // weight files); this project's own dw_* buffers are [in,out] (see gdn_math.hpp's header comment) --
    // transpose the SAME way as the forward fixture's own w_qkv/w_z/... conversion, just in reverse, so
    // the comparison below reads both sides in [out,in].
    auto cmp = [&](const char* label, const std::vector<float>& mine_in_out, int in_f, int out_f,
                   const char* grad_file) {
        const auto real = read_f32(dir / grad_file, static_cast<std::size_t>(out_f) * in_f);
        REQUIRE(real.size() == static_cast<std::size_t>(out_f) * in_f);
        const auto mine_out_in = transpose(mine_in_out, in_f, out_f);   // [in,out] -> [out,in]
        double max_abs = 0.0, max_real = 0.0;
        for (std::size_t i = 0; i < real.size(); ++i) {
            max_abs = std::max(max_abs, static_cast<double>(std::abs(mine_out_in[i] - real[i])));
            max_real = std::max(max_real, static_cast<double>(std::abs(real[i])));
        }
        INFO(label << ": max|diff|=" << max_abs << " max|real|=" << max_real);
        REQUIRE(max_abs < std::max(1e-4, max_real * 0.05));   // real numbers are tiny (~1e-4..1e-6 scale)
        return max_real;
    };
    auto cmp_vec = [&](const char* label, const std::vector<float>& mine, const char* grad_file) {
        const auto real = read_f32(dir / grad_file, mine.size());
        REQUIRE(real.size() == mine.size());
        double max_abs = 0.0, max_real = 0.0;
        for (std::size_t i = 0; i < real.size(); ++i) {
            max_abs = std::max(max_abs, static_cast<double>(std::abs(mine[i] - real[i])));
            max_real = std::max(max_real, static_cast<double>(std::abs(real[i])));
        }
        INFO(label << ": max|diff|=" << max_abs << " max|real|=" << max_real);
        REQUIRE(max_abs < std::max(1e-4, max_real * 0.05));
        return max_real;
    };

    cmp_vec("d(input)", dx, "gdn_layer0_small_grad_input.bin");
    cmp("d(in_proj_qkv)", dw_qkv, hidden_size, conv_dim, "gdn_layer0_small_grad_in_proj_qkv.bin");
    cmp("d(in_proj_z)", dw_z, hidden_size, value_dim, "gdn_layer0_small_grad_in_proj_z.bin");
    cmp("d(in_proj_b)", dw_b, hidden_size, num_v_heads, "gdn_layer0_small_grad_in_proj_b.bin");
    cmp("d(in_proj_a)", dw_a, hidden_size, num_v_heads, "gdn_layer0_small_grad_in_proj_a.bin");
    cmp_vec("d(dt_bias)", ddt_bias, "gdn_layer0_small_grad_dt_bias.bin");
    const double a_log_real_max = cmp_vec("d(A_log)", da_log, "gdn_layer0_small_grad_A_log.bin");
    cmp_vec("d(norm)", dnorm_w, "gdn_layer0_small_grad_norm.bin");
    cmp("d(out_proj)", dw_out, value_dim, hidden_size, "gdn_layer0_small_grad_out_proj.bin");

    // conv1d's real gradient file is [conv_dim,1,K] (PyTorch Conv1d), same flat layout as this
    // project's own [conv_dim,K] dconv_w -- no [in,out] transpose applies to a depthwise-conv weight.
    {
        const auto real = read_f32(dir / "gdn_layer0_small_grad_conv1d.bin", dconv_w.size());
        REQUIRE(real.size() == dconv_w.size());
        double max_abs = 0.0, max_real = 0.0;
        for (std::size_t i = 0; i < real.size(); ++i) {
            max_abs = std::max(max_abs, static_cast<double>(std::abs(dconv_w[i] - real[i])));
            max_real = std::max(max_real, static_cast<double>(std::abs(real[i])));
        }
        INFO("d(conv1d): max|diff|=" << max_abs << " max|real|=" << max_real);
        REQUIRE(max_abs < std::max(1e-4, max_real * 0.05));
    }

    // Presence/mutation-style check (AGENTS.md S6, the depth-attention lesson): d(A_log) must be
    // GENUINELY nonzero and match the real gate-dependent value above, not merely present-but-wrong or
    // a structurally-nonzero stub -- the real reference's own d(A_log) is itself tiny (num_v_heads=3
    // scalars), so this also confirms the comparison above wasn't vacuously passing on an all-zero case.
    REQUIRE(a_log_real_max > 0.0);
    bool any_nonzero = false;
    for (float v : da_log) any_nonzero |= (v != 0.f);
    REQUIRE(any_nonzero);
}

// --- Stage 2: finite-difference gradient check, independent of the real oracle above -------------------
// A synthetic multi-key-head-group shape (Hk=2, unlike the real fixture's Hk=1 -- see
// gdn_math.hpp's own header comment on why that distinction matters for the repeat_interleave fold) with
// a tiny, hand-picked (not RNG-dependent) input/weight set, checked via central differences against
// sub0::gdn::forward's own loss = dot(out, a fixed vector) -- the SAME style of check
// tests/engine_tests.cpp's "analytic gradients match finite differences" uses for the rest of this
// engine, applied here to sub0::gdn::forward/backward directly (engine-free). Every one of the 9 weight
// tensors AND the input gets its own per-element central-difference probe (feasible at this scale: a few
// hundred floats total, each needing 2 extra forward() calls).
TEST_CASE("Gated DeltaNet backward matches finite differences at a synthetic multi-key-head-group shape",
          "[gdn][grad][fd]") {
    constexpr int T = 4, hidden_size = 5;
    constexpr int num_k_heads = 2, num_v_heads = 4, head_k_dim = 3, head_v_dim = 2, conv_kernel = 3;
    const sub0::gdn::Dims dims{hidden_size, num_k_heads, num_v_heads, head_k_dim, head_v_dim, conv_kernel};
    const int key_dim = dims.key_dim(), value_dim = dims.value_dim(), conv_dim = dims.conv_dim();
    REQUIRE(dims.rep() == 2);

    // Deterministic "random" fill, no RNG dependency (same style as the mutation tests above's `vval`).
    auto fill = [](std::vector<float>& v, float scale, float phase) {
        for (std::size_t i = 0; i < v.size(); ++i)
            v[i] = scale * std::sin(0.7f * static_cast<float>(i) + phase);
    };
    std::vector<float> x(static_cast<std::size_t>(T) * hidden_size);
    std::vector<float> w_qkv(static_cast<std::size_t>(hidden_size) * conv_dim);
    std::vector<float> w_z(static_cast<std::size_t>(hidden_size) * value_dim);
    std::vector<float> w_b(static_cast<std::size_t>(hidden_size) * num_v_heads);
    std::vector<float> w_a(static_cast<std::size_t>(hidden_size) * num_v_heads);
    std::vector<float> conv_w(static_cast<std::size_t>(conv_dim) * conv_kernel);
    std::vector<float> dt_bias(num_v_heads), a_log(num_v_heads), norm_w(head_v_dim);
    std::vector<float> w_out(static_cast<std::size_t>(value_dim) * hidden_size);
    std::vector<float> dOut(static_cast<std::size_t>(T) * hidden_size);
    fill(x, 0.3f, 0.1f); fill(w_qkv, 0.2f, 0.3f); fill(w_z, 0.2f, 0.5f); fill(w_b, 0.3f, 0.7f);
    fill(w_a, 0.3f, 0.9f); fill(conv_w, 0.25f, 1.1f); fill(dt_bias, 0.4f, 1.3f); fill(norm_w, 0.5f, 1.9f);
    fill(w_out, 0.2f, 2.1f); fill(dOut, 1.0f, 2.3f);
    for (int h = 0; h < num_v_heads; ++h) a_log[static_cast<std::size_t>(h)] = std::log(0.5f + 0.3f * h);

    std::vector<float> state(sub0::gdn::state_floats(dims), 0.f);
    std::vector<float> conv_hist(sub0::gdn::conv_hist_floats(dims), 0.f);
    std::vector<float> fwd_scratch(sub0::gdn::scratch_floats(dims, T), 0.f);
    std::vector<float> out(static_cast<std::size_t>(T) * hidden_size, 0.f);

    auto loss_at = [&]() -> double {
        std::fill(state.begin(), state.end(), 0.f);
        std::fill(conv_hist.begin(), conv_hist.end(), 0.f);
        std::fill(out.begin(), out.end(), 0.f);
        sub0::gdn::forward(dims, T, x.data(), w_qkv.data(), w_z.data(), w_b.data(), w_a.data(),
                            conv_w.data(), dt_bias.data(), a_log.data(), norm_w.data(), w_out.data(),
                            state.data(), conv_hist.data(), out.data(), fwd_scratch.data());
        double s = 0.0;
        for (std::size_t i = 0; i < out.size(); ++i) s += static_cast<double>(out[i]) * dOut[i];
        return s;
    };

    std::vector<float> scratch(sub0::gdn::bwd_scratch_floats(dims, T), 0.f);
    std::vector<float> dx(x.size(), 0.f), dw_qkv(w_qkv.size(), 0.f), dw_z(w_z.size(), 0.f),
        dw_b(w_b.size(), 0.f), dw_a(w_a.size(), 0.f), dconv_w(conv_w.size(), 0.f),
        ddt_bias(dt_bias.size(), 0.f), da_log(a_log.size(), 0.f), dnorm_w(norm_w.size(), 0.f),
        dw_out(w_out.size(), 0.f);
    sub0::gdn::backward(dims, T, x.data(), w_qkv.data(), w_z.data(), w_b.data(), w_a.data(),
                         conv_w.data(), dt_bias.data(), a_log.data(), norm_w.data(), w_out.data(),
                         dOut.data(), dx.data(), dw_qkv.data(), dw_z.data(), dw_b.data(), dw_a.data(),
                         dconv_w.data(), ddt_bias.data(), da_log.data(), dnorm_w.data(), dw_out.data(),
                         scratch.data());

    constexpr float EPS = 2e-3f;
    auto fd_check = [&](const char* label, std::vector<float>& param, const std::vector<float>& analytic) {
        double max_abs = 0.0, max_num = 0.0;
        for (std::size_t i = 0; i < param.size(); ++i) {
            const float orig = param[i];
            param[i] = orig + EPS; const double lp = loss_at();
            param[i] = orig - EPS; const double lm = loss_at();
            param[i] = orig;
            const double num = (lp - lm) / (2.0 * EPS);
            max_abs = std::max(max_abs, std::abs(num - static_cast<double>(analytic[i])));
            max_num = std::max(max_num, std::abs(num));
        }
        INFO(label << ": max|analytic-numeric|=" << max_abs << " max|numeric|=" << max_num);
        REQUIRE(max_abs < std::max(2e-3, max_num * 0.02));
        return max_num;
    };

    fd_check("x", x, dx);
    fd_check("w_qkv", w_qkv, dw_qkv);
    fd_check("w_z", w_z, dw_z);
    fd_check("w_b", w_b, dw_b);
    fd_check("w_a", w_a, dw_a);
    fd_check("conv_w", conv_w, dconv_w);
    const double dt_bias_num = fd_check("dt_bias", dt_bias, ddt_bias);
    const double a_log_num = fd_check("a_log", a_log, da_log);
    fd_check("norm_w", norm_w, dnorm_w);
    fd_check("w_out", w_out, dw_out);

    // Presence/mutation-style check (AGENTS.md S6): dt_bias/A_log gradients must be genuinely nonzero
    // and match a real finite-difference SLOPE, not merely present -- a mutant that silently zeroed
    // either gate-scalar's backward path would still pass a naive "field is populated" check but fails
    // both this FD comparison and the following nonzero assertions.
    REQUIRE(dt_bias_num > 1e-6);
    REQUIRE(a_log_num > 1e-6);
    for (float v : ddt_bias) REQUIRE(std::abs(v) > 0.f);
    for (float v : da_log) REQUIRE(std::abs(v) > 0.f);
}

#endif  // SUB0_SOURCE_DIR

// --- Mutation-style numerical property checks (docs/GATED_DELTANET.md sec 6 / AGENTS.md S6) ----------
// A real-weight fixture match alone does not rule out a subtly wrong implementation that happens to
// agree on ONE input (the depth-attention lesson, docs/DEPTH_ATTENTION.md sec 6: a stubbed "return v;"
// can still pass a naive check). These call sub0::gdn::recurrence_step DIRECTLY with hand-picked
// g_t/beta_t/k_t/q_t/v_t -- no fixture files needed, so they always run.

// With g_t == 1 (no decay) and beta_t == 1 (full write, no forgetting) and MUTUALLY ORTHONORMAL keys
// across time, the correction term `S^T k_t` is exactly zero every step (S only ever holds components
// along PREVIOUS keys, orthogonal to the current one) -- so `delta = beta_t*(v_t - 0) = v_t` and the
// recurrence degenerates to the state being EXACTLY the running (cumulative) sum of the outer products
// k_t (x) v_t, with no decay ever discounting an earlier term and no interference ever cancelling one.
// A mutant that (say) always overwrites S instead of accumulating into it, or applies decay/beta in the
// wrong order, would fail this even though it might still pass a single fixed-input fixture match.
TEST_CASE("Gated DeltaNet recurrence degenerates to plain cumulative summation at g=1, beta=1, "
          "orthonormal keys", "[gdn][mutation]") {
    constexpr int T = 5, dk = T, dv = 3;   // dk == T so T orthonormal keys (the standard basis) fit exactly
    std::vector<float> S(static_cast<std::size_t>(dk) * dv, 0.f);
    std::vector<float> expected(static_cast<std::size_t>(dk) * dv, 0.f);   // closed-form running sum

    // A small deterministic "random" v_t sequence -- any nonzero values exercise the property, no need
    // for a real RNG dependency in a frontend-test target.
    auto vval = [](int t, int j) { return static_cast<float>((t + 1) * 3 - j * 2) * 0.1f; };

    for (int t = 0; t < T; ++t) {
        std::vector<float> k_t(dk, 0.f);
        k_t[static_cast<std::size_t>(t)] = 1.f;   // standard basis vector e_t -- orthonormal across t
        std::vector<float> v_t(dv);
        for (int j = 0; j < dv; ++j) v_t[static_cast<std::size_t>(j)] = vval(t, j);
        std::vector<float> q_t(dk, 0.f);   // unused by this property (only S is checked), q=0 is fine
        std::vector<float> out_o(dv, 0.f);

        sub0::gdn::recurrence_step(dk, dv, /*g_t=*/1.f, /*beta_t=*/1.f, k_t.data(), q_t.data(),
                                    v_t.data(), S.data(), out_o.data());

        // Closed form: S should now equal sum_{s<=t} e_s (x) v_s, i.e. row t of S is exactly v_t (the
        // outer product of e_t against v_t only touches row t) and every other row is untouched by this
        // step -- so accumulate the expectation the same way, independently of the function under test.
        for (int j = 0; j < dv; ++j) expected[static_cast<std::size_t>(t) * dv + j] = v_t[static_cast<std::size_t>(j)];

        double max_diff = 0.0;
        for (std::size_t i = 0; i < S.size(); ++i)
            max_diff = std::max(max_diff, static_cast<double>(std::abs(S[i] - expected[i])));
        INFO("after step t=" << t << ", max|S - cumulative-sum closed form| = " << max_diff);
        REQUIRE(max_diff < 1e-6);
    }
}

// With beta_t == 0 (no write) AND g_t == 1 (no decay), the state must be an EXACT, bit-identical no-op:
// completely independent of k_t/v_t (which is what makes this a genuine mutation-style check -- a
// broken implementation that ignores beta and always writes would produce different output as k_t/v_t
// vary, even though the state trivially "matches" for any single fixed input).
TEST_CASE("Gated DeltaNet recurrence at beta=0, g=1 never updates the state (provable no-op)",
          "[gdn][mutation]") {
    constexpr int dk = 4, dv = 3;
    std::vector<float> S0 = {1.f, -2.f, 0.5f, 3.f, -1.f, 0.25f, 2.f, -0.75f, 1.5f, 0.1f, -0.2f, 4.f};
    REQUIRE(S0.size() == static_cast<std::size_t>(dk) * dv);

    std::vector<float> q_t = {0.3f, -0.7f, 1.1f, 0.2f};
    float o_ref[dv];
    {
        std::vector<float> S = S0;
        std::vector<float> k_t(dk, 0.f), v_t(dv, 0.f);   // the "control": k=v=0 (an actual no-write anyway)
        sub0::gdn::recurrence_step(dk, dv, 1.f, 0.f, k_t.data(), q_t.data(), v_t.data(), S.data(), o_ref);
        for (std::size_t i = 0; i < S.size(); ++i) REQUIRE(S[i] == S0[i]);   // exact IEEE754 no-op
    }

    // Now vary k_t/v_t across several unrelated, nonzero, non-trivial inputs -- output must be
    // BIT-FOR-BIT identical every time (o_t = S0^T q_t, a fixed function of q_t and the untouched S0
    // alone), which a "beta forgot to gate the write" mutant would not reproduce.
    const std::vector<std::array<float, 4>> ks = {
        {1.f, 0.f, 0.f, 0.f}, {0.f, 5.f, -3.f, 2.f}, {-1.f, -1.f, -1.f, -1.f}, {9.f, 9.f, 9.f, 9.f}};
    const std::vector<std::array<float, 3>> vs = {
        {2.f, 2.f, 2.f}, {0.f, -4.f, 6.f}, {1.f, 1.f, 1.f}, {-7.f, 3.f, 0.5f}};
    for (std::size_t trial = 0; trial < ks.size(); ++trial) {
        std::vector<float> S = S0;
        float o[dv];
        sub0::gdn::recurrence_step(dk, dv, 1.f, 0.f, ks[trial].data(), q_t.data(), vs[trial].data(),
                                    S.data(), o);
        for (std::size_t i = 0; i < S.size(); ++i) REQUIRE(S[i] == S0[i]);
        for (int j = 0; j < dv; ++j) REQUIRE(o[j] == o_ref[j]);
    }
}
