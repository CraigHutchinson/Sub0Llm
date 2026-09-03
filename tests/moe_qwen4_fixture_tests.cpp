// moe_qwen4_fixture_tests.cpp -- Mixture of Experts Stage 1's PRIMARY correctness gate (docs/MOE.md S7):
// sub0::moe::forward_row (include/sub0/moe_math.hpp) run against the REAL extracted
// Qwen/Qwen3.8-Flash-Next weights + activations at tests/fixtures/qwen4_preview/moe_layer0_small_*.
//
// Engine-free, mirroring gdn_qwen4_fixture_tests.cpp's/gated_residual_qwen4_fixture_tests.cpp's own
// pattern: calls sub0::moe:: directly with the fixture's own raw dims (hidden_size=16, d_ff=8,
// num_experts=512, num_experts_per_tok=10) rather than routing through the compiled engine's own
// NUM_EXPERTS/EXPERTS_PER_TOK build -- keeps this test independent of whatever this binary happens to be
// configured for. num_experts/num_experts_per_tok are kept at their REAL values (512/10, docs/MOE.md S8)
// -- the one axis this whole mechanism exists to test would degenerate at a small num_experts.

#include <catch2/catch_test_macros.hpp>

#include "sub0/moe_math.hpp"

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
// row-major -- the TRANSPOSE of this project's own [rows=in, cols=out] convention moe_math.hpp expects
// (see that file's header comment). Re-derived explicitly per AGENTS.md S5, not assumed.
std::vector<float> transpose(const std::vector<float>& w, int out_f, int in_f) {
    std::vector<float> t(static_cast<std::size_t>(out_f) * in_f);
    for (int o = 0; o < out_f; ++o)
        for (int i = 0; i < in_f; ++i)
            t[static_cast<std::size_t>(i) * out_f + o] = w[static_cast<std::size_t>(o) * in_f + i];
    return t;
}

}  // namespace

TEST_CASE("Mixture of Experts CPU forward matches the real Qwen4-preview reference at layer 0's small "
          "mlp fixture", "[moe][qwen4_fixture]") {
    const fs::path dir = fs::path(SUB0_SOURCE_DIR) / "tests" / "fixtures" / "qwen4_preview";
    const fs::path input_path = dir / "moe_layer0_small_input.bin";
    if (!fs::exists(input_path)) {
        WARN("Qwen4 MoE fixtures not present at " << dir.string() << " -- skipping (optional)");
        return;
    }

    // Real config_small (moe_layer0_small_manifest.json). num_experts/num_experts_per_tok kept at their
    // REAL values (512/10) -- only hidden_size/d_ff are sliced (docs/MOE.md S8).
    constexpr int hidden_size = 16, d_ff = 8, num_experts = 512, experts_per_tok = 10, kFetched = 12;
    const sub0::moe::Dims dims{hidden_size, d_ff, num_experts, experts_per_tok};

    const auto input = read_f32(input_path, static_cast<std::size_t>(hidden_size));
    REQUIRE(input.size() == static_cast<std::size_t>(hidden_size));

    const auto router_raw = read_f32(dir / "moe_layer0_small_weight_router.bin",
                                      static_cast<std::size_t>(num_experts) * hidden_size);
    REQUIRE(router_raw.size() == static_cast<std::size_t>(num_experts) * hidden_size);
    const auto router_w = transpose(router_raw, num_experts, hidden_size);   // -> [hidden_size, num_experts]

    const auto expert_ids_f = read_f32(dir / "moe_layer0_small_expert_ids.bin", kFetched);
    REQUIRE(expert_ids_f.size() == kFetched);
    std::vector<int> expert_ids(kFetched);
    for (int i = 0; i < kFetched; ++i) expert_ids[static_cast<std::size_t>(i)] = static_cast<int>(expert_ids_f[static_cast<std::size_t>(i)]);

    const auto gate_stack_raw = read_f32(dir / "moe_layer0_small_weight_experts_gate.bin",
                                          static_cast<std::size_t>(kFetched) * d_ff * hidden_size);
    const auto up_stack_raw   = read_f32(dir / "moe_layer0_small_weight_experts_up.bin",
                                          static_cast<std::size_t>(kFetched) * d_ff * hidden_size);
    const auto down_stack_raw = read_f32(dir / "moe_layer0_small_weight_experts_down.bin",
                                          static_cast<std::size_t>(kFetched) * hidden_size * d_ff);
    REQUIRE(gate_stack_raw.size() == static_cast<std::size_t>(kFetched) * d_ff * hidden_size);
    REQUIRE(up_stack_raw.size() == static_cast<std::size_t>(kFetched) * d_ff * hidden_size);
    REQUIRE(down_stack_raw.size() == static_cast<std::size_t>(kFetched) * hidden_size * d_ff);

    // Default every one of the 512 slots to a shared, all-zero buffer -- never dereferenced in practice
    // (the real router's own top-10 selection on this real weight only ever names the kFetched=12
    // fetched experts' own ids, verified against the real transformers module's own selection during
    // extraction, see the manifest) but a safe fallback rather than a crash if float32 reduction-order
    // differences ever moved a boundary selection by one expert.
    std::vector<float> zero_gate(static_cast<std::size_t>(hidden_size) * d_ff, 0.f);
    std::vector<float> zero_down(static_cast<std::size_t>(d_ff) * hidden_size, 0.f);
    std::vector<const float*> gate_w(static_cast<std::size_t>(num_experts), zero_gate.data());
    std::vector<const float*> up_w(static_cast<std::size_t>(num_experts), zero_gate.data());
    std::vector<const float*> down_w(static_cast<std::size_t>(num_experts), zero_down.data());

    std::vector<std::vector<float>> gate_t(static_cast<std::size_t>(kFetched)), up_t(static_cast<std::size_t>(kFetched)),
        down_t(static_cast<std::size_t>(kFetched));
    for (int k = 0; k < kFetched; ++k) {
        const std::size_t gu_off = static_cast<std::size_t>(k) * d_ff * hidden_size;
        const std::size_t dn_off = static_cast<std::size_t>(k) * hidden_size * d_ff;
        std::vector<float> gate_raw(gate_stack_raw.begin() + static_cast<std::ptrdiff_t>(gu_off),
                                     gate_stack_raw.begin() + static_cast<std::ptrdiff_t>(gu_off) + d_ff * hidden_size);
        std::vector<float> up_raw(up_stack_raw.begin() + static_cast<std::ptrdiff_t>(gu_off),
                                   up_stack_raw.begin() + static_cast<std::ptrdiff_t>(gu_off) + d_ff * hidden_size);
        std::vector<float> down_raw(down_stack_raw.begin() + static_cast<std::ptrdiff_t>(dn_off),
                                     down_stack_raw.begin() + static_cast<std::ptrdiff_t>(dn_off) + hidden_size * d_ff);
        gate_t[static_cast<std::size_t>(k)] = transpose(gate_raw, d_ff, hidden_size);    // -> [hidden_size, d_ff]
        up_t[static_cast<std::size_t>(k)]   = transpose(up_raw, d_ff, hidden_size);      // -> [hidden_size, d_ff]
        down_t[static_cast<std::size_t>(k)] = transpose(down_raw, hidden_size, d_ff);    // -> [d_ff, hidden_size]
        const int e = expert_ids[static_cast<std::size_t>(k)];
        REQUIRE(e >= 0); REQUIRE(e < num_experts);
        gate_w[static_cast<std::size_t>(e)] = gate_t[static_cast<std::size_t>(k)].data();
        up_w[static_cast<std::size_t>(e)]   = up_t[static_cast<std::size_t>(k)].data();
        down_w[static_cast<std::size_t>(e)] = down_t[static_cast<std::size_t>(k)].data();
    }

    const auto shared_gate_raw = read_f32(dir / "moe_layer0_small_weight_shared_gate.bin",
                                           static_cast<std::size_t>(d_ff) * hidden_size);
    const auto shared_up_raw   = read_f32(dir / "moe_layer0_small_weight_shared_up.bin",
                                           static_cast<std::size_t>(d_ff) * hidden_size);
    const auto shared_down_raw = read_f32(dir / "moe_layer0_small_weight_shared_down.bin",
                                           static_cast<std::size_t>(hidden_size) * d_ff);
    const auto shared_gate_proj_raw = read_f32(dir / "moe_layer0_small_weight_shared_gate_proj.bin",
                                                static_cast<std::size_t>(hidden_size));
    REQUIRE(shared_gate_raw.size() == static_cast<std::size_t>(d_ff) * hidden_size);
    REQUIRE(shared_up_raw.size() == static_cast<std::size_t>(d_ff) * hidden_size);
    REQUIRE(shared_down_raw.size() == static_cast<std::size_t>(hidden_size) * d_ff);
    REQUIRE(shared_gate_proj_raw.size() == static_cast<std::size_t>(hidden_size));

    const auto shared_gate_w = transpose(shared_gate_raw, d_ff, hidden_size);   // -> [hidden_size, d_ff]
    const auto shared_up_w   = transpose(shared_up_raw, d_ff, hidden_size);     // -> [hidden_size, d_ff]
    const auto shared_down_w = transpose(shared_down_raw, hidden_size, d_ff);   // -> [d_ff, hidden_size]
    // shared_gate_proj is [1, hidden_size] (out_features=1) -- transpose is a no-op shape-wise, but the
    // convention flip still matters conceptually; a 1-row matrix's transpose is itself byte-for-byte.
    const auto& shared_gate_proj_w = shared_gate_proj_raw;

    const auto expected = read_f32(dir / "moe_layer0_small_output.bin", static_cast<std::size_t>(hidden_size));
    REQUIRE(expected.size() == static_cast<std::size_t>(hidden_size));

    std::vector<float> scratch(sub0::moe::scratch_floats(dims), 0.f);
    std::vector<float> out(static_cast<std::size_t>(hidden_size), 0.f);
    sub0::moe::forward_row(dims, input.data(), router_w.data(), gate_w.data(), up_w.data(), down_w.data(),
                            shared_gate_w.data(), shared_up_w.data(), shared_down_w.data(),
                            shared_gate_proj_w.data(), out.data(), scratch.data(), /*norm_topk_prob=*/true);

    double max_abs_diff = 0.0, sum_abs_expected = 0.0;
    for (std::size_t i = 0; i < out.size(); ++i) {
        const double diff = static_cast<double>(out[i]) - static_cast<double>(expected[i]);
        max_abs_diff = std::max(max_abs_diff, std::abs(diff));
        sum_abs_expected += std::abs(static_cast<double>(expected[i]));
    }
    INFO("max |out - expected| = " << max_abs_diff << ", sum |expected| = " << sum_abs_expected
         << " over " << out.size() << " values");
    REQUIRE(max_abs_diff < 5e-7);
    REQUIRE(sum_abs_expected > 0.0);   // sanity: not a degenerate all-zero fixture

    // --- Presence/mutation check using the SAME real fixture data (docs/MOE.md S7): force the router's
    // real top-10 selection's first two slots to the two DECOY experts (also real, extracted weights,
    // see the manifest's mutant_selection) via a raw call to expert_ffn_row + a manual weighted combine
    // (bypassing router_topk_row's own selection), and compare against the real reference's OWN
    // mutant output (moe_layer0_small_output_mutant.bin) -- a mutant CPU port that always consulted a
    // FIXED set of experts regardless of routing could still pass the correctness check above on this
    // one input; it cannot also match a SECOND, genuinely different-routing real reference.
    const auto expected_mutant = read_f32(dir / "moe_layer0_small_output_mutant.bin",
                                           static_cast<std::size_t>(hidden_size));
    REQUIRE(expected_mutant.size() == static_cast<std::size_t>(hidden_size));

    // Re-derive the real router's own top-10 + weights via router_topk_row directly (the same call
    // forward_row made above), then force indices[0] and indices[1] to the two decoy experts (manifest's
    // "decoy_experts", the last two entries of expert_ids by construction of the extraction script).
    std::vector<float> probs(static_cast<std::size_t>(num_experts), 0.f);
    float topk_w[sub0::moe::TOPK_MAX];
    int topk_idx[sub0::moe::TOPK_MAX];
    sub0::moe::router_topk_row(dims, input.data(), router_w.data(), probs.data(), topk_w, topk_idx, true);

    // manifest's mutant_selection names the exact expected post-mutation index list; the two decoys are
    // exactly the two expert_ids NOT among the real top-10 (the last two of the 12 fetched, by
    // construction -- see moe_layer0_small_manifest.json's own "decoy_experts"/"top10_real_selection").
    std::vector<int> is_top10(static_cast<std::size_t>(num_experts), 0);
    for (int k = 0; k < experts_per_tok; ++k) is_top10[static_cast<std::size_t>(topk_idx[k])] = 1;
    std::vector<int> decoys;
    for (int e : expert_ids) if (!is_top10[static_cast<std::size_t>(e)]) decoys.push_back(e);
    REQUIRE(decoys.size() == 2);
    int mutant_idx[sub0::moe::TOPK_MAX];
    for (int k = 0; k < experts_per_tok; ++k) mutant_idx[k] = topk_idx[k];
    mutant_idx[0] = decoys[0];
    mutant_idx[1] = decoys[1];

    std::vector<float> ffn_scratch(static_cast<std::size_t>(d_ff), 0.f);
    std::vector<float> g_scratch(static_cast<std::size_t>(d_ff), 0.f);
    std::vector<float> expert_out(static_cast<std::size_t>(hidden_size), 0.f);
    std::vector<float> out_mutant(static_cast<std::size_t>(hidden_size), 0.f);
    for (int k = 0; k < experts_per_tok; ++k) {
        const int e = mutant_idx[k];
        sub0::moe::expert_ffn_row(dims, input.data(), gate_w[static_cast<std::size_t>(e)],
                                   up_w[static_cast<std::size_t>(e)], down_w[static_cast<std::size_t>(e)],
                                   expert_out.data(), ffn_scratch.data(), g_scratch.data());
        for (int j = 0; j < hidden_size; ++j) out_mutant[static_cast<std::size_t>(j)] += topk_w[k] * expert_out[static_cast<std::size_t>(j)];
    }
    sub0::moe::expert_ffn_row(dims, input.data(), shared_gate_w.data(), shared_up_w.data(), shared_down_w.data(),
                               expert_out.data(), ffn_scratch.data(), g_scratch.data());
    float gate_logit = 0.f;
    for (int i = 0; i < hidden_size; ++i) gate_logit += input[static_cast<std::size_t>(i)] * shared_gate_proj_w[static_cast<std::size_t>(i)];
    const float sg = 1.f / (1.f + std::exp(-gate_logit));
    for (int j = 0; j < hidden_size; ++j) out_mutant[static_cast<std::size_t>(j)] += sg * expert_out[static_cast<std::size_t>(j)];

    double mutant_max_abs_diff = 0.0;
    for (std::size_t i = 0; i < out_mutant.size(); ++i)
        mutant_max_abs_diff = std::max(mutant_max_abs_diff,
            std::abs(static_cast<double>(out_mutant[i]) - static_cast<double>(expected_mutant[i])));
    INFO("mutant: max |out_mutant - expected_mutant| = " << mutant_max_abs_diff);
    REQUIRE(mutant_max_abs_diff < 5e-7);

    // The real point of this check: the mutant output must GENUINELY differ from the correct output --
    // proof this fixture's own two references (real vs. mutant routing) are not coincidentally equal,
    // and that OUR implementation reproduces that real difference rather than collapsing to one fixed
    // answer regardless of which experts are actually consulted.
    double real_vs_mutant_diff = 0.0;
    for (std::size_t i = 0; i < out.size(); ++i)
        real_vs_mutant_diff = std::max(real_vs_mutant_diff,
            std::abs(static_cast<double>(out[i]) - static_cast<double>(out_mutant[i])));
    INFO("max |out(real routing) - out_mutant(forced decoys)| = " << real_vs_mutant_diff);
    REQUIRE(real_vs_mutant_diff > 1e-6);
}

// A different mutant shape, needing no fixture data at all (mirroring gated_residual_qwen4_fixture_
// tests.cpp's own "identical-stream" style check): a synthetic router with num_experts=4,
// experts_per_tok=2 where two DISTINCT, hand-verified experts are always the top two regardless of the
// input's magnitude (a monotonically increasing router logit per expert index) -- confirms the selected
// SET actually changes as the input is perturbed to favor a different expert, proving router_topk_row
// genuinely reads the router's own output rather than returning a fixed index list.
TEST_CASE("Mixture of Experts router_topk_row genuinely selects DIFFERENT experts for different inputs",
          "[moe][mutation]") {
    constexpr int hidden_size = 2, d_ff = 1, num_experts = 4, experts_per_tok = 2;
    const sub0::moe::Dims dims{hidden_size, d_ff, num_experts, experts_per_tok};

    // router_w: [hidden_size, num_experts], this project's own [in,out] convention. Expert e's logit is
    // x[0]*row0[e] + x[1]*row1[e]; row0 favors LOW expert indices, row1 favors HIGH ones.
    const std::vector<float> router_w = {
        4.f, 3.f, 2.f, 1.f,     // row for x[0]: favors experts 0,1
        1.f, 2.f, 3.f, 4.f,     // row for x[1]: favors experts 2,3
    };
    std::vector<float> probs(num_experts, 0.f);
    float topk_w[sub0::moe::TOPK_MAX];
    int topk_idx[sub0::moe::TOPK_MAX];

    const std::vector<float> x_low_favored = {1.f, 0.f};    // should pick experts {0,1}
    sub0::moe::router_topk_row(dims, x_low_favored.data(), router_w.data(), probs.data(), topk_w, topk_idx, true);
    std::vector<int> sel_a{topk_idx[0], topk_idx[1]};
    std::sort(sel_a.begin(), sel_a.end());
    REQUIRE(sel_a == std::vector<int>{0, 1});

    const std::vector<float> x_high_favored = {0.f, 1.f};   // should pick experts {2,3}
    sub0::moe::router_topk_row(dims, x_high_favored.data(), router_w.data(), probs.data(), topk_w, topk_idx, true);
    std::vector<int> sel_b{topk_idx[0], topk_idx[1]};
    std::sort(sel_b.begin(), sel_b.end());
    REQUIRE(sel_b == std::vector<int>{2, 3});

    // norm_topk_prob=true: the two selected weights must sum to 1 (renormalized), regardless of input.
    float wsum = topk_w[0] + topk_w[1];
    REQUIRE(std::abs(wsum - 1.f) < 1e-5f);
}

#endif  // SUB0_SOURCE_DIR
