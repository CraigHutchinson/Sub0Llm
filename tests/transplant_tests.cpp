// transplant_tests.cpp -- unit tests for sub0::transplant's mapping table and array operations
// (docs/WP4_SCOPE.md S4c levels 1 and 2, the parts that do not need a real file or a fixture).
//
// Engine-free, like gguf_tests.cpp: transplant.hpp deliberately does not include sub0_config.hpp, so
// none of this needs a compiled model. The fixture REPLAY (levels 3 and 4) lives in its own file,
// transplant_fixture_tests.cpp.
//
// The array ops are checked against expectations derived here, element by element, rather than against
// a second call to the same function -- and each has at least one NEGATIVE case pinning the specific
// wrong-but-plausible implementation it exists to rule out (a missed transpose, a down-the-middle
// per-head split, a swapped concat order). A shape assertion passes for all three.

#include "sub0/transplant.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

using namespace sub0::transplant;

namespace {

// A deterministic, index-distinguishable filler: every element of every test matrix is unique, so a
// permutation bug cannot hide behind repeated values.
std::vector<float> ramp(std::size_t n, float base = 0.f) {
    std::vector<float> v(n);
    for (std::size_t i = 0; i < n; ++i) v[i] = base + static_cast<float>(i);
    return v;
}

}  // namespace

TEST_CASE("transplant: transpose_out_in maps (o,i) -> (i,o) on a non-square matrix", "[transplant]") {
    // 3 outputs x 4 inputs. Non-square deliberately: a square matrix cannot distinguish a transpose
    // from a copy by SIZE, and this project's own Wo stopped being square at WP4b blocker A.
    constexpr int out_f = 3, in_f = 4;
    const std::vector<float> src = ramp(out_f * in_f);   // src[o*4 + i] == 4o + i
    std::vector<float> dst(src.size(), -1.f);
    transpose_out_in(src.data(), out_f, in_f, dst.data());
    for (int o = 0; o < out_f; ++o)
        for (int i = 0; i < in_f; ++i) {
            INFO("o " << o << " i " << i);
            CHECK(dst[static_cast<std::size_t>(i) * out_f + o] == static_cast<float>(o * in_f + i));
        }
    // Negative: a straight copy would leave dst[1] == 1, but the transpose puts src[4] there.
    CHECK(dst[1] != src[1]);
    CHECK(dst[1] == 4.f);
    // Applying it twice returns the original -- so a DOUBLE transpose is a real hazard the element
    // check above is what catches, not the size (which is identical either way).
    std::vector<float> back(src.size(), -1.f);
    transpose_out_in(dst.data(), in_f, out_f, back.data());
    CHECK(back == src);
}

TEST_CASE("transplant: per_head_half_transpose splits PER HEAD, not down the middle", "[transplant]") {
    // 2 heads x head_dim 3, so the fused output axis is 12 rows laid out as
    //   [q_h0 (3) | g_h0 (3) | q_h1 (3) | g_h1 (3)].
    // Splitting the row block down the middle would take rows 0..5 as "query", which is q_h0 followed
    // by GATE h0 -- correct for head 0's first head_dim only, and wrong for everything after. Two heads
    // is the smallest shape where the two answers differ at all (docs/QSA.md S2b.4).
    constexpr int n_heads = 2, head_dim = 3, in_f = 2;
    const std::vector<float> src = ramp(n_heads * 2 * head_dim * in_f);   // src[row*2 + i]

    std::vector<float> q(static_cast<std::size_t>(in_f) * n_heads * head_dim, -1.f);
    std::vector<float> g(q.size(), -1.f);
    per_head_half_transpose(src.data(), n_heads, head_dim, in_f, /*half=*/0, q.data());
    per_head_half_transpose(src.data(), n_heads, head_dim, in_f, /*half=*/1, g.data());

    const int out_f = n_heads * head_dim;
    for (int h = 0; h < n_heads; ++h)
        for (int d = 0; d < head_dim; ++d)
            for (int i = 0; i < in_f; ++i) {
                const int q_row = h * 2 * head_dim + d;                 // query half
                const int g_row = h * 2 * head_dim + head_dim + d;      // gate half
                const std::size_t at = static_cast<std::size_t>(i) * out_f + h * head_dim + d;
                INFO("h " << h << " d " << d << " i " << i);
                CHECK(q[at] == src[static_cast<std::size_t>(q_row) * in_f + i]);
                CHECK(g[at] == src[static_cast<std::size_t>(g_row) * in_f + i]);
            }

    // The negative case, stated as the actual wrong implementation: a down-the-middle split takes the
    // first n_heads*head_dim rows as the query. Head 1's slot must DISAGREE with that.
    std::vector<float> naive(q.size(), -1.f);
    transpose_out_in(src.data(), out_f, in_f, naive.data());   // rows 0..5 == "first half"
    bool differs = false;
    for (std::size_t k = 0; k < q.size(); ++k) differs |= (q[k] != naive[k]);
    CHECK(differs);
    // ...and specifically: head 1's first query column must come from row 6, not row 3.
    CHECK(q[static_cast<std::size_t>(0) * out_f + head_dim] == src[6 * in_f + 0]);
}

TEST_CASE("transplant: concat_out_transpose joins along the output axis, q-half first", "[transplant]") {
    // The indexer's asymmetric split: n_heads*head_dim query outputs then kv_heads*head_dim key
    // outputs (docs/QSA.md S1a's torch.split([n*hd, kv*hd], dim=-1)). out_a != out_b deliberately, so a
    // swapped concat order is not merely a value error but changes which columns are which.
    constexpr int out_a = 3, out_b = 2, in_f = 4;
    const std::vector<float> a = ramp(out_a * in_f, 100.f);
    const std::vector<float> b = ramp(out_b * in_f, 900.f);
    std::vector<float> dst(static_cast<std::size_t>(in_f) * (out_a + out_b), -1.f);
    concat_out_transpose(a.data(), out_a, b.data(), out_b, in_f, dst.data());

    const int out_f = out_a + out_b;
    for (int o = 0; o < out_a; ++o)
        for (int i = 0; i < in_f; ++i)
            CHECK(dst[static_cast<std::size_t>(i) * out_f + o] == a[static_cast<std::size_t>(o) * in_f + i]);
    for (int o = 0; o < out_b; ++o)
        for (int i = 0; i < in_f; ++i)
            CHECK(dst[static_cast<std::size_t>(i) * out_f + out_a + o] == b[static_cast<std::size_t>(o) * in_f + i]);
    // Column out_a is the FIRST key column, i.e. b's first row -- not a's last.
    CHECK(dst[static_cast<std::size_t>(0) * out_f + out_a] == 900.f);
    CHECK(dst[static_cast<std::size_t>(0) * out_f + out_a - 1] == 100.f + 2.f * in_f);
}

TEST_CASE("transplant: gguf_name substitutes the layer index, and passes patterns without one through",
          "[transplant]") {
    CHECK(gguf_name("blk.%d.attn_qkv.weight", 0) == "blk.0.attn_qkv.weight");
    CHECK(gguf_name("blk.%d.ssm_dt.bias", 37) == "blk.37.ssm_dt.bias");
    CHECK(gguf_name("output_hc_norm.weight", 3) == "output_hc_norm.weight");
    CHECK(gguf_name("token_embd.weight", 0) == "token_embd.weight");
    CHECK(gguf_name(nullptr, 0).empty());
}

TEST_CASE("transplant: every destination has a well-formed recipe", "[transplant]") {
    int synthetic = 0, expert_slice = 0, concat = 0, per_head = 0;
    for (int d = 0; d < static_cast<int>(Dest::Count); ++d) {
        const Recipe r = recipe_for(static_cast<Dest>(d));
        INFO("Dest " << d);
        if (r.op == Op::Synthetic) {
            ++synthetic;
            CHECK(r.src == nullptr);          // a synthetic destination must not claim a source
            continue;
        }
        REQUIRE(r.src != nullptr);            // everything else must name one
        CHECK(std::string(r.src).find(' ') == std::string::npos);
        CHECK((r.op == Op::ConcatOut) == (r.src2 != nullptr));
        if (r.op == Op::ConcatOut) ++concat;
        if (r.op == Op::ExpertSlice) ++expert_slice;
        if (r.op == Op::PerHeadHalf) ++per_head;
    }
    // Pinned counts, so a future edit that (say) turns a Transpose into a Copy is a test failure and
    // not a silent behaviour change. ONE synthetic = LmBias, the only destination the real file has no
    // source for (the real head is bias-free). It was two while an LnF destination still existed; the
    // real model has no final norm at all, so make_param_layout() stopped emitting the slot under
    // USE_GATED_RESIDUAL and Dest::LnF was removed rather than left synthesizing an unused identity
    // gain (see recipe_for's own comment -- this is the finding, not an oversight).
    CHECK(synthetic == 1);
    CHECK(concat == 1);        // the QSA indexer's q|k pair
    CHECK(expert_slice == 3);  // MoE gate/up/down
    CHECK(per_head == 2);      // QSA query and gate halves of the fused attn_q

    // The two same-shaped, opposite-treatment embeddings -- the single easiest thing to get backwards
    // in this whole table, since token_embd.weight and output.weight declare IDENTICAL dims.
    CHECK(recipe_for(Dest::TokEmb).op == Op::Copy);
    CHECK(recipe_for(Dest::LmHead).op == Op::Transpose);
    // The depthwise conv, likewise: GGUF's [kernel, channels] ne IS the destination's [C, K] bytes.
    CHECK(recipe_for(Dest::GdnConv).op == Op::Copy);
    // And the pair that is shaped exactly like the dt_bias/A_log identity swap WP-GDN Stage 3 found:
    // both [hidden, num_v_heads], so only the NAMES distinguish them.
    CHECK(std::string(recipe_for(Dest::GdnInProjA).src) == "blk.%d.ssm_alpha.weight");
    CHECK(std::string(recipe_for(Dest::GdnInProjB).src) == "blk.%d.ssm_beta.weight");
}

TEST_CASE("transplant: stats_of and stats_consistent detect a wrong permutation", "[transplant]") {
    const std::vector<float> src = ramp(12, 1.f);          // 1..12
    const Stats a = stats_of(src);
    CHECK(a.n == 12);
    CHECK(a.min == 1.0);
    CHECK(a.max == 12.0);
    CHECK(a.mean == Catch::Approx(6.5));
    CHECK(a.nonfinite == 0);

    // A genuine permutation (the transpose) leaves every statistic alone.
    std::vector<float> t(src.size());
    transpose_out_in(src.data(), 3, 4, t.data());
    CHECK(stats_consistent(a, stats_of(t)));
    CHECK(t != src);   // ...while the DATA genuinely moved, so this is not vacuous

    // A wrong SLICE (same count, different elements) is caught by the exact extrema comparison.
    std::vector<float> wrong = src;
    wrong[0] = 13.f;
    CHECK_FALSE(stats_consistent(a, stats_of(wrong)));
    // So is a truncation, and so is a NaN appearing on one side only.
    CHECK_FALSE(stats_consistent(a, stats_of(std::span<const float>(src.data(), 11))));
    std::vector<float> nan_side = src;
    nan_side[3] = std::nanf("");
    CHECK_FALSE(stats_consistent(a, stats_of(nan_side)));
    CHECK(stats_of(nan_side).nonfinite == 1);

    // What it deliberately does NOT catch, stated so the limitation is on the record rather than
    // discovered later: two DIFFERENT tensors that happen to be permutations of each other -- i.e.
    // exactly the same-shaped identity swap (dt_bias/A_log, ssm_alpha/ssm_beta) that levels 3 and 4
    // of the gate exist for. Here, a reversal.
    std::vector<float> reversed(src.rbegin(), src.rend());
    CHECK(stats_consistent(a, stats_of(reversed)));
}

// --- WP4f: the converter's own value/order conventions ---------------------------------------------
// These cases pin the corrections whose ABSENCE made the first real cross-comparison against llama.cpp
// diverge at decoder layer 0 (transplant.hpp's "A GGUF IS NOT A COPY OF THE CHECKPOINT"). They live
// here, and not only in the fixture replay, because the fixture is num_k_heads == 1 -- where the
// V-head reorder is the identity and therefore cannot be tested at all.

TEST_CASE("transplant: fold_for names exactly the tensors llama.cpp's converter rewrites",
          "[transplant]") {
    // Every Qwen4ExpTextRMSNorm gain in the file is (1 + w) EXCEPT GDN's ssm_norm, which the
    // converter's own `not name.endswith("linear_attn.norm.weight")` excludes -- and gdn_math.hpp
    // correspondingly uses the gain directly while gr/qsa add the 1 themselves. Getting this list
    // wrong in EITHER direction is a silent, model-changing error, so it is pinned name by name.
    for (Dest d : {Dest::GrAttnNorm, Dest::GrFfnNorm, Dest::GrExitNorm, Dest::QsaQNorm, Dest::QsaKNorm,
                   Dest::QsaIdxQNorm, Dest::QsaIdxKNorm}) {
        INFO("dest " << static_cast<int>(d));
        CHECK(fold_for(d) == Fold::ZeroCentredGamma);
    }
    CHECK(fold_for(Dest::GdnNorm) == Fold::None);       // the one real exception
    CHECK(fold_for(Dest::GdnALog) == Fold::NegExpALog);
    CHECK(fold_for(Dest::GdnDtBias) == Fold::None);     // same shape as ssm_a, a DIFFERENT convention
    for (Dest d : {Dest::TokEmb, Dest::LmHead, Dest::GdnInProjQkv, Dest::MoeRouter, Dest::MoeDown,
                   Dest::GrAttnDown, Dest::QsaQProj}) {
        INFO("dest " << static_cast<int>(d));
        CHECK(fold_for(d) == Fold::None);
    }
}

TEST_CASE("transplant: apply_fold inverts the converter, exactly", "[transplant]") {
    // ZeroCentredGamma: the file holds 1 + w, the destination must hold w.
    std::vector<float> gamma{1.f, 0.5f, 2.25f, -0.75f};
    REQUIRE(apply_fold(Fold::ZeroCentredGamma, gamma.data(), gamma.size()));
    CHECK(gamma[0] == 0.f);
    CHECK(gamma[1] == -0.5f);
    CHECK(gamma[2] == 1.25f);
    CHECK(gamma[3] == -1.75f);

    // NegExpALog: the file holds -exp(A_log), the destination must hold A_log, so that gdn_math's own
    // `-exp(a_log)` reproduces the file's value. Checked as a ROUND TRIP through that consumer's
    // formula, not against a literal -- what has to hold is that the gate is unchanged.
    const std::vector<float> a_log{-2.f, -0.25f, 0.f, 1.5f, 5.0625f};
    std::vector<float> stored(a_log.size());
    for (std::size_t i = 0; i < a_log.size(); ++i) stored[i] = -std::exp(a_log[i]);
    REQUIRE(apply_fold(Fold::NegExpALog, stored.data(), stored.size()));
    for (std::size_t i = 0; i < a_log.size(); ++i) {
        INFO("i " << i);
        CHECK(stored[i] == Catch::Approx(a_log[i]).margin(1e-5));
        CHECK(-std::exp(stored[i]) == Catch::Approx(-std::exp(a_log[i])).epsilon(1e-6));
    }

    // A non-negative entry means the source is NOT `-exp(...)`, so the inverse must REFUSE rather than
    // write a NaN weight, which would look like a plausible number everywhere downstream.
    std::vector<float> bad{-1.f, 0.f, -2.f};
    CHECK_FALSE(apply_fold(Fold::NegExpALog, bad.data(), bad.size()));

    // None leaves the buffer alone.
    std::vector<float> untouched = ramp(5, 1.f);
    const std::vector<float> copy = untouched;
    REQUIRE(apply_fold(Fold::None, untouched.data(), untouched.size()));
    CHECK(untouched == copy);
}

TEST_CASE("transplant: ungroup_v_heads undoes the converter's grouped->tiled V-head reorder",
          "[transplant]") {
    // A GENUINE multi-key-head shape: 2 key heads, 6 value heads, rep 3.
    constexpr int n_k = 2, n_v = 6, rep = n_v / n_k, hk = 2, hv = 2;

    SECTION("columns, the whole axis, head_dim wide (in_proj_z)") {
        constexpr int rows = 3, cols = n_v * hv;
        const std::vector<float> src = ramp(rows * cols);
        std::vector<float> dst(src.size(), -1.f);
        ungroup_v_heads(src.data(), rows, cols, VPerm{VAxis::Cols, false, true}, n_k, n_v, hk, hv,
                        dst.data());
        for (int row = 0; row < rows; ++row)
            for (int k = 0; k < n_k; ++k)
                for (int r = 0; r < rep; ++r)
                    for (int g = 0; g < hv; ++g) {
                        INFO("row " << row << " k " << k << " r " << r << " g " << g);
                        CHECK(dst[static_cast<std::size_t>(row) * cols + (k * rep + r) * hv + g] ==
                              src[static_cast<std::size_t>(row) * cols + (r * n_k + k) * hv + g]);
                    }
        // It is a genuine permutation -- every statistic survives it, which is the point: level 2 of
        // the WP4c gate is provably blind to this whole class of defect.
        CHECK(stats_consistent(stats_of(src), stats_of(dst)));
        CHECK(dst != src);   // ...and it really did move, so the check above is not vacuous
    }

    SECTION("columns, per-head scalars, no head_dim (ssm_alpha / ssm_a / dt_bias)") {
        constexpr int rows = 2, cols = n_v;
        const std::vector<float> src = ramp(rows * cols);
        std::vector<float> dst(src.size(), -1.f);
        ungroup_v_heads(src.data(), rows, cols, VPerm{VAxis::Cols, false, false}, n_k, n_v, hk, hv,
                        dst.data());
        // HF v-heads 0..5 are (k,r) = (0,0)(0,1)(0,2)(1,0)(1,1)(1,2), which sit in tiled slots
        // r*n_k + k = 0, 2, 4, 1, 3, 5.
        const int expect[n_v] = {0, 2, 4, 1, 3, 5};
        for (int row = 0; row < rows; ++row)
            for (int h = 0; h < n_v; ++h) {
                INFO("row " << row << " h " << h);
                CHECK(dst[static_cast<std::size_t>(row) * cols + h] ==
                      src[static_cast<std::size_t>(row) * cols + expect[h]]);
            }
    }

    SECTION("columns after the two key blocks (in_proj_qkv)") {
        constexpr int key = n_k * hk, base = 2 * key, cols = base + n_v * hv, rows = 2;
        const std::vector<float> src = ramp(rows * cols);
        std::vector<float> dst(src.size(), -1.f);
        ungroup_v_heads(src.data(), rows, cols, VPerm{VAxis::Cols, true, true}, n_k, n_v, hk, hv,
                        dst.data());
        // The Q and K blocks are NOT reordered -- only the V third is. Shuffling the key heads too
        // would leave every shape and every statistic right and compute a different model.
        for (int row = 0; row < rows; ++row)
            for (int c = 0; c < base; ++c) {
                INFO("row " << row << " c " << c);
                CHECK(dst[static_cast<std::size_t>(row) * cols + c] ==
                      src[static_cast<std::size_t>(row) * cols + c]);
            }
        for (int row = 0; row < rows; ++row)
            for (int h = 0; h < n_v; ++h)
                for (int g = 0; g < hv; ++g) {
                    const int k = h / rep, r = h % rep;
                    INFO("row " << row << " h " << h << " g " << g);
                    CHECK(dst[static_cast<std::size_t>(row) * cols + base + h * hv + g] ==
                          src[static_cast<std::size_t>(row) * cols + base + (r * n_k + k) * hv + g]);
                }
    }

    SECTION("rows (out_proj's input axis, and the conv's channels)") {
        constexpr int rows = n_v * hv, cols = 3;
        const std::vector<float> src = ramp(rows * cols);
        std::vector<float> dst(src.size(), -1.f);
        ungroup_v_heads(src.data(), rows, cols, VPerm{VAxis::Rows, false, true}, n_k, n_v, hk, hv,
                        dst.data());
        for (int h = 0; h < n_v; ++h) {
            const int k = h / rep, r = h % rep;
            for (int g = 0; g < hv; ++g)
                for (int c = 0; c < cols; ++c) {
                    INFO("h " << h << " g " << g << " c " << c);
                    CHECK(dst[static_cast<std::size_t>(h * hv + g) * cols + c] ==
                          src[static_cast<std::size_t>((r * n_k + k) * hv + g) * cols + c]);
                }
        }
    }

    SECTION("it is the IDENTITY at one key head -- which is why the fixture replay cannot see it") {
        constexpr int rows = 2, cols = 3 * hv;
        const std::vector<float> src = ramp(rows * cols);
        std::vector<float> dst(src.size(), -1.f);
        ungroup_v_heads(src.data(), rows, cols, VPerm{VAxis::Cols, false, true}, /*num_k_heads=*/1,
                        /*num_v_heads=*/3, hk, hv, dst.data());
        CHECK(dst == src);
    }

    SECTION("VAxis::None copies through untouched") {
        const std::vector<float> src = ramp(12);
        std::vector<float> dst(src.size(), -1.f);
        ungroup_v_heads(src.data(), 3, 4, VPerm{}, n_k, n_v, hk, hv, dst.data());
        CHECK(dst == src);
    }
}

TEST_CASE("transplant: vperm_for names every v-head-indexed GDN tensor and nothing else",
          "[transplant]") {
    CHECK(vperm_for(Dest::GdnInProjQkv).axis == VAxis::Cols);
    CHECK(vperm_for(Dest::GdnInProjQkv).after_keys);          // Q|K first, then V
    CHECK(vperm_for(Dest::GdnInProjQkv).wide);
    CHECK(vperm_for(Dest::GdnInProjZ).axis == VAxis::Cols);
    CHECK_FALSE(vperm_for(Dest::GdnInProjZ).after_keys);
    CHECK(vperm_for(Dest::GdnConv).axis == VAxis::Rows);
    CHECK(vperm_for(Dest::GdnConv).after_keys);
    CHECK(vperm_for(Dest::GdnOutProj).axis == VAxis::Rows);   // out_proj's INPUT axis is the v-head one
    CHECK_FALSE(vperm_for(Dest::GdnOutProj).after_keys);
    for (Dest d : {Dest::GdnInProjA, Dest::GdnInProjB, Dest::GdnALog, Dest::GdnDtBias}) {
        INFO("dest " << static_cast<int>(d));
        CHECK(vperm_for(d).axis == VAxis::Cols);
        CHECK_FALSE(vperm_for(d).wide);                       // one slot per head, not head_v_dim
    }
    // ssm_norm is [1, head_v_dim], SHARED across heads -- it has no v-head axis at all, and permuting
    // it would corrupt a tensor the converter never touched.
    CHECK(vperm_for(Dest::GdnNorm).axis == VAxis::None);
    for (Dest d : {Dest::TokEmb, Dest::LmHead, Dest::MoeGate, Dest::QsaQProj, Dest::GrAttnNorm}) {
        INFO("dest " << static_cast<int>(d));
        CHECK(vperm_for(d).axis == VAxis::None);
    }
}
