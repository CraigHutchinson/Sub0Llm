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
    // not a silent behaviour change. Two synthetics = LnF and LmBias, the two destinations the real
    // file has NO source for (see recipe_for's own comment -- this is the finding, not an oversight).
    CHECK(synthetic == 2);
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
