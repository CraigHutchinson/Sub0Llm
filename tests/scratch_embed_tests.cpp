// scratch_embed_tests.cpp -- the content-derived scratch-slot embedding scaffold (include/sub0/
// scratch_slots.hpp + backend_cpu's op_embed/forward_one branch). Validates the pure mean-pool encoder
// math (exact), that the engine actually FEEDS a bound slot's content-derived embedding into forward
// (a differential against a manually-set mean embedding), and that with no bindings the path is unchanged.

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "sub0/core.hpp"
#include "sub0/scratch_slots.hpp"

#include <random>
#include <span>
#include <string>
#include <vector>

using sub0::SlotEncoding;

// --- 1. The pure encoder math (engine-free): mean-pool forward + its adjoint -----------------------
TEST_CASE("scratch encoder: mean-pool forward + backward adjoint are exact", "[scratch][embed]") {
    constexpr int C = 4;
    // A tiny fake embedding table: rows 0..3, distinct values.
    std::vector<float> tab = {
        1, 2, 3, 4,      // row 0
        5, 6, 7, 8,      // row 1
        9, 10, 11, 12,   // row 2
        0, 0, 0, 0,      // row 3
    };
    const std::vector<int> frags = {0, 1, 2};

    float out[C];
    sub0::encode_slot(tab.data(), C, frags, SlotEncoding::MeanPool, out);
    for (int j = 0; j < C; ++j)
        REQUIRE(out[j] == (tab[0 * C + j] + tab[1 * C + j] + tab[2 * C + j]) / 3.f);   // mean of the rows

    // Backward: a row-grad dout scatters dout/nfrags into each fragment row; untouched rows stay 0.
    std::vector<float> grad(tab.size(), 0.f);
    const float dout[C] = {3, 6, 9, 12};
    sub0::encode_slot_bwd(dout, C, frags, SlotEncoding::MeanPool, grad.data());
    for (int f : frags)
        for (int j = 0; j < C; ++j) REQUIRE(grad[f * C + j] == dout[j] / 3.f);
    for (int j = 0; j < C; ++j) REQUIRE(grad[3 * C + j] == 0.f);   // row 3 not a fragment -> no grad

    // Empty fragments -> zero row, no grad.
    float zero[C];
    sub0::encode_slot(tab.data(), C, {}, SlotEncoding::MeanPool, zero);
    for (int j = 0; j < C; ++j) REQUIRE(zero[j] == 0.f);
}

// --- 1b. CharEncoder (learned per-fragment projection + relu, sum-pooled): forward + backward vs FD ---
TEST_CASE("scratch encoder: CharEncoder forward + backward match finite differences", "[scratch][embed]") {
    constexpr int C = 3;
    std::vector<float> tab = { 0.5f, -0.3f, 0.2f,   -0.1f, 0.4f, 0.6f,   0.f, 0.f, 0.f };   // rows 0,1 used
    std::vector<float> W(static_cast<std::size_t>(C) * C);
    std::mt19937 rng(7);
    std::normal_distribution<float> nd(0.f, 0.6f);
    for (float& x : W) x = nd(rng);
    const std::vector<int> frags = {0, 1};
    const float dout[C] = {1.0f, -0.5f, 0.7f};

    auto loss = [&](const std::vector<float>& tabv, const std::vector<float>& Wv) {
        float out[C];
        sub0::encode_slot(tabv.data(), C, frags, SlotEncoding::CharEncoder, out, Wv.data());
        float L = 0.f; for (int c = 0; c < C; ++c) L += dout[c] * out[c]; return L;   // <dout, out>
    };

    std::vector<float> Wg(W.size(), 0.f), Tg(tab.size(), 0.f);
    sub0::encode_slot_bwd(dout, C, frags, SlotEncoding::CharEncoder, Tg.data(), tab.data(), W.data(), Wg.data());

    const float eps = 1e-3f;
    for (std::size_t i = 0; i < W.size(); ++i) {                       // dW vs finite difference
        auto wp = W, wm = W; wp[i] += eps; wm[i] -= eps;
        REQUIRE(Wg[i] == Catch::Approx((loss(tab, wp) - loss(tab, wm)) / (2 * eps)).margin(2e-2));
    }
    for (int f : frags) for (int k = 0; k < C; ++k) {                  // d(fragment row) vs finite difference
        auto tp = tab, tm = tab;
        tp[static_cast<std::size_t>(f) * C + k] += eps;
        tm[static_cast<std::size_t>(f) * C + k] -= eps;
        REQUIRE(Tg[static_cast<std::size_t>(f) * C + k] ==
                Catch::Approx((loss(tp, W) - loss(tm, W)) / (2 * eps)).margin(2e-2));
    }
}

// --- 1b-ii. Hash (RoPE-style positional binding): forward + backward vs finite differences ------------
TEST_CASE("scratch encoder: Hash forward + backward match finite differences", "[scratch][embed]") {
    constexpr int C = 6;   // even, so no odd-tail branch here (that's covered by the order test below)
    std::vector<float> tab = {
        0.5f, -0.3f, 0.2f, 0.1f, -0.4f, 0.6f,     // row 0
       -0.1f,  0.4f, 0.6f, -0.2f, 0.3f, -0.5f,    // row 1
        0.2f, -0.6f, 0.1f,  0.4f, -0.1f, 0.3f,    // row 2
    };
    const std::vector<int> frags = {0, 1, 2};   // three fragments -> positions 0,1,2
    const float dout[C] = {1.0f, -0.5f, 0.7f, -0.2f, 0.3f, 0.4f};

    auto loss = [&](const std::vector<float>& tabv) {
        float out[C];
        sub0::encode_slot(tabv.data(), C, frags, SlotEncoding::Hash, out);
        float L = 0.f; for (int c = 0; c < C; ++c) L += dout[c] * out[c]; return L;
    };

    std::vector<float> Tg(tab.size(), 0.f);
    sub0::encode_slot_bwd(dout, C, frags, SlotEncoding::Hash, Tg.data());

    const float eps = 1e-3f;
    for (int f : frags) for (int k = 0; k < C; ++k) {
        auto tp = tab, tm = tab;
        tp[static_cast<std::size_t>(f) * C + k] += eps;
        tm[static_cast<std::size_t>(f) * C + k] -= eps;
        REQUIRE(Tg[static_cast<std::size_t>(f) * C + k] ==
                Catch::Approx((loss(tp) - loss(tm)) / (2 * eps)).margin(2e-2));
    }
}

// --- 1b-iii. Hash is actually ORDER-SENSITIVE (unlike MeanPool/CharEncoder, which are provably
// permutation-invariant by the Deep Sets canonical form -- see the TODO(order-sensitive-slot-encoding)
// comment in scratch_slots.hpp). Two direct, training-free checks: (a) reordering the SAME fragments
// changes the encoded vector, and (b) a lone fragment at position 0 passes through as its own raw row
// (angle 0 = identity), the deterministic anchor a downstream model could learn to read off directly.
TEST_CASE("scratch encoder: Hash is order-sensitive, unlike MeanPool/CharEncoder", "[scratch][embed]") {
    constexpr int C = 8;   // even C: exercises the interleaved-pair path with no odd tail
    std::mt19937 rng(11);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> tab(static_cast<std::size_t>(4) * C);
    for (float& x : tab) x = nd(rng);

    const std::vector<int> fwd = {0, 1, 2};
    const std::vector<int> rev = {2, 1, 0};   // same SET, reversed order

    float out_fwd[C], out_rev[C];
    sub0::encode_slot(tab.data(), C, fwd, SlotEncoding::Hash, out_fwd);
    sub0::encode_slot(tab.data(), C, rev, SlotEncoding::Hash, out_rev);
    bool differs = false;
    for (int j = 0; j < C; ++j) if (out_fwd[j] != Catch::Approx(out_rev[j]).margin(1e-5)) differs = true;
    REQUIRE(differs);   // MeanPool/CharEncoder would be IDENTICAL here (order cannot matter to a sum/mean)

    // Sanity floor: a MeanPool/CharEncoder-style check would ALSO see identical output on reorder --
    // confirm that premise on the SAME fragments so this test's "differs" claim above isn't a fluke of
    // the specific vectors chosen (MeanPool must be order-invariant BY CONSTRUCTION, not just empirically).
    // Approx, not `==`: summing the same three rows in a different ORDER can differ in the last float ULPs
    // (IEEE-754 addition isn't associative) even though the mathematical mean is order-invariant -- Hash's
    // difference above (1e-5 margin) is many orders of magnitude larger than this rounding noise.
    float mp_fwd[C], mp_rev[C];
    sub0::encode_slot(tab.data(), C, fwd, SlotEncoding::MeanPool, mp_fwd);
    sub0::encode_slot(tab.data(), C, rev, SlotEncoding::MeanPool, mp_rev);
    for (int j = 0; j < C; ++j) REQUIRE(mp_fwd[j] == Catch::Approx(mp_rev[j]).margin(1e-5));

    // A lone fragment at position 0 passes through as its own raw row (angle 0 -> identity rotation).
    const std::vector<int> solo = {1};
    float out_solo[C];
    sub0::encode_slot(tab.data(), C, solo, SlotEncoding::Hash, out_solo);
    for (int j = 0; j < C; ++j) REQUIRE(out_solo[j] == Catch::Approx(tab[1 * C + j]).margin(1e-5));
}

// --- 1b-iv. ConvPool (Kim et al. char-CNN style: width-2 conv + relu + maxpool): forward + backward vs FD ---
// enc_w is packed as [2,C,C] (w0, then w1) -- reuses CharEncoder's own enc_w/enc_w_grad pointers, no new
// fields. Maxpool is differentiable almost everywhere (not AT a tie between window positions); the test
// data below is chosen with enough separation between per-channel window activations that finite
// differences (which can only see one side of an argmax flip near a tie) don't spuriously fail.
TEST_CASE("scratch encoder: ConvPool forward + backward match finite differences", "[scratch][embed]") {
    constexpr int C = 3;
    // Three fragments -> two width-2 windows: (0,1) and (1,2). Rows chosen with enough separation that
    // each channel's argmax window has a clear margin at the tested weights.
    std::vector<float> tab = {
        0.6f, -0.2f, 0.3f,     // row 0
       -0.4f,  0.5f, -0.1f,    // row 1
        0.2f,  0.1f,  0.7f,    // row 2
    };
    std::vector<float> W(static_cast<std::size_t>(2) * C * C);
    std::mt19937 rng(13);
    std::normal_distribution<float> nd(0.f, 0.5f);
    for (float& x : W) x = nd(rng);
    const std::vector<int> frags = {0, 1, 2};
    const float dout[C] = {0.8f, -0.6f, 1.1f};

    auto loss = [&](const std::vector<float>& tabv, const std::vector<float>& Wv) {
        float out[C];
        sub0::encode_slot(tabv.data(), C, frags, SlotEncoding::ConvPool, out, Wv.data());
        float L = 0.f; for (int c = 0; c < C; ++c) L += dout[c] * out[c]; return L;
    };

    std::vector<float> Wg(W.size(), 0.f), Tg(tab.size(), 0.f);
    sub0::encode_slot_bwd(dout, C, frags, SlotEncoding::ConvPool, Tg.data(), tab.data(), W.data(), Wg.data());

    const float eps = 1e-3f;
    for (std::size_t i = 0; i < W.size(); ++i) {
        auto wp = W, wm = W; wp[i] += eps; wm[i] -= eps;
        REQUIRE(Wg[i] == Catch::Approx((loss(tab, wp) - loss(tab, wm)) / (2 * eps)).margin(2e-2));
    }
    for (int f : frags) for (int k = 0; k < C; ++k) {
        auto tp = tab, tm = tab;
        tp[static_cast<std::size_t>(f) * C + k] += eps;
        tm[static_cast<std::size_t>(f) * C + k] -= eps;
        REQUIRE(Tg[static_cast<std::size_t>(f) * C + k] ==
                Catch::Approx((loss(tp, W) - loss(tm, W)) / (2 * eps)).margin(2e-2));
    }
}

// --- 1b-v. ConvPool is order-sensitive (unlike MeanPool/CharEncoder), and a lone fragment (no window)
// passes through unchanged -- same shape as the Hash order-sensitivity test above.
TEST_CASE("scratch encoder: ConvPool is order-sensitive, unlike MeanPool/CharEncoder", "[scratch][embed]") {
    constexpr int C = 4;
    std::vector<float> W(static_cast<std::size_t>(2) * C * C);
    std::mt19937 wr(21);
    std::normal_distribution<float> wnd(0.f, 0.5f);
    for (float& x : W) x = wnd(wr);

    std::mt19937 rng(11);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> tab(static_cast<std::size_t>(4) * C);
    for (float& x : tab) x = nd(rng);

    const std::vector<int> fwd = {0, 1, 2};
    const std::vector<int> rev = {2, 1, 0};

    float out_fwd[C], out_rev[C];
    sub0::encode_slot(tab.data(), C, fwd, SlotEncoding::ConvPool, out_fwd, W.data());
    sub0::encode_slot(tab.data(), C, rev, SlotEncoding::ConvPool, out_rev, W.data());
    bool differs = false;
    for (int j = 0; j < C; ++j) if (out_fwd[j] != Catch::Approx(out_rev[j]).margin(1e-5)) differs = true;
    REQUIRE(differs);

    // A lone fragment (no width-2 window) passes through as its own raw row, unmodified.
    const std::vector<int> solo = {1};
    float out_solo[C];
    sub0::encode_slot(tab.data(), C, solo, SlotEncoding::ConvPool, out_solo, W.data());
    for (int j = 0; j < C; ++j) REQUIRE(out_solo[j] == Catch::Approx(tab[1 * C + j]).margin(1e-5));
}

// --- 1b-vi. HRR (Holographic Reduced Representations, Plate 1995): circular-convolution binding of a
// fixed pseudo-random per-position "role" vector with each fragment, summed. No learned params (like
// Hash), but a mathematically DIFFERENT VSA binding operator (circular convolution vs. rotation) -- see
// scratch_slots.hpp's HRR comment and project memory meanpool-alternatives-prior-art-and-math.
TEST_CASE("scratch encoder: HRR forward + backward match finite differences", "[scratch][embed]") {
    constexpr int C = 6;
    std::vector<float> tab = {
        0.5f, -0.3f, 0.2f, 0.1f, -0.4f, 0.6f,     // row 0
       -0.1f,  0.4f, 0.6f, -0.2f, 0.3f, -0.5f,    // row 1
        0.2f, -0.6f, 0.1f,  0.4f, -0.1f, 0.3f,    // row 2
    };
    const std::vector<int> frags = {0, 1, 2};
    const float dout[C] = {1.0f, -0.5f, 0.7f, -0.2f, 0.3f, 0.4f};

    auto loss = [&](const std::vector<float>& tabv) {
        float out[C];
        sub0::encode_slot(tabv.data(), C, frags, SlotEncoding::HRR, out);
        float L = 0.f; for (int c = 0; c < C; ++c) L += dout[c] * out[c]; return L;
    };

    std::vector<float> Tg(tab.size(), 0.f);
    sub0::encode_slot_bwd(dout, C, frags, SlotEncoding::HRR, Tg.data());

    const float eps = 1e-3f;
    for (int f : frags) for (int k = 0; k < C; ++k) {
        auto tp = tab, tm = tab;
        tp[static_cast<std::size_t>(f) * C + k] += eps;
        tm[static_cast<std::size_t>(f) * C + k] -= eps;
        REQUIRE(Tg[static_cast<std::size_t>(f) * C + k] ==
                Catch::Approx((loss(tp) - loss(tm)) / (2 * eps)).margin(2e-2));
    }
}

// --- 1b-vii. HRR is order-sensitive (unlike MeanPool/CharEncoder). Unlike Hash, HRR has no "position 0 =
// identity" special case (the role vector at position 0 is a random binding, not an impulse), so this
// test checks order-sensitivity + a same-fragments-single-vs-repeated sanity check instead of a
// passthrough case.
TEST_CASE("scratch encoder: HRR is order-sensitive, unlike MeanPool/CharEncoder", "[scratch][embed]") {
    constexpr int C = 8;
    std::mt19937 rng(11);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> tab(static_cast<std::size_t>(4) * C);
    for (float& x : tab) x = nd(rng);

    const std::vector<int> fwd = {0, 1, 2};
    const std::vector<int> rev = {2, 1, 0};   // same SET, reversed order

    float out_fwd[C], out_rev[C];
    sub0::encode_slot(tab.data(), C, fwd, SlotEncoding::HRR, out_fwd);
    sub0::encode_slot(tab.data(), C, rev, SlotEncoding::HRR, out_rev);
    bool differs = false;
    for (int j = 0; j < C; ++j) if (out_fwd[j] != Catch::Approx(out_rev[j]).margin(1e-5)) differs = true;
    REQUIRE(differs);

    // Same MeanPool sanity floor as the Hash/ConvPool tests: confirm MeanPool stays order-invariant on
    // these SAME fragments (Approx, not `==` -- summation order can differ in float ULPs).
    float mp_fwd[C], mp_rev[C];
    sub0::encode_slot(tab.data(), C, fwd, SlotEncoding::MeanPool, mp_fwd);
    sub0::encode_slot(tab.data(), C, rev, SlotEncoding::MeanPool, mp_rev);
    for (int j = 0; j < C; ++j) REQUIRE(mp_fwd[j] == Catch::Approx(mp_rev[j]).margin(1e-5));

    // A single fragment's encoding is deterministic and depends only on ITS row + role[0] -- re-deriving
    // it independently (not via encode_slot) confirms the circular-convolution math itself, not just that
    // encode_slot is self-consistent.
    const std::vector<int> solo = {2};
    float out_solo[C];
    sub0::encode_slot(tab.data(), C, solo, SlotEncoding::HRR, out_solo);
    const std::vector<float>& roles = sub0::hrr_role_table(C);
    const float* role0 = roles.data();
    const float* filler = tab.data() + 2 * C;
    for (int n = 0; n < C; ++n) {
        float s = 0.f;
        for (int k = 0; k < C; ++k) { int idx = n - k; if (idx < 0) idx += C; s += role0[k] * filler[idx]; }
        REQUIRE(out_solo[n] == Catch::Approx(s).margin(1e-5));
    }
}

// --- 1c. Scalar: fixed scientific encoding of the value the fragments spell (the brain-swap re-entry) ---
namespace {
std::vector<int> frags_of(const std::string& s) {   // digit/sign/point/exp chars -> byte token ids
    std::vector<int> f; f.reserve(s.size());
    for (char c : s) f.push_back(static_cast<unsigned char>(c));
    return f;
}
}  // namespace

TEST_CASE("scratch encoder: Scalar parses value into (sign, exponent, mantissa)", "[scratch][embed]") {
    using sub0::SCALAR_MANT_DIGITS;

    {   // plain integer 12345 = 1.2345e4
        auto p = sub0::parse_scalar(std::span<const int>(frags_of("12345")));
        REQUIRE(p.ok); REQUIRE(p.sign == 1); REQUIRE(p.exp == 4);
        const int want[SCALAR_MANT_DIGITS] = {1, 2, 3, 4};
        for (int d = 0; d < SCALAR_MANT_DIGITS; ++d) REQUIRE(p.mant[d] == want[d]);
    }
    {   // negative scientific -6.02e23
        const auto f = frags_of("-6.02e23");
        auto p = sub0::parse_scalar(std::span<const int>(f));
        REQUIRE(p.ok); REQUIRE(p.sign == -1); REQUIRE(p.exp == 23);
        const int want[SCALAR_MANT_DIGITS] = {6, 0, 2, 0};
        for (int d = 0; d < SCALAR_MANT_DIGITS; ++d) REQUIRE(p.mant[d] == want[d]);
    }
    {   // 2^100 expanded (31 digits) -- a power maps into the SAME bounded encoding, E=30
        const auto f = frags_of("1267650600228229401496703205376");
        auto p = sub0::parse_scalar(std::span<const int>(f));
        REQUIRE(p.ok); REQUIRE(p.sign == 1); REQUIRE(p.exp == 30);
        const int want[SCALAR_MANT_DIGITS] = {1, 2, 6, 7};
        for (int d = 0; d < SCALAR_MANT_DIGITS; ++d) REQUIRE(p.mant[d] == want[d]);
    }
    {   // small fraction 0.00456 = 4.56e-3 (leading zeros handled)
        const auto f = frags_of("0.00456");
        auto p = sub0::parse_scalar(std::span<const int>(f));
        REQUIRE(p.ok); REQUIRE(p.sign == 1); REQUIRE(p.exp == -3);
        const int want[SCALAR_MANT_DIGITS] = {4, 5, 6, 0};
        for (int d = 0; d < SCALAR_MANT_DIGITS; ++d) REQUIRE(p.mant[d] == want[d]);
    }
    {   // exact zero -> sign 0
        const auto f = frags_of("0");
        auto p = sub0::parse_scalar(std::span<const int>(f));
        REQUIRE(p.ok); REQUIRE(p.sign == 0); REQUIRE(p.exp == 0);
    }
    {   // an un-evaluated symbolic form is NOT a plain literal -> ok=false (encoder emits a zero row)
        const auto f = frags_of("2^100");
        auto p = sub0::parse_scalar(std::span<const int>(f));
        REQUIRE_FALSE(p.ok);
    }
}

TEST_CASE("scratch encoder: Scalar writes the bounded layout and has no gradient", "[scratch][embed]") {
    using sub0::SCALAR_MANT_DIGITS; using sub0::SCALAR_EXP_SCALE; using sub0::SCALAR_AMP;
    constexpr int C = 8;
    const auto f = frags_of("-6.02e23");
    float out[C];
    sub0::encode_slot(nullptr, C, std::span<const int>(f), SlotEncoding::Scalar, out);   // tok_emb unused
    REQUIRE(out[0] == -1.f * SCALAR_AMP);                       // sign
    REQUIRE(out[1] == (23.f / SCALAR_EXP_SCALE) * SCALAR_AMP);  // exponent
    const int mant[SCALAR_MANT_DIGITS] = {6, 0, 2, 0};
    for (int d = 0; d < SCALAR_MANT_DIGITS; ++d) REQUIRE(out[2 + d] == (mant[d] / 9.f) * SCALAR_AMP);
    for (int j = 2 + SCALAR_MANT_DIGITS; j < C; ++j) REQUIRE(out[j] == 0.f);   // rest zero (bounded, sparse)

    // Backward is a no-op: a fixed encoding of discrete digit tokens carries no gradient.
    std::vector<float> tok_grad(4 * C, 1.f);   // sentinel; must stay untouched
    const float dout[C] = {1, 1, 1, 1, 1, 1, 1, 1};
    sub0::encode_slot_bwd(dout, C, std::span<const int>(f), SlotEncoding::Scalar, tok_grad.data());
    for (float g : tok_grad) REQUIRE(g == 1.f);
}

// --- 2. The engine actually feeds the content-derived embedding into forward ----------------------
// Differential: forwarding a sequence with a BOUND slot must match forwarding it with the slot's tok_emb
// row manually overwritten to the fragment mean (bindings off). The two feed the identical embedding, so
// the hidden states -- and every logit column except the slot's own (which a TIED head reads from the
// slot ROW, an intentionally-separate concern) -- are identical.
TEST_CASE("scratch embed: a bound slot embeds as its fragment mean in forward", "[scratch][embed]") {
    sub0::build_model();
    const int C = D_MODEL;
    float* P = sub0::params_ptr();                 // tok_emb is [VOCAB, C] at param offset 0
    const int slot = sub0::scratch_slot_id(0);
    const std::vector<int> frags = {'a', 'b', 'c'};   // byte tokens, all < VOCAB

    std::vector<std::vector<int>> tbl(sub0::SCRATCH_SLOT_COUNT);
    tbl[0] = frags;
    const sub0::ScratchBindings binds{ std::span<const std::vector<int>>(tbl), SlotEncoding::MeanPool };

    std::vector<float> mean(C, 0.f);
    for (int f : frags) for (int j = 0; j < C; ++j) mean[j] += P[static_cast<std::size_t>(f) * C + j];
    for (int j = 0; j < C; ++j) mean[j] /= static_cast<float>(frags.size());

    const std::vector<int> seq = {'x', slot, 'y'};

    // A) content-derived: bind the slot, forward.
    sub0::set_scratch_bindings(&binds);
    sub0::graph_reset();
    sub0::Node* a = sub0::forward(seq.data(), static_cast<int>(seq.size()));
    const std::vector<float> logits_scratch(a->data.begin(), a->data.end());
    sub0::set_scratch_bindings(nullptr);

    // B) plain: overwrite tok_emb[slot] := mean, forward with no bindings, then restore.
    std::vector<float> saved(P + static_cast<std::size_t>(slot) * C, P + static_cast<std::size_t>(slot) * C + C);
    for (int j = 0; j < C; ++j) P[static_cast<std::size_t>(slot) * C + j] = mean[j];
    sub0::graph_reset();
    sub0::Node* b = sub0::forward(seq.data(), static_cast<int>(seq.size()));
    const std::vector<float> logits_plain(b->data.begin(), b->data.end());
    for (int j = 0; j < C; ++j) P[static_cast<std::size_t>(slot) * C + j] = saved[j];

    REQUIRE(logits_scratch.size() == logits_plain.size());
    const int T = static_cast<int>(seq.size());
    bool any_checked = false;
    for (int t = 0; t < T; ++t)
        for (int v = 0; v < VOCAB; ++v) {
            if (v == slot) continue;               // tied-head reads the slot ROW -> excluded by design
            const std::size_t i = static_cast<std::size_t>(t) * VOCAB + v;
            REQUIRE(logits_scratch[i] == Catch::Approx(logits_plain[i]).margin(1e-4));
            any_checked = true;
        }
    REQUIRE(any_checked);
}

// --- 2b. train_batch actually applies the per-window content bindings (the training-integration path) --
TEST_CASE("scratch embed: train_batch applies per-window content bindings", "[scratch][embed]") {
    sub0::build_model();
    const int slot = sub0::scratch_slot_id(0);
    // A single 3-position window (needs 4 tokens: 3 inputs + the last shifted target). The slot is input 1.
    const std::vector<int> data = {'x', slot, 'y', 'z'};
    const std::vector<std::size_t> starts = {0};

    std::vector<std::vector<int>> tbl(sub0::SCRATCH_SLOT_COUNT);
    tbl[0] = {'a', 'b', 'c'};
    const sub0::ScratchBindings binds{ std::span<const std::vector<int>>(tbl), SlotEncoding::MeanPool };
    const sub0::ScratchBindings* wb[1] = { &binds };

    // Same weights (train_batch updates only grads, not params); the only difference is the slot's
    // embedding source, so the forward loss must differ when the content bindings are threaded in.
    const float loss_plain = sub0::train_batch(data.data(), starts.data(), 1, 3, nullptr, nullptr, nullptr);
    const float loss_ce    = sub0::train_batch(data.data(), starts.data(), 1, 3, nullptr, nullptr, wb);
    REQUIRE(loss_plain != loss_ce);
}

// --- 3. No bindings => byte-identical to the plain path -------------------------------------------
TEST_CASE("scratch embed: no bindings leaves forward unchanged", "[scratch][embed]") {
    sub0::build_model();
    const int slot = sub0::scratch_slot_id(0);
    const std::vector<int> seq = {'x', slot, 'y'};

    sub0::set_scratch_bindings(nullptr);
    sub0::graph_reset();
    sub0::Node* a = sub0::forward(seq.data(), static_cast<int>(seq.size()));
    const std::vector<float> l0(a->data.begin(), a->data.end());

    // An EMPTY binding table (slot present but unbound) must also be inert -- bound() is false.
    std::vector<std::vector<int>> empty(sub0::SCRATCH_SLOT_COUNT);
    const sub0::ScratchBindings none{ std::span<const std::vector<int>>(empty), SlotEncoding::MeanPool };
    sub0::set_scratch_bindings(&none);
    sub0::graph_reset();
    sub0::Node* b = sub0::forward(seq.data(), static_cast<int>(seq.size()));
    const std::vector<float> l1(b->data.begin(), b->data.end());
    sub0::set_scratch_bindings(nullptr);

    REQUIRE(l0.size() == l1.size());
    for (std::size_t i = 0; i < l0.size(); ++i) REQUIRE(l0[i] == l1[i]);   // bit-identical
}
