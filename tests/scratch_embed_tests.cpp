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
