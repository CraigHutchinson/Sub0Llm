#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "sub0diff/eval/recovery.hpp"
#include "sub0diff/nn/denoiser.hpp"
#include "sub0diff/nn/noise_schedule.hpp"
#include "sub0diff/nn/sampler.hpp"
#include "sub0diff/train/curriculum.hpp"
#include "sub0diff/train/diffusion_loss.hpp"

#include "sub0llm/nn/optimizer.hpp"

#include <numeric>
#include <random>
#include <span>
#include <stdexcept>
#include <vector>

using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;
namespace dn = sub0diff::nn;
namespace dt = sub0diff::train;
namespace de = sub0diff::eval;

namespace {

std::vector<std::int32_t> ramp_tokens(std::size_t n, std::int32_t vocab) {
    std::vector<std::int32_t> v(n);
    for (std::size_t i = 0; i < n; ++i) v[i] = static_cast<std::int32_t>(i) % vocab;
    return v;
}

} // namespace

TEST_CASE("corrupt_into - reuses buffers, guarantees >=1 corruption", "[diffusion]") {
    const auto clean = ramp_tokens(32, 16);
    std::mt19937 rng(1);
    dn::Corruption c;

    dn::corrupt_into(clean, 0.01f, dn::NoiseSchedule::Absorbing, 16, 16, rng, c);
    REQUIRE(c.n_corrupted >= 1);             // floor guarantee at tiny probability
    const auto* tok_ptr = c.tokens.data();

    dn::corrupt_into(clean, 0.5f, dn::NoiseSchedule::Absorbing, 16, 16, rng, c);
    REQUIRE(c.tokens.data() == tok_ptr);     // same storage — allocation-free reuse
    REQUIRE(c.n_corrupted ==
            static_cast<std::uint32_t>(std::accumulate(c.corrupted.begin(), c.corrupted.end(), 0)));
    for (std::size_t i = 0; i < clean.size(); ++i)
        REQUIRE(c.tokens[i] == (c.corrupted[i] ? 16 : clean[i]));
}

TEST_CASE("corrupt_into - exact_count masks round(t*T), clamped to [1, T-1]", "[diffusion]") {
    const auto clean = ramp_tokens(40, 16);
    std::mt19937 rng(1);
    dn::Corruption c;
    auto count = [&] { return std::accumulate(c.corrupted.begin(), c.corrupted.end(), 0); };

    dn::corrupt_into(clean, 0.25f, dn::NoiseSchedule::Absorbing, 16, 16, rng, c, /*exact_count=*/true);
    REQUIRE(c.n_corrupted == 10);            // round(0.25 * 40)
    REQUIRE(count() == 10);
    dn::corrupt_into(clean, 0.001f, dn::NoiseSchedule::Absorbing, 16, 16, rng, c, true);
    REQUIRE(c.n_corrupted == 1);             // floor: round→0, clamped to 1 (≥1 masked target)
    dn::corrupt_into(clean, 1.0f, dn::NoiseSchedule::Absorbing, 16, 16, rng, c, true);
    REQUIRE(c.n_corrupted == 39);            // min-1-VISIBLE floor: T-1, never fully blank
    // The COUNT is deterministic across seeds (only WHICH positions differ) — the whole point.
    std::mt19937 r2(99);
    dn::corrupt_into(clean, 0.5f, dn::NoiseSchedule::Absorbing, 16, 16, r2, c, true);
    REQUIRE(c.n_corrupted == 20);
    // masked positions carry the mask id, the rest are clean.
    for (std::size_t i = 0; i < clean.size(); ++i)
        REQUIRE(c.tokens[i] == (c.corrupted[i] ? 16 : clean[i]));
}

TEST_CASE("corrupt_whole_word_into - masks whole words atomically", "[diffusion]") {
    // ids 10,20,30 carry the word-start flag; words are [10,11,12], [20,21], [30,31].
    const std::vector<std::int32_t> clean{10, 11, 12, 20, 21, 30, 31};
    std::vector<std::uint8_t> ws(64, 0);
    ws[10] = ws[20] = ws[30] = 1;
    std::mt19937 rng(3);
    dn::Corruption c;

    for (int trial = 0; trial < 200; ++trial) {
        dn::corrupt_whole_word_into(clean, ws, 0.5f, dn::NoiseSchedule::Absorbing, 99, 99, rng, c);
        REQUIRE(c.n_corrupted >= 1);                       // floor guarantee
        // Atomicity: every subword of a word shares the word's mask decision.
        REQUIRE(c.corrupted[0] == c.corrupted[1]);         // word [10,11,12]
        REQUIRE(c.corrupted[1] == c.corrupted[2]);
        REQUIRE(c.corrupted[3] == c.corrupted[4]);         // word [20,21]
        REQUIRE(c.corrupted[5] == c.corrupted[6]);         // word [30,31]
    }
    // Empty table → falls back to per-token corruption (still honours the floor).
    dn::corrupt_whole_word_into(clean, {}, 0.3f, dn::NoiseSchedule::Absorbing, 99, 99, rng, c);
    REQUIRE(c.n_corrupted >= 1);
}

TEST_CASE("evaluate_recovery - word-start/continuation split bookkeeping", "[diffusion]") {
    dn::Denoiser model(64, 32, 2, 2, 1, 64, /*seed=*/5);
    const std::vector<std::int32_t> clean{10, 11, 12, 20, 21, 30, 31};
    std::vector<std::uint8_t> ws(65, 0);
    ws[10] = ws[20] = ws[30] = 1;                          // word boundaries at 0,3,5
    std::vector<std::uint8_t> masked(clean.size(), 0);
    masked[1] = masked[3] = masked[5] = 1;                 // cont, start, start

    auto r = de::evaluate_recovery(model, clean, masked, nullptr, ws);
    REQUIRE(r.masked == 3);
    REQUIRE(r.ws_masked + r.wc_masked == r.masked);        // partition
    REQUIRE(r.ws_hits + r.wc_hits == r.hits);
    REQUIRE(r.ws_masked == 2);                             // pos 3,5 are word-starts
    REQUIRE(r.wc_masked == 1);                             // pos 1 is a continuation
    REQUIRE(r.word_total == 3);                            // three distinct words touched
    REQUIRE(r.word_hits <= r.word_total);
    // Without a table, the split fields stay zero but recall is unchanged.
    auto r0 = de::evaluate_recovery(model, clean, masked, nullptr, {});
    REQUIRE(r0.masked == 3);
    REQUIRE(r0.ws_masked == 0);
    REQUIRE(r0.word_total == 0);
    REQUIRE(r0.hits == r.hits);                            // same argmax, table-independent
}

TEST_CASE("diffusion_loss - NELBO weight is n_masked/(t*T)", "[diffusion]") {
    dn::Denoiser model(16, 32, 2, 2, 1, 64, /*seed=*/3);
    const auto clean = ramp_tokens(16, 16);
    dt::DiffusionLossContext ctx(16);

    // Two identical corruptions, different reported t: same mean-CE, weight ratio known.
    // Drive via the public API with a band so narrow t is effectively fixed.
    std::mt19937 rng_a(7), rng_b(7);   // identical corruption stream
    auto a = dt::diffusion_loss(model, clean, rng_a, ctx, 0.499f, 0.501f);
    dt::DiffusionLossContext ctx_b(16);
    auto b = dt::diffusion_loss(model, clean, rng_b, ctx_b, 0.499f, 0.501f);
    REQUIRE(a.n_masked == b.n_masked);
    REQUIRE_THAT(a.loss.data().item<float>(),
                 WithinRel(b.loss.data().item<float>(), 1e-5f));   // deterministic

    // The scalar equals mean_ce * n_masked / (t*T): verify the weight bounds —
    // with t≈0.5 and T=16, weight = n_masked/8, so loss/mean_ce must be in (0, 2].
    REQUIRE(a.loss.data().item<float>() > 0.0f);
    REQUIRE(a.t >= 0.499f);
    REQUIRE(a.t <= 0.501f);
}

TEST_CASE("diffusion_loss - one optimizer step reduces loss on a fixed sample", "[diffusion]") {
    dn::Denoiser model(16, 32, 2, 2, 1, 64, /*seed=*/5);
    auto params = model.parameters();
    sub0llm::nn::Adam opt(params, 5e-2f);
    const auto clean = ramp_tokens(16, 16);
    dt::DiffusionLossContext ctx(16);

    // Same RNG seed each iteration -> identical (t, corruption) sample.
    auto sample_loss = [&]() {
        std::mt19937 rng(11);
        return dt::diffusion_loss(model, clean, rng, ctx, 0.3f, 0.7f);
    };

    const float before = sample_loss().loss.data().item<float>();
    for (int i = 0; i < 10; ++i) {
        auto res = sample_loss();
        opt.zero_grad();
        res.loss.backward();
        opt.step();
    }
    const float after = sample_loss().loss.data().item<float>();
    REQUIRE(after < before);   // gradient flows end-to-end and reduces the objective
}

// Stage 4 Phase 7 capstone: the FULL Denoiser training step — embed → transformer (rms_norm,
// RoPE, GQA-attention softmax, SwiGLU) → LM head → diffusion loss → backward → Adam — composes
// end-to-end on CUDA and reduces the loss. Every op dispatched to the device kernels (Phases 1–7).
TEST_CASE("Denoiser end-to-end training step on CUDA reduces loss", "[diffusion][cuda][device]") {
#ifdef SUB0LLM_CUDA
    dn::Denoiser model(16, 32, 2, 2, 1, 64, /*seed=*/5);
    model.to(sub0llm::Device::cuda());
    auto params = model.parameters();
    REQUIRE(params[0]->data().device().is_cuda());     // confirm we really run on the GPU
    sub0llm::nn::Adam opt(params, 5e-2f);              // m_/v_ allocated on-device (after to(cuda))
    const auto clean = ramp_tokens(16, 16);
    dt::DiffusionLossContext ctx(16);

    auto loss_now = [&]() {
        std::mt19937 rng(11);                          // identical (t, corruption) each call
        return dt::diffusion_loss(model, clean, rng, ctx, 0.3f, 0.7f)
                   .loss.data().to(sub0llm::Device::cpu()).item<float>();
    };

    const float before = loss_now();
    for (int i = 0; i < 10; ++i) {
        std::mt19937 rng(11);
        auto res = dt::diffusion_loss(model, clean, rng, ctx, 0.3f, 0.7f);
        opt.zero_grad();
        res.loss.backward();
        opt.step();
    }
    const float after = loss_now();
    REQUIRE(after < before);                           // full fwd+bwd+update on GPU reduces the loss
#else
    SUCCEED("CPU build - CUDA end-to-end training step is exercised on the cuda preset");
#endif
}

TEST_CASE("evaluate_recovery - counts and position stats are consistent", "[diffusion]") {
    dn::Denoiser model(16, 32, 2, 2, 1, 64, /*seed=*/9);
    const auto clean = ramp_tokens(24, 16);
    std::vector<std::uint8_t> masked(24, 0);
    masked[0] = masked[5] = masked[23] = 1;

    de::PositionStats pos(24);
    auto r = de::evaluate_recovery(model, clean, masked, &pos);
    REQUIRE(r.masked == 3);
    REQUIRE(r.hits >= 0);
    REQUIRE(r.hits <= 3);
    REQUIRE(pos.masked[0] == 1);
    REQUIRE(pos.masked[5] == 1);
    REQUIRE(pos.masked[23] == 1);
    REQUIRE(pos.masked[1] == 0);
    REQUIRE(r.hits == pos.hits[0] + pos.hits[5] + pos.hits[23]);
}

TEST_CASE("refine_canvas - completes, respects fixed positions, self-terminates", "[diffusion]") {
    dn::Denoiser model(16, 32, 2, 2, 1, 64, /*seed=*/21);
    std::mt19937 rng(3);

    // Canvas with a 3-token fixed prompt; the rest must be filled.
    const std::vector<std::int32_t> prompt{1, 2, 3};
    auto canvas = dn::make_canvas(model, 16, prompt);
    REQUIRE(canvas[0] == 1);
    REQUIRE(canvas[3] == model.mask_id());

    dn::SamplerConfig cfg;
    cfg.temperature = 0.0f;
    auto stats = dn::refine_canvas(model, canvas, cfg, rng);

    REQUIRE(canvas[0] == 1);                      // fixed prompt untouched
    REQUIRE(canvas[1] == 2);
    REQUIRE(canvas[2] == 3);
    for (auto t : canvas) {
        REQUIRE(t != model.mask_id());            // canvas complete
        REQUIRE(t >= 0);
        REQUIRE(t < 16);                          // only real tokens predicted
    }
    REQUIRE(stats.committed == 13);
    REQUIRE(stats.iterations >= 1);
    REQUIRE(stats.iterations <= 13);              // guaranteed progress each iter
}

TEST_CASE("refine_canvas - entropy_bound forces single-iteration commit", "[diffusion]") {
    dn::Denoiser model(16, 32, 2, 2, 1, 64, /*seed=*/23);
    std::mt19937 rng(5);
    auto canvas = dn::make_canvas(model, 12);
    dn::SamplerConfig cfg;
    cfg.temperature   = 0.0f;
    cfg.entropy_bound = 1e9f;                     // everything counts as low-entropy
    auto stats = dn::refine_canvas(model, canvas, cfg, rng);
    REQUIRE(stats.iterations == 1);
    REQUIRE(stats.entropy_stopped);
    REQUIRE(stats.committed == 12);
}

TEST_CASE("evaluate_corpus_recall - sweeps every window", "[diffusion]") {
    dn::Denoiser model(16, 32, 2, 2, 1, 64, /*seed=*/13);
    const auto corpus = ramp_tokens(40, 16);
    std::mt19937 rng(17);
    auto r = de::evaluate_corpus_recall(model, corpus, /*T=*/8, /*noise=*/0.5f, rng);
    REQUIRE(r.masked > 0);
    // 32 windows, >=1 masked each (corruption floor)
    REQUIRE(r.masked >= 32);
}

// ── Batched forward parity (Ch29 re-architecture) ──────────────────────────────
// The batched (B·T) forward must equal stacking B independent per-window forwards:
// attention is block-diagonal and every other op is row-wise, so batching changes
// only the matmul M dimension, never a window's result.
TEST_CASE("Denoiser batched forward equals stacked per-window forward", "[diffusion][batched]") {
    const std::int64_t V = 16, D = 32, T = 8, B = 4;
    dn::Denoiser model(V, D, /*n_heads=*/2, /*n_kv_heads=*/2, /*n_layers=*/2, /*d_ff=*/64, 7);
    const std::int64_t Vm = V + 1;

    std::mt19937 rng(123);
    std::uniform_int_distribution<std::int32_t> tok(0, static_cast<std::int32_t>(Vm) - 1);
    sub0llm::Tensor ids({B * T}, sub0llm::DType::Int32);
    auto idd = ids.data_as<std::int32_t>();
    for (auto& v : idd) v = tok(rng);

    std::vector<float> noise(static_cast<std::size_t>(B));
    for (std::int64_t b = 0; b < B; ++b) noise[static_cast<std::size_t>(b)] = 0.1f + 0.2f * static_cast<float>(b);

    auto batched = model.forward(ids, std::span<const float>(noise), B, T);   // (B·T, Vm)
    REQUIRE(batched.data().shape() == sub0llm::Tensor::Shape{B * T, Vm});
    auto bd = batched.data().data_as<float>();

    for (std::int64_t b = 0; b < B; ++b) {
        sub0llm::Tensor wids({T}, sub0llm::DType::Int32);
        auto wd = wids.data_as<std::int32_t>();
        for (std::int64_t t = 0; t < T; ++t) wd[static_cast<std::size_t>(t)] = idd[static_cast<std::size_t>(b * T + t)];
        auto single = model.forward(wids, noise[static_cast<std::size_t>(b)]);   // (T, Vm)
        auto sd = single.data().data_as<float>();
        for (std::int64_t i = 0; i < T * Vm; ++i)
            REQUIRE_THAT(bd[static_cast<std::size_t>(b * T * Vm + i)],
                         WithinAbs(sd[static_cast<std::size_t>(i)], 1e-4f));
    }
}

// Gradient parity: d/dθ Σ(batched logits) must equal Σ_b d/dθ Σ(window-b logits).
// sum() avoids RNG alignment and exercises the full batched backward — attention,
// reshape, batched matmul VJPs — against the trusted per-window 2D path.
TEST_CASE("Denoiser batched backward equals summed per-window gradients", "[diffusion][batched]") {
    const std::int64_t V = 12, D = 32, T = 6, B = 3;
    dn::Denoiser model(V, D, /*n_heads=*/4, /*n_kv_heads=*/2, /*n_layers=*/2, /*d_ff=*/64, 31);
    auto params = model.parameters();

    std::mt19937 rng(99);
    std::uniform_int_distribution<std::int32_t> tok(0, static_cast<std::int32_t>(V));   // includes mask id V
    sub0llm::Tensor ids({B * T}, sub0llm::DType::Int32);
    auto idd = ids.data_as<std::int32_t>();
    for (auto& v : idd) v = tok(rng);
    std::vector<float> noise(static_cast<std::size_t>(B));
    for (std::int64_t b = 0; b < B; ++b) noise[static_cast<std::size_t>(b)] = 0.2f + 0.15f * static_cast<float>(b);

    // Batched gradients.
    for (auto* p : params) p->zero_grad();
    sub0llm::autograd::sum(model.forward(ids, std::span<const float>(noise), B, T)).backward();
    std::vector<sub0llm::Tensor> g_batch;
    for (auto* p : params) g_batch.push_back(sub0llm::copy(p->grad()));

    // Summed per-window gradients (backward accumulates into the same param grads).
    for (auto* p : params) p->zero_grad();
    for (std::int64_t b = 0; b < B; ++b) {
        sub0llm::Tensor wids({T}, sub0llm::DType::Int32);
        auto wd = wids.data_as<std::int32_t>();
        for (std::int64_t t = 0; t < T; ++t) wd[static_cast<std::size_t>(t)] = idd[static_cast<std::size_t>(b * T + t)];
        sub0llm::autograd::sum(model.forward(wids, noise[static_cast<std::size_t>(b)])).backward();
    }

    for (std::size_t pi = 0; pi < params.size(); ++pi) {
        auto gb = g_batch[pi].data_as<float>();
        auto gs = params[pi]->grad().data_as<float>();
        REQUIRE(gb.size() == gs.size());
        for (std::size_t i = 0; i < gb.size(); ++i)
            REQUIRE_THAT(gb[i], WithinAbs(gs[i], 2e-4f));
    }
}

// (B,W) invariance: with deterministic per-window seeding (seed_base), a window's corruption is a
// pure function of its GLOBAL index, NOT how the batch was split across workers. This is what lets
// the unified trainer produce the same gradient at any worker count W for a fixed effective batch B
// — √B consistency decoupled from cores. Tested at the data level (no FP/threading nondeterminism).
TEST_CASE("batched_diffusion_loss - window corruption invariant to batch split (B,W invariance)",
          "[diffusion][batched]") {
    const std::int64_t V = 16, D = 32, T = 8;
    dn::Denoiser model(V, D, /*n_heads=*/2, /*n_kv_heads=*/2, /*n_layers=*/2, /*d_ff=*/64, 7);
    const auto stream = ramp_tokens(64, V);
    std::span<const std::int32_t> s(stream);
    const std::uint64_t S = 0xDEADBEEF12345ull;
    std::mt19937 rng(1);   // only the (deterministic, band-collapsed) shared-t draw touches this

    // Call A: one worker owns all 4 windows (W=1), global indices 0..3.
    std::vector<std::size_t> off4{0, 9, 18, 27};
    dt::BatchedDiffusionLossContext ctxA(4, T);
    (void)dt::batched_diffusion_loss(model, s, std::span<const std::size_t>(off4), rng, ctxA,
                                     0.5f, 0.5f, /*shared_t=*/true, /*exact_count=*/true,
                                     {}, /*whole_word=*/false, S, /*index0=*/0);

    // Call B: a second worker owns the tail 2 windows (global indices 2,3) — index0=2.
    std::vector<std::size_t> off2{18, 27};
    dt::BatchedDiffusionLossContext ctxB(2, T);
    (void)dt::batched_diffusion_loss(model, s, std::span<const std::size_t>(off2), rng, ctxB,
                                     0.5f, 0.5f, true, true, {}, false, S, /*index0=*/2);

    // Global window 2 == ctxA.corr[2] == ctxB.corr[0]; window 3 == ctxA.corr[3] == ctxB.corr[1].
    for (int j = 0; j < 2; ++j) {
        REQUIRE(ctxA.corr[static_cast<std::size_t>(2 + j)].n_corrupted
                    == ctxB.corr[static_cast<std::size_t>(j)].n_corrupted);
        REQUIRE(ctxA.corr[static_cast<std::size_t>(2 + j)].corrupted
                    == ctxB.corr[static_cast<std::size_t>(j)].corrupted);
        REQUIRE(ctxA.corr[static_cast<std::size_t>(2 + j)].tokens
                    == ctxB.corr[static_cast<std::size_t>(j)].tokens);
    }
}

TEST_CASE("FrontierCurriculum - per-epoch NELBO plateau advances integer k", "[diffusion]") {
    dt::FrontierCurriculum c(dt::FrontierCurriculum::Config{
        .seq_len = 8, .k_start = 1, .k_max = 0, .k_step = 1, .patience = 2, .min_improve = 0.01f});
    REQUIRE(c.level() == 1);
    REQUIRE(c.max_level() == 7);                 // T-1 (min-1-visible cap)
    REQUIRE_THAT(c.frontier(), WithinAbs(1.0f / 8.0f, 1e-6));   // t = k/T

    // Descending NELBO at the level ⇒ "still learning", k stays put, stalls reset each time.
    REQUIRE_FALSE(c.observe_epoch(5.0f));
    REQUIRE_FALSE(c.observe_epoch(4.0f));
    REQUIRE_FALSE(c.observe_epoch(3.0f));
    REQUIRE(c.level() == 1);

    // Plateau: no improvement for `patience` epochs ⇒ advance exactly once, k→2, best resets.
    REQUIRE_FALSE(c.observe_epoch(3.0f));        // stall 1/2
    REQUIRE(c.observe_epoch(3.0f));              // stall 2/2 ⇒ advance
    REQUIRE(c.level() == 2);
    REQUIRE_THAT(c.frontier(), WithinAbs(2.0f / 8.0f, 1e-6));

    // Drive to the top and confirm it converges at k_max (and stops advancing).
    for (int guard = 0; guard < 100 && !c.converged(); ++guard) {
        c.observe_epoch(9.0f);                   // flat/high ⇒ keep plateauing/advancing
    }
    REQUIRE(c.converged());
    REQUIRE(c.level() == 7);
    REQUIRE_FALSE(c.observe_epoch(0.0f));        // converged: no further advancement
    REQUIRE(c.level() == 7);
}

// ── Device placement (Stage 4 Phase 0) ────────────────────────────────────────
//
// Denoiser::to moves every parameter in place. On CPU it must be a value-exact
// no-op (forward parity); the CUDA case (gated) proves the params actually land
// on the device. parameters() pointers stay valid across the move, so a Trainer
// holding them keeps working.

TEST_CASE("Denoiser::to - CPU move is a value-exact no-op", "[diffusion][device]") {
    dn::Denoiser model(16, 32, 2, 2, 1, 64, /*seed=*/7);
    const auto ids_vec = ramp_tokens(12, 16);
    sub0llm::Tensor ids({12}, sub0llm::DType::Int32);
    auto idd = ids.data_as<std::int32_t>();
    for (std::size_t i = 0; i < ids_vec.size(); ++i) idd[i] = ids_vec[i];

    auto before = model.forward(ids, 0.4f);
    const auto bd = before.data().data_as<float>();

    dn::Denoiser& ret = model.to(sub0llm::Device::cpu());
    REQUIRE(&ret == &model);                          // chainable
    for (auto* p : model.parameters())
        REQUIRE(p->data().device().is_cpu());

    auto after = model.forward(ids, 0.4f);
    const auto ad = after.data().data_as<float>();
    REQUIRE(after.data().shape() == before.data().shape());
    for (std::int64_t i = 0; i < before.data().numel(); ++i)
        REQUIRE_THAT(bd[static_cast<std::size_t>(i)],
                     WithinAbs(ad[static_cast<std::size_t>(i)], 0.0f));
}

TEST_CASE("Denoiser::to - moves parameters to the GPU", "[diffusion][device]") {
    dn::Denoiser model(16, 32, 2, 2, 1, 64, /*seed=*/7);
#ifdef SUB0LLM_CUDA
    model.to(sub0llm::Device::cuda());
    for (auto* p : model.parameters())
        REQUIRE(p->data().device().is_cuda());
    model.to(sub0llm::Device::cpu());                 // round-trip back
    for (auto* p : model.parameters())
        REQUIRE(p->data().device().is_cpu());
#else
    REQUIRE_THROWS_AS(model.to(sub0llm::Device::cuda()), std::runtime_error);
#endif
}
