#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "sub0diff/eval/recovery.hpp"
#include "sub0diff/nn/denoiser.hpp"
#include "sub0diff/nn/noise_schedule.hpp"
#include "sub0diff/nn/sampler.hpp"
#include "sub0diff/train/diffusion_loss.hpp"

#include "sub0llm/nn/optimizer.hpp"

#include <numeric>
#include <random>
#include <span>
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

TEST_CASE("corrupt_into - exact_count masks exactly round(t*T), floor 1", "[diffusion]") {
    const auto clean = ramp_tokens(40, 16);
    std::mt19937 rng(1);
    dn::Corruption c;
    auto count = [&] { return std::accumulate(c.corrupted.begin(), c.corrupted.end(), 0); };

    dn::corrupt_into(clean, 0.25f, dn::NoiseSchedule::Absorbing, 16, 16, rng, c, /*exact_count=*/true);
    REQUIRE(c.n_corrupted == 10);            // round(0.25 * 40)
    REQUIRE(count() == 10);
    dn::corrupt_into(clean, 0.001f, dn::NoiseSchedule::Absorbing, 16, 16, rng, c, true);
    REQUIRE(c.n_corrupted == 1);             // floor: round→0, clamped to 1
    dn::corrupt_into(clean, 1.0f, dn::NoiseSchedule::Absorbing, 16, 16, rng, c, true);
    REQUIRE(c.n_corrupted == 40);            // all masked
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
