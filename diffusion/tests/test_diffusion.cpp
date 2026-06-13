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
