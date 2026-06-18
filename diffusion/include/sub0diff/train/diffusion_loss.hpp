#pragma once

// diffusion_loss.hpp (Ch29) — the FORMAL masked-diffusion training objective.
//
// Ch28 trained the denoiser with an ad-hoc masked cross-entropy at a curriculum-
// chosen noise level. This header is the principled version: the negative evidence
// lower bound (NELBO) of an absorbing-state discrete diffusion model, which
// Rao-Blackwellizes to a strikingly simple form (MDLM, Sahoo et al.; LLaDA):
//
//     L  =  E_{t ~ U(0,1]}  E_{x_t ~ q(·|x_0, t)}  [ (1/t) · Σ_{i masked} -log p_θ(x_0[i] | x_t) ]
//
// i.e. sample a noise level t, mask each position independently with probability t,
// and weight the summed cross-entropy of the masked positions by 1/t. Intuition for
// the 1/t: a low-noise sample masks few tokens, so each masked token observed there
// is a rarer, more informative event — the weight makes the estimator unbiased for
// the integral over all noise levels.
//
// Our weighted_cross_entropy returns the MEAN over masked positions (it normalizes
// by the weight sum), so the per-sample NELBO term is recovered by scaling with
//     w  =  n_masked / (t · T)
// (≈1 in expectation since E[n_masked] = t·T — the weight only corrects the
// sampling variance, which is exactly the Rao-Blackwellization).
//
// Ch28's measured curriculum findings (frontier-point sampling, token-gated dwell)
// are a practical *variant* of choosing the distribution of t; this header also
// accepts a [t_min, t_max] band so a caller can reproduce those schedules. The
// formal objective is t ~ U(t_min, 1].

#include "sub0diff/nn/denoiser.hpp"
#include "sub0diff/nn/noise_schedule.hpp"

#include "sub0llm/autograd/ops.hpp"
#include "sub0llm/core/tensor.hpp"

#include <algorithm>
#include <cstdint>
#include <random>
#include <span>
#include <vector>

namespace sub0diff::train {

// Reusable per-step buffers — hoist one of these out of the training loop and the
// loss computation performs no heap allocation beyond the autograd graph.
struct DiffusionLossContext {
    sub0llm::Tensor  ids_input, ids_clean, weights;
    nn::Corruption   corruption;

    explicit DiffusionLossContext(std::int64_t T)
        : ids_input({T}, sub0llm::DType::Int32),
          ids_clean({T}, sub0llm::DType::Int32),
          weights({T}, sub0llm::DType::Float32) {}
};

struct DiffusionLossResult {
    sub0llm::autograd::Variable loss;       // scalar — NELBO estimate for this sample
    float                       t = 0.0f;   // sampled noise level
    std::uint32_t               n_masked = 0;
    float                       mean_ce = 0.0f;  // RAW mean masked CE (nats), before 1/t scaling —
                                                 // the per-token NLL, for the per-t diagnostic curve
};

// One NELBO Monte-Carlo sample on a clean window.
//   t is drawn from U(t_min, t_max]; pass t_max < 1 to train under a curriculum
//   ceiling (Ch28-style), or the defaults for the formal objective.
template<class RNG>
[[nodiscard]] DiffusionLossResult diffusion_loss(const nn::Denoiser& model,
                                                 std::span<const std::int32_t> clean,
                                                 RNG& rng,
                                                 DiffusionLossContext& ctx,
                                                 float t_min = 0.02f,
                                                 float t_max = 1.0f,
                                                 std::span<const std::uint8_t> is_word_start = {},
                                                 bool whole_word = false,
                                                 bool exact_count = false,
                                                 bool contiguous = false) {
    namespace ag = sub0llm::autograd;
    const auto T = static_cast<float>(clean.size());

    std::uniform_real_distribution<float> t_dist(t_min, t_max);
    const float t = t_dist(rng);

    if (whole_word)
        nn::corrupt_whole_word_into(clean, is_word_start, t, spec::NoiseSchedule::Absorbing,
                                    model.mask_id(), model.real_vocab(), rng, ctx.corruption);
    else
        nn::corrupt_into(clean, t, spec::NoiseSchedule::Absorbing,
                         model.mask_id(), model.real_vocab(), rng, ctx.corruption, exact_count, contiguous);
    const auto& corr = ctx.corruption;

    std::ranges::copy(corr.tokens, ctx.ids_input.data_as<std::int32_t>().begin());
    std::ranges::copy(clean, ctx.ids_clean.data_as<std::int32_t>().begin());
    auto w = ctx.weights.data_as<float>();
    for (std::size_t i = 0; i < clean.size(); ++i) w[i] = corr.corrupted[i];

    const float actual_noise = static_cast<float>(corr.n_corrupted) / T;
    ag::Variable logits = model.forward(ctx.ids_input, actual_noise);
    ag::Variable mean_ce = ag::weighted_cross_entropy(logits, ctx.ids_clean, ctx.weights);
    const float mean_ce_val = mean_ce.data().item<float>();

    // mean-over-masked → NELBO term: scale by n_masked / (t·T).
    const float nelbo_w = static_cast<float>(corr.n_corrupted) / (t * T);
    return {ag::scale(mean_ce, nelbo_w), t, corr.n_corrupted, mean_ce_val};
}

// ── Batched objective (Ch29 re-architecture) ──────────────────────────────────
// One forward+backward over B windows stacked as (B·T) rows — the keystone that
// turns tiny per-window passes into big GEMMs and amortizes the autograd/allocation
// overhead by ~B×. The loss is the MEAN per-window NELBO (same estimator the data-
// parallel trainer averaged across W replicas), so convergence semantics match.
struct BatchedDiffusionLossContext {
    std::int64_t                 B, T;
    sub0llm::Tensor              ids_input;    // (B·T,) int32 — the corrupted batch
    sub0llm::Tensor              ids_clean;    // (B·T,) int32 — clean targets (fast path)
    sub0llm::Tensor              weights;      // (B·T,) f32  — masked-position flags (fast path)
    std::vector<nn::Corruption>  corr;         // per-window corruption (reused buffers)
    std::vector<float>           noise;        // per-window actual noise fraction (size B)

    BatchedDiffusionLossContext(std::int64_t B_, std::int64_t T_)
        : B(B_), T(T_),
          ids_input({B_ * T_}, sub0llm::DType::Int32),
          ids_clean({B_ * T_}, sub0llm::DType::Int32),
          weights({B_ * T_}, sub0llm::DType::Float32),
          corr(static_cast<std::size_t>(B_)),
          noise(static_cast<std::size_t>(B_)) {}
};

struct BatchedDiffusionLossResult {
    sub0llm::autograd::Variable loss;            // scalar — mean per-window NELBO
    std::uint64_t               n_masked = 0;    // summed over the batch
    float                       mean_t   = 0.0f; // mean sampled noise level
};

// Fast integer hash for per-window deterministic seeding (Steele/Vigna splitmix64).
[[nodiscard]] inline std::uint64_t splitmix64(std::uint64_t x) noexcept {
    x += 0x9E3779B97F4A7C15ull;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
    return x ^ (x >> 31);
}

// Corrupt B windows (one per offset), run a single batched forward, and reduce to
// the mean per-window NELBO. offsets.size() must equal ctx.B.
//
// shared_t (variance reduction, the Ch31-sandbox lever): draw ONE t for all B windows so the
// B-window average sharpens the per-noise-level gradient (within-level consistency ∝ √B) instead
// of diluting across the near-orthogonal noise levels. Still unbiased (t ~ U over steps). This is
// the single-thread analog of the pool's --shared-t, but B is decoupled from the core count, so a
// single core can buy arbitrarily high consistency. exact_count: mask EXACTLY round(t·T) positions
// (removes the Bernoulli count variance) — the other half of the consistency win.
template<class RNG>
[[nodiscard]] BatchedDiffusionLossResult
batched_diffusion_loss(const nn::Denoiser& model,
                       std::span<const std::int32_t> stream,
                       std::span<const std::size_t> offsets,
                       RNG& rng, BatchedDiffusionLossContext& ctx,
                       float t_min = 0.02f, float t_max = 1.0f,
                       bool shared_t = false, bool exact_count = false,
                       std::span<const std::uint8_t> is_word_start = {},
                       bool whole_word = false,
                       std::uint64_t seed_base = 0, std::int64_t index0 = 0,
                       bool contiguous = false) {
    namespace ag = sub0llm::autograd;
    const std::int64_t B = ctx.B, T = ctx.T;
    auto idd = ctx.ids_input.data_as<std::int32_t>();
    std::uniform_real_distribution<float> t_dist(t_min, t_max);
    // Per-window DETERMINISTIC seeding (seed_base != 0): window's corruption + (independent) t are a
    // pure function of its GLOBAL index, not which worker ran it — so the master gradient is bitwise
    // identical for any worker count W at fixed B. seed_base == 0 ⇒ legacy: draw from the shared rng.
    const float shared = shared_t ? t_dist(rng) : 0.0f;   // one level for the whole batch

    std::vector<float>         ts(static_cast<std::size_t>(B));
    std::vector<std::uint32_t> nm(static_cast<std::size_t>(B));
    std::uint64_t total_masked = 0;
    double        t_accum = 0.0;

    for (std::int64_t b = 0; b < B; ++b) {
        auto clean = stream.subspan(offsets[static_cast<std::size_t>(b)],
                                    static_cast<std::size_t>(T));
        // Window-local generator: deterministic from the global index when seed_base is set,
        // else freshly drawn from the shared rng (legacy callers). shared-t makes t identical
        // across workers (band collapsed to ts by the caller), so only mask POSITIONS vary here.
        std::mt19937 wrng(static_cast<std::uint32_t>(
            seed_base ? splitmix64(seed_base + static_cast<std::uint64_t>(index0 + b))
                      : static_cast<std::uint64_t>(rng())));
        const float t = shared_t ? shared : t_dist(wrng);
        if (whole_word)
            nn::corrupt_whole_word_into(clean, is_word_start, t, spec::NoiseSchedule::Absorbing,
                                        model.mask_id(), model.real_vocab(), wrng,
                                        ctx.corr[static_cast<std::size_t>(b)]);
        else
            nn::corrupt_into(clean, t, spec::NoiseSchedule::Absorbing,
                             model.mask_id(), model.real_vocab(), wrng,
                             ctx.corr[static_cast<std::size_t>(b)], exact_count, contiguous);
        const auto& c = ctx.corr[static_cast<std::size_t>(b)];
        std::copy_n(c.tokens.begin(), T, idd.begin() + b * T);
        ts[static_cast<std::size_t>(b)] = t;
        nm[static_cast<std::size_t>(b)] = c.n_corrupted;
        ctx.noise[static_cast<std::size_t>(b)] = static_cast<float>(c.n_corrupted) / static_cast<float>(T);
        total_masked += c.n_corrupted;
        t_accum      += t;
    }

    ag::Variable logits = model.forward(ctx.ids_input,
                                        std::span<const float>(ctx.noise), B, T);  // (B·T, Vm)

    // FAST PATH (shared_t + exact_count, the --batch default): every window masks EXACTLY the
    // same count n=round(t·T) at the SAME t, so the per-window NELBO weight n_b/(t_b·T) is
    // identical, and (1/B)Σ_b mean_ce_b = mean over ALL masked positions. The whole reduction
    // collapses to ONE weighted_cross_entropy over the (B·T) batch × a single scalar — no B
    // serial CE ops, no per-window narrow/alloc. Exact, not an approximation.
    if (shared_t && exact_count && !whole_word) {
        auto cl = ctx.ids_clean.data_as<std::int32_t>();
        auto wt = ctx.weights.data_as<float>();
        for (std::int64_t b = 0; b < B; ++b) {
            auto clean = stream.subspan(offsets[static_cast<std::size_t>(b)], static_cast<std::size_t>(T));
            const auto& c = ctx.corr[static_cast<std::size_t>(b)];
            for (std::int64_t i = 0; i < T; ++i) {
                cl[static_cast<std::size_t>(b * T + i)] = clean[static_cast<std::size_t>(i)];
                wt[static_cast<std::size_t>(b * T + i)] = static_cast<float>(c.corrupted[static_cast<std::size_t>(i)]);
            }
        }
        ag::Variable mean_ce = ag::weighted_cross_entropy(logits, ctx.ids_clean, ctx.weights);
        const float w = static_cast<float>(nm[0]) / (ts[0] * static_cast<float>(T));
        return {ag::scale(mean_ce, w), total_masked, static_cast<float>(t_accum / static_cast<double>(B))};
    }

    // General path (independent t and/or Bernoulli count): per-window weighted CE on the shared
    // logits graph. loss = (1/B) Σ_b NELBO_b, NELBO_b = mean_ce_b·n_b/(t_b·T).
    ag::Variable loss;
    for (std::int64_t b = 0; b < B; ++b) {
        const auto& c = ctx.corr[static_cast<std::size_t>(b)];
        auto clean = stream.subspan(offsets[static_cast<std::size_t>(b)],
                                    static_cast<std::size_t>(T));
        sub0llm::Tensor tb({T}, sub0llm::DType::Int32);
        sub0llm::Tensor wb({T}, sub0llm::DType::Float32);
        auto tbd = tb.data_as<std::int32_t>();
        auto wbd = wb.data_as<float>();
        for (std::int64_t i = 0; i < T; ++i) {
            tbd[static_cast<std::size_t>(i)] = clean[static_cast<std::size_t>(i)];
            wbd[static_cast<std::size_t>(i)] = static_cast<float>(c.corrupted[static_cast<std::size_t>(i)]);
        }
        ag::Variable logits_b = ag::narrow(logits, b * T, T);                   // (T, Vm)
        ag::Variable mean_ce  = ag::weighted_cross_entropy(logits_b, tb, wb);
        const float  w        = static_cast<float>(nm[static_cast<std::size_t>(b)]) /
                                (ts[static_cast<std::size_t>(b)] * static_cast<float>(T));
        ag::Variable nelbo_b  = ag::scale(mean_ce, w);
        loss = (b == 0) ? nelbo_b : ag::add(loss, nelbo_b);
    }
    loss = ag::scale(loss, 1.0f / static_cast<float>(B));
    return {loss, total_masked, static_cast<float>(t_accum / static_cast<double>(B))};
}

} // namespace sub0diff::train
