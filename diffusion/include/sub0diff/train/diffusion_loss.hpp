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
                                                 float t_max = 1.0f) {
    namespace ag = sub0llm::autograd;
    const auto T = static_cast<float>(clean.size());

    std::uniform_real_distribution<float> t_dist(t_min, t_max);
    const float t = t_dist(rng);

    nn::corrupt_into(clean, t, spec::NoiseSchedule::Absorbing,
                     model.mask_id(), model.real_vocab(), rng, ctx.corruption);
    const auto& corr = ctx.corruption;

    std::ranges::copy(corr.tokens, ctx.ids_input.data_as<std::int32_t>().begin());
    std::ranges::copy(clean, ctx.ids_clean.data_as<std::int32_t>().begin());
    auto w = ctx.weights.data_as<float>();
    for (std::size_t i = 0; i < clean.size(); ++i) w[i] = corr.corrupted[i];

    const float actual_noise = static_cast<float>(corr.n_corrupted) / T;
    ag::Variable logits = model.forward(ctx.ids_input, actual_noise);
    ag::Variable mean_ce = ag::weighted_cross_entropy(logits, ctx.ids_clean, ctx.weights);

    // mean-over-masked → NELBO term: scale by n_masked / (t·T).
    const float nelbo_w = static_cast<float>(corr.n_corrupted) / (t * T);
    return {ag::scale(mean_ce, nelbo_w), t, corr.n_corrupted};
}

} // namespace sub0diff::train
