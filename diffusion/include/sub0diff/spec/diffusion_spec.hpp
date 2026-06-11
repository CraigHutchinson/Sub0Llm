#pragma once

// diffusion_spec.hpp (Ch28) — the compile-time contract for a text-diffusion model.
//
// This refines the autoregressive StaticSpec (Ch27, sub0llm/nn/static_spec.hpp) with
// the axes a *diffusion* forward/sampling path needs to fold at compile time:
//   • canvas_len            — tokens refined in parallel per diffusion block (256 for DiffusionGemma)
//   • n_diffusion_steps     — default reverse-process step budget
//   • mask_token_id         — the absorbing-state token used by the forward corruption
//   • schedule              — Absorbing ([MASK]) vs Uniform (random-token) corruption
//   • n_experts/n_active    — MoE sparsity (0 = dense); DiffusionGemma is 26B-A4B MoE
//
// A `sub0diff-specialize`d binary includes a generated struct meeting DiffusionStaticSpec
// and the denoiser monomorphizes against it — every shape and the step schedule constant-fold.

#include "sub0llm/nn/static_spec.hpp"

#include <concepts>
#include <cstddef>
#include <cstdint>

namespace sub0diff::spec {

// How the forward (corruption) process replaces a token.
//   Absorbing — token -> a dedicated [MASK] id (BERT/LLaDA/MDLM style). Loss is masked-MLM.
//   Uniform   — token -> a uniformly random vocab id (DiffusionGemma "Uniform State Diffusion").
enum class NoiseSchedule : int { Absorbing = 0, Uniform = 1 };

// A type S is a DiffusionStaticSpec if it is a valid model StaticSpec *and* exposes the
// diffusion-specific axes as compile-time constants of the expected types.
template<class S>
concept DiffusionStaticSpec = sub0llm::nn::StaticSpec<S> && requires {
    requires std::same_as<std::remove_cv_t<decltype(S::canvas_len)>,        int64_t>;
    requires std::same_as<std::remove_cv_t<decltype(S::n_diffusion_steps)>, int64_t>;
    requires std::same_as<std::remove_cv_t<decltype(S::mask_token_id)>,     int64_t>;
    requires std::same_as<std::remove_cv_t<decltype(S::n_experts)>,         int64_t>;
    requires std::same_as<std::remove_cv_t<decltype(S::n_active_experts)>,  int64_t>;
    { S::schedule } -> std::convertible_to<NoiseSchedule>;
};

// Constants derived from a diffusion spec — the things a sampler/denoiser indexes with.
template<DiffusionStaticSpec S>
struct DiffusionSpecDerived : sub0llm::nn::SpecDerived<S> {
    // MoE: 0 experts means a dense FFN; otherwise top-`n_active_experts` routing.
    static constexpr bool is_moe = S::n_experts > 0;

    // Absorbing-state diffusion uses a dedicated [MASK] id; uniform-state does not.
    static constexpr bool uses_mask_token =
        S::schedule == NoiseSchedule::Absorbing;

    // Default fraction of canvas positions revealed per step under a linear schedule.
    // (Samplers may override; this is the compile-time default the loop is sized around.)
    static constexpr float tokens_revealed_per_step =
        S::n_diffusion_steps > 0
            ? static_cast<float>(S::canvas_len) / static_cast<float>(S::n_diffusion_steps)
            : static_cast<float>(S::canvas_len);

    static_assert(S::canvas_len > 0, "canvas_len must be positive");
    static_assert(S::n_diffusion_steps > 0, "n_diffusion_steps must be positive");
    static_assert(S::n_active_experts <= S::n_experts || S::n_experts == 0,
                  "n_active_experts cannot exceed n_experts");
};

} // namespace sub0diff::spec
