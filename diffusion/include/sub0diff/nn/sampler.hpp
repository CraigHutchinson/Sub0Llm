#pragma once

// sampler.hpp (Ch30) — the REVERSE process: iterative canvas refinement.
//
// Ch28/29 trained a denoiser that predicts every masked position in ONE forward
// pass — good when most of the canvas is context, hopeless when most of it is
// masked. The reverse process closes that gap by iterating:
//
//   1. denoise the whole canvas (bidirectional, all positions in parallel)
//   2. COMMIT only the predictions the model is confident about
//   3. keep the rest masked and denoise again — every committed token is new
//      context that sharpens the next round ("the canvas snaps into focus")
//
// Confidence-based commitment (MaskGIT / LLaDA low-confidence remasking) plus an
// entropy bound for early stopping (DiffusionGemma's `entropy_bound`): when every
// remaining masked position is already low-entropy, finish in one final commit.
//
// The loop is SELF-TERMINATING and adaptive: each iteration commits at least
// ceil(n_masked/T)·guaranteed progress, confident canvases finish in few rounds,
// hard ones take more — iteration count is an output, not an input.
//
// The same machinery doubles as ITERATIVE RECOVERY: seed the canvas with a
// corrupted window (unmasked positions fixed) and refine — directly comparable
// to Ch29's one-step recovery numbers.

#include "sub0diff/nn/denoiser.hpp"

#include <cstdint>
#include <functional>
#include <random>
#include <span>
#include <vector>

namespace sub0diff::nn {

struct SamplerConfig {
    float conf_threshold = 0.9f;   // commit positions with max-prob ≥ this
    float entropy_bound  = 0.1f;   // mean masked entropy below this → commit all & stop
    float min_commit_frac = 0.10f; // progress guarantee: ≥ this fraction of masked/iter
    float temperature    = 1.0f;   // 0 = greedy argmax; >0 = sample from softmax
    std::size_t max_iters = 0;     // 0 = canvas length (one-per-iter worst case)

    // Low-confidence REMASKING: each iteration, generated (non-fixed) tokens whose
    // probability under the CURRENT context falls below this are returned to [MASK]
    // for reconsideration — commitments are revisable, so early errors don't
    // permanently poison the context (the error-compounding failure of pure
    // commit-only refinement, measured in Ch30). 0 = off. Net progress is enforced
    // (never remask more than was committed this iteration, minus one).
    float remask_threshold = 0.0f;
};

struct SamplerStats {
    std::size_t iterations = 0;    // denoiser forwards used (self-chosen)
    std::size_t committed  = 0;    // positions filled by the sampler
    bool entropy_stopped   = false;
    double seconds         = 0.0;
};

// Refine `canvas` in place. Positions equal to model.mask_id() are generated;
// all other positions are FIXED context (prompt tokens or surviving corruption
// context). on_iter (optional) observes the canvas after each iteration.
SamplerStats refine_canvas(const Denoiser& model,
                           std::span<std::int32_t> canvas,
                           const SamplerConfig& cfg,
                           std::mt19937& rng,
                           const std::function<void(std::span<const std::int32_t>,
                                                    std::size_t)>& on_iter = {});

// Convenience: an all-masked canvas of length T with `prompt` fixed at the front.
[[nodiscard]] std::vector<std::int32_t> make_canvas(const Denoiser& model,
                                                    std::int64_t T,
                                                    std::span<const std::int32_t> prompt = {});

} // namespace sub0diff::nn
