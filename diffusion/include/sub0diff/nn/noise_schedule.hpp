#pragma once

// noise_schedule.hpp (Ch28) — the FORWARD (corruption) process of a text diffusion model.
//
// Diffusion training never sees a clean sequence: it sees a *corrupted* one and learns to
// undo the corruption. The forward process is fixed (no parameters) — it takes clean tokens
// and a noise level in [0,1] and randomly damages a fraction of positions:
//
//   • Absorbing-state (BERT / LLaDA / MDLM): a damaged token becomes a dedicated [MASK] id.
//     The model only has to predict the masked positions → a masked-LM objective. Simplest,
//     so we teach it first.
//   • Uniform-state (DiffusionGemma "Uniform State Diffusion"): a damaged token becomes a
//     uniformly random vocab token. The model must also *detect* which positions are wrong.
//
// The mask id is the real vocabulary size: the diffusion model is built with vocab_size+1
// rows so [MASK] is one slot above every real token. This means the base tokenizer needs no
// changes — the diffusion layer owns the extra id. The reverse (sampling) process in Ch30
// inverts this schedule step by step.

#include "sub0diff/spec/diffusion_spec.hpp"

#include <algorithm>
#include <cstdint>
#include <random>
#include <span>
#include <vector>

namespace sub0diff::nn {

using sub0diff::spec::NoiseSchedule;

// Result of corrupting a clean sequence.
struct Corruption {
    std::vector<std::int32_t> tokens;     // the corrupted sequence fed to the denoiser
    std::vector<std::uint8_t> corrupted;  // 1 where a position was damaged (a loss target)
    int                       n_corrupted = 0;
};

// Linear noise level for a reverse-process step index: step 0 = clean, n_steps = fully masked.
// Training samples a level in [0,1]; this helper maps an integer step onto that range.
[[nodiscard]] inline float mask_ratio_linear(std::int64_t step, std::int64_t n_steps) noexcept {
    if (n_steps <= 0) return 1.0f;
    const float r = static_cast<float>(step) / static_cast<float>(n_steps);
    return std::clamp(r, 0.0f, 1.0f);
}

// Corrupt `clean` by independently damaging each position with probability `mask_prob`.
//   mask_id      — the [MASK] token (== real vocab size); used by Absorbing only.
//   real_vocab   — number of real tokens; Uniform draws a replacement in [0, real_vocab).
template<class RNG>
[[nodiscard]] Corruption corrupt(std::span<const std::int32_t> clean,
                                 float            mask_prob,
                                 NoiseSchedule    schedule,
                                 std::int32_t     mask_id,
                                 std::int64_t     real_vocab,
                                 RNG&             rng) {
    Corruption c;
    c.tokens.assign(clean.begin(), clean.end());
    c.corrupted.assign(clean.size(), 0);

    std::bernoulli_distribution coin(std::clamp(mask_prob, 0.0f, 1.0f));
    std::uniform_int_distribution<std::int32_t> uni(0, static_cast<std::int32_t>(real_vocab) - 1);

    for (std::size_t i = 0; i < clean.size(); ++i) {
        if (!coin(rng)) continue;
        c.corrupted[i] = 1;
        ++c.n_corrupted;
        if (schedule == NoiseSchedule::Absorbing) {
            c.tokens[i] = mask_id;
        } else {  // Uniform: replace with a random real token (never the mask id)
            std::int32_t r = uni(rng);
            if (r == clean[i] && real_vocab > 1) r = (r + 1) % static_cast<std::int32_t>(real_vocab);
            c.tokens[i] = r;
        }
    }
    return c;
}

} // namespace sub0diff::nn
