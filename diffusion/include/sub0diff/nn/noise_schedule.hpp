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
#include <cmath>
#include <cstdint>
#include <random>
#include <span>
#include <vector>

namespace sub0diff::nn {

using sub0diff::spec::NoiseSchedule;

// Result of corrupting a clean sequence.
struct Corruption {
    std::vector<std::int32_t> tokens;       // the corrupted sequence fed to the denoiser
    std::vector<std::uint8_t> corrupted;    // 1 where a position was damaged (a loss target)
    std::uint32_t             n_corrupted = 0U;
};

// Linear noise level for a reverse-process step index: step 0 = clean, n_steps = fully masked.
// Training samples a level in [0,1]; this helper maps an integer step onto that range.
[[nodiscard]] inline float mask_ratio_linear(std::int64_t step, std::int64_t n_steps) noexcept {
    if (n_steps <= 0) return 1.0f;
    const float r = static_cast<float>(step) / static_cast<float>(n_steps);
    return std::clamp(r, 0.0f, 1.0f);
}

// Corrupt `clean` into `c`, independently damaging each position with probability
// `mask_prob`. Guarantees at least one position is corrupted (Bernoulli can return 0
// at low probabilities).
//   mask_id      — the [MASK] token (== real vocab size); used by Absorbing only.
//   real_vocab   — number of real tokens; Uniform draws a replacement in [0, real_vocab).
// Allocation-free when `c` is reused across calls with the same sequence length —
// a training loop corrupts every step, so pass one Corruption hoisted out of the loop.
//   exact_count — when true, mask EXACTLY k = round(mask_prob·T) positions (≥1), chosen
//   uniformly via Knuth's Algorithm S (selection sampling — no buffer, no heap), instead of an
//   independent Bernoulli per position. This makes the realised noise equal the nominal EXACTLY
//   (so the model's
//   noise conditioning is exact, not a random Binomial), removes the Bernoulli COUNT variance
//   from the gradient, and makes the min-1 floor a clean k≥1 clamp rather than a low-t hack.
//   The remaining randomness is only WHICH positions — a deterministic, even, unbiased count.
//   The count is clamped to [1, T-1] (T≥2): ≥1 masked gives a loss TARGET, and ≥1 VISIBLE gives
//   a context SIGNAL — a fully-masked window (t→1) teaches only the marginal and is the highest-
//   variance, lowest-information case (a measured t_max A/B: dropping the high-t tail trains a
//   better model per unit compute). This min-1-visible floor is the symmetric twin of min-1-masked.
template<class RNG>
void corrupt_into(std::span<const std::int32_t> clean,
                  float            mask_prob,
                  NoiseSchedule    schedule,
                  std::int32_t     mask_id,
                  std::int64_t     real_vocab,
                  RNG&             rng,
                  Corruption&      c,
                  bool             exact_count = false,
                  bool             contiguous  = false) {
    c.tokens.assign(clean.begin(), clean.end());
    c.corrupted.assign(clean.size(), 0);
    c.n_corrupted = 0;

    std::uniform_int_distribution<std::int32_t> uni(0, static_cast<std::int32_t>(real_vocab) - 1);

    // Local helper: apply schedule-specific corruption to a single position.
    // Encapsulates the Absorbing/Uniform conditional so the fallback doesn't duplicate it.
    // IDEMPOTENT: corrupting an already-corrupted position is a no-op, so n_corrupted always
    // equals the true number of damaged positions (no double-count can skew the NELBO weight),
    // robust to any caller that might revisit a position.
    auto corrupt_pos = [&](std::size_t pos) {
        if (c.corrupted[pos]) return;
        c.corrupted[pos] = 1;
        ++c.n_corrupted;
        if (schedule == NoiseSchedule::Absorbing) {
            c.tokens[pos] = mask_id;
        } else {  // Uniform: replace with a random real token (never the mask id)
            std::int32_t r = uni(rng);
            if (r == clean[pos] && real_vocab > 1) r = (r + 1) % static_cast<std::int32_t>(real_vocab);
            c.tokens[pos] = r;
        }
    };

    if (exact_count) {
        const std::int64_t T = static_cast<std::int64_t>(clean.size());
        std::int64_t k = std::llround(std::clamp(mask_prob, 0.0f, 1.0f) * static_cast<float>(T));
        // [1, T-1]: always ≥1 masked (a target) AND ≥1 visible (context). T=1 is degenerate
        // (can't have both) → fall back to k=1 (=T, fully masked, unavoidable at length 1).
        k = std::clamp<std::int64_t>(k, 1, std::max<std::int64_t>(1, T - 1));
        if (contiguous) {
            // CONTIGUOUS span of length k at a random start in [0, T-k]: the masked tokens lack
            // LOCAL context (only the span edges abut visible tokens), so the model must use
            // LONG-RANGE context to fill it — the continuation/infill capability the scattered
            // objective never trains (TRAINING_DESIGN §13.4, the local-interpolator finding).
            std::uniform_int_distribution<std::int64_t> sd(0, T - k);
            const std::int64_t s = sd(rng);
            for (std::int64_t i = s; i < s + k; ++i) corrupt_pos(static_cast<std::size_t>(i));
            return;
        }
        // Knuth Algorithm S — select EXACTLY k of T positions uniformly with NO buffer/heap:
        // visit each position once, take it with probability (remaining_k / remaining_n).
        std::int64_t rem_k = k, rem_n = T;
        std::uniform_real_distribution<float> u(0.0f, 1.0f);
        for (std::size_t i = 0; i < clean.size() && rem_k > 0; ++i, --rem_n)
            if (u(rng) * static_cast<float>(rem_n) < static_cast<float>(rem_k)) {
                corrupt_pos(i);
                --rem_k;
            }
        return;
    }

    std::bernoulli_distribution coin(std::clamp(mask_prob, 0.0f, 1.0f));
    for (std::size_t i = 0; i < clean.size(); ++i) {
        if (!coin(rng)) continue;
        corrupt_pos(i);
    }

    // Guarantee at least one corrupted position — Bernoulli can return 0 at low mask_prob.
    // With mask_prob=0.04 and T=24: P(0) ≈ 37%, so this branch is hit frequently.
    if (c.n_corrupted == 0) {
        std::uniform_int_distribution<std::size_t> pos_dist(0, clean.size() - 1);
        corrupt_pos( pos_dist(rng) );
    }
    // Symmetric min-1-VISIBLE floor: if every position got masked (mask_prob→1), restore one
    // so the window keeps a context signal (mirrors the exact-count [1, T-1] clamp). T≥2 only.
    if (clean.size() >= 2 && c.n_corrupted == static_cast<std::uint32_t>(clean.size())) {
        std::uniform_int_distribution<std::size_t> pos_dist(0, clean.size() - 1);
        const std::size_t pos = pos_dist(rng);
        c.tokens[pos] = clean[pos];
        c.corrupted[pos] = 0;
        --c.n_corrupted;
    }
}

// WHOLE-WORD corruption: instead of damaging independent BPE tokens, damage whole words
// (all subwords of a chosen word together). A word boundary is `pos == 0` or a token that
// carries the Ġ leading-space marker (is_word_start[token_id] != 0). Masking per-word makes
// the task honest — the model can no longer "complete" a partially-visible word (`Ham`→`let`)
// and must predict whole words from context. `mask_prob` is the per-WORD probability; the
// realised token-level noise is similar in expectation (words are short). Guarantees ≥1
// masked word. Falls back to per-token corruption if `is_word_start` is empty.
template<class RNG>
void corrupt_whole_word_into(std::span<const std::int32_t> clean,
                             std::span<const std::uint8_t>  is_word_start,
                             float            mask_prob,
                             NoiseSchedule    schedule,
                             std::int32_t     mask_id,
                             std::int64_t     real_vocab,
                             RNG&             rng,
                             Corruption&      c) {
    if (is_word_start.empty()) {
        corrupt_into(clean, mask_prob, schedule, mask_id, real_vocab, rng, c);
        return;
    }
    c.tokens.assign(clean.begin(), clean.end());
    c.corrupted.assign(clean.size(), 0);
    c.n_corrupted = 0;

    std::bernoulli_distribution coin(std::clamp(mask_prob, 0.0f, 1.0f));
    std::uniform_int_distribution<std::int32_t> uni(0, static_cast<std::int32_t>(real_vocab) - 1);

    auto damage = [&](std::size_t pos) {
        if (c.corrupted[pos]) return;   // idempotent — never double-count a position
        c.corrupted[pos] = 1;
        ++c.n_corrupted;
        if (schedule == NoiseSchedule::Absorbing) {
            c.tokens[pos] = mask_id;
        } else {
            std::int32_t r = uni(rng);
            if (r == clean[pos] && real_vocab > 1) r = (r + 1) % static_cast<std::int32_t>(real_vocab);
            c.tokens[pos] = r;
        }
    };
    auto starts_word = [&](std::size_t t) {
        return t == 0 || is_word_start[static_cast<std::size_t>(clean[t])] != 0;
    };

    // Walk words; decide once per word, apply to all its subwords.
    for (std::size_t i = 0; i < clean.size();) {
        const bool mask_this = coin(rng);
        std::size_t j = i + 1;
        while (j < clean.size() && !starts_word(j)) ++j;   // [i, j) is one word
        if (mask_this) for (std::size_t k = i; k < j; ++k) damage(k);
        i = j;
    }

    // Guarantee ≥1 masked word (mask the word containing a random position).
    if (c.n_corrupted == 0 && !clean.empty()) {
        std::uniform_int_distribution<std::size_t> pos_dist(0, clean.size() - 1);
        std::size_t p = pos_dist(rng);
        while (p > 0 && !starts_word(p)) --p;                 // back up to the word start
        std::size_t q = p + 1;
        while (q < clean.size() && !starts_word(q)) ++q;      // forward to next start
        for (std::size_t k = p; k < q; ++k) damage(k);
    }
}

// Convenience form returning a fresh Corruption (allocates; fine outside hot loops).
template<class RNG>
[[nodiscard]] Corruption corrupt(std::span<const std::int32_t> clean,
                                 float            mask_prob,
                                 NoiseSchedule    schedule,
                                 std::int32_t     mask_id,
                                 std::int64_t     real_vocab,
                                 RNG&             rng) {
    Corruption c;
    corrupt_into(clean, mask_prob, schedule, mask_id, real_vocab, rng, c);
    return c;
}

} // namespace sub0diff::nn
