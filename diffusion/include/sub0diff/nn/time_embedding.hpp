#pragma once

// time_embedding.hpp (Ch28) — conditioning the denoiser on the noise level.
//
// A diffusion denoiser must know *how corrupted* its input is: predicting from a 90%-masked
// canvas is a very different job from polishing a 10%-masked one. We inject the scalar noise
// level t ∈ [0,1] the same way the original transformer injected absolute position — a fixed
// sinusoidal embedding (no parameters), added to every token's embedding row.
//
//   emb[2i]   = sin(t · ω_i),   emb[2i+1] = cos(t · ω_i),   ω_i = SCALE · 10000^(-2i/D)
//
// The SCALE spreads a level in [0,1] across many phases so nearby levels stay distinguishable.
// A learned projection of this embedding is a natural Ch29 upgrade; the intro keeps it
// parameter-free so the mechanism is transparent.

#include "sub0llm/core/tensor.hpp"

#include <cmath>
#include <cstdint>

namespace sub0diff::nn {

inline constexpr float kTimeEmbedScale = 1000.0f;

// Sinusoidal embedding of a single noise level into a (D,) vector.
//
// AMPLITUDE matters: token embeddings initialise at N(0, 1/sqrt(D)) (~0.09/dim at
// D=128), so unit-amplitude sinusoids would be ~11x louder than the content signal —
// token identity becomes <1% of the input energy and learning slows ~10x (measured:
// Ch29 stuck near uniform NELBO until this was matched). Scale to the same 1/sqrt(D).
[[nodiscard]] inline sub0llm::Tensor time_embedding_vector(float level, std::int64_t D) {
    sub0llm::Tensor v = sub0llm::zeros({D});
    auto s = v.data_as<float>();
    const float amp = 1.0f / std::sqrt(static_cast<float>(D));
    const std::size_t half = static_cast<std::size_t>(D / 2);
    for (std::size_t i = 0; i < half; ++i) {
        const float omega = kTimeEmbedScale *
                            std::pow(10000.0f, -2.0f * static_cast<float>(i) / static_cast<float>(D));
        const float ang = level * omega;
        s[2 * i]     = amp * std::sin(ang);
        s[2 * i + 1] = amp * std::cos(ang);
    }
    if (D & 1) s[static_cast<std::size_t>(D - 1)] = amp * std::sin(level * kTimeEmbedScale);  // odd D
    return v;
}

// The same vector broadcast to every row of a (T, D) tensor — ready to add to token embeddings.
[[nodiscard]] inline sub0llm::Tensor time_embedding_rows(float level, std::int64_t T, std::int64_t D) {
    sub0llm::Tensor row = time_embedding_vector(level, D);
    auto rs = row.data_as<float>();
    sub0llm::Tensor m = sub0llm::zeros({T, D});
    auto ms = m.data_as<float>();
    for (std::size_t t = 0; t < static_cast<std::size_t>(T); ++t)
        for (std::size_t d = 0; d < static_cast<std::size_t>(D); ++d)
            ms[t * static_cast<std::size_t>(D) + d] = rs[d];
    return m;
}

} // namespace sub0diff::nn
