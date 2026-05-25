#include "sub0llm/nn/positional_encoding.hpp"

#include <cmath>
#include <format>
#include <stdexcept>

namespace sub0llm::nn {

// ── sinusoidal_encoding ───────────────────────────────────────────────────────

Tensor sinusoidal_encoding(std::int64_t seq_len, std::int64_t embed_dim) {
    if (seq_len <= 0)
        throw std::runtime_error(std::format(
            "sinusoidal_encoding: seq_len must be positive, got {}", seq_len));
    if (embed_dim % 2 != 0)
        throw std::runtime_error(std::format(
            "sinusoidal_encoding: embed_dim must be even, got {}", embed_dim));

    Tensor pe  = zeros({seq_len, embed_dim});
    auto   ped = pe.data_as<float>();
    const std::size_t T = static_cast<std::size_t>(seq_len);
    const std::size_t D = static_cast<std::size_t>(embed_dim);

    for (std::size_t pos = 0; pos < T; ++pos) {
        for (std::size_t i = 0; i < D / 2; ++i) {
            const float freq = 1.0f / std::pow(10000.0f, 2.0f * static_cast<float>(i) / static_cast<float>(D));
            ped[pos * D + 2 * i]     = std::sin(static_cast<float>(pos) * freq);
            ped[pos * D + 2 * i + 1] = std::cos(static_cast<float>(pos) * freq);
        }
    }
    return pe;
}

// ── LearnedPositionalEncoding ─────────────────────────────────────────────────

LearnedPositionalEncoding::LearnedPositionalEncoding(
    std::int64_t max_seq_len, std::int64_t embed_dim, std::uint64_t seed)
    : emb_(max_seq_len, embed_dim, seed) {}

autograd::Variable LearnedPositionalEncoding::forward(std::int64_t seq_len) const {
    if (seq_len <= 0)
        throw std::runtime_error(std::format(
            "LearnedPositionalEncoding::forward: seq_len must be positive, got {}", seq_len));
    if (seq_len > emb_.vocab_size())
        throw std::runtime_error(std::format(
            "LearnedPositionalEncoding::forward: seq_len {} exceeds max_seq_len {}",
            seq_len, emb_.vocab_size()));
    // Build position indices [0, 1, ..., seq_len-1] as int32.
    Tensor pos = zeros({seq_len}, DType::Int32);
    auto   ps  = pos.data_as<int32_t>();
    for (std::size_t i = 0; i < static_cast<std::size_t>(seq_len); ++i)
        ps[i] = static_cast<int32_t>(i);
    return emb_.forward(pos);
}

std::int64_t LearnedPositionalEncoding::max_seq_len() const noexcept {
    return emb_.vocab_size();
}
std::int64_t LearnedPositionalEncoding::embed_dim() const noexcept {
    return emb_.embed_dim();
}

// ── apply_rope ────────────────────────────────────────────────────────────────
//
// Rotates adjacent dimension pairs (2i, 2i+1) of each token by its position
// angle:  theta_i(pos) = pos / base^(2i / D)
//
//   out[pos, 2i]   = x[pos, 2i]   * cos(theta) - x[pos, 2i+1] * sin(theta)
//   out[pos, 2i+1] = x[pos, 2i] * sin(theta) + x[pos, 2i+1] * cos(theta)

Tensor apply_rope(const Tensor& x, float base) {
    if (x.ndim() != 2)
        throw std::runtime_error("apply_rope: input must be 2D (T, D)");
    const std::size_t T = static_cast<std::size_t>(x.shape()[0]);
    const std::size_t D = static_cast<std::size_t>(x.shape()[1]);
    if (D % 2 != 0)
        throw std::runtime_error(std::format(
            "apply_rope: embed_dim must be even, got {}", D));

    Tensor out    = zeros({static_cast<int64_t>(T), static_cast<int64_t>(D)});
    const Tensor xc = x.contiguous();
    const auto xs  = xc.data_as<float>();
    auto       os  = out.data_as<float>();

    for (std::size_t pos = 0; pos < T; ++pos) {
        for (std::size_t i = 0; i < D / 2; ++i) {
            const float theta = static_cast<float>(pos) /
                std::pow(base, 2.0f * static_cast<float>(i) / static_cast<float>(D));
            const float cos_t = std::cos(theta);
            const float sin_t = std::sin(theta);
            const float x0    = xs[pos * D + 2 * i];
            const float x1    = xs[pos * D + 2 * i + 1];
            os[pos * D + 2 * i]     = x0 * cos_t - x1 * sin_t;
            os[pos * D + 2 * i + 1] = x0 * sin_t + x1 * cos_t;
        }
    }
    return out;
}

} // namespace sub0llm::nn
