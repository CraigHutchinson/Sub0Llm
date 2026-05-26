#include "sub0llm/nn/sampler.hpp"

#include "sub0llm/core/ops.hpp"

#include <algorithm>
#include <cmath>
#include <format>
#include <numeric>
#include <stdexcept>
#include <utility>
#include <vector>

namespace sub0llm::nn {

// ── Internal helpers ──────────────────────────────────────────────────────────

namespace {

// Extract the last-row logits as a flat float span.
// Accepts 1-D (V,) or 2-D (T, V) input; returns span into the final row.
// Throws on dtype mismatch, non-1/2D input, or empty vocab.
std::span<const float> last_row(const Tensor& logits) {
    if (logits.dtype() != DType::Float32)
        throw std::runtime_error(
            std::format("sampler: expected float32 logits, got {}",
                        dtype_name(logits.dtype())));
    if (logits.ndim() == 0 || logits.ndim() > 2)
        throw std::runtime_error(
            std::format("sampler: logits must be 1-D or 2-D, got {}D",
                        logits.ndim()));

    const int64_t V = logits.shape(logits.ndim() - 1);
    if (V <= 0)
        throw std::runtime_error("sampler: vocab dimension must be > 0");
    if (logits.numel() < V)
        throw std::runtime_error(
            std::format("sampler: numel={} is less than vocab dim V={}",
                        logits.numel(), V));

    auto all = logits.data_as<float>();
    // Last row starts at offset (numel - V); safe because numel >= V checked above.
    const std::size_t offset = static_cast<std::size_t>(logits.numel() - V);
    return all.subspan(offset, static_cast<std::size_t>(V));
}

// Apply temperature scaling then build a normalised probability vector.
// Returns (probs, vocab_size) — probs is indexed 0..V-1.
std::vector<float> to_probs(std::span<const float> row, float temp) {
    if (temp <= 0.0f)
        throw std::runtime_error(
            std::format("sampler: temperature must be > 0, got {}", temp));

    const std::size_t V = row.size();
    std::vector<float> probs(V);

    // Numerically stable softmax with temperature.
    float mx = *std::max_element(row.begin(), row.end());
    if (!std::isfinite(mx))
        throw std::runtime_error("sampler: all logits are -inf or NaN");

    float s = 0.0f;
    for (std::size_t i = 0; i < V; ++i) {
        probs[i] = std::exp((row[i] - mx) / temp);
        s        += probs[i];
    }
    for (float& p : probs) p /= s;
    return probs;
}

// Draw a token index from `probs` using std::discrete_distribution.
int32_t multinomial(const std::vector<float>& probs, std::mt19937& rng) {
    std::discrete_distribution<int32_t> dist(probs.begin(), probs.end());
    return dist(rng);
}

} // anonymous namespace

// ── greedy_sample ─────────────────────────────────────────────────────────────

int32_t greedy_sample(const Tensor& logits) {
    auto row = last_row(logits);
    const std::size_t idx =
        static_cast<std::size_t>(
            std::max_element(row.begin(), row.end()) - row.begin());
    return static_cast<int32_t>(idx);
}

// ── temperature_sample ────────────────────────────────────────────────────────

int32_t temperature_sample(const Tensor& logits, float temp, std::mt19937& rng) {
    auto row   = last_row(logits);
    auto probs = to_probs(row, temp);
    return multinomial(probs, rng);
}

// ── top_k_sample ──────────────────────────────────────────────────────────────

int32_t top_k_sample(const Tensor& logits, int64_t k, float temp, std::mt19937& rng) {
    if (k < 1)
        throw std::runtime_error(
            std::format("top_k_sample: k must be >= 1, got {}", k));
    if (temp <= 0.0f)
        throw std::runtime_error(
            std::format("top_k_sample: temperature must be > 0, got {}", temp));

    auto row = last_row(logits);
    const std::size_t V  = row.size();
    const std::size_t ek = static_cast<std::size_t>(std::min<int64_t>(k, static_cast<int64_t>(V)));

    // Indices sorted descending by logit value.
    std::vector<std::size_t> idx(V);
    std::iota(idx.begin(), idx.end(), 0);
    std::partial_sort(idx.begin(), idx.begin() + static_cast<std::ptrdiff_t>(ek), idx.end(),
                      [&](std::size_t a, std::size_t b) { return row[a] > row[b]; });

    // Build masked logit vector — keep top-k positions, fill rest with -inf.
    constexpr float kNegInf = -std::numeric_limits<float>::infinity();
    std::vector<float> masked(V, kNegInf);
    for (std::size_t i = 0; i < ek; ++i) masked[idx[i]] = row[idx[i]];

    float mx = *std::max_element(masked.begin(), masked.end());
    if (!std::isfinite(mx))
        throw std::runtime_error("top_k_sample: all top-k logits are -inf or NaN");

    std::vector<float> probs(V);
    float s = 0.0f;
    for (std::size_t i = 0; i < V; ++i) {
        probs[i] = (masked[i] == kNegInf) ? 0.0f : std::exp((masked[i] - mx) / temp);
        s        += probs[i];
    }
    if (s == 0.0f)
        throw std::runtime_error("top_k_sample: probability mass is zero after masking");
    for (float& p : probs) p /= s;

    return multinomial(probs, rng);
}

// ── top_p_sample ──────────────────────────────────────────────────────────────

int32_t top_p_sample(const Tensor& logits, float p, float temp, std::mt19937& rng) {
    if (p <= 0.0f || p > 1.0f)
        throw std::runtime_error(
            std::format("top_p_sample: p must be in (0, 1], got {}", p));

    auto row   = last_row(logits);
    auto probs = to_probs(row, temp);   // full normalised distribution

    const std::size_t V = probs.size();

    // Sort indices by descending probability.
    std::vector<std::size_t> idx(V);
    std::iota(idx.begin(), idx.end(), 0);
    std::sort(idx.begin(), idx.end(),
              [&](std::size_t a, std::size_t b) { return probs[a] > probs[b]; });

    // Find the smallest prefix whose cumulative mass >= p.
    float cumsum  = 0.0f;
    std::size_t cutoff = V;            // number of tokens to keep
    for (std::size_t i = 0; i < V; ++i) {
        cumsum += probs[idx[i]];
        if (cumsum >= p) {
            cutoff = i + 1;
            break;
        }
    }

    // Zero out everything outside the nucleus.
    std::vector<float> masked(V, 0.0f);
    for (std::size_t i = 0; i < cutoff; ++i) masked[idx[i]] = probs[idx[i]];

    // Renormalise (cumsum may be slightly off 1.0 after truncation).
    float s = 0.0f;
    for (float v : masked) s += v;
    for (float& v : masked) v /= s;

    return multinomial(masked, rng);
}

// ── generate ─────────────────────────────────────────────────────────────────

std::vector<int32_t> generate(ModernGPT&                  model,
                               const std::vector<int32_t>& prompt,
                               int64_t                     max_new,
                               const SamplingConfig&       cfg,
                               std::mt19937&               rng) {
    if (max_new < 1)
        throw std::runtime_error(
            std::format("generate: max_new must be >= 1, got {}", max_new));
    if (prompt.empty())
        throw std::runtime_error("generate: prompt must not be empty");

    std::vector<int32_t> tokens(prompt);
    tokens.reserve(prompt.size() + static_cast<std::size_t>(max_new));

    for (int64_t step = 0; step < max_new; ++step) {
        // Build a (T,) Int32 token-id tensor from the current sequence.
        const int64_t T = static_cast<int64_t>(tokens.size());
        Tensor ids({T}, DType::Int32);
        {
            auto sp = ids.data_as<int32_t>();
            for (std::size_t i = 0; i < tokens.size(); ++i)
                sp[i] = tokens[i];
        }

        // Forward pass — returns Variable wrapping logits (T, V).
        // Copy the Tensor (shared storage) so it survives the temporary Variable.
        Tensor logits = model.forward(ids).data();

        int32_t next_token{};
        switch (cfg.mode) {
            case SamplingMode::Greedy:
                next_token = greedy_sample(logits);
                break;
            case SamplingMode::Temperature:
                next_token = temperature_sample(logits, cfg.temperature, rng);
                break;
            case SamplingMode::TopK:
                next_token = top_k_sample(logits, cfg.top_k, cfg.temperature, rng);
                break;
            case SamplingMode::TopP:
                next_token = top_p_sample(logits, cfg.top_p, cfg.temperature, rng);
                break;
            default:
                throw std::runtime_error(
                    std::format("generate: unhandled SamplingMode {}",
                                static_cast<int>(cfg.mode)));
        }

        tokens.push_back(next_token);
    }

    return tokens;
}

} // namespace sub0llm::nn
