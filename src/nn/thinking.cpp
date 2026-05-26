#include "sub0llm/nn/thinking.hpp"
#include "sub0llm/nn/sampler.hpp"

#include <algorithm>
#include <format>
#include <map>
#include <stdexcept>

namespace sub0llm::nn {

namespace {

static int32_t sample_one(const Tensor& logits_1d, const SamplingConfig& cfg, std::mt19937& rng) {
    switch (cfg.mode) {
        case SamplingMode::Greedy:      return greedy_sample(logits_1d);
        case SamplingMode::Temperature: return temperature_sample(logits_1d, cfg.temperature, rng);
        case SamplingMode::TopK:        return top_k_sample(logits_1d, cfg.top_k, cfg.temperature, rng);
        case SamplingMode::TopP:        return top_p_sample(logits_1d, cfg.top_p, cfg.temperature, rng);
        default: throw std::runtime_error(std::format("sample_one: unknown SamplingMode {}", static_cast<int>(cfg.mode)));
    }
}

// Build a 1-D Tensor {V} from the last row of a (T, V) logits tensor.
static Tensor extract_last_row(const Tensor& logits_full) {
    const int64_t T = logits_full.shape(0);
    const int64_t V = logits_full.shape(1);
    Tensor last({V}, DType::Float32);
    auto src = logits_full.data_as<float>();
    auto dst = last.data_as<float>();
    const std::size_t offset = static_cast<std::size_t>((T - 1) * V);
    for (std::size_t j = 0; j < static_cast<std::size_t>(V); ++j)
        dst[j] = src[offset + j];
    return last;
}

// Build a (T,) Int32 token-id tensor from a vector.
static Tensor make_ids(const std::vector<int32_t>& tokens) {
    const int64_t T = static_cast<int64_t>(tokens.size());
    Tensor ids({T}, DType::Int32);
    auto sp = ids.data_as<int32_t>();
    for (std::size_t i = 0; i < tokens.size(); ++i)
        sp[i] = tokens[i];
    return ids;
}

} // anonymous namespace

ThinkingResult generate_with_thinking(
    ModernGPT&                  model,
    const std::vector<int32_t>& prompt,
    const ThinkingConfig&       cfg,
    std::mt19937&               rng)
{
    if (cfg.think_bos_id < 0)
        throw std::runtime_error(
            std::format("generate_with_thinking: think_bos_id must be >= 0, got {}", cfg.think_bos_id));
    if (cfg.think_eos_id < 0)
        throw std::runtime_error(
            std::format("generate_with_thinking: think_eos_id must be >= 0, got {}", cfg.think_eos_id));
    if (prompt.empty())
        throw std::runtime_error("generate_with_thinking: prompt must not be empty");
    if (cfg.max_think_tokens < 1)
        throw std::runtime_error(
            std::format("generate_with_thinking: max_think_tokens must be >= 1, got {}", cfg.max_think_tokens));
    if (cfg.max_answer_tokens < 1)
        throw std::runtime_error(
            std::format("generate_with_thinking: max_answer_tokens must be >= 1, got {}", cfg.max_answer_tokens));

    ThinkingResult result;
    result.full_sequence.reserve(
        prompt.size() + 1 +
        static_cast<std::size_t>(cfg.max_think_tokens) + 1 +
        static_cast<std::size_t>(cfg.max_answer_tokens));

    // Build initial context: prompt + think_bos_id
    std::vector<int32_t> context(prompt);
    context.push_back(cfg.think_bos_id);

    // Copy prompt into full_sequence
    result.full_sequence.insert(result.full_sequence.end(), prompt.begin(), prompt.end());
    result.full_sequence.push_back(cfg.think_bos_id);

    // Phase 1: thinking loop
    bool thinking_complete = false;
    for (int64_t step = 0; step < cfg.max_think_tokens; ++step) {
        Tensor ids = make_ids(context);
        Tensor logits_full = model.forward(ids).data();
        Tensor last = extract_last_row(logits_full);

        int32_t token = sample_one(last, cfg.inner_cfg, rng);

        if (token == cfg.think_eos_id) {
            thinking_complete = true;
            result.full_sequence.push_back(cfg.think_eos_id);
            context.push_back(cfg.think_eos_id);
            break;
        }

        result.thinking_tokens.push_back(token);
        result.full_sequence.push_back(token);
        context.push_back(token);
    }

    // If budget exhausted without seeing think_eos, force it
    if (!thinking_complete) {
        result.full_sequence.push_back(cfg.think_eos_id);
        context.push_back(cfg.think_eos_id);
    }

    result.thinking_complete = thinking_complete;

    // Phase 2: answer loop
    for (int64_t step = 0; step < cfg.max_answer_tokens; ++step) {
        Tensor ids = make_ids(context);
        Tensor logits_full = model.forward(ids).data();
        Tensor last = extract_last_row(logits_full);

        int32_t token = sample_one(last, cfg.answer_cfg, rng);

        result.answer_tokens.push_back(token);
        result.full_sequence.push_back(token);
        context.push_back(token);
    }

    return result;
}

int32_t think_self_consistency(
    ModernGPT&                  model,
    const std::vector<int32_t>& prompt,
    const ThinkingConfig&       cfg,
    int                         n_samples,
    std::mt19937&               rng)
{
    if (n_samples < 1)
        throw std::runtime_error(
            std::format("think_self_consistency: n_samples must be >= 1, got {}", n_samples));

    std::map<int32_t, int> votes;

    for (int i = 0; i < n_samples; ++i) {
        auto result = generate_with_thinking(model, prompt, cfg, rng);
        if (result.answer_tokens.empty())
            throw std::runtime_error(
                std::format("think_self_consistency: run {} produced empty answer_tokens", i));
        votes[result.answer_tokens[0]]++;
    }

    // Return plurality winner; tie-break: lowest token id
    int32_t best_token = votes.begin()->first;
    int best_count = votes.begin()->second;
    for (auto& [tok, cnt] : votes) {
        if (cnt > best_count || (cnt == best_count && tok < best_token)) {
            best_token = tok;
            best_count = cnt;
        }
    }

    return best_token;
}

} // namespace sub0llm::nn
