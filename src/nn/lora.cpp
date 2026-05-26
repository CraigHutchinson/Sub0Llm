#include "sub0llm/nn/lora.hpp"

#include "sub0llm/autograd/ops.hpp"
#include "sub0llm/core/tensor.hpp"

#include <algorithm>
#include <cmath>
#include <format>
#include <random>
#include <stdexcept>

namespace sub0llm::nn {

LoRALinear::LoRALinear(int64_t in_features, int64_t out_features,
                       int64_t rank, float alpha, uint32_t seed)
    : scaling_(0.0f), in_(in_features), out_(out_features), rank_(rank)
{
    if (in_features <= 0)
        throw std::runtime_error(
            std::format("LoRALinear: in_features={} must be positive", in_features));
    if (out_features <= 0)
        throw std::runtime_error(
            std::format("LoRALinear: out_features={} must be positive", out_features));
    if (rank < 0)
        throw std::runtime_error(
            std::format("LoRALinear: rank={} must be >= 0", rank));
    if (rank > std::min(in_features, out_features))
        throw std::runtime_error(
            std::format("LoRALinear: rank={} exceeds min(in={}, out={})",
                        rank, in_features, out_features));
    if (alpha <= 0.0f)
        throw std::runtime_error(
            std::format("LoRALinear: alpha={} must be positive", alpha));

    std::mt19937 rng(seed);

    // Base weight: Xavier uniform, shape (out, in), frozen.
    {
        const float limit = std::sqrt(6.0f / static_cast<float>(in_features + out_features));
        std::uniform_real_distribution<float> dist(-limit, limit);
        Tensor w = zeros({out_features, in_features}, DType::Float32);
        auto   s = w.data_as<float>();
        for (auto& v : s) v = dist(rng);
        base_ = autograd::Variable(std::move(w), /*requires_grad=*/false, "lora.base");
    }

    if (rank > 0) {
        scaling_ = alpha / static_cast<float>(rank);

        // A: Kaiming normal (std=1/sqrt(in)), shape (in, rank), trainable.
        {
            const float std_a = 1.0f / std::sqrt(static_cast<float>(in_features));
            std::normal_distribution<float> dist(0.0f, std_a);
            Tensor a = zeros({in_features, rank}, DType::Float32);
            auto   s = a.data_as<float>();
            for (auto& v : s) v = dist(rng);
            lora_A_ = autograd::Variable(std::move(a), /*requires_grad=*/true, "lora.A");
        }

        // B: zeros, shape (rank, out), trainable — ensures ΔW = 0 at init.
        {
            Tensor b = zeros({rank, out_features}, DType::Float32);
            lora_B_ = autograd::Variable(std::move(b), /*requires_grad=*/true, "lora.B");
        }
    }
}

autograd::Variable LoRALinear::forward(const autograd::Variable& x) const {
    using namespace autograd;
    auto base_out = matmul(x, transpose2d(base_));
    if (rank_ == 0) return base_out;
    auto lora_out = scale(matmul(matmul(x, lora_A_), lora_B_), scaling_);
    return add(base_out, lora_out);
}

std::vector<autograd::Variable*> LoRALinear::lora_parameters() {
    if (rank_ == 0) return {};
    return {&lora_A_, &lora_B_};
}

std::vector<autograd::Variable*> LoRALinear::all_parameters() {
    if (rank_ == 0) return {&base_};
    return {&base_, &lora_A_, &lora_B_};
}

} // namespace sub0llm::nn
