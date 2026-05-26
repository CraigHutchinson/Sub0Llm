#pragma once

#include "sub0llm/autograd/variable.hpp"
#include "sub0llm/core/tensor.hpp"
#include <cstdint>
#include <vector>

namespace sub0llm::nn {

// LoRALinear: frozen base weight W plus trainable low-rank delta ΔW = (alpha/r) * A @ B.
// Forward: y = x @ W^T + (alpha/r) * (x @ A) @ B
// W: (out, in) — frozen (requires_grad=false)
// A: (in, rank) — trainable, init Kaiming normal std=1/sqrt(in)
// B: (rank, out) — trainable, init zeros → ΔW = 0 at start
// When rank=0: plain frozen linear (no LoRA delta).
class LoRALinear {
public:
    LoRALinear(int64_t in_features, int64_t out_features,
               int64_t rank, float alpha = 1.0f, uint32_t seed = 42);

    [[nodiscard]] autograd::Variable forward(const autograd::Variable& x) const;

    [[nodiscard]] std::vector<autograd::Variable*> lora_parameters();
    [[nodiscard]] std::vector<autograd::Variable*> all_parameters();

    [[nodiscard]] int64_t in_features()  const noexcept { return in_; }
    [[nodiscard]] int64_t out_features() const noexcept { return out_; }
    [[nodiscard]] int64_t rank()         const noexcept { return rank_; }
    [[nodiscard]] float   scaling()      const noexcept { return scaling_; }

private:
    autograd::Variable base_;    // (out, in), requires_grad=false
    autograd::Variable lora_A_;  // (in, rank), requires_grad=true
    autograd::Variable lora_B_;  // (rank, out), requires_grad=true
    float    scaling_;           // alpha / rank (0 if rank==0)
    int64_t  in_, out_, rank_;
};

} // namespace sub0llm::nn
