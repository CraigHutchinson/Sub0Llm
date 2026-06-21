#pragma once
// Single-GPU-stream trainer — the CUDA counterpart to ParallelTrainer's CPU worker pool.
// No threads, no barriers, no replica/reduce: one model on the GPU, one batched forward+backward
// per step(). The optimizer in main() steps the SAME Variables this writes grads into, so there is
// no cross-device copy on the hot path. Implements ITrainer so main()'s loop is back-end-agnostic.
//
// Prereqs (all landed in Stage 4 Phase 7): batched_diffusion_loss on CUDA, cuda-safe
// clip_grad_norm (sum_squares), checkpoint D2H, Tensor::item() D2H, Adam on CUDA.
#include "sub0diff/nn/denoiser.hpp"
#include "sub0diff/train/diffusion_loss.hpp"
#include "sub0diff/train/parallel.hpp"   // ITrainer, TrainStepResult, splitmix64

#include "sub0llm/autograd/variable.hpp"
#include "sub0llm/core/ops.hpp"          // zeros

#include <chrono>
#include <cstdint>
#include <random>
#include <span>
#include <vector>

namespace sub0diff::train {

class GpuTrainer : public ITrainer {
public:
    // `model` must already be on CUDA; `master_params` = model.parameters() (also on CUDA, the same
    // pointers the caller's optimizer steps). Knobs mirror ParallelTrainer's so behaviour matches a
    // W=1 CPU step (minus the multi-worker reduce, which is a no-op for a single stream).
    GpuTrainer(nn::Denoiser& model, std::vector<sub0llm::autograd::Variable*> master_params,
               std::int64_t seq_len, float t_min, float t_max, std::uint64_t seed,
               std::int64_t batch_size, bool shared_t, bool exact_count,
               std::span<const std::uint8_t> is_word_start, bool whole_word, bool contiguous)
        : model_(model), params_(std::move(master_params)),
          bctx_(batch_size, seq_len), rng_(static_cast<std::uint32_t>(seed)),
          t_min_(t_min), t_max_(t_max), seed_(seed), batch_size_(batch_size),
          shared_t_(shared_t), exact_count_(exact_count),
          is_word_start_(is_word_start), whole_word_(whole_word), contiguous_(contiguous) {
        for (auto* p : params_)   // master grads must exist on-device before backward accumulates
            if (p->grad().numel() == 0)
                p->grad() = sub0llm::zeros(p->data().shape(), p->data().dtype(), p->data().device());
    }

    void set_t_range(float t_min, float t_max) noexcept override { t_min_ = t_min; t_max_ = t_max; }
    [[nodiscard]] std::int64_t batch_size() const noexcept override { return batch_size_; }

    TrainStepResult step(std::span<const std::int32_t> stream,
                         std::span<const std::size_t> offsets) override {
        const auto t0 = std::chrono::steady_clock::now();
        for (auto* p : params_)   // ZERO grads on-device (the optimizer reads these same Variables)
            if (p->grad().numel() > 0)
                p->grad() = sub0llm::zeros(p->data().shape(), p->data().dtype(), p->data().device());

        const std::uint64_t step_seed = splitmix64(seed_ + 0xA5A5A5A5ull + step_count_) | 1ull;
        ++step_count_;
        // shared-t is sampled inside batched_diffusion_loss from [t_min_, t_max_]; for a single
        // stream the whole batch shares one t when shared_t_ — same √B consistency as the pool.
        auto res = batched_diffusion_loss(model_, stream, offsets, rng_, bctx_,
                                          t_min_, t_max_, shared_t_, exact_count_,
                                          is_word_start_, whole_word_, step_seed,
                                          /*window_base=*/0, contiguous_);
        res.loss.backward();
        const auto t1 = std::chrono::steady_clock::now();

        TrainStepResult r;
        r.mean_loss     = res.loss.data().item<float>();   // D2H scalar (item() brings to host)
        r.masked_tokens = res.n_masked;
        r.last_t        = res.mean_t;
        r.compute_s     = std::chrono::duration<double>(t1 - t0).count();
        return r;
    }

private:
    nn::Denoiser&                              model_;
    std::vector<sub0llm::autograd::Variable*>  params_;
    BatchedDiffusionLossContext                bctx_;
    std::mt19937                               rng_;
    float                                      t_min_, t_max_;
    std::uint64_t                              seed_, step_count_ = 0;
    std::int64_t                               batch_size_;
    bool                                       shared_t_, exact_count_, whole_word_, contiguous_;
    std::span<const std::uint8_t>              is_word_start_;
};

}  // namespace sub0diff::train
