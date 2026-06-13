#pragma once

// parallel.hpp (Ch29 optimization) — data-parallel gradient accumulation.
//
// One training "step" becomes W windows, one per worker thread:
//
//   1. SYNC     each worker memcpys the master weights into its own replica
//               (~3.4 MB at Ch29 scale — ~0.2 ms done in parallel)
//   2. COMPUTE  each worker corrupts its window and runs forward+backward on its
//               OWN replica — no shared mutable state, no locks in the hot path
//   3. REDUCE   workers partition the parameter list (i % W == wid) and write the
//               mean of all replicas' gradients straight into the master's grad
//               tensors (assignment, so no zeroing pass is needed)
//
// The caller then clips + steps the master optimizer as usual. This is ALSO the
// gradient-accumulation convergence lever: the effective batch is W windows, so
// each Adam update sees a W× less noisy gradient.
//
// Thread safety rests on replica isolation: autograd graphs, loss contexts and
// RNGs are all per-worker. The master's tensors are only read during SYNC/COMPUTE
// and only written during REDUCE, with std::barrier separating the phases.
//
// FTZ note: workers inherit MXCSR from the thread that constructs the pool, so
// call sub0llm::init_cpu_compute() before constructing (main() already does).

#include "sub0diff/nn/denoiser.hpp"
#include "sub0diff/train/diffusion_loss.hpp"

#include "sub0llm/core/cpu_topology.hpp"

#include <atomic>
#include <barrier>
#include <cstring>
#include <random>
#include <span>
#include <thread>
#include <vector>

namespace sub0diff::train {

class ParallelTrainer {
public:
    struct Arch {
        std::int64_t vocab_size, embed_dim, n_layers, d_ff, seq_len;
        std::size_t  n_heads, n_kv_heads;
    };
    struct StepResult {
        float         mean_loss = 0.0f;   // mean per-window NELBO this step
        std::uint64_t masked_tokens = 0;  // summed over all workers
        float         last_t = 0.0f;      // a representative sampled noise level
    };

    ParallelTrainer(std::size_t n_workers, const Arch& arch,
                    std::vector<sub0llm::autograd::Variable*> master_params,
                    float t_min, float t_max, std::uint64_t seed)
        : master_(std::move(master_params)),
          t_min_(t_min), t_max_(t_max),
          barrier_(static_cast<std::ptrdiff_t>(n_workers + 1)) {
        // Master grads must exist before workers assign into them.
        for (auto* p : master_)
            if (p->grad().numel() == 0)
                p->grad() = sub0llm::zeros(p->data().shape());

        // Compute-bound pool: pin workers P-cores-first via the shared topology
        // (sub0llm/core/cpu_topology.hpp) — the project-wide thread-placement model.
        const auto pins = sub0llm::compute_pin_set(sub0llm::detect_cpu_topology(), n_workers);

        workers_.reserve(n_workers);
        for (std::size_t w = 0; w < n_workers; ++w) {
            auto& wk = *workers_.emplace_back(std::make_unique<Worker>(arch, seed + 1000 * (w + 1)));
            const int pin = pins[w];
            wk.thread = std::thread([this, &wk, w, n_workers, pin] {
                sub0llm::pin_current_thread(pin);
                worker_loop(wk, w, n_workers);
            });
        }
    }

    ParallelTrainer(const ParallelTrainer&) = delete;
    ParallelTrainer& operator=(const ParallelTrainer&) = delete;

    ~ParallelTrainer() {
        stop_.store(true, std::memory_order_release);
        barrier_.arrive_and_wait();          // release workers into the stop check
        for (auto& w : workers_) w->thread.join();
    }

    [[nodiscard]] std::size_t n_workers() const noexcept { return workers_.size(); }

    // Run one accumulated step: window w = stream[offsets[w] .. +seq_len).
    // offsets.size() must equal n_workers(). Master grads hold the MEAN gradient
    // on return; caller clips and steps its optimizer.
    StepResult step(std::span<const std::int32_t> stream,
                    std::span<const std::size_t> offsets) {
        stream_  = stream;
        offsets_ = offsets;
        barrier_.arrive_and_wait();          // → SYNC + COMPUTE
        barrier_.arrive_and_wait();          // compute done → REDUCE
        barrier_.arrive_and_wait();          // reduce done

        StepResult r;
        for (const auto& w : workers_) {
            r.mean_loss     += w->loss;
            r.masked_tokens += w->masked;
            r.last_t         = w->t;
        }
        r.mean_loss /= static_cast<float>(workers_.size());
        return r;
    }

private:
    struct Worker {
        nn::Denoiser                              model;
        std::vector<sub0llm::autograd::Variable*> params;
        DiffusionLossContext                      ctx;
        std::mt19937                              rng;
        std::thread                               thread;
        float         loss = 0.0f, t = 0.0f;
        std::uint32_t masked = 0;

        Worker(const Arch& a, std::uint64_t seed)
            : model(a.vocab_size, a.embed_dim, a.n_heads, a.n_kv_heads,
                    a.n_layers, a.d_ff, /*seed=*/42),   // arch identical to master
              ctx(a.seq_len),
              rng(static_cast<std::uint32_t>(seed)) {
            params = model.parameters();
        }
    };

    void worker_loop(Worker& wk, std::size_t wid, std::size_t n_workers) {
        const std::int64_t T = wk.ctx.ids_input.numel();
        for (;;) {
            barrier_.arrive_and_wait();      // wait for step() (or shutdown)
            if (stop_.load(std::memory_order_acquire)) {
                barrier_.arrive_and_drop();
                return;
            }

            // SYNC: master weights → replica (grad cleared by fresh accumulate).
            for (std::size_t i = 0; i < master_.size(); ++i) {
                auto&       dst = wk.params[i]->data();
                const auto& src = master_[i]->data();
                std::memcpy(dst.raw_ptr(), src.raw_ptr(),
                            static_cast<std::size_t>(src.numel()) * sizeof(float));
                auto& g = wk.params[i]->grad();
                if (g.numel() > 0)
                    std::memset(g.raw_ptr(), 0,
                                static_cast<std::size_t>(g.numel()) * sizeof(float));
            }

            // COMPUTE: this worker's window, forward + backward on the replica.
            const auto window = stream_.subspan(offsets_[wid], static_cast<std::size_t>(T));
            auto res = diffusion_loss(wk.model, window, wk.rng, wk.ctx, t_min_, t_max_);
            res.loss.backward();
            wk.loss   = res.loss.data().item<float>();
            wk.t      = res.t;
            wk.masked = res.n_masked;

            barrier_.arrive_and_wait();      // all replicas done → REDUCE

            // REDUCE: this worker owns params i ≡ wid (mod W); master.grad =
            // mean over replicas (assignment — no prior zeroing required).
            const float inv_w = 1.0f / static_cast<float>(n_workers);
            for (std::size_t i = wid; i < master_.size(); i += n_workers) {
                auto  mg = master_[i]->grad().data_as<float>();
                const std::size_t n = mg.size();
                for (std::size_t k = 0; k < n; ++k) mg[k] = 0.0f;
                for (const auto& other : workers_) {
                    const auto& og = other->params[i]->grad();
                    if (og.numel() == 0) continue;   // param unused this pass
                    const float* os = og.data_as<float>().data();
                    for (std::size_t k = 0; k < n; ++k) mg[k] += inv_w * os[k];
                }
            }

            barrier_.arrive_and_wait();      // reduce done → caller resumes
        }
    }

    std::vector<sub0llm::autograd::Variable*>  master_;
    std::vector<std::unique_ptr<Worker>>       workers_;
    std::span<const std::int32_t>              stream_;
    std::span<const std::size_t>               offsets_;
    float                                      t_min_, t_max_;
    std::atomic<bool>                          stop_{false};
    std::barrier<>                             barrier_;
};

} // namespace sub0diff::train
