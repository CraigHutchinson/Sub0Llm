#pragma once

// parallel.hpp (Ch29 optimization) — data-parallel gradient accumulation.
//
// One training "step" becomes W windows, one per worker thread:
//
//   1. ZERO     each worker clears its own (private) gradient accumulators
//   2. COMPUTE  each worker corrupts its window and runs forward+backward reading
//               the SHARED master weights — no per-step weight copy, no locks
//   3. REDUCE   workers partition the parameter list (i % W == wid) and write the
//               mean of all replicas' gradients straight into the master's grad
//               tensors (assignment, so no zeroing pass is needed)
//
// The caller then clips + steps the master optimizer as usual. This is ALSO the
// gradient-accumulation convergence lever: the effective batch is W windows, so
// each Adam update sees a W× less noisy gradient.
//
// SHARED WEIGHTS (not replicated): at construction each worker's leaf parameter
// *data* storage is aliased onto the master's buffer, so all W workers + master read
// ONE 6.4 MB weight copy. This was the key throughput fix — W private replicas total
// W×6.4 MB, which blows past L3 (~30 MB) at W≳5 and forces every forward to stream
// weights from DRAM; the workers then contend for memory bandwidth and compute stops
// scaling. One shared copy stays L3-resident. The master mutates the weights in place
// in opt.step() AFTER pool.step() returns (all barriers passed), so the read-only
// COMPUTE phase never races the write. Only the gradient accumulators stay private.
//
// Thread safety: gradients/autograd graphs/loss contexts/RNGs are all per-worker;
// weight DATA is shared but read-only within COMPUTE. The std::barrier phases
// separate every worker read of the weights from the single master write.
//
// FTZ note: each worker thread sets FTZ+DAZ ITSELF (init_cpu_compute in the thread body) —
// MXCSR is per-thread and a new std::thread does NOT inherit it on Windows. main() also calls
// it, but that only covers the main/reduce thread, not these workers where the compute runs.

#include "sub0diff/nn/denoiser.hpp"
#include "sub0diff/train/diffusion_loss.hpp"

#include "sub0llm/core/cpu_topology.hpp"
#include "sub0llm/core/runtime.hpp"      // init_cpu_compute (FTZ+DAZ) — set per worker thread
#include "sub0llm/core/thread_pool.hpp"

#include <atomic>
#include <barrier>
#include <chrono>
#include <cstring>
#include <random>
#include <span>
#include <thread>
#include <vector>

namespace sub0diff::train {

// One step's outcome, shared across trainer back-ends (CPU pool / single GPU stream).
struct TrainStepResult {
    float         mean_loss = 0.0f;   // mean per-window NELBO this step
    std::uint64_t masked_tokens = 0;  // summed over all windows
    float         last_t = 0.0f;      // a representative sampled noise level
    // Master-side phase timing (≈free: 2 clock reads/step) — lets a caller see
    // the parallel-overhead split. compute_s = SYNC+forward+backward (the useful,
    // perfectly-parallel part); reduce_s = the barrier-bounded gradient reduction.
    double        compute_s = 0.0;
    double        reduce_s  = 0.0;
};

// Trainer interface: main()'s loop drives a back-end through set_t_range()/step() and is
// agnostic to whether it's the CPU worker pool or the single-GPU-stream trainer (gpu_trainer.hpp).
class ITrainer {
public:
    virtual ~ITrainer() = default;
    virtual void set_t_range(float t_min, float t_max) noexcept = 0;
    virtual TrainStepResult step(std::span<const std::int32_t> stream,
                                 std::span<const std::size_t> offsets) = 0;
    [[nodiscard]] virtual std::int64_t batch_size() const noexcept = 0;
};

class ParallelTrainer : public ITrainer {
public:
    using StepResult = TrainStepResult;   // back-compat alias (internal use)
    struct Arch {
        std::int64_t vocab_size, embed_dim, n_layers, d_ff, seq_len;
        std::size_t  n_heads, n_kv_heads;
    };

    ParallelTrainer(std::size_t n_workers, const Arch& arch,
                    std::vector<sub0llm::autograd::Variable*> master_params,
                    float t_min, float t_max, std::uint64_t seed,
                    bool share_weights = true,
                    std::span<const std::uint8_t> is_word_start = {},
                    bool whole_word = false,
                    std::vector<int> pin_override = {},
                    bool exact_count = false,
                    std::int64_t batch_size = 0, // total windows/step B (0 ⇒ = n_workers)
                    bool shared_t = false,
                    bool contiguous = false)
        : master_(std::move(master_params)),
          t_min_(t_min), t_max_(t_max), share_weights_(share_weights),
          is_word_start_(is_word_start), whole_word_(whole_word), exact_count_(exact_count),
          shared_t_(shared_t), contiguous_(contiguous),
          windows_per_worker_(batch_size <= 0 ? 1 : batch_size / static_cast<std::int64_t>(n_workers)),
          batch_size_(windows_per_worker_ * static_cast<std::int64_t>(n_workers)),
          master_rng_(static_cast<std::uint32_t>(seed ^ 0x9E3779B9u)),
          seed_(seed),
          barrier_(static_cast<std::ptrdiff_t>(n_workers + 1)) {
        // Decouple consistency (B = batch windows/step → gradient quality ∝ √B) from parallelism
        // (W = n_workers → throughput). Each worker computes a B/W-window batched mini-gradient;
        // the reduce averages over workers, so the master gradient is the mean over all B windows
        // and is IDENTICAL for any W (only wall-time changes). W=1 is just a pool of one.
        // Master grads must exist before workers assign into them.
        for (auto* p : master_)
            if (p->grad().numel() == 0)
                p->grad() = sub0llm::zeros(p->data().shape());

        // Compute-bound pool: pin workers P-cores-first via the shared topology
        // (sub0llm/core/cpu_topology.hpp) — the project-wide thread-placement model.
        // An explicit pin_override (from --pin) lets concurrent trainers take DISJOINT
        // cores so they don't oversubscribe the same P-cores.
        const auto pins = pin_override.empty()
            ? sub0llm::compute_pin_set(sub0llm::detect_cpu_topology(), n_workers)
            : pin_override;

        workers_.reserve(n_workers);
        for (std::size_t w = 0; w < n_workers; ++w) {
            auto& wk = *workers_.emplace_back(
                std::make_unique<Worker>(arch, seed + 1000 * (w + 1), windows_per_worker_));
            // Alias each replica's weight DATA onto the master's storage (shallow
            // Tensor copy shares the underlying buffer): one L3-resident weight copy,
            // no per-step SYNC, and the worker's own randomly-initialised weight
            // buffers are released here. Grads stay private (not aliased).
            if (share_weights_)
                for (std::size_t i = 0; i < master_.size(); ++i)
                    wk.params[i]->data() = master_[i]->data();
            const int pin = pins[w];
            wk.thread = std::thread([this, &wk, w, n_workers, pin] {
                sub0llm::pin_current_thread(pin);
                // FTZ+DAZ must be set PER THREAD: a freshly created std::thread starts with the
                // DEFAULT MXCSR on Windows (it does NOT inherit the constructing thread's), so the
                // main()-level init_cpu_compute() never reached these workers — and since ALL the
                // training compute runs here, the model slid into denormal-stall territory as the
                // loss fell (the exact 50-100×/op penalty runtime.hpp documents). Set it in-thread.
                sub0llm::init_cpu_compute();
                // W>1: each replica IS the parallel unit — keep its matmuls serial so we don't
                // oversubscribe (W workers × an 8-way GEMM each). W==1: the lone worker fans its
                // (B·T)-row GEMMs across all cores (no other worker to contend with).
                sub0llm::intra_op_threading_enabled() = (n_workers == 1);
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

    // Update the noise band workers sample t from (the curriculum's dynamic ceiling).
    // Read by each worker at the top of its next step — set it before step().
    void set_t_range(float t_min, float t_max) noexcept override {
        t_min_.store(t_min, std::memory_order_relaxed);
        t_max_.store(t_max, std::memory_order_relaxed);
    }

    [[nodiscard]] std::int64_t batch_size() const noexcept override { return batch_size_; }

    // Run one accumulated step over B = batch_size() windows: offsets.size() must equal batch_size().
    // Worker w processes offsets[w·B/W .. (w+1)·B/W). Master grads hold the MEAN gradient over
    // all B windows on return (IDENTICAL for any W); caller clips and steps its optimizer.
    StepResult step(std::span<const std::int32_t> stream,
                    std::span<const std::size_t> offsets) override {
        stream_  = stream;
        offsets_ = offsets;
        // shared-t: ONE noise level for the whole step (all B windows) → within-level consistency
        // ∝ √B. Sampled once on the master from the current band; workers read it. Independent-t
        // (shared_t_=false) lets each window draw its own t (the diluted baseline).
        if (shared_t_) {
            std::uniform_real_distribution<float> td(t_min_.load(std::memory_order_relaxed),
                                                     t_max_.load(std::memory_order_relaxed));
            shared_ts_.store(td(master_rng_), std::memory_order_relaxed);
        }
        // Per-window seed base for THIS step — workers hash (this | global window index) so the
        // masking is a pure function of (step, window), identical for any worker count W.
        step_seed_.store(splitmix64(seed_ + 0xA5A5A5A5ull + step_count_) | 1ull,
                         std::memory_order_relaxed);
        ++step_count_;
        const auto t0 = std::chrono::steady_clock::now();
        barrier_.arrive_and_wait();          // → SYNC + COMPUTE
        barrier_.arrive_and_wait();          // compute done → REDUCE
        const auto t1 = std::chrono::steady_clock::now();
        barrier_.arrive_and_wait();          // reduce done
        const auto t2 = std::chrono::steady_clock::now();

        StepResult r;
        r.compute_s = std::chrono::duration<double>(t1 - t0).count();
        r.reduce_s  = std::chrono::duration<double>(t2 - t1).count();
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
        BatchedDiffusionLossContext               bctx;   // this worker's B/W-window mini-batch
        std::mt19937                              rng;
        std::thread                               thread;
        float         loss = 0.0f, t = 0.0f;
        std::uint32_t masked = 0;

        Worker(const Arch& a, std::uint64_t seed, std::int64_t per_worker)
            : model(a.vocab_size, a.embed_dim, a.n_heads, a.n_kv_heads,
                    a.n_layers, a.d_ff, /*seed=*/42),   // arch identical to master
              bctx(per_worker, a.seq_len),
              rng(static_cast<std::uint32_t>(seed)) {
            params = model.parameters();
        }
    };

    void worker_loop(Worker& wk, std::size_t wid, std::size_t n_workers) {
        const std::int64_t per = windows_per_worker_;          // windows this worker owns
        for (;;) {
            barrier_.arrive_and_wait();      // wait for step() (or shutdown)
            if (stop_.load(std::memory_order_acquire)) {
                barrier_.arrive_and_drop();
                return;
            }

            // ZERO (+ optional SYNC): weights are SHARED read-only with the master
            // (aliased once at construction) so there is normally no per-step weight
            // copy — workers observe the master's in-place opt.step() directly. With
            // share_weights=false (A/B fallback) we instead memcpy master→replica.
            // Either way the private gradient accumulators are cleared before backward.
            for (std::size_t i = 0; i < master_.size(); ++i) {
                if (!share_weights_)
                    std::memcpy(wk.params[i]->data().raw_ptr(), master_[i]->data().raw_ptr(),
                                static_cast<std::size_t>(master_[i]->data().numel()) * sizeof(float));
                auto& g = wk.params[i]->grad();
                if (g.numel() > 0)
                    std::memset(g.raw_ptr(), 0,
                                static_cast<std::size_t>(g.numel()) * sizeof(float));
            }

            // COMPUTE: this worker's B/W-window mini-batch, one batched forward + backward
            // (reads shared weights). For shared-t the band collapses to the master's single ts so
            // all windows (across all workers) mask at the same level → global √B consistency.
            auto my_offsets = offsets_.subspan(static_cast<std::size_t>(wid) * static_cast<std::size_t>(per),
                                               static_cast<std::size_t>(per));
            const float lo = shared_t_ ? shared_ts_.load(std::memory_order_relaxed)
                                       : t_min_.load(std::memory_order_relaxed);
            const float hi = shared_t_ ? shared_ts_.load(std::memory_order_relaxed)
                                       : t_max_.load(std::memory_order_relaxed);
            auto res = batched_diffusion_loss(wk.model, stream_, my_offsets, wk.rng, wk.bctx,
                                              lo, hi, shared_t_, exact_count_,
                                              is_word_start_, whole_word_,
                                              step_seed_.load(std::memory_order_relaxed),
                                              static_cast<std::int64_t>(wid) * per, contiguous_);
            res.loss.backward();
            wk.loss   = res.loss.data().item<float>();
            wk.t      = res.mean_t;
            wk.masked = static_cast<std::uint32_t>(res.n_masked);

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
    std::atomic<float>                         t_min_, t_max_;
    bool                                       share_weights_ = true;
    std::span<const std::uint8_t>              is_word_start_;   // read-only, owned by caller
    bool                                       whole_word_ = false;
    bool                                       exact_count_ = false;
    bool                                       shared_t_ = false;
    bool                                       contiguous_ = false;  // mask a contiguous span (§13.4)
    std::int64_t                               windows_per_worker_ = 1;  // windows/worker = B/W
    std::int64_t                               batch_size_ = 0;       // total windows/step B = per·W
    std::atomic<float>                         shared_ts_{0.0f}; // the step's shared noise level
    std::mt19937                               master_rng_;      // shared-t sampling (master only)
    std::uint64_t                              seed_ = 0;        // base seed for per-window hashing
    std::uint64_t                              step_count_ = 0;  // master-only step counter
    std::atomic<std::uint64_t>                 step_seed_{1};    // this step's per-window seed base
    std::atomic<bool>                          stop_{false};
    std::barrier<>                             barrier_;
};

} // namespace sub0diff::train
