#pragma once

// thread_pool.hpp — a persistent fork-join pool for INTRA-OP parallelism.
//
// The Ch29 measurement settled the threading model: per-window data parallelism
// (one autograd graph per worker) collapses on the global heap-allocator lock
// (~12% CPU at 21 threads). The work is ~85% compute, and a single GEMM's output
// rows partition across cores at 4.2× on 8 P-cores / 6× at 12 — with NO allocator
// contention because ONE thread builds the graph and only the matmul kernels fan
// out. This pool is that fan-out: a batched forward/backward runs on the caller
// thread and each large matmul calls parallel_for() to split its M rows.
//
// Design:
//   • Persistent workers pinned P-cores-first (shared cpu_topology model) — created
//     once, so per-call cost is just a wakeup, not thread spawn.
//   • Fork-join: parallel_for() blocks until the whole range is done; the CALLING
//     thread runs partition 0 itself (so an n-thread pool needs n-1 worker threads).
//   • No nested parallelism: the graph thread calls parallel_for serially, one op at
//     a time, so workers never re-enter. Not safe to call parallel_for concurrently
//     from multiple threads (single-producer by construction).

#include "sub0llm/core/cpu_topology.hpp"

#include <algorithm>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

namespace sub0llm {

// Per-thread opt-out for intra-op (matmul) threading. A thread that is ITSELF a
// data-parallel worker (e.g. a ParallelTrainer replica) sets this false so its
// matmuls stay serial — otherwise W workers each fanning out an 8-way GEMM would
// oversubscribe (W×8 threads). Default true: a lone graph thread fans GEMMs out.
[[nodiscard]] inline bool& intra_op_threading_enabled() {
    static thread_local bool enabled = true;
    return enabled;
}

class ThreadPool {
public:
    // Default size = physical P-cores (compute-bound sweet spot; E-cores add little
    // and hurt past ~12, per the GEMM probe). Clamped to [1, P-cores].
    static int default_threads() {
        const auto topo = detect_cpu_topology();
        const int pc = topo.n_perf_cores();
        return std::max(1, pc > 0 ? pc : topo.n_logical);
    }

    static ThreadPool& global() {
        static ThreadPool pool(default_threads());
        return pool;
    }

    explicit ThreadPool(int n_threads) : n_(std::max(1, n_threads)) {
        const auto pins = compute_pin_set(detect_cpu_topology(), static_cast<std::size_t>(n_));
        // Worker 0 is the calling thread (pinned lazily on first parallel_for); spawn
        // n_-1 persistent workers for partitions 1..n_-1.
        workers_.reserve(static_cast<std::size_t>(n_ - 1));
        for (int w = 1; w < n_; ++w) {
            const int pin = pins[static_cast<std::size_t>(w)];
            workers_.emplace_back([this, w, pin] {
                pin_current_thread(pin);
                worker_loop(w);
            });
        }
        caller_pin_ = pins.empty() ? -1 : pins[0];
    }

    ThreadPool(const ThreadPool&)            = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;

    ~ThreadPool() {
        {
            std::lock_guard<std::mutex> lk(m_);
            stop_ = true;
            ++generation_;
        }
        cv_.notify_all();
        for (auto& t : workers_) t.join();
    }

    [[nodiscard]] int size() const noexcept { return n_; }

    // Run fn(begin, end) over a contiguous partition of [0, n). Blocks until done.
    // Falls back to a single inline call when n is below `grain` or the pool is 1-wide,
    // so small ops pay zero synchronization cost.
    void parallel_for(std::int64_t n, const std::function<void(std::int64_t, std::int64_t)>& fn,
                      std::int64_t grain = 1) {
        if (n <= 0) return;
        const int nthreads = (n < grain * 2 || n_ == 1)
                                 ? 1
                                 : static_cast<int>(std::min<std::int64_t>(n_, (n + grain - 1) / grain));
        if (nthreads == 1) { fn(0, n); return; }

        const std::int64_t chunk = (n + nthreads - 1) / nthreads;
        {
            std::lock_guard<std::mutex> lk(m_);
            fn_       = &fn;
            n_items_  = n;
            chunk_    = chunk;
            active_   = nthreads;       // partitions 0..nthreads-1
            pending_  = nthreads - 1;   // workers (caller handles its own)
            ++generation_;
        }
        cv_.notify_all();

        run_partition(0);              // the caller runs partition 0

        std::unique_lock<std::mutex> lk(m_);
        done_cv_.wait(lk, [this] { return pending_ == 0; });
        fn_ = nullptr;
    }

private:
    void run_partition(int idx) {
        const std::int64_t begin = static_cast<std::int64_t>(idx) * chunk_;
        if (begin >= n_items_) return;
        const std::int64_t end = std::min(n_items_, begin + chunk_);
        (*fn_)(begin, end);
    }

    void worker_loop(int idx) {
        std::int64_t seen = 0;
        for (;;) {
            std::unique_lock<std::mutex> lk(m_);
            cv_.wait(lk, [&] { return generation_ != seen; });
            seen = generation_;
            if (stop_) return;
            const bool participate = idx < active_;
            lk.unlock();

            if (participate) run_partition(idx);

            lk.lock();
            if (participate && --pending_ == 0) {
                lk.unlock();
                done_cv_.notify_one();
            }
        }
    }

    int                                                  n_;
    int                                                  caller_pin_ = -1;
    std::vector<std::thread>                             workers_;
    const std::function<void(std::int64_t, std::int64_t)>* fn_ = nullptr;
    std::int64_t                                         n_items_ = 0, chunk_ = 0;
    int                                                  active_ = 0, pending_ = 0;
    std::int64_t                                         generation_ = 0;
    bool                                                 stop_ = false;
    std::mutex                                           m_;
    std::condition_variable                              cv_, done_cv_;
};

} // namespace sub0llm
