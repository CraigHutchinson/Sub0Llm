// bench_parallel.cpp — isolated throughput benchmark for the Ch29 data-parallel
// trainer. No BPE, no eval, no checkpoints: just the hot loop, so we can see how
// pool->step() scales with worker count against the serial diffusion_loss path.
//
// Usage: ch29b_diffusion_training [--seq-len 64] [--iters 600] [--max-threads N]

#include "sub0diff/nn/denoiser.hpp"
#include "sub0diff/train/diffusion_loss.hpp"
#include "sub0diff/train/parallel.hpp"

#include "sub0llm/compat/print.hpp"
#include "sub0llm/core/cpu_topology.hpp"
#include "sub0llm/core/ops.hpp"
#include "sub0llm/core/pool_stats.hpp"
#include "sub0llm/core/runtime.hpp"
#include "sub0llm/core/tensor.hpp"
#include "sub0llm/core/thread_pool.hpp"
#include "sub0llm/nn/optimizer.hpp"

#include "sub0llm/autograd/ops.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <new>
#include <memory>
#include <random>
#include <span>
#include <string>
#include <thread>
#include <vector>

using namespace sub0llm;
namespace ag = sub0llm::autograd;
namespace dn = sub0diff::nn;
namespace dt = sub0diff::train;

// ── Allocation profiler (ch29b binary only — does NOT affect the library/tests) ──
// A global operator new/delete that COUNTS allocations (and histograms their size)
// when armed, then forwards to malloc/free. Lets --alloc-count measure exactly how
// many heap allocations a single training step does and at what sizes, so the
// zero-alloc substrate work targets the real offenders instead of guessing.
namespace allocprof {
    struct Counters {
        std::atomic<std::uint64_t> count{0};
        std::atomic<std::uint64_t> bytes{0};
        std::array<std::atomic<std::uint64_t>, 8> buckets{};   // <=32,64,128,256,512,1k,4k,>4k
        bool armed = false;
    };
    inline Counters& c() { static Counters g; return g; }
    inline void note(std::size_t n) {
        auto& g = c();
        if (!g.armed) return;
        g.count.fetch_add(1, std::memory_order_relaxed);
        g.bytes.fetch_add(n, std::memory_order_relaxed);
        int b = n <= 32 ? 0 : n <= 64 ? 1 : n <= 128 ? 2 : n <= 256 ? 3
              : n <= 512 ? 4 : n <= 1024 ? 5 : n <= 4096 ? 6 : 7;
        g.buckets[static_cast<std::size_t>(b)].fetch_add(1, std::memory_order_relaxed);
    }
}
void* operator new(std::size_t n) {
    allocprof::note(n);
    void* p = std::malloc(n ? n : 1);
    if (!p) throw std::bad_alloc{};
    return p;
}
void  operator delete(void* p) noexcept { std::free(p); }
void  operator delete(void* p, std::size_t) noexcept { std::free(p); }
void* operator new[](std::size_t n) { return ::operator new(n); }
void  operator delete[](void* p) noexcept { std::free(p); }
void  operator delete[](void* p, std::size_t) noexcept { std::free(p); }

int main(int argc, char** argv) {
    sub0llm::init_cpu_compute();

    std::int64_t seq_len = 64;
    std::uint64_t iters  = 600;
    int max_threads = 0;   // 0 = up to physical P-cores
    std::int64_t alloc_batch = 16;   // --alloc-count effective batch (per-worker = B/W)
    const std::int64_t V = 512, D = 128, n_layers = 4, d_ff = 384;
    const std::size_t n_heads = 8, n_kv_heads = 4;

    for (int i = 1; i < argc; ++i) {
        std::string_view a = argv[i];
        auto next = [&] { return std::string(argv[++i]); };
        if      (a == "--seq-len")     seq_len = std::stoll(next());
        else if (a == "--iters")       iters   = std::stoull(next());
        else if (a == "--max-threads") max_threads = std::stoi(next());
        else if (a == "--alloc-batch") alloc_batch = std::stoll(next());
    }

    bool alloc_probe = false, affinity_test = false, batch_bench = false, gemm_probe = false,
         profile_batch = false, alloc_count = false;
    for (int i = 1; i < argc; ++i) {
        std::string_view a = argv[i];
        if (a == "--alloc-probe")   alloc_probe = true;
        if (a == "--affinity-test") affinity_test = true;
        if (a == "--batch-bench")   batch_bench = true;
        if (a == "--gemm-probe")    gemm_probe = true;
        if (a == "--profile-batch") profile_batch = true;
        if (a == "--alloc-count")   alloc_count = true;
    }

    // ── Allocation profile: how many heap allocations does ONE step do? ──────────
    // Targets the zero-alloc substrate work at the real offenders. Uses the FOUNDED arch
    // (the real default: D=256/L6/d_ff=1024) and the BATCHED path with a real effective
    // batch B — i.e. one whole training step — so the count reflects what production runs.
    // NOTE: operator-new counts BlockPool (::operator new) + std/misc, but NOT TensorPool
    // buffers (those use _aligned_malloc). The pool-stats block below (SUB0LLM_POOL_STATS)
    // is the authoritative source for TensorPool MISS/EVICT — the heap-thrash signature.
    if (alloc_count) {
        const std::int64_t fV = 512, fD = 256, fL = 6, fdff = 1024;
        const std::size_t  fH = 8, fKV = 4;
        const std::int64_t Bb = alloc_batch;   // per-worker batch (real run: B/W); sweep with --alloc-batch
        dn::Denoiser model(fV, fD, fH, fKV, fL, fdff, /*seed=*/42);
        auto params = model.parameters();
        nn::Adam opt(params, 1e-3f);
        dt::BatchedDiffusionLossContext bctx(Bb, seq_len);
        std::mt19937 rng(1234);
        std::vector<std::int32_t> stream(1u << 16);
        std::uniform_int_distribution<std::int32_t> tk(0, static_cast<std::int32_t>(fV) - 1);
        for (auto& t : stream) t = tk(rng);
        std::span<const std::int32_t> ss(stream);
        std::uniform_int_distribution<std::size_t> od(0, stream.size() - static_cast<std::size_t>(seq_len) - 1);
        std::vector<std::size_t> offs(static_cast<std::size_t>(Bb));
        auto one = [&] {
            for (auto& o : offs) o = od(rng);
            auto r = dt::batched_diffusion_loss(model, ss, offs, rng, bctx, 0.02f, 1.0f);
            opt.zero_grad(); r.loss.backward();
            (void)nn::clip_grad_norm(params, 5.0f); opt.step();
        };
        for (int i = 0; i < 30; ++i) one();             // warmup (pools settle)
#ifdef SUB0LLM_POOL_STATS
        const auto ps0 = sub0llm::poolstats::snapshot();
#endif
        auto& gc = allocprof::c();                        // reset counters
        gc.count = 0; gc.bytes = 0;
        for (auto& bkt : gc.buckets) bkt = 0;
        gc.armed = true;
        const int N = 50;
        for (int i = 0; i < N; ++i) one();
        allocprof::c().armed = false;
        auto& g = allocprof::c();
        std::println("alloc-count: FOUNDED batched step (V={} D={} layers={} d_ff={} T={} B={})\n",
                     fV, fD, fL, fdff, seq_len, Bb);
        std::println("  {:.1f} operator-new/step   {:.1f} KB/step   ({:.2f} new/window)",
                     static_cast<double>(g.count.load()) / N,
                     static_cast<double>(g.bytes.load()) / N / 1024.0,
                     static_cast<double>(g.count.load()) / N / static_cast<double>(Bb));
        const char* names[] = {"<=32", "<=64", "<=128", "<=256", "<=512", "<=1K", "<=4K", ">4K"};
        std::println("  operator-new size histogram (allocs/step):");
        for (std::size_t b = 0; b < 8; ++b)
            std::println("    {:>5}  {:8.1f}", names[b],
                         static_cast<double>(g.buckets[b].load()) / N);
#ifdef SUB0LLM_POOL_STATS
        const auto ps1 = sub0llm::poolstats::snapshot();
        auto per = [&](std::uint64_t a, std::uint64_t b) { return static_cast<double>(b - a) / N; };
        std::println("\n  pool stats/step (SUB0LLM_POOL_STATS) — MISS+EVICT = real heap thrash:");
        std::println("    TensorPool  hit {:8.1f}  miss {:8.1f}  cache {:8.1f}  EVICT {:8.1f}",
                     per(ps0.tp_hit, ps1.tp_hit), per(ps0.tp_miss, ps1.tp_miss),
                     per(ps0.tp_cache, ps1.tp_cache), per(ps0.tp_evict, ps1.tp_evict));
        std::println("    BlockPool   hit {:8.1f}  miss {:8.1f}  cache {:8.1f}  EVICT {:8.1f}",
                     per(ps0.bp_hit, ps1.bp_hit), per(ps0.bp_miss, ps1.bp_miss),
                     per(ps0.bp_cache, ps1.bp_cache), per(ps0.bp_evict, ps1.bp_evict));
#else
        std::println("\n  (build with -DSUB0LLM_POOL_STATS for TensorPool/BlockPool miss/evict attribution)");
#endif
        return 0;
    }

    const auto topo = sub0llm::detect_cpu_topology();
    // ALL physical cores (P + E) — Arrow Lake-HX has no SMT, so every logical CPU is
    // its own core and compute-bound workers can use E-cores too (pinned P-first).
    const int n_phys = topo.n_perf_cores() + static_cast<int>(topo.eff_primary.size());
    if (max_threads <= 0) max_threads = std::max(1, n_phys);

    // ── Affinity self-test ──────────────────────────────────────────────────────
    // Spawn `max_threads` workers, pin each with the SAME compute_pin_set() the
    // ParallelTrainer uses, then have each busy-spin while sampling current_cpu().
    // A correctly pinned worker reports exactly ONE core == its pin every sample;
    // a worker seen on many cores is migrating (affinity silently failed) and would
    // wreck cache locality + scaling. This is the sanity check for "all cores show a
    // little load instead of N cores pegged".
    if (affinity_test) {
        const auto pins = sub0llm::compute_pin_set(topo, static_cast<std::size_t>(max_threads));
        std::println("affinity self-test: {} workers, pinning P-cores-first\n", max_threads);
        std::vector<int>  pin_ok(static_cast<std::size_t>(max_threads), 0);
        std::vector<int>  observed_distinct(static_cast<std::size_t>(max_threads), 0);
        std::vector<int>  observed_mode(static_cast<std::size_t>(max_threads), -1);
        std::vector<int>  stray(static_cast<std::size_t>(max_threads), 0);
        std::vector<std::thread> ts;
        std::atomic<int> ready{0};
        for (int w = 0; w < max_threads; ++w) {
            ts.emplace_back([&, w] {
                const int pin = pins[static_cast<std::size_t>(w)];
                pin_ok[static_cast<std::size_t>(w)] = sub0llm::pin_current_thread(pin) ? 1 : 0;
                ready.fetch_add(1);
                while (ready.load() < max_threads) {}          // start together
                // Busy-spin ~300 ms, tallying which core we actually run on.
                std::array<int, 64> seen{};
                volatile std::uint64_t sink = 0;
                const auto t_end = std::chrono::steady_clock::now() + std::chrono::milliseconds(300);
                int samples = 0, off_pin = 0;
                while (std::chrono::steady_clock::now() < t_end) {
                    for (int k = 0; k < 50000; ++k) sink += static_cast<std::uint64_t>(k) * k;
                    const int c = sub0llm::current_cpu();
                    if (c >= 0 && c < 64) ++seen[static_cast<std::size_t>(c)];
                    if (c != pin) ++off_pin;
                    ++samples;
                }
                int distinct = 0, mode = -1, modec = 0;
                for (int c = 0; c < 64; ++c) if (seen[static_cast<std::size_t>(c)]) {
                    ++distinct;
                    if (seen[static_cast<std::size_t>(c)] > modec) { modec = seen[static_cast<std::size_t>(c)]; mode = c; }
                }
                observed_distinct[static_cast<std::size_t>(w)] = distinct;
                observed_mode[static_cast<std::size_t>(w)]     = mode;
                stray[static_cast<std::size_t>(w)]             = (samples ? (100 * off_pin / samples) : 0);
            });
        }
        for (auto& t : ts) t.join();
        int bad = 0;
        std::println("  {:>6}  {:>4}  {:>8}  {:>8}  {:>9}  {}",
                     "worker", "pin", "ran-on", "distinct", "off-pin%", "verdict");
        for (int w = 0; w < max_threads; ++w) {
            const bool ok = pin_ok[static_cast<std::size_t>(w)] &&
                            observed_distinct[static_cast<std::size_t>(w)] == 1 &&
                            observed_mode[static_cast<std::size_t>(w)] == pins[static_cast<std::size_t>(w)];
            if (!ok) ++bad;
            std::println("  {:>6}  {:>4}  {:>8}  {:>8}  {:>8}%  {}",
                         w, pins[static_cast<std::size_t>(w)], observed_mode[static_cast<std::size_t>(w)],
                         observed_distinct[static_cast<std::size_t>(w)], stray[static_cast<std::size_t>(w)],
                         ok ? "OK" : (pin_ok[static_cast<std::size_t>(w)] ? "MIGRATING" : "PIN-FAILED"));
        }
        std::println("\n  {} / {} workers correctly pinned{}", max_threads - bad, max_threads,
                     bad ? "  ← AFFINITY BROKEN" : "  ← affinity OK");
        return 0;
    }

    // ── Allocation-contention probe ─────────────────────────────────────────────
    // Mimics the autograd graph's churn (make_shared<Node> + edge closures + small
    // vectors) with NO model maths, across 1..N threads pinned P-cores-first. If
    // this slows down per-thread as threads grow, the global heap lock — not compute
    // or bandwidth — is the scaling wall.
    if (alloc_probe) {
        std::println("alloc-probe: pure small-allocation churn (no model), {} reps/thread\n",
                     200000);
        constexpr int reps = 200000;
        auto churn = [] {
            // ~ the per-op allocation shape: a control block, a vector, two closures.
            volatile std::size_t sink = 0;
            for (int r = 0; r < reps; ++r) {
                auto n = std::make_shared<std::array<float, 8>>();
                std::vector<int> edges(3);
                std::function<int(int)> f = [n](int x) { return x + static_cast<int>((*n)[0]); };
                std::function<int(int)> g = [v = edges](int x) { return x + v[0]; };
                sink += reinterpret_cast<std::size_t>(n.get()) + f(1) + g(1);
            }
            return sink;
        };
        double base_tput = 0.0;
        for (int W = 1; W <= max_threads; ++W) {
            const auto pins = sub0llm::compute_pin_set(topo, static_cast<std::size_t>(W));
            auto t0 = std::chrono::steady_clock::now();
            std::vector<std::thread> ts;
            for (int w = 0; w < W; ++w)
                ts.emplace_back([&, w] { sub0llm::pin_current_thread(pins[static_cast<std::size_t>(w)]); churn(); });
            for (auto& t : ts) t.join();
            const double el = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - t0).count();
            const double tput = static_cast<double>(reps) * W / el;   // alloc-sets/s
            if (W == 1) base_tput = tput;
            std::println("  W={:<2}  {:8.2f} M-alloc-sets/s   {:6.2f} ms/thread   "
                         "(scaling {:.2f}x)",
                         W, tput / 1e6, 1e3 * el, tput / base_tput);
        }
        return 0;
    }
    // ── GEMM scaling probe ──────────────────────────────────────────────────────
    // Decisive test for Route B: does ONE big GEMM's work scale when its M=B·T output
    // rows are partitioned across cores? Buffers are PREALLOCATED (zero allocation in
    // the timed loop), so this isolates compute/bandwidth from the allocator. If
    // GFLOP/s scales ~linearly, the batched compute is parallelizable (Route B works);
    // if it plateaus, a memory-bandwidth wall caps us regardless of approach.
    if (gemm_probe) {
        const int64_t Bb = 32, Mrows = Bb * seq_len;     // batched FFN/LM-head M
        std::println("gemm-probe: partition M={} (B={},T={}) across cores; preallocated, best-of-3\n",
                     Mrows, Bb, seq_len);
        struct Shape { const char* name; int64_t M, K, N; bool bt; };
        const Shape shapes[] = {
            {"FFN-up    A(M,128)*W(128,384)", Mrows, D,            d_ff, false},
            {"FFN-down  A(M,384)*W(384,128)", Mrows, d_ff,         D,    false},
            {"LM-head   A(M,128)*Wt(513,128)", Mrows, D, static_cast<int64_t>(V) + 1, true},
        };
        for (const auto& sh : shapes) {
            sub0llm::Tensor A = sub0llm::randn({sh.M, sh.K});
            sub0llm::Tensor Bm = sh.bt ? sub0llm::randn({sh.N, sh.K}) : sub0llm::randn({sh.K, sh.N});
            sub0llm::Tensor C  = sub0llm::zeros({sh.M, sh.N});
            const double flop = 2.0 * static_cast<double>(sh.M) * sh.K * sh.N;
            double base = 0.0;
            std::println("  {}", sh.name);
            for (int W : {1, 2, 4, 6, 8, 12, 16, 24}) {
                if (W > max_threads) break;
                const auto pins = sub0llm::compute_pin_set(topo, static_cast<std::size_t>(W));
                double best = 0.0;
                for (int round = 0; round < 3; ++round) {
                    const int64_t chunk = (sh.M + W - 1) / W;
                    std::atomic<int> ready{0};
                    auto t0 = std::chrono::steady_clock::now();
                    std::vector<std::thread> ts;
                    for (int w = 0; w < W; ++w)
                        ts.emplace_back([&, w] {
                            sub0llm::pin_current_thread(pins[static_cast<std::size_t>(w)]);
                            const int64_t r0 = w * chunk, r1 = std::min(sh.M, r0 + chunk);
                            if (r1 <= r0) { ready.fetch_add(1); return; }
                            sub0llm::Tensor Ablk = sub0llm::ops::narrow(A, r0, r1 - r0);
                            sub0llm::Tensor Cblk = sub0llm::ops::narrow(C, r0, r1 - r0);
                            ready.fetch_add(1);
                            while (ready.load() < W) {}
                            for (int it = 0; it < 200; ++it) {
                                if (sh.bt) sub0llm::ops::matmul_bt_into(Ablk, Bm, Cblk);
                                else       sub0llm::ops::matmul_into(Ablk, Bm, Cblk);
                            }
                        });
                    for (auto& t : ts) t.join();
                    const double el = std::chrono::duration<double>(
                        std::chrono::steady_clock::now() - t0).count();
                    best = std::max(best, 200.0 * flop / el / 1e9);   // GFLOP/s
                }
                if (W == 1) base = best;
                std::println("    W={:<2}  {:7.1f} GFLOP/s   {:.2f}x", W, best, best / base);
            }
        }
        return 0;
    }

    std::println("bench_parallel: V={} D={} layers={} d_ff={} seq_len={} iters={}",
                 V, D, n_layers, d_ff, seq_len, iters);
    std::println("cpu: {} logical, {} physical cores = {} P-cores + {} E-cores (sweeping to {})\n",
                 topo.n_logical, n_phys, topo.n_perf_cores(), topo.eff_primary.size(), max_threads);

    // Synthetic token stream — large enough that offsets never collide trivially.
    std::mt19937 srng(123);
    std::uniform_int_distribution<std::int32_t> tok(0, static_cast<std::int32_t>(V) - 1);
    std::vector<std::int32_t> stream(1u << 16);
    for (auto& t : stream) t = tok(srng);
    const std::span<const std::int32_t> stream_span(stream);
    std::uniform_int_distribution<std::size_t> off_dist(
        0, stream.size() - static_cast<std::size_t>(seq_len) - 1);

    const double mean_masked = 0.5 * (0.02 + 1.0) * static_cast<double>(seq_len);

    // ── Batched-step profile: where does the per-window time actually go? ────────
    if (profile_batch) {
        const int64_t Bb = 32;
        std::println("profile-batch: B={}, T={}  (isolating forward / loss+backward / opt; pool size {})\n",
                     Bb, seq_len, sub0llm::ThreadPool::global().size());

        // Pool sanity: does parallel_for actually parallelize a heavy compute loop?
        {
            const std::int64_t N = 200'000'000;
            auto heavy = [](std::int64_t a, std::int64_t b) {
                volatile double s = 0; for (std::int64_t i = a; i < b; ++i) s += (i & 7); (void)s; };
            auto t1 = std::chrono::steady_clock::now(); heavy(0, N);
            const double serial = std::chrono::duration<double>(std::chrono::steady_clock::now() - t1).count();
            auto t2 = std::chrono::steady_clock::now();
            sub0llm::ThreadPool::global().parallel_for(N, [&](std::int64_t a, std::int64_t b){ heavy(a, b); }, 1000);
            const double par = std::chrono::duration<double>(std::chrono::steady_clock::now() - t2).count();
            std::println("  pool sanity: serial {:.3f}s  parallel {:.3f}s  -> {:.2f}x speedup\n", serial, par, serial / par);
        }

        dn::Denoiser model(V, D, n_heads, n_kv_heads, n_layers, d_ff, /*seed=*/42);
        auto params = model.parameters();
        nn::Adam opt(params, 1e-3f);
        std::mt19937 rng(1234);
        dt::BatchedDiffusionLossContext bctx(Bb, seq_len);
        std::vector<std::size_t> offs(static_cast<std::size_t>(Bb));
        for (auto& o : offs) o = off_dist(rng);
        std::vector<float> noise(static_cast<std::size_t>(Bb), 0.5f);
        sub0llm::Tensor ids({Bb * seq_len}, sub0llm::DType::Int32);
        for (auto& v : ids.data_as<std::int32_t>()) v = 0;

        const int reps = 60, warm = 15;
        auto timeit = [&](const char* label, auto&& body) {
            for (int i = 0; i < warm; ++i) body();
            auto t0 = std::chrono::steady_clock::now();
            for (int i = 0; i < reps; ++i) body();
            const double el = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            const double per_window_ms = 1e3 * el / reps / static_cast<double>(Bb);
            std::println("  {:<28} {:7.2f} ms/step   {:6.3f} ms/window   {:7.1f} windows/s",
                         label, 1e3 * el / reps, per_window_ms, reps * Bb / el);
        };

        timeit("forward only", [&] {
            volatile float sink = model.forward(ids, std::span<const float>(noise), Bb, seq_len)
                                      .data().data_as<float>()[0]; (void)sink; });
        timeit("forward + sum-loss + bwd", [&] {
            opt.zero_grad();
            ag::sum(model.forward(ids, std::span<const float>(noise), Bb, seq_len)).backward(); });
        timeit("forward + CE-loss + bwd + opt", [&] {
            auto r = dt::batched_diffusion_loss(model, stream_span, offs, rng, bctx, 0.02f, 1.0f);
            opt.zero_grad(); r.loss.backward();
            (void)nn::clip_grad_norm(params, 5.0f); opt.step(); });
        return 0;
    }

    // ── Batched-forward throughput (Ch29 F1): one (B·T) forward vs B per-window ──
    // Single-thread comparison isolating the architectural win (no threads): the
    // batched path amortizes per-op autograd + allocation overhead across B windows.
    if (batch_bench) {
        std::println("batch-bench: V={} D={} layers={} d_ff={} seq_len={}  (single thread)\n",
                     V, D, n_layers, d_ff, seq_len);
        dn::Denoiser model(V, D, n_heads, n_kv_heads, n_layers, d_ff, /*seed=*/42);
        auto params = model.parameters();
        nn::Adam opt(params, 1e-3f);
        std::mt19937 rng(1234);

        // Serial per-window reference.
        {
            dt::DiffusionLossContext ctx(seq_len);
            const std::uint64_t W = std::max<std::uint64_t>(40, iters / 4);
            for (std::uint64_t s = 0; s < W / 4 + 1; ++s) {   // warmup
                auto w = stream_span.subspan(off_dist(rng), static_cast<std::size_t>(seq_len));
                auto r = dt::diffusion_loss(model, w, rng, ctx, 0.02f, 1.0f);
                opt.zero_grad(); r.loss.backward(); opt.step();
            }
            std::uint64_t masked = 0;
            auto t0 = std::chrono::steady_clock::now();
            for (std::uint64_t s = 0; s < W; ++s) {
                auto w = stream_span.subspan(off_dist(rng), static_cast<std::size_t>(seq_len));
                auto r = dt::diffusion_loss(model, w, rng, ctx, 0.02f, 1.0f);
                opt.zero_grad(); r.loss.backward();
                (void)nn::clip_grad_norm(params, 5.0f); opt.step();
                masked += r.n_masked;
            }
            const double el = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            std::println("serial per-window     {:8.1f} windows/s   {:9.0f} masked-tok/s", W / el, masked / el);
        }

        // Batched: B windows per forward.
        for (int Bb : {4, 8, 16, 32, 64}) {
            dt::BatchedDiffusionLossContext bctx(Bb, seq_len);
            std::vector<std::size_t> offs(static_cast<std::size_t>(Bb));
            const std::uint64_t steps = std::max<std::uint64_t>(10, (iters / 4) / static_cast<std::uint64_t>(Bb));
            for (std::uint64_t s = 0; s < steps / 4 + 1; ++s) {  // warmup
                for (auto& o : offs) o = off_dist(rng);
                auto r = dt::batched_diffusion_loss(model, stream_span, offs, rng, bctx, 0.02f, 1.0f);
                opt.zero_grad(); r.loss.backward(); opt.step();
            }
            std::uint64_t masked = 0;
            auto t0 = std::chrono::steady_clock::now();
            for (std::uint64_t s = 0; s < steps; ++s) {
                for (auto& o : offs) o = off_dist(rng);
                auto r = dt::batched_diffusion_loss(model, stream_span, offs, rng, bctx, 0.02f, 1.0f);
                opt.zero_grad(); r.loss.backward();
                (void)nn::clip_grad_norm(params, 5.0f); opt.step();
                masked += r.n_masked;
            }
            const double el = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            const double windows = static_cast<double>(steps) * Bb;
            std::println("batched B={:<3}         {:8.1f} windows/s   {:9.0f} masked-tok/s", Bb, windows / el, masked / el);
        }
        return 0;
    }

    // ── Robust measurement: best-of-N with warmup, configs interleaved ──────────
    // Thermal drift makes a single cool→hot sweep over-report serial and under-report
    // high-W. We therefore run every config once per ROUND and take the BEST windows/s
    // across rounds (best = the least throttled / least contended sample, i.e. the
    // hardware's real capability). Configs alternate within a round so drift hits all.
    constexpr int rounds = 3;

    // One timed measurement of a serial (W==1) or pooled (W>=2, share toggle) config.
    // Returns achieved windows/s. Each call warms up (untimed) before the timed window.
    auto measure_serial = [&](std::uint64_t timed) {
        dn::Denoiser model(V, D, n_heads, n_kv_heads, n_layers, d_ff, /*seed=*/42);
        auto params = model.parameters();
        nn::Adam opt(params, 1e-3f);
        dt::DiffusionLossContext ctx(seq_len);
        std::mt19937 rng(1234);
        auto one = [&] {
            auto w = stream_span.subspan(off_dist(rng), static_cast<std::size_t>(seq_len));
            auto res = dt::diffusion_loss(model, w, rng, ctx, 0.02f, 1.0f);
            opt.zero_grad();
            res.loss.backward();
            (void)nn::clip_grad_norm(params, 5.0f);
            opt.step();
        };
        for (std::uint64_t s = 0; s < timed / 4 + 1; ++s) one();        // warmup
        auto t0 = std::chrono::steady_clock::now();
        for (std::uint64_t s = 0; s < timed; ++s) one();
        const double el = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - t0).count();
        return static_cast<double>(timed) / el;                          // windows/s
    };
    struct PoolStat { double win_s = 0, compute = 0, reduce = 0, opt = 0; };
    auto measure_pool = [&](int W, bool share, std::uint64_t timed_windows) {
        dn::Denoiser model(V, D, n_heads, n_kv_heads, n_layers, d_ff, /*seed=*/42);
        auto params = model.parameters();
        nn::Adam opt(params, 1e-3f);
        std::mt19937 rng(1234);
        dt::ParallelTrainer pool(
            static_cast<std::size_t>(W),
            dt::ParallelTrainer::Arch{V, D, n_layers, d_ff, seq_len, n_heads, n_kv_heads},
            params, 0.02f, 1.0f, 1234, share);
        std::vector<std::size_t> offsets(static_cast<std::size_t>(W));
        const std::uint64_t steps = std::max<std::uint64_t>(
            1, timed_windows / static_cast<std::uint64_t>(W));
        auto one = [&] {
            for (auto& o : offsets) o = off_dist(rng);
            auto res = pool.step(stream_span, offsets);
            (void)nn::clip_grad_norm(params, 5.0f);
            opt.step();
            return res;
        };
        for (std::uint64_t s = 0; s < steps / 4 + 1; ++s) one();        // warmup
        double compute_s = 0, reduce_s = 0, optclip_s = 0;
        auto t0 = std::chrono::steady_clock::now();
        for (std::uint64_t s = 0; s < steps; ++s) {
            auto res = one();
            compute_s += res.compute_s;
            reduce_s  += res.reduce_s;
        }
        const double el = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - t0).count();
        optclip_s = el - compute_s - reduce_s;   // master-side clip+opt+barrier latency
        const double windows = static_cast<double>(steps) * static_cast<double>(W);
        return PoolStat{windows / el, compute_s / el, reduce_s / el, optclip_s / el};
    };

    // W schedule: fine over the 8 P-cores, coarse over the 16 E-cores (P-cores fill
    // first, so 8 marks the P→E boundary). Trimmed to the machine's physical count.
    std::vector<int> wsched;
    for (int w : {2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 20, 24, 32})
        if (w <= max_threads) wsched.push_back(w);
    if (wsched.empty() || wsched.back() != max_threads) wsched.push_back(max_threads);

    // Accumulate best-of-rounds. Index 0 = serial; [W][share] for pooled.
    double best_serial = 0.0;
    std::vector<std::array<PoolStat, 2>> best(static_cast<std::size_t>(max_threads) + 1);
    const std::uint64_t timed = std::max<std::uint64_t>(200, iters / 2);
    for (int r = 0; r < rounds; ++r) {
        best_serial = std::max(best_serial, measure_serial(timed));
        for (int W : wsched)
            for (int sh = 1; sh >= 0; --sh) {        // shared first, then replicated
                auto st = measure_pool(W, sh != 0, timed);
                auto& slot = best[static_cast<std::size_t>(W)][static_cast<std::size_t>(sh)];
                if (st.win_s > slot.win_s) slot = st;
            }
    }

    const double tokens_per_win = mean_masked;
    std::println("serial (1 worker):       {:7.1f} windows/s   {:8.0f} masked-tok/s",
                 best_serial, best_serial * tokens_per_win);
    std::println("");
    std::println("  {:>2}  {:>10} {:>10}   {:>6}  region  | phase (shared) compute/reduce/opt",
                 "W", "shared w/s", "replica w/s", "speedup");
    const int pc = topo.n_perf_cores();
    for (int W : wsched) {
        const auto& sh = best[static_cast<std::size_t>(W)][1];
        const auto& rp = best[static_cast<std::size_t>(W)][0];
        const char* region = (W <= pc) ? "P    " : "P+E  ";
        std::println("  {:>2}  {:10.1f} {:10.1f}   {:5.2f}x  {}   | {:4.0f}% / {:4.0f}% / {:4.0f}%  "
                     "(shared {:+.0f}%)",
                     W, sh.win_s, rp.win_s, sh.win_s / best_serial, region,
                     100.0 * sh.compute, 100.0 * sh.reduce, 100.0 * sh.opt,
                     100.0 * (sh.win_s / rp.win_s - 1.0));
    }
    return 0;
}
