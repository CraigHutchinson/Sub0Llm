#include "sub0llm/nn/gemma.hpp"

#include "../backends/cpu/kernels.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <format>
#include <functional>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>

#if defined(_M_X64) || defined(__x86_64__) || defined(__i386__)
#  include <immintrin.h>
#  define GEMMA_CPU_RELAX() _mm_pause()
#else
#  define GEMMA_CPU_RELAX() std::this_thread::yield()
#endif

#ifdef _WIN32
#  ifndef NOMINMAX
#    define NOMINMAX
#  endif
#  include <windows.h>
#endif

namespace sub0llm::nn {

// Configurable GEMV thread count (0 = auto). Set via set_gemma_threads().
namespace { int g_gemma_threads = 0; }
void set_gemma_threads(int n) { g_gemma_threads = n < 0 ? 0 : n; }

// GEMV fusion toggle (Q/K/V and gate/up under one barrier). On by default; the off path
// dispatches each output separately — for drift-free A/B of the fusion optimization.
namespace { bool g_gemma_fuse = true; }
void set_gemma_fuse(bool on) { g_gemma_fuse = on; }

namespace {

namespace cpu = backend::cpu;

constexpr int64_t QK = cpu::QK8_0;   // 32

void pin_thread_to_cpu(int cpu) {
#ifdef _WIN32
    SetThreadAffinityMask(GetCurrentThread(), static_cast<DWORD_PTR>(1) << (cpu & 63));
#else
    (void)cpu;
#endif
}

// One output of a fused GEMV group: y[m] = dot(W + m*kblk, xq, kblk), m in [0, M).
// Several jobs that share the SAME activation (same K) are dispatched under ONE barrier
// (e.g. Q/K/V from attn_norm, or gate/up from ffn_norm) — fewer sync points per token and
// a larger, better-balanced row space than three small separate dispatches.
struct GemvJob { const cpu::BlockQ8_0* W; float* y; int64_t M; };

// Persistent thread pool for the Q8 GEMVs, tuned for a single-token decode (which is
// DRAM-bandwidth-bound: one core can't saturate memory, so the win is many cores
// streaming weights concurrently). Three things make this scale where the naive
// per-GEMV fork-join didn't, matching ggml's threadpool design:
//   1. Persistent workers — spawned once, not per GEMV (~340 / token).
//   2. SPIN-WAIT barrier (not a condition variable) — at ~340 sync points per token,
//      a CV's ~10 us kernel wake dominates; workers instead busy-poll an atomic
//      generation counter with _mm_pause, so dispatch latency is tens of ns.
//   3. Thread AFFINITY — each worker (and the caller) is pinned to a distinct logical
//      CPU (SetThreadAffinityMask). This stops the OS migrating threads (which thrashes
//      the per-core L2 holding the hot quantized activation) and, on this hybrid
//      8 P-core + 16 E-core part, places the first/few threads on the P-cores.
// The caller participates as one of the N threads (so N total = nworkers + 1), and rows
// are handed out as CHUNK-row tiles from an atomic cursor — dynamic load-balancing that
// naturally compensates for the slower E-cores.
class GemmaPool {
public:
    explicit GemmaPool(int total_threads) : nworkers_(std::max(0, total_threads - 1)) {
        pin_thread_to_cpu(0);                       // caller runs on CPU 0
        for (int i = 0; i < nworkers_; ++i)
            workers_.emplace_back([this, i] { worker(i + 1); });
    }
    ~GemmaPool() {
        stop_.store(true, std::memory_order_relaxed);
        gen_.fetch_add(1, std::memory_order_release);   // release the spinning workers
        for (auto& t : workers_) t.join();
    }
    [[nodiscard]] int total() const noexcept { return nworkers_ + 1; }

    // Run `njobs` GEMV outputs (sharing the activation `xq`, kblk blocks each) under one
    // barrier. `offsets` is the prefix sum of job row counts (length njobs+1); rows are
    // the concatenated space [0, offsets[njobs]). Caller participates, then waits.
    void run(const GemvJob* jobs, int njobs, const int64_t* offsets,
             const cpu::BlockQ8_0* xq, int64_t kblk) {
        jobs_ = jobs; njobs_ = njobs; offsets_ = offsets;
        xq_ = xq; kblk_ = kblk; total_ = offsets[njobs];
        gtask_ = nullptr;                               // GEMV mode
        next_.store(0, std::memory_order_relaxed);
        remaining_.store(nworkers_, std::memory_order_relaxed);
        gen_.fetch_add(1, std::memory_order_release);   // publishes the job state
        work();                                         // caller participates
        while (remaining_.load(std::memory_order_acquire) != 0) GEMMA_CPU_RELAX();
    }

    // Generic parallel-for: call (*task)(lo, hi) over tiles covering [0, n). Used to move
    // the per-head attention work (independent across heads) off the single caller thread.
    void run_generic(int64_t n, const std::function<void(int64_t, int64_t)>* task) {
        gtask_ = task; total_ = n;
        next_.store(0, std::memory_order_relaxed);
        remaining_.store(nworkers_, std::memory_order_relaxed);
        gen_.fetch_add(1, std::memory_order_release);
        work();
        while (remaining_.load(std::memory_order_acquire) != 0) GEMMA_CPU_RELAX();
    }

private:
    static constexpr int64_t CHUNK = 32;

    void work() {
        if (gtask_) { work_generic(); return; }
        const GemvJob* jobs = jobs_; const int64_t* off = offsets_;
        const cpu::BlockQ8_0* xq = xq_; const int64_t kblk = kblk_, total = total_;
        const int nj = njobs_;
        for (;;) {
            const int64_t i = next_.fetch_add(CHUNK, std::memory_order_relaxed);
            if (i >= total) break;
            const int64_t hi = std::min(total, i + CHUNK);
            int j = 0; while (j + 1 < nj && i >= off[j + 1]) ++j;    // job owning row i
            for (int64_t g = i; g < hi; ++g) {
                while (j + 1 < nj && g >= off[j + 1]) ++j;           // advance across boundary
                const int64_t lr = g - off[j];
                jobs[j].y[lr] = cpu::dot_q8_0_q8_0(jobs[j].W + lr * kblk, xq, kblk);
            }
        }
    }
    // Small index spaces (head counts) — grab one index at a time for even distribution.
    void work_generic() {
        const auto& task = *gtask_; const int64_t total = total_;
        for (;;) {
            const int64_t i = next_.fetch_add(1, std::memory_order_relaxed);
            if (i >= total) break;
            task(i, i + 1);
        }
    }
    void worker(int cpu) {
        pin_thread_to_cpu(cpu);
        int local = 0;
        for (;;) {
            while (gen_.load(std::memory_order_acquire) == local) GEMMA_CPU_RELAX();
            local = gen_.load(std::memory_order_relaxed);
            if (stop_.load(std::memory_order_relaxed)) return;
            work();
            remaining_.fetch_sub(1, std::memory_order_release);  // publishes y writes
        }
    }

    int                          nworkers_;
    std::vector<std::thread>     workers_;
    std::atomic<int>             gen_{0};
    std::atomic<bool>            stop_{false};
    std::atomic<int>             remaining_{0};
    std::atomic<int64_t>         next_{0};
    const GemvJob*               jobs_    = nullptr;
    const int64_t*               offsets_ = nullptr;
    const cpu::BlockQ8_0*        xq_      = nullptr;
    const std::function<void(int64_t, int64_t)>* gtask_ = nullptr;  // non-null = generic
    int                          njobs_   = 0;
    int64_t                      total_   = 0, kblk_ = 0;
};

int desired_threads() {
    if (g_gemma_threads > 0) return g_gemma_threads;
    unsigned hw = std::thread::hardware_concurrency();
    if (hw == 0) return 1;
    // Single-token decode is DRAM-bandwidth-bound and saturates at the memory knee, well
    // before all cores; past it the extra (E-)cores add contention not bandwidth, and
    // spin-wait workers on every core starve the OS. MEASURED on this 8 P + 16 E (24-thread)
    // part: 16 threads is the peak for BOTH us (6.47 tok/s) and llama (5.94) — 16 beats 20
    // (6.02) and 24 (5.15). Default to ~2/3 of cores (= 16 here); -t overrides per machine.
    const unsigned n = hw > 6 ? (hw * 2u) / 3u : hw;
    return static_cast<int>(std::clamp<unsigned>(n, 1u, 32u));
}

// Get (lazily build / rebuild on thread-count change) the process-wide pool. Called
// only from the single generation thread, so no locking on the pool pointer is needed.
GemmaPool* get_pool() {
    static std::unique_ptr<GemmaPool> pool;
    const int want = desired_threads();
    if (want <= 1) { pool.reset(); return nullptr; }
    if (!pool || pool->total() != want) pool = std::make_unique<GemmaPool>(want);
    return pool.get();
}

// Fused parallel Q8 GEMV: dispatch up to `njobs` outputs (each y[m] = W·x, W row-major
// (out,in) Q8_0) that share the same f32 activation `x` (K wide) under ONE barrier.
// Quantizes x once. Used to fuse Q/K/V (from attn_norm) and gate/up (from ffn_norm).
void pmatvec_multi(const GemvJob* jobs, int njobs, const float* x, int64_t K) {
    const int64_t nb = K / QK;
    std::vector<cpu::BlockQ8_0> xq(static_cast<std::size_t>(nb));
    cpu::quantize_row_q8_0(x, xq.data(), K);

    int64_t offsets[8];
    offsets[0] = 0;
    for (int j = 0; j < njobs; ++j) offsets[j + 1] = offsets[j] + jobs[j].M;
    const int64_t total = offsets[njobs];

    GemmaPool* pool = get_pool();
    if (!pool || total < 128) {                   // tiny: not worth dispatching
        for (int j = 0; j < njobs; ++j)
            for (int64_t m = 0; m < jobs[j].M; ++m)
                jobs[j].y[m] = cpu::dot_q8_0_q8_0(jobs[j].W + m * nb, xq.data(), nb);
        return;
    }
    if (g_gemma_fuse) {
        pool->run(jobs, njobs, offsets, xq.data(), nb);
    } else {                                       // A/B: one barrier per output
        for (int j = 0; j < njobs; ++j) {
            const int64_t off[2] = {0, jobs[j].M};
            pool->run(&jobs[j], 1, off, xq.data(), nb);
        }
    }
}

// Single Q8 GEMV: y[M] = W[M,K]·x[K]. Thin wrapper over the fused path (one job).
void pmatvec(const std::vector<cpu::BlockQ8_0>& W, const float* x, float* y,
             int64_t M, int64_t K) {
    const GemvJob job{W.data(), y, M};
    pmatvec_multi(&job, 1, x, K);
}

// Parallelize an independent per-index loop (e.g. attention heads) across the pool —
// moves serial main-thread work onto the idle cores. `body(i)` for i in [0, n).
// `g_gemma_fuse` also gates this (the attention parallelization is part of the same
// orchestration optimization), so --no-fuse measures the fully-serial baseline.
template <typename F>
void parallel_for(int64_t n, F body) {
    GemmaPool* pool = get_pool();
    if (!pool || !g_gemma_fuse || n <= 1) {
        for (int64_t i = 0; i < n; ++i) body(i);
        return;
    }
    const std::function<void(int64_t, int64_t)> task =
        [&](int64_t lo, int64_t hi) { for (int64_t i = lo; i < hi; ++i) body(i); };
    pool->run_generic(n, &task);
}

// Batched Q8 GEMM for prompt prefill: Y[M,T] = W[M,K]·Xqᵀ, W row-major (out,in) Q8_0,
// Xq the T pre-quantized activation columns (T, K/QK row-major). Parallel over the M
// output rows; each W row is streamed ONCE from RAM and reused across all T columns
// (cache-resident) — so weight traffic is M·K, not T·M·K as T separate GEMVs would be.
// That row-reuse is exactly why batched prefill beats sequential forward_one for T>1.
void pmatmul(const std::vector<cpu::BlockQ8_0>& W, const cpu::BlockQ8_0* Xq, float* Y,
             int64_t M, int64_t K, int64_t T) {
    const int64_t nb = K / QK;
    const cpu::BlockQ8_0* Wp = W.data();
    // Each weight row is 4-column register-blocked across the T tokens (gemm_row_q8_0):
    // the per-block weight decode (|w| + f16 scale) is done once and reused across 4
    // columns, turning the compute-bound prefill GEMM from per-column GEMVs into a tiled
    // matmul. Parallel over the M output rows.
    parallel_for(M, [&](int64_t m) {
        cpu::gemm_row_q8_0(Wp + m * nb, Xq, Y + m * T, nb, T);
    });
}

// Transpose an (M, T) row-major matrix to (T, M) row-major — used to turn a GEMM's
// (out_features, token) output into per-token-contiguous rows for the per-head norm/RoPE.
void transpose_mt(const float* src, float* dst, int64_t M, int64_t T) {
    for (int64_t m = 0; m < M; ++m)
        for (int64_t t = 0; t < T; ++t)
            dst[t * M + m] = src[m * T + t];
}

// Quantize T contiguous activation rows of width K (T, K) → (T, K/QK) Q8 blocks.
void quantize_rows(const float* X, cpu::BlockQ8_0* Xq, int64_t T, int64_t K) {
    const int64_t nb = K / QK;
    for (int64_t t = 0; t < T; ++t)
        cpu::quantize_row_q8_0(X + t * K, Xq + t * nb, K);
}

// Plain RMSNorm matching ggml: y = x/sqrt(mean(x^2)+eps) * w. The Gemma (1+weight)
// is already baked into `w` at GGUF conversion, so this is a straight multiply.
// The sum-of-squares uses the SIMD reduction (the compiler can't auto-vectorize a float
// reduction without -ffast-math); the scale pass is a countable elementwise loop the
// compiler does vectorize. (float accumulation here is far below the Q8 weight error.)
void rmsnorm(const float* x, const float* w, float* y, int64_t n, float eps) {
    const float ss  = cpu::sum_squares_f32(x, n);
    const float inv = 1.0f / std::sqrt(ss / static_cast<float>(n) + eps);
    for (int64_t i = 0; i < n; ++i) y[i] = x[i] * inv * w[i];
}

// RMSNorm with no learned weight (Gemma V projection): y = x/sqrt(mean(x^2)+eps).
void rmsnorm_noweight(const float* x, float* y, int64_t n, float eps) {
    const float ss  = cpu::sum_squares_f32(x, n);
    const float inv = 1.0f / std::sqrt(ss / static_cast<float>(n) + eps);
    for (int64_t i = 0; i < n; ++i) y[i] = x[i] * inv;
}

// NEOX (half-split) RoPE on a single head vector of dim `dh`, in place.
//   pair i in [0, dh/2): theta = pos * base^(-2i/dh) / ff[i]; rotate (x[i], x[i+dh/2]).
// ff is the global-layer freq_factors (nullptr for local = no scaling).
void rope_neox(float* x, int64_t dh, int64_t pos, float base, const float* ff) {
    const int64_t half = dh / 2;
    const float   p    = static_cast<float>(pos);
    for (int64_t i = 0; i < half; ++i) {
        float theta = p * std::pow(base, -2.0f * static_cast<float>(i) / static_cast<float>(dh));
        if (ff) theta /= ff[i];
        const float c = std::cos(theta), s = std::sin(theta);
        const float a = x[i], b = x[i + half];
        x[i]        = a * c - b * s;
        x[i + half] = a * s + b * c;
    }
}

std::vector<float> load_norm(const GGUFReader& r, const std::string& name) {
    return r.has_tensor(name) ? r.load_tensor(name) : std::vector<float>{};
}

// Look up a stringified metadata value (gemma4.* keys) and parse as float.
float meta_float(const GGUFReader& r, const std::string& key, float fallback) {
    for (const auto& [k, v] : r.metadata())
        if (k == key) { try { return std::stof(v); } catch (...) { return fallback; } }
    return fallback;
}

} // namespace

GemmaModel GemmaModel::load_q8(const GGUFReader& reader) {
    const GGUFModelConfig& cfg = reader.config();
    if (cfg.arch != "gemma4")
        throw std::runtime_error(std::format(
            "GemmaModel::load_q8: arch is '{}', expected 'gemma4'", cfg.arch));

    GemmaModel m;
    m.V_    = cfg.vocab_size;
    m.D_    = cfg.embed_dim;
    m.d_ff_ = cfg.d_ff;
    m.eps_  = cfg.norm_eps;
    m.embed_scale_   = std::sqrt(static_cast<float>(m.D_));
    m.final_softcap_ = meta_float(reader, "gemma4.final_logit_softcapping", 30.0f);

    const int64_t n_layers = cfg.n_layers;
    const int64_t n_head   = static_cast<int64_t>(cfg.n_heads);   // 16
    const float   base_glb = cfg.rope_base;                       // 1e6
    const float   base_loc = meta_float(reader, "gemma4.rope.freq_base_swa", 10000.0f);
    const int64_t window   = cfg.sliding_window;                  // 1024

    if (m.D_ % QK != 0)
        throw std::runtime_error("GemmaModel: embed_dim must be a multiple of 32");

    // rope_freqs (global freq_factors) + final norm + tied embedding/LM head.
    if (reader.has_tensor("rope_freqs.weight"))
        m.rope_freqs_ = reader.load_tensor("rope_freqs.weight");
    m.output_norm_ = load_norm(reader, "output_norm.weight");
    const std::string emb = reader.has_tensor("output.weight") ? "output.weight"
                                                               : "token_embd.weight";
    m.token_embd_ = reader.load_tensor_q8(emb);
    m.n_params_   = static_cast<int64_t>(m.token_embd_.size()) * QK;

    m.layers_.resize(static_cast<std::size_t>(n_layers));
    for (int64_t l = 0; l < n_layers; ++l) {
        GemmaLayer& L = m.layers_[static_cast<std::size_t>(l)];

        // Per-layer shape comes from the tensors (authoritative): head_dim = q_out/n_head,
        // n_kv = k_out/head_dim. Global layers have the large head_dim / single kv head.
        const auto& qi = reader.tensors().at(std::format("blk.{}.attn_q.weight", l));
        const auto& ki = reader.tensors().at(std::format("blk.{}.attn_k.weight", l));
        const int64_t q_out   = qi.shape[0];
        const int64_t head_dim = q_out / n_head;
        const int64_t n_kv    = ki.shape[0] / head_dim;

        L.head_dim  = head_dim;
        L.n_head    = n_head;
        L.n_kv_head = n_kv;
        L.is_global = (n_kv == 1);   // global layers use MQA (1 kv head)
        L.rope_base = L.is_global ? base_glb : base_loc;
        L.window    = L.is_global ? 0 : window;

        if (head_dim % QK != 0)
            throw std::runtime_error(std::format(
                "GemmaModel: layer {} head_dim {} not a multiple of 32", l, head_dim));

        L.attn_norm      = load_norm(reader, std::format("blk.{}.attn_norm.weight", l));
        L.post_attn_norm = load_norm(reader, std::format("blk.{}.post_attention_norm.weight", l));
        L.ffn_norm       = load_norm(reader, std::format("blk.{}.ffn_norm.weight", l));
        L.post_ffw_norm  = load_norm(reader, std::format("blk.{}.post_ffw_norm.weight", l));
        L.q_norm         = load_norm(reader, std::format("blk.{}.attn_q_norm.weight", l));
        L.k_norm         = load_norm(reader, std::format("blk.{}.attn_k_norm.weight", l));

        if (auto os = std::format("blk.{}.layer_output_scale.weight", l); reader.has_tensor(os)) {
            const auto v = reader.load_tensor(os);
            if (!v.empty()) L.out_scale = v[0];
        }

        L.wq   = reader.load_tensor_q8(std::format("blk.{}.attn_q.weight", l));
        L.wk   = reader.load_tensor_q8(std::format("blk.{}.attn_k.weight", l));
        L.has_wv = reader.has_tensor(std::format("blk.{}.attn_v.weight", l));
        if (L.has_wv)
            L.wv = reader.load_tensor_q8(std::format("blk.{}.attn_v.weight", l));
        L.wo   = reader.load_tensor_q8(std::format("blk.{}.attn_output.weight", l));
        L.gate = reader.load_tensor_q8(std::format("blk.{}.ffn_gate.weight", l));
        L.up   = reader.load_tensor_q8(std::format("blk.{}.ffn_up.weight", l));
        L.down = reader.load_tensor_q8(std::format("blk.{}.ffn_down.weight", l));

        for (const auto* w : {&L.wq, &L.wk, &L.wv, &L.wo, &L.gate, &L.up, &L.down})
            m.n_params_ += static_cast<int64_t>(w->size()) * QK;
    }
    return m;
}

GemmaKVCache GemmaModel::make_cache(int64_t max_pos) const {
    GemmaKVCache kv;
    kv.max_pos = max_pos;
    kv.len     = 0;
    kv.layers.resize(layers_.size());
    for (std::size_t l = 0; l < layers_.size(); ++l) {
        const GemmaLayer& L = layers_[l];
        kv.layers[l].n_kv_head = L.n_kv_head;
        kv.layers[l].head_dim  = L.head_dim;
        const std::size_t n = static_cast<std::size_t>(L.n_kv_head * max_pos * L.head_dim);
        kv.layers[l].k.assign(n, 0.0f);
        kv.layers[l].v.assign(n, 0.0f);
    }
    return kv;
}

std::vector<float> GemmaModel::forward_one(int32_t token, int64_t pos,
                                           GemmaKVCache& kv, bool apply_softcap,
                                           bool compute_logits) const {
    const int64_t D = D_;
    const int64_t nbD = D / QK;

    // ── embedding lookup + scale ────────────────────────────────────────────────
    std::vector<float> x(static_cast<std::size_t>(D));
    cpu::dequantize_row_q8_0(token_embd_.data()
                                 + static_cast<std::size_t>(token) * static_cast<std::size_t>(nbD),
                             x.data(), D);
    for (std::size_t i = 0; i < x.size(); ++i) x[i] *= embed_scale_;

    std::vector<float> h(static_cast<std::size_t>(D));         // pre-attn / pre-ffn norm output
    std::vector<float> attn_out(static_cast<std::size_t>(D));

    for (std::size_t li = 0; li < layers_.size(); ++li) {
        const GemmaLayer& L = layers_[li];
        GemmaKVCache::Layer& C = kv.layers[li];
        const int64_t dh = L.head_dim, nH = L.n_head, nKV = L.n_kv_head;
        const float*  ff = L.is_global && !rope_freqs_.empty() ? rope_freqs_.data() : nullptr;

        // ── attention ───────────────────────────────────────────────────────────
        rmsnorm(x.data(), L.attn_norm.data(), h.data(), D, eps_);

        std::vector<float> q(static_cast<std::size_t>(nH * dh));
        std::vector<float> kcur(static_cast<std::size_t>(nKV * dh));
        std::vector<float> vcur(static_cast<std::size_t>(nKV * dh));
        // Fuse Q, K, (V) into one dispatch — all project the same attn_norm output `h`,
        // so quantize h once and run the concatenated rows under a single barrier.
        // Global layers omit attn_v → V is the raw K projection (Vcur = Kcur), so only
        // Q,K are projected and vcur is copied from kcur afterward.
        GemvJob qkv[3];
        int njobs = 0;
        qkv[njobs++] = {L.wq.data(), q.data(),    nH * dh};
        qkv[njobs++] = {L.wk.data(), kcur.data(), nKV * dh};
        if (L.has_wv) qkv[njobs++] = {L.wv.data(), vcur.data(), nKV * dh};
        pmatvec_multi(qkv, njobs, h.data(), D);
        if (!L.has_wv) vcur = kcur;

        // Q-norm+RoPE (per q-head) and K-norm+RoPE / V-norm + cache-store (per kv-head)
        // are all independent — dispatch them together across the pool. Indices [0,nH)
        // are q-heads; [nH, nH+nKV) are kv-heads. (For !has_wv, vcur is the raw-K copy.)
        parallel_for(nH + nKV, [&](int64_t idx) {
            if (idx < nH) {
                float* qh = q.data() + idx * dh;
                rmsnorm(qh, L.q_norm.data(), qh, dh, eps_);
                rope_neox(qh, dh, pos, L.rope_base, ff);
            } else {
                const int64_t g = idx - nH;
                float* kh = kcur.data() + g * dh;
                float* vh = vcur.data() + g * dh;
                rmsnorm(kh, L.k_norm.data(), kh, dh, eps_);
                rope_neox(kh, dh, pos, L.rope_base, ff);
                rmsnorm_noweight(vh, vh, dh, eps_);
                const std::size_t base = static_cast<std::size_t>(g * kv.max_pos * dh + pos * dh);
                std::copy(kh, kh + dh, C.k.begin() + static_cast<std::ptrdiff_t>(base));
                std::copy(vh, vh + dh, C.v.begin() + static_cast<std::ptrdiff_t>(base));
            }
        });

        // Per query head, attend over the cache (scale = 1.0; sliding window for local).
        // Independent across heads → parallelize; each head uses its own scores slice.
        const int64_t kv_lo = L.window > 0 ? std::max<int64_t>(0, pos - L.window + 1) : 0;
        const int64_t kv_hi = pos;                    // inclusive
        const int64_t kvlen = kv_hi - kv_lo + 1;
        const int64_t group = nH / nKV;               // q heads per kv head
        std::vector<float> attn_concat(static_cast<std::size_t>(nH * dh));
        std::vector<float> scores_all(static_cast<std::size_t>(nH * kvlen));
        parallel_for(nH, [&](int64_t hd) {
            const float* qh = q.data() + hd * dh;
            const int64_t g = hd / group;
            const float* Kbase = C.k.data() + static_cast<std::size_t>(g * kv.max_pos * dh);
            const float* Vbase = C.v.data() + static_cast<std::size_t>(g * kv.max_pos * dh);
            float* scores = scores_all.data() + hd * kvlen;

            float mx = -std::numeric_limits<float>::infinity();
            for (int64_t t = kv_lo; t <= kv_hi; ++t) {           // QKᵀ score (SIMD dot)
                const float dot = cpu::dot_f32(qh, Kbase + static_cast<std::size_t>(t * dh), dh);
                scores[t - kv_lo] = dot;   // scale = 1.0
                mx = std::max(mx, dot);
            }
            float sum = 0.0f;
            for (int64_t t = 0; t < kvlen; ++t) { scores[t] = std::exp(scores[t] - mx); sum += scores[t]; }
            const float invsum = 1.0f / sum;

            float* oh = attn_concat.data() + hd * dh;
            for (int64_t d = 0; d < dh; ++d) oh[d] = 0.0f;
            for (int64_t t = kv_lo; t <= kv_hi; ++t) {           // weighted V sum (SIMD axpy)
                const float* vt = Vbase + static_cast<std::size_t>(t * dh);
                cpu::axpy_f32(scores[t - kv_lo] * invsum, vt, oh, dh);
            }
        });

        // Output projection (D, nH*dh) → D, post-attn norm, residual.
        std::vector<float> ao(static_cast<std::size_t>(D));
        pmatvec(L.wo, attn_concat.data(), ao.data(), D, nH * dh);
        rmsnorm(ao.data(), L.post_attn_norm.data(), ao.data(), D, eps_);
        for (std::size_t i = 0; i < attn_out.size(); ++i) attn_out[i] = x[i] + ao[i];

        // ── FFN (GeGLU) ───────────────────────────────────────────────────────────
        rmsnorm(attn_out.data(), L.ffn_norm.data(), h.data(), D, eps_);
        std::vector<float> g(static_cast<std::size_t>(d_ff_));
        std::vector<float> u(static_cast<std::size_t>(d_ff_));
        // Fuse gate + up — both project the same ffn_norm output `h` — into one barrier.
        const GemvJob gu[2] = {{L.gate.data(), g.data(), d_ff_},
                               {L.up.data(),   u.data(), d_ff_}};
        pmatvec_multi(gu, 2, h.data(), D);
        cpu::gelu_f32(g.data(), g.data(), static_cast<std::size_t>(d_ff_));
        for (std::size_t i = 0; i < g.size(); ++i) g[i] *= u[i];

        std::vector<float> ff_out(static_cast<std::size_t>(D));
        pmatvec(L.down, g.data(), ff_out.data(), D, d_ff_);
        rmsnorm(ff_out.data(), L.post_ffw_norm.data(), ff_out.data(), D, eps_);

        // Residual + per-layer output scale → next layer input.
        for (std::size_t i = 0; i < x.size(); ++i)
            x[i] = (attn_out[i] + ff_out[i]) * L.out_scale;
    }

    kv.len = pos + 1;

    // Prompt prefill of non-final tokens: KV cache is filled, logits unused → stop here
    // and skip the V-wide LM head (the single biggest GEMV per token).
    if (!compute_logits) return {};

    // ── final norm + tied LM head + logit soft-cap ──────────────────────────────
    rmsnorm(x.data(), output_norm_.data(), x.data(), D, eps_);
    std::vector<float> logits(static_cast<std::size_t>(V_));
    pmatvec(token_embd_, x.data(), logits.data(), V_, D);
    // Soft-cap is monotonic → order-preserving; skip it when only the argmax matters.
    if (apply_softcap && final_softcap_ > 0.0f) {
        const float cap = final_softcap_;
        for (auto& z : logits) z = cap * std::tanh(z / cap);
    }
    return logits;
}

// ── batched prompt prefill ──────────────────────────────────────────────────────
// Processes T prompt tokens in one pass per layer via the row-reuse GEMM (pmatmul):
// weights stream once and are reused across all T tokens, so prefill is compute-bound
// rather than paying the per-token bandwidth wall T times. Numerically equivalent to T
// sequential forward_one calls (same per-token norms/RoPE/causal attention). The big
// (out,T) GEMM outputs are transposed to (T,out) so the per-head norm/RoPE work on
// contiguous head vectors, exactly as forward_one does.
std::vector<float> GemmaModel::forward_prefill(const int32_t* tokens, int64_t T,
                                               int64_t start_pos, GemmaKVCache& kv,
                                               bool apply_softcap) const {
    if (T <= 1)
        return forward_one(tokens[0], start_pos, kv, apply_softcap, /*compute_logits=*/true);

    const int64_t D = D_, nbD = D / QK, dff = d_ff_;
    const auto Z = [](int64_t n) { return static_cast<std::size_t>(n); };

    // X (T, D): embed lookup + ×sqrt(D) scale.
    std::vector<float> X(Z(T * D));
    for (int64_t t = 0; t < T; ++t) {
        cpu::dequantize_row_q8_0(
            token_embd_.data() + Z(tokens[t]) * Z(nbD), X.data() + t * D, D);
        for (int64_t i = 0; i < D; ++i) X[Z(t * D + i)] *= embed_scale_;
    }

    std::vector<float>           H(Z(T * D));            // norm output (reused attn/ffn)
    std::vector<cpu::BlockQ8_0>  Hq(Z(T * nbD));         // quantized H
    std::vector<float>           attn_out(Z(T * D));

    for (std::size_t li = 0; li < layers_.size(); ++li) {
        const GemmaLayer& L = layers_[li];
        GemmaKVCache::Layer& C = kv.layers[li];
        const int64_t dh = L.head_dim, nH = L.n_head, nKV = L.n_kv_head;
        const int64_t qM = nH * dh, kvM = nKV * dh;
        const float* ff = L.is_global && !rope_freqs_.empty() ? rope_freqs_.data() : nullptr;

        // ── attention ───────────────────────────────────────────────────────────
        for (int64_t t = 0; t < T; ++t)
            rmsnorm(X.data() + t * D, L.attn_norm.data(), H.data() + t * D, D, eps_);
        quantize_rows(H.data(), Hq.data(), T, D);

        std::vector<float> Qmt(Z(qM * T)), Kmt(Z(kvM * T)), Vmt;
        pmatmul(L.wq, Hq.data(), Qmt.data(), qM, D, T);
        pmatmul(L.wk, Hq.data(), Kmt.data(), kvM, D, T);
        if (L.has_wv) { Vmt.resize(Z(kvM * T)); pmatmul(L.wv, Hq.data(), Vmt.data(), kvM, D, T); }

        std::vector<float> Q(Z(T * qM)), Kt(Z(T * kvM)), Vt(Z(T * kvM));
        transpose_mt(Qmt.data(), Q.data(),  qM,  T);
        transpose_mt(Kmt.data(), Kt.data(), kvM, T);
        if (L.has_wv) transpose_mt(Vmt.data(), Vt.data(), kvM, T);
        else          Vt = Kt;                              // V = raw K projection (pre-norm)

        // per token: Q/K rmsnorm+RoPE, V plain rmsnorm, store K/V to cache (all before attn)
        parallel_for(T, [&](int64_t t) {
            const int64_t p = start_pos + t;
            for (int64_t h = 0; h < nH; ++h) {
                float* qh = Q.data() + t * qM + h * dh;
                rmsnorm(qh, L.q_norm.data(), qh, dh, eps_);
                rope_neox(qh, dh, p, L.rope_base, ff);
            }
            for (int64_t g = 0; g < nKV; ++g) {
                float* kh = Kt.data() + t * kvM + g * dh;
                float* vh = Vt.data() + t * kvM + g * dh;
                rmsnorm(kh, L.k_norm.data(), kh, dh, eps_);
                rope_neox(kh, dh, p, L.rope_base, ff);
                rmsnorm_noweight(vh, vh, dh, eps_);
                const std::size_t b = Z(g * kv.max_pos * dh + p * dh);
                std::copy(kh, kh + dh, C.k.begin() + static_cast<std::ptrdiff_t>(b));
                std::copy(vh, vh + dh, C.v.begin() + static_cast<std::ptrdiff_t>(b));
            }
        });

        // causal attention per (token, head): position p attends to cache [kv_lo, p].
        std::vector<float> attn_concat(Z(T * qM));
        const int64_t group = nH / nKV;
        parallel_for(T * nH, [&](int64_t th) {
            const int64_t t = th / nH, h = th % nH, p = start_pos + t;
            const int64_t kv_lo = L.window > 0 ? std::max<int64_t>(0, p - L.window + 1) : 0;
            const int64_t g = h / group;
            const float* qh    = Q.data() + t * qM + h * dh;
            const float* Kbase = C.k.data() + Z(g * kv.max_pos * dh);
            const float* Vbase = C.v.data() + Z(g * kv.max_pos * dh);
            std::vector<float> scores(Z(p - kv_lo + 1));
            float mx = -std::numeric_limits<float>::infinity();
            for (int64_t s = kv_lo; s <= p; ++s) {
                const float d = cpu::dot_f32(qh, Kbase + Z(s * dh), dh);   // scale = 1.0
                scores[Z(s - kv_lo)] = d;
                mx = std::max(mx, d);
            }
            float sum = 0.0f;
            for (auto& sc : scores) { sc = std::exp(sc - mx); sum += sc; }
            const float inv = 1.0f / sum;
            float* oh = attn_concat.data() + t * qM + h * dh;
            for (int64_t d = 0; d < dh; ++d) oh[d] = 0.0f;
            for (int64_t s = kv_lo; s <= p; ++s)
                cpu::axpy_f32(scores[Z(s - kv_lo)] * inv, Vbase + Z(s * dh), oh, dh);
        });

        // output projection (D,T) → transpose → post-attn norm → residual
        std::vector<cpu::BlockQ8_0> ACq(Z(T * (qM / QK)));
        quantize_rows(attn_concat.data(), ACq.data(), T, qM);
        std::vector<float> AOmt(Z(D * T)), AO(Z(T * D));
        pmatmul(L.wo, ACq.data(), AOmt.data(), D, qM, T);
        transpose_mt(AOmt.data(), AO.data(), D, T);
        for (int64_t t = 0; t < T; ++t) {
            float* ao = AO.data() + t * D;
            rmsnorm(ao, L.post_attn_norm.data(), ao, D, eps_);
            for (int64_t i = 0; i < D; ++i) attn_out[Z(t * D + i)] = X[Z(t * D + i)] + ao[i];
        }

        // ── FFN (GeGLU), batched ──────────────────────────────────────────────────
        for (int64_t t = 0; t < T; ++t)
            rmsnorm(attn_out.data() + t * D, L.ffn_norm.data(), H.data() + t * D, D, eps_);
        quantize_rows(H.data(), Hq.data(), T, D);
        std::vector<float> Gmt(Z(dff * T)), Umt(Z(dff * T));
        pmatmul(L.gate, Hq.data(), Gmt.data(), dff, D, T);
        pmatmul(L.up,   Hq.data(), Umt.data(), dff, D, T);
        cpu::gelu_f32(Gmt.data(), Gmt.data(), Z(dff * T));
        for (std::size_t i = 0; i < Gmt.size(); ++i) Gmt[i] *= Umt[i];   // GeGLU gate ⊙ up
        std::vector<float> Gtt(Z(T * dff));
        transpose_mt(Gmt.data(), Gtt.data(), dff, T);
        std::vector<cpu::BlockQ8_0> Gq(Z(T * (dff / QK)));
        quantize_rows(Gtt.data(), Gq.data(), T, dff);
        std::vector<float> FFmt(Z(D * T)), FF(Z(T * D));
        pmatmul(L.down, Gq.data(), FFmt.data(), D, dff, T);
        transpose_mt(FFmt.data(), FF.data(), D, T);
        for (int64_t t = 0; t < T; ++t) {
            float* fo = FF.data() + t * D;
            rmsnorm(fo, L.post_ffw_norm.data(), fo, D, eps_);
            for (int64_t i = 0; i < D; ++i)
                X[Z(t * D + i)] = (attn_out[Z(t * D + i)] + fo[i]) * L.out_scale;
        }
    }
    kv.len = start_pos + T;

    // ── final norm + tied LM head (last token only) + soft-cap ──────────────────
    float* xl = X.data() + (T - 1) * D;
    rmsnorm(xl, output_norm_.data(), xl, D, eps_);
    std::vector<float> logits(Z(V_));
    pmatvec(token_embd_, xl, logits.data(), V_, D);
    if (apply_softcap && final_softcap_ > 0.0f) {
        const float cap = final_softcap_;
        for (auto& z : logits) z = cap * std::tanh(z / cap);
    }
    return logits;
}

} // namespace sub0llm::nn
