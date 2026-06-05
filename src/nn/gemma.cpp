#include "sub0llm/nn/gemma.hpp"

#include "../backends/cpu/kernels.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <format>
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

    // y[m] = dot(W + m*nb, xq, nb) for m in [0, M), across the pool + caller.
    void run(const cpu::BlockQ8_0* W, const cpu::BlockQ8_0* xq, float* y,
             int64_t M, int64_t nb) {
        W_ = W; xq_ = xq; y_ = y; M_ = M; nb_ = nb;
        next_.store(0, std::memory_order_relaxed);
        remaining_.store(nworkers_, std::memory_order_relaxed);
        gen_.fetch_add(1, std::memory_order_release);   // publishes W_..nb_ to workers
        work();                                         // caller participates
        while (remaining_.load(std::memory_order_acquire) != 0) GEMMA_CPU_RELAX();
    }

private:
    static constexpr int64_t CHUNK = 32;

    void work() {
        const cpu::BlockQ8_0* W = W_;
        const cpu::BlockQ8_0* xq = xq_;
        float* y = y_; const int64_t M = M_, nb = nb_;
        for (;;) {
            const int64_t i = next_.fetch_add(CHUNK, std::memory_order_relaxed);
            if (i >= M) break;
            const int64_t hi = std::min(M, i + CHUNK);
            for (int64_t m = i; m < hi; ++m)
                y[m] = cpu::dot_q8_0_q8_0(W + m * nb, xq, nb);
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
    const cpu::BlockQ8_0*        W_  = nullptr;
    const cpu::BlockQ8_0*        xq_ = nullptr;
    float*                       y_  = nullptr;
    int64_t                      M_ = 0, nb_ = 0;
};

int desired_threads() {
    if (g_gemma_threads > 0) return g_gemma_threads;
    unsigned hw = std::thread::hardware_concurrency();
    if (hw == 0) return 1;
    // Single-token decode is DRAM-bandwidth-bound and saturates well before all cores;
    // meanwhile spin-wait workers starve the OS if they occupy every core. Leaving a few
    // cores free is measurably faster (on a 24-core part, 20 threads beat 24). Default to
    // hw-4 for large core counts; use everything on small machines.
    const unsigned n = hw > 8 ? hw - 4 : hw;
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

// Parallel Q8 GEMV: y[M] = W[M,K]·x[K], W row-major (out,in) Q8_0, x f32.
// Quantizes the activation once, then dots each output row (across the pool).
void pmatvec(const std::vector<cpu::BlockQ8_0>& W, const float* x, float* y,
             int64_t M, int64_t K) {
    const int64_t nb = K / QK;
    std::vector<cpu::BlockQ8_0> xq(static_cast<std::size_t>(nb));
    cpu::quantize_row_q8_0(x, xq.data(), K);

    GemmaPool* pool = get_pool();
    if (!pool || M < 128) {                       // tiny GEMV: not worth dispatching
        for (int64_t m = 0; m < M; ++m)
            y[m] = cpu::dot_q8_0_q8_0(W.data() + m * nb, xq.data(), nb);
        return;
    }
    pool->run(W.data(), xq.data(), y, M, nb);
}

// Plain RMSNorm matching ggml: y = x/sqrt(mean(x^2)+eps) * w. The Gemma (1+weight)
// is already baked into `w` at GGUF conversion, so this is a straight multiply.
void rmsnorm(const float* x, const float* w, float* y, int64_t n, float eps) {
    double ss = 0.0;
    for (int64_t i = 0; i < n; ++i) ss += static_cast<double>(x[i]) * x[i];
    const float inv = 1.0f / std::sqrt(static_cast<float>(ss / static_cast<double>(n)) + eps);
    for (int64_t i = 0; i < n; ++i) y[i] = x[i] * inv * w[i];
}

// RMSNorm with no learned weight (Gemma V projection): y = x/sqrt(mean(x^2)+eps).
void rmsnorm_noweight(const float* x, float* y, int64_t n, float eps) {
    double ss = 0.0;
    for (int64_t i = 0; i < n; ++i) ss += static_cast<double>(x[i]) * x[i];
    const float inv = 1.0f / std::sqrt(static_cast<float>(ss / static_cast<double>(n)) + eps);
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
                                           GemmaKVCache& kv) const {
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
        pmatvec(L.wq, h.data(), q.data(),    nH * dh,  D);
        pmatvec(L.wk, h.data(), kcur.data(), nKV * dh, D);   // raw K projection
        // V source: a dedicated wv projection, or — on global layers that omit attn_v —
        // the raw K projection itself (matching llama.cpp's "Vcur = Kcur" fallback).
        if (L.has_wv) pmatvec(L.wv, h.data(), vcur.data(), nKV * dh, D);
        else          vcur = kcur;

        // Q: per-head rmsnorm(q_norm) + RoPE.
        for (int64_t hd = 0; hd < nH; ++hd) {
            float* qh = q.data() + hd * dh;
            rmsnorm(qh, L.q_norm.data(), qh, dh, eps_);
            rope_neox(qh, dh, pos, L.rope_base, ff);
        }
        // K: per-head rmsnorm(k_norm) + RoPE; V: per-head plain rmsnorm (no weight).
        // Store K/V for this position into the cache.
        for (int64_t g = 0; g < nKV; ++g) {
            float* kh = kcur.data() + g * dh;
            float* vh = vcur.data() + g * dh;
            rmsnorm(kh, L.k_norm.data(), kh, dh, eps_);
            rope_neox(kh, dh, pos, L.rope_base, ff);
            rmsnorm_noweight(vh, vh, dh, eps_);
            const std::size_t base = static_cast<std::size_t>(g * kv.max_pos * dh + pos * dh);
            std::copy(kh, kh + dh, C.k.begin() + static_cast<std::ptrdiff_t>(base));
            std::copy(vh, vh + dh, C.v.begin() + static_cast<std::ptrdiff_t>(base));
        }

        // Per query head, attend over the cache (scale = 1.0; sliding window for local).
        const int64_t kv_lo = L.window > 0 ? std::max<int64_t>(0, pos - L.window + 1) : 0;
        const int64_t kv_hi = pos;                    // inclusive
        const int64_t group = nH / nKV;               // q heads per kv head
        std::vector<float> attn_concat(static_cast<std::size_t>(nH * dh));
        std::vector<float> scores(static_cast<std::size_t>(kv_hi - kv_lo + 1));
        for (int64_t hd = 0; hd < nH; ++hd) {
            const float* qh = q.data() + hd * dh;
            const int64_t g = hd / group;
            const float* Kbase = C.k.data() + static_cast<std::size_t>(g * kv.max_pos * dh);
            const float* Vbase = C.v.data() + static_cast<std::size_t>(g * kv.max_pos * dh);

            float mx = -std::numeric_limits<float>::infinity();
            for (int64_t t = kv_lo; t <= kv_hi; ++t) {
                const float* kt = Kbase + static_cast<std::size_t>(t * dh);
                float dot = 0.0f;
                for (int64_t d = 0; d < dh; ++d) dot += qh[d] * kt[d];
                scores[static_cast<std::size_t>(t - kv_lo)] = dot;   // scale = 1.0
                mx = std::max(mx, dot);
            }
            float sum = 0.0f;
            for (auto& s : scores) { s = std::exp(s - mx); sum += s; }
            const float invsum = 1.0f / sum;

            float* oh = attn_concat.data() + hd * dh;
            for (int64_t d = 0; d < dh; ++d) oh[d] = 0.0f;
            for (int64_t t = kv_lo; t <= kv_hi; ++t) {
                const float w  = scores[static_cast<std::size_t>(t - kv_lo)] * invsum;
                const float* vt = Vbase + static_cast<std::size_t>(t * dh);
                for (int64_t d = 0; d < dh; ++d) oh[d] += w * vt[d];
            }
        }

        // Output projection (D, nH*dh) → D, post-attn norm, residual.
        std::vector<float> ao(static_cast<std::size_t>(D));
        pmatvec(L.wo, attn_concat.data(), ao.data(), D, nH * dh);
        rmsnorm(ao.data(), L.post_attn_norm.data(), ao.data(), D, eps_);
        for (std::size_t i = 0; i < attn_out.size(); ++i) attn_out[i] = x[i] + ao[i];

        // ── FFN (GeGLU) ───────────────────────────────────────────────────────────
        rmsnorm(attn_out.data(), L.ffn_norm.data(), h.data(), D, eps_);
        std::vector<float> g(static_cast<std::size_t>(d_ff_));
        std::vector<float> u(static_cast<std::size_t>(d_ff_));
        pmatvec(L.gate, h.data(), g.data(), d_ff_, D);
        pmatvec(L.up,   h.data(), u.data(), d_ff_, D);
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

    // ── final norm + tied LM head + logit soft-cap ──────────────────────────────
    rmsnorm(x.data(), output_norm_.data(), x.data(), D, eps_);
    std::vector<float> logits(static_cast<std::size_t>(V_));
    pmatvec(token_embd_, x.data(), logits.data(), V_, D);
    if (final_softcap_ > 0.0f) {
        const float cap = final_softcap_;
        for (auto& z : logits) z = cap * std::tanh(z / cap);
    }
    return logits;
}

} // namespace sub0llm::nn
