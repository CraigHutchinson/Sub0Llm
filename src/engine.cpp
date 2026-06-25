// engine.cpp — shared differentiable substrate (libsub0_core).
//
// Statically allocated: every model dimension is a compile-time constant (from
// the generated config header), so all parameter and activation memory lives in
// fixed-size std::array buffers allocated once in BSS and reused. There is no
// per-step heap allocation. Reverse-mode autograd is a fixed enum-tagged node
// pool walked in reverse; activations come from a bump-allocated arena reset each
// forward pass. See include/sub0/core.hpp for the exposed API.

#include "sub0/core.hpp"
#include "sub0/casing.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <mdspan>
#include <print>
#include <random>
#include <ranges>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#if defined(_OPENMP)
#include <omp.h>
#else
static inline int omp_get_thread_num()  { return 0; }
static inline int omp_get_num_threads() { return 1; }
static inline int omp_get_max_threads() { return 1; }
#endif

// x86 SSE/AVX control register access, used to flush subnormal floats to zero
// (FTZ/DAZ) in the hot loops, where a subnormal operand triggers a slow assist.
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#include <immintrin.h>
#define SUB0_X86 1
#endif

namespace sub0 {

// ============================================================================
//  Derived compile-time sizes
// ============================================================================

constexpr int NUM_PARAMS = 2 + 10 * N_LAYERS + 3;

consteval size_t calc_param_floats() {
    size_t n = 0;
    n += (size_t)VOCAB * D_MODEL;       // tok_emb
    n += (size_t)SEQ_LEN * D_MODEL;     // pos_emb
    for (int l = 0; l < N_LAYERS; ++l) {
        n += D_MODEL + D_MODEL;                 // ln1, ln2
        n += (size_t)4 * D_MODEL * D_MODEL;     // Wq,Wk,Wv,Wo
        n += (size_t)D_MODEL * D_FF + D_FF;      // W1, b1
        n += (size_t)D_FF * D_MODEL + D_MODEL;   // W2, b2
    }
    n += D_MODEL;                       // ln_f
    n += (size_t)D_MODEL * VOCAB + VOCAB;        // lm_head, lm_bias
    return n;
}
constexpr size_t PARAM_FLOATS = calc_param_floats();

consteval size_t calc_act_cap() {
    const size_t T = SEQ_LEN, C = D_MODEL, F = D_FF, H = N_HEADS, V = VOCAB;
    size_t base = 3 * T * C;
    size_t per  = 10 * T * C + 2 * T * F + H * T * T + 2 * T
                + (USE_TERNARY ? (size_t)4 * C * C + 2 * C * F : 0);
    size_t fin  = T * C + 2 * T * V + 64;
    return base + (size_t)N_LAYERS * per + fin;
}
constexpr size_t ACT_CAP   = calc_act_cap() * 3 / 2 + 8192;
constexpr size_t MAX_NODES = 16 + 16 * (size_t)N_LAYERS;

// ============================================================================
//  Static storage
// ============================================================================

// Shared across all worker threads: the weights (read-only during fwd/bwd), the
// optimizer moments (touched only by AdamW::step on the main thread), and the
// REDUCED gradient that AdamW::step consumes.
static std::array<float, PARAM_FLOATS> g_param_data{};
static std::array<float, PARAM_FLOATS> g_param_grad{};
static std::array<float, PARAM_FLOATS> g_param_m{};
static std::array<float, PARAM_FLOATS> g_param_vel{};

struct ParamView { size_t off, n; bool decay; };

// All per-thread state for a data-parallel window lives in one Worker. Workers are
// a STATICALLY allocated pool indexed by OpenMP thread id -- not thread_local: a
// multi-MB thread_local in a DLL overruns Windows' static-TLS block and faults. BSS
// is demand-paged, so only the slots a run actually touches become resident. Each
// thread binds its slot once into the thread_local handle `W` (see
// ensure_thread_built), keeping the hot paths pointer-direct. Slot 0 serves all
// single-window work (gen / eval / bench). The pool is sized to MAX_WORKERS, a
// hardware-scaled constant cached into the generated config by sub0-configure.
struct Worker {
    std::array<float, PARAM_FLOATS> grad{};        // gradient accumulator (this slot)
    std::array<float, ACT_CAP>      act_data{};    // activation arena: values
    std::array<float, ACT_CAP>      act_grad{};    // activation arena: grads
    std::array<Node, NUM_PARAMS>    param_nodes{}; // parameter leaves (data->shared, grad->this)
    std::array<Node, MAX_NODES>     pool{};        // forward-graph node pool
    std::array<ParamView, NUM_PARAMS> views{};     // optimizer parameter spans
    size_t act_used = 0, pool_used = 0, pcount = 0, pused = 0;
};
static std::array<Worker, MAX_WORKERS> g_workers{};
static std::atomic<bool> g_params_init{false};
// nullptr until this thread runs ensure_thread_built(); a non-null W is the single
// per-thread "bound and laid out" flag (its own TLS lifetime tracks the OS thread,
// so each thread builds its g_model exactly once even if OpenMP reuses slots).
thread_local Worker* W = nullptr;

static std::pair<std::span<float>, std::span<float>> arena_alloc(size_t n) {
    if (W->act_used + n > ACT_CAP) {
        std::println(stderr, "fatal: activation arena overflow (need {}, cap {})",
                     W->act_used + n, ACT_CAP);
        std::abort();
    }
    size_t off = W->act_used;
    W->act_used += n;
    std::span<float> d(W->act_data.data() + off, n);
    std::span<float> gr(W->act_grad.data() + off, n);
    std::fill(d.begin(), d.end(), 0.f);
    std::fill(gr.begin(), gr.end(), 0.f);
    return {d, gr};
}

// When a model was loaded with weights already in their final (ternary) form,
// the linear op must not re-quantize them (absmean re-quantization is not
// idempotent). Training keeps this false so the straight-through estimator runs.
static bool g_packed_inference = false;

static Node* mk_param(int r, int c, bool decay) {
    size_t n = (size_t)r * c, off = W->pused;
    W->pused += n;
    Node& nd = W->param_nodes[W->pcount];
    nd = Node{};
    nd.op = Op::Leaf; nd.rows = r; nd.cols = c;
    nd.data = std::span<float>(g_param_data.data() + off, n);   // shared weights
    nd.grad = std::span<float>(W->grad.data() + off, n);        // this thread's grad accumulator
    W->views[W->pcount] = {off, n, decay};
    ++W->pcount;
    return &nd;
}

static Node* mk_node(Op op, int r, int c) {
    if (W->pool_used >= MAX_NODES) { std::println(stderr, "fatal: node pool overflow"); std::abort(); }
    Node& nd = W->pool[W->pool_used++];
    nd = Node{};
    nd.op = op; nd.rows = r; nd.cols = c;
    auto [d, gr] = arena_alloc((size_t)r * c);
    nd.data = d; nd.grad = gr;
    return &nd;
}

// 2D row-major view over a Node's flat [rows x cols] span (data or grad). Replaces
// hand-rolled i*cols+j indexing in the scalar/scatter paths; the hot vectorized
// kernels keep their __restrict row pointers.
using Mat = std::mdspan<float, std::dextents<std::size_t, 2>>;
static inline Mat mat(std::span<float> s, int rows, int cols) {
    return Mat(s.data(), static_cast<std::size_t>(rows), static_cast<std::size_t>(cols));
}

// ============================================================================
//  Fast transcendental math (vectorizable), selected at compile time
// ============================================================================
// Softmax exp and GELU dominate the non-GEMM forward/backward cost and std::exp/
// std::erf are scalar libm calls. These branchless approximations let clang
// vectorize the softmax/GELU loops; the exact path stays behind FAST_MATH. The
// choice is a compile-time constant (define SUB0_EXACT_MATH to take the exact
// path), so the unused branch is eliminated rather than tested per element.
#ifdef SUB0_EXACT_MATH
constexpr bool FAST_MATH = false;
#else
constexpr bool FAST_MATH = true;
#endif

// exp(x) to ~1e-6 over the softmax/GELU range. Range-reduce x = k*ln2 + r, then
// exp(x) = 2^k * poly(r); assemble 2^k from the float exponent bits. Branchless so
// `#pragma omp simd` vectorizes the calling loop.
static inline float fast_exp(float x) {
    x = x < -87.f ? -87.f : (x > 88.f ? 88.f : x);
    const float k = std::floor(x * 1.4426950409f + 0.5f);
    const float r = x - k * 0.6931471805f;
    float p = 0.0013888939f;
    p = p * r + 0.0083333680f;
    p = p * r + 0.0416666418f;
    p = p * r + 0.1666666664f;
    p = p * r + 0.5000000000f;
    p = p * r + 1.0000000000f;
    p = p * r + 1.0000000000f;
    const std::uint32_t bits = static_cast<std::uint32_t>(static_cast<int>(k) + 127) << 23;
    return p * std::bit_cast<float>(bits);
}
static inline float fast_tanh(float x) {            // tanh via fast_exp, saturates to +-1
    const float e = fast_exp(-2.f * x);
    return (1.f - e) / (1.f + e);
}
// tanh-form GELU (matches GPT-2's approximation, ~3e-4 vs the erf form): value and
// derivative, kept mutually consistent so the gradient check passes in fast mode.
constexpr float GELU_C = 0.7978845608f;             // sqrt(2/pi)
constexpr float GELU_A = 0.0447150000f;
static inline float gelu_fast(float v) {
    return 0.5f * v * (1.f + fast_tanh(GELU_C * (v + GELU_A * v * v * v)));
}
static inline float dgelu_fast(float v) {
    const float t  = fast_tanh(GELU_C * (v + GELU_A * v * v * v));
    const float du = GELU_C * (1.f + 3.f * GELU_A * v * v);
    return 0.5f * (1.f + t) + 0.5f * v * (1.f - t * t) * du;
}

// ============================================================================
//  Differentiable ops (forward builders)
// ============================================================================

static Node* op_embed(Node* table, const int* ids, int T) {
    const int C = table->cols;
    Node* out = mk_node(Op::Embed, T, C);
    out->w = table; out->ids = ids;
    Mat o = mat(out->data, T, C), tab = mat(table->data, table->rows, C);
    for (int t = 0; t < T; ++t)
        for (int j = 0; j < C; ++j) o[t, j] = tab[ids[t], j];
    return out;
}

static Node* op_add(Node* a, Node* b) {
    Node* out = mk_node(Op::Add, a->rows, a->cols);
    out->a = a; out->b = b;
    const size_t n = out->data.size();
    #pragma omp simd
    for (size_t i = 0; i < n; ++i) out->data[i] = a->data[i] + b->data[i];
    return out;
}

static void ternarize_into(std::span<const float> w, std::span<float> q) {
    const size_t n = w.size();
    double s = 0.0;
    #pragma omp simd reduction(+ : s)
    for (size_t i = 0; i < n; ++i) s += std::fabs(w[i]);
    float scale = (float)(s / std::max<size_t>(1, n)) + 1e-8f;
    #pragma omp simd
    for (size_t i = 0; i < n; ++i) {
        float r = w[i] / scale;
        float t = r > 0.5f ? 1.f : (r < -0.5f ? -1.f : 0.f);
        q[i] = t * scale;
    }
}

static Node* op_linear(Node* x, Node* W, Node* bias, bool ternary) {
    const int T = x->rows, in = x->cols, out = W->cols;
    Node* y = mk_node(Op::Linear, T, out);
    y->a = x; y->w = W; y->bias = bias; y->ternary = ternary;
    const float* Wf = W->data.data();
    if (ternary && !g_packed_inference) {
        auto [sd, sg] = arena_alloc(W->data.size());
        ternarize_into(W->data, sd);
        y->scratch = sd;
        Wf = sd.data();
    }
    const float* X = x->data.data();
    for (int t = 0; t < T; ++t) {
        float* __restrict Yr        = y->data.data() + (size_t)t * out;
        const float* __restrict Xr  = X + (size_t)t * in;
        for (int p = 0; p < in; ++p) {
            const float xtp = Xr[p];
            if (xtp == 0.f) continue;                       // ternary weights make this sparse
            const float* __restrict Wr = Wf + (size_t)p * out;
            for (int o = 0; o < out; ++o) Yr[o] += xtp * Wr[o];  // contiguous axpy -> vectorizes
        }
    }
    if (bias) {
        Mat ym = mat(y->data, T, out);
        for (int t = 0; t < T; ++t)
            for (int o = 0; o < out; ++o) ym[t, o] += bias->data[o];
    }
    return y;
}

static Node* op_rmsnorm(Node* x, Node* gamma) {
    const int T = x->rows, C = x->cols;
    const float eps = 1e-5f;
    Node* y = mk_node(Op::RMSNorm, T, C);
    y->a = x; y->w = gamma;
    auto [rinv, rinv_g] = arena_alloc(T);
    y->scratch = rinv;
    const float* __restrict G = gamma->data.data();
    for (int t = 0; t < T; ++t) {
        const float* __restrict xr = x->data.data() + (size_t)t * C;
        float* __restrict yr       = y->data.data() + (size_t)t * C;
        float ms = 0.f;
        #pragma omp simd reduction(+ : ms)
        for (int j = 0; j < C; ++j) ms += xr[j] * xr[j];
        ms /= C;
        float r = 1.f / std::sqrt(ms + eps);
        rinv[t] = r;
        for (int j = 0; j < C; ++j) yr[j] = xr[j] * r * G[j];
    }
    return y;
}

static Node* op_gelu(Node* x) {
    Node* y = mk_node(Op::GELU, x->rows, x->cols);
    y->a = x;
    const float* __restrict xd = x->data.data();
    float* __restrict yd       = y->data.data();
    const size_t n = x->data.size();
    if constexpr (FAST_MATH) {
        #pragma omp simd
        for (size_t i = 0; i < n; ++i) yd[i] = gelu_fast(xd[i]);
    } else {
        const float inv_sqrt2 = 0.70710678f;
        for (size_t i = 0; i < n; ++i) yd[i] = 0.5f * xd[i] * (1.f + std::erf(xd[i] * inv_sqrt2));
    }
    return y;
}

static Node* op_attn(Node* q, Node* k, Node* v, int H) {
    const int T = q->rows, C = q->cols, d = C / H;
    const float scale = 1.f / std::sqrt((float)d);
    Node* out = mk_node(Op::Attn, T, C);
    out->a = q; out->b = k; out->bias = v; out->heads = H;
    auto [P, Pg] = arena_alloc((size_t)H * T * T);
    out->scratch = P;
    auto Pidx = [T](int h, int i, int j) { return ((size_t)h * T + i) * T + j; };
    for (int h = 0; h < H; ++h) {
        int off = h * d;
        for (int i = 0; i < T; ++i) {
            const float* __restrict qi = q->data.data() + (size_t)i * C + off;
            float* __restrict oi       = out->data.data() + (size_t)i * C + off;
            float mx = -1e30f;
            std::array<float, SEQ_LEN> sc{};
            for (int j = 0; j <= i; ++j) {
                const float* __restrict kj = k->data.data() + (size_t)j * C + off;
                float s = 0.f;
                #pragma omp simd reduction(+ : s)
                for (int a = 0; a < d; ++a) s += qi[a] * kj[a];
                s *= scale; sc[j] = s; mx = std::max(mx, s);
            }
            float Z = 0.f;
            if constexpr (FAST_MATH) for (int j = 0; j <= i; ++j) { sc[j] = fast_exp(sc[j] - mx); Z += sc[j]; }
            else                     for (int j = 0; j <= i; ++j) { sc[j] = std::exp(sc[j] - mx);  Z += sc[j]; }
            for (int j = 0; j <= i; ++j) {
                float p = sc[j] / Z;
                P[Pidx(h, i, j)] = p;
                const float* __restrict vj = v->data.data() + (size_t)j * C + off;
                for (int a = 0; a < d; ++a) oi[a] += p * vj[a];      // contiguous axpy
            }
        }
    }
    return out;
}

static Node* op_cross_entropy(Node* logits, const int* targets) {
    const int T = logits->rows, V = logits->cols;
    Node* loss = mk_node(Op::CrossEnt, 1, 1);
    loss->a = logits; loss->ids = targets;
    auto [probs, pg] = arena_alloc((size_t)T * V);
    loss->scratch = probs;
    float total = 0.f;
    for (int t = 0; t < T; ++t) {
        const float* __restrict lr = logits->data.data() + (size_t)t * V;
        float* __restrict pr       = probs.data() + (size_t)t * V;
        float mx = -1e30f;
        for (int j = 0; j < V; ++j) mx = std::max(mx, lr[j]);
        float Z = 0.f;
        if constexpr (FAST_MATH) {
            #pragma omp simd reduction(+ : Z)
            for (int j = 0; j < V; ++j) { float e = fast_exp(lr[j] - mx); pr[j] = e; Z += e; }
        } else {
            for (int j = 0; j < V; ++j) { float e = std::exp(lr[j] - mx); pr[j] = e; Z += e; }
        }
        const float invZ = 1.f / Z;
        for (int j = 0; j < V; ++j) pr[j] *= invZ;
        total += -std::log(std::max(1e-9f, probs[(size_t)t * V + targets[t]]));
    }
    loss->data[0] = total / T;
    return loss;
}

// ============================================================================
//  Backward dispatch
// ============================================================================

static void backward_node(Node& n) {
    switch (n.op) {
    case Op::Leaf: break;
    case Op::Embed: {
        const int T = n.rows, C = n.cols;
        Mat wg = mat(n.w->grad, n.w->rows, C), ng = mat(n.grad, T, C);
        for (int t = 0; t < T; ++t)
            for (int j = 0; j < C; ++j) wg[n.ids[t], j] += ng[t, j];
        break;
    }
    case Op::Add: {
        for (size_t i = 0; i < n.grad.size(); ++i) { n.a->grad[i] += n.grad[i]; n.b->grad[i] += n.grad[i]; }
        break;
    }
    case Op::Linear: {
        Node* x = n.a; Node* W = n.w;
        const int T = x->rows, in = x->cols, out = W->cols;
        const float* Wf = (n.ternary && !n.scratch.empty()) ? n.scratch.data() : W->data.data();
        const float* dY = n.grad.data();
        const float* X = x->data.data();
        // dX = dY . W^T : inner loop over `o` is contiguous in both operands (a dot
        // product reduction). Per-window; data-parallelism is at the batch level.
        for (int t = 0; t < T; ++t) {
            const float* __restrict dYr = dY + (size_t)t * out;
            float* __restrict xg        = x->grad.data() + (size_t)t * in;
            for (int p = 0; p < in; ++p) {
                const float* __restrict Wr = Wf + (size_t)p * out;
                float s = 0.f;
                #pragma omp simd reduction(+ : s)            // allow vectorizing the dot product
                for (int o = 0; o < out; ++o) s += dYr[o] * Wr[o];
                xg[p] += s;
            }
        }
        // dW = X^T . dY : iterate (p, t, o) so the inner loop over `o` is a contiguous
        // axpy into W->grad's row (the old (p,o,t) order strided over t -> no vectorize).
        for (int p = 0; p < in; ++p) {
            float* __restrict Wg = W->grad.data() + (size_t)p * out;
            for (int t = 0; t < T; ++t) {
                const float xtp = X[(size_t)t * in + p];
                const float* __restrict dYr = dY + (size_t)t * out;
                for (int o = 0; o < out; ++o) Wg[o] += xtp * dYr[o];
            }
        }
        if (n.bias)
            for (int t = 0; t < T; ++t) {
                #pragma omp simd
                for (int o = 0; o < out; ++o) n.bias->grad[o] += dY[(size_t)t * out + o];
            }
        break;
    }
    case Op::RMSNorm: {
        Node* x = n.a; Node* g = n.w;
        const int T = x->rows, C = x->cols;
        std::span<float> rinv = n.scratch;
        Mat gy = mat(n.grad, T, C), xd = mat(x->data, T, C), xg = mat(x->grad, T, C);
        for (int t = 0; t < T; ++t) {
            float S = 0.f;
            #pragma omp simd reduction(+ : S)
            for (int j = 0; j < C; ++j) S += gy[t, j] * g->data[j] * xd[t, j];
            float r = rinv[t], r3 = r * r * r;
            #pragma omp simd
            for (int j = 0; j < C; ++j) {
                float xj = xd[t, j], dy = gy[t, j], gj = g->data[j];
                xg[t, j] += r * dy * gj - (xj * r3 / C) * S;
                g->grad[j] += dy * xj * r;
            }
        }
        break;
    }
    case Op::GELU: {
        Node* x = n.a;
        const float* __restrict xd  = x->data.data();
        const float* __restrict gy  = n.grad.data();
        float* __restrict gx        = x->grad.data();
        const size_t n_el = x->data.size();
        if constexpr (FAST_MATH) {
            #pragma omp simd
            for (size_t i = 0; i < n_el; ++i) gx[i] += gy[i] * dgelu_fast(xd[i]);
        } else {
            const float inv_sqrt2 = 0.70710678f, inv_sqrt2pi = 0.39894228f;
            for (size_t i = 0; i < n_el; ++i) {
                float v = xd[i];
                float cdf = 0.5f * (1.f + std::erf(v * inv_sqrt2));
                float pdf = inv_sqrt2pi * std::exp(-0.5f * v * v);
                gx[i] += gy[i] * (cdf + v * pdf);
            }
        }
        break;
    }
    case Op::Attn: {
        Node* q = n.a; Node* k = n.b; Node* v = n.bias;
        const int T = q->rows, C = q->cols, H = n.heads, d = C / H;
        const float scale = 1.f / std::sqrt((float)d);
        std::span<float> P = n.scratch;
        auto Pidx = [T](int h, int i, int j) { return ((size_t)h * T + i) * T + j; };
        Mat ng = mat(n.grad, T, C), vg = mat(v->grad, T, C), vd = mat(v->data, T, C);
        Mat qg = mat(q->grad, T, C), kg = mat(k->grad, T, C);
        Mat qd = mat(q->data, T, C), kd = mat(k->data, T, C);
        for (int h = 0; h < H; ++h) {
            int off = h * d;
            for (int i = 0; i < T; ++i) {
                std::array<float, SEQ_LEN> dP{};
                for (int j = 0; j <= i; ++j) {
                    float p = P[Pidx(h, i, j)], dp = 0.f;
                    #pragma omp simd reduction(+ : dp)
                    for (int a = 0; a < d; ++a) {
                        float dout = ng[i, off + a];
                        vg[j, off + a] += p * dout;
                        dp += dout * vd[j, off + a];
                    }
                    dP[j] = dp;
                }
                float dot = 0.f;
                #pragma omp simd reduction(+ : dot)
                for (int j = 0; j <= i; ++j) dot += P[Pidx(h, i, j)] * dP[j];
                for (int j = 0; j <= i; ++j) {
                    float ds = P[Pidx(h, i, j)] * (dP[j] - dot) * scale;
                    for (int a = 0; a < d; ++a) {
                        qg[i, off + a] += ds * kd[j, off + a];
                        kg[j, off + a] += ds * qd[i, off + a];
                    }
                }
            }
        }
        break;
    }
    case Op::CrossEnt: {
        Node* logits = n.a;
        const int T = logits->rows, V = logits->cols;
        std::span<float> probs = n.scratch;
        float g = n.grad[0] / T;
        Mat lg = mat(logits->grad, T, V);
        for (int t = 0; t < T; ++t)
            for (int j = 0; j < V; ++j) {
                float p = probs[(size_t)t * V + j];
                lg[t, j] += g * (p - (j == n.ids[t] ? 1.f : 0.f));
            }
        break;
    }
    }
}

// ============================================================================
//  Model (internal): parameter layout + forward pass
// ============================================================================

namespace {
struct Layer { Node *ln1, *ln2, *Wq, *Wk, *Wv, *Wo, *W1, *b1, *W2, *b2; };

struct Model {
    Node* tok_emb;
    Node* pos_emb;
    std::array<Layer, N_LAYERS> layers;
    Node* ln_f;
    Node* lm_head;
    Node* lm_bias;

    // Lay out the parameter nodes for the CALLING thread: data spans into the shared
    // weights, grad spans into this thread's accumulator. Deterministic offsets, so
    // every thread agrees on the layout. No weight initialization here.
    void build_layout() {
        W->pused = 0; W->pcount = 0;
        tok_emb = mk_param(VOCAB, D_MODEL, false);
        pos_emb = mk_param(SEQ_LEN, D_MODEL, false);
        for (auto& L : layers) {
            L.ln1 = mk_param(1, D_MODEL, false);
            L.ln2 = mk_param(1, D_MODEL, false);
            L.Wq = mk_param(D_MODEL, D_MODEL, true);
            L.Wk = mk_param(D_MODEL, D_MODEL, true);
            L.Wv = mk_param(D_MODEL, D_MODEL, true);
            L.Wo = mk_param(D_MODEL, D_MODEL, true);
            L.W1 = mk_param(D_MODEL, D_FF, true);
            L.b1 = mk_param(1, D_FF, false);
            L.W2 = mk_param(D_FF, D_MODEL, true);
            L.b2 = mk_param(1, D_MODEL, false);
        }
        ln_f = mk_param(1, D_MODEL, false);
        lm_head = mk_param(D_MODEL, VOCAB, true);
        lm_bias = mk_param(1, VOCAB, false);
    }

    // Randomly initialize the SHARED weights through this thread's node layout.
    // Called once (main thread); biases stay zero from the static arena.
    void init_weights() {
        std::mt19937 rng(1234);
        auto randn = [&](Node* t, float std) {
            std::normal_distribution<float> nd(0.f, std);
            for (auto& x : t->data) x = nd(rng);
        };
        auto ones = [](Node* t) { std::fill(t->data.begin(), t->data.end(), 1.f); };
        randn(tok_emb, 0.02f);
        randn(pos_emb, 0.02f);
        for (auto& L : layers) {
            ones(L.ln1); ones(L.ln2);
            randn(L.Wq, 0.02f); randn(L.Wk, 0.02f); randn(L.Wv, 0.02f); randn(L.Wo, 0.02f);
            randn(L.W1, 0.02f); randn(L.W2, 0.02f);
        }
        ones(ln_f);
        randn(lm_head, 0.02f);
    }

    Node* forward(const int* ids, int T) {
        // Persistent per-thread: op_embed stores this pointer and backward reads it
        // after forward returns, so it must outlive the call (a local would dangle).
        static thread_local int pos_ids[SEQ_LEN];
        for (int t = 0; t < T; ++t) pos_ids[t] = t;
        constexpr bool q = USE_TERNARY;
        Node* h = op_add(op_embed(tok_emb, ids, T), op_embed(pos_emb, pos_ids, T));
        for (auto& L : layers) {
            Node* a = op_rmsnorm(h, L.ln1);
            Node* att = op_attn(op_linear(a, L.Wq, nullptr, q),
                                op_linear(a, L.Wk, nullptr, q),
                                op_linear(a, L.Wv, nullptr, q), N_HEADS);
            h = op_add(h, op_linear(att, L.Wo, nullptr, q));
            Node* f = op_rmsnorm(h, L.ln2);
            f = op_linear(op_gelu(op_linear(f, L.W1, L.b1, q)), L.W2, L.b2, q);
            h = op_add(h, f);
        }
        h = op_rmsnorm(h, ln_f);
        return op_linear(h, lm_head, lm_bias, false);  // head stays full precision
    }
};

thread_local Model g_model;

// Flush-to-zero (FTZ) + denormals-are-zero (DAZ). A subnormal float operand traps
// into a slow microcode assist on x86 (often ~100x a normal op); the fast-math
// approximations and decaying gradients can produce them in the hot loops. MXCSR is
// per-thread, so every compute thread sets this once (via ensure_thread_built). The
// model tolerates flushing these near-zero values -- they are underflow noise here.
static inline void set_flush_denormals() {
#if defined(SUB0_X86)
    _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
    _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
#endif
}

// Bind the calling thread to its Worker slot and lay out its parameter nodes once.
// `W` (thread_local) doubles as the guard: once non-null the thread is bound and
// g_model is populated, so the hot path early-outs without touching the OpenMP
// runtime. Buffers are static BSS (not heap); the reduction reads each slot's grad
// directly, so there is nothing to register.
static void ensure_thread_built() {
    if (W) return;
    set_flush_denormals();                                // FTZ/DAZ for this thread's MXCSR
    W = &g_workers[omp_get_thread_num() % MAX_WORKERS];   // clamp; train_batch caps the width
    g_model.build_layout();                               // grad spans now reference this slot
}
}  // anonymous namespace

// ============================================================================
//  Serialization (header validates the generated config)
// ============================================================================

namespace {
struct Header {
    char magic[4] = {'S', '0', 'L', '4'};
    int d_model = D_MODEL, n_layers = N_LAYERS, n_heads = N_HEADS;
    int d_ff = D_FF, seq_len = SEQ_LEN, vocab = VOCAB, ternary = USE_TERNARY;
    uint64_t param_floats = PARAM_FLOATS;
};
}

// ============================================================================
//  Exposed API
// ============================================================================

void build_model() {
    ensure_thread_built();                       // this (main) thread's node layout
    if (!g_params_init.exchange(true)) g_model.init_weights();   // randomize shared weights once
}

void save_model(const char* path) {
    std::ofstream os(path, std::ios::binary);
    Header h;
    os.write((const char*)&h, sizeof(h));
    os.write((const char*)g_param_data.data(), (std::streamsize)(PARAM_FLOATS * sizeof(float)));
}

bool load_model(const char* path) {
    std::ifstream is(path, std::ios::binary);
    if (!is) return false;
    Header h, ref;
    is.read((char*)&h, sizeof(h));
    if (std::memcmp(h.magic, ref.magic, 4) != 0 ||
        h.d_model != ref.d_model || h.n_layers != ref.n_layers || h.n_heads != ref.n_heads ||
        h.d_ff != ref.d_ff || h.seq_len != ref.seq_len || h.vocab != ref.vocab ||
        h.ternary != ref.ternary || h.param_floats != ref.param_floats) {
        std::println(stderr, "error: model was built with a different (constexpr) config");
        return false;
    }
    is.read((char*)g_param_data.data(), (std::streamsize)(PARAM_FLOATS * sizeof(float)));
    return (bool)is;
}

void print_config() {
    std::println("model: d={} L={} H={} ff={} seq={} vocab={}{} | params: {:.2f}M | "
                 "static mem: params {:.1f}MB acts {:.1f}MB | math: {}",
                 D_MODEL, N_LAYERS, N_HEADS, D_FF, SEQ_LEN, VOCAB,
                 USE_TERNARY ? " (ternary)" : "", PARAM_FLOATS / 1e6,
                 4 * PARAM_FLOATS * sizeof(float) / 1e6, 2 * ACT_CAP * sizeof(float) / 1e6,
                 FAST_MATH ? "fast" : "exact");
}

bool fast_math() { return FAST_MATH; }

const char* default_corpus()     { return DEFAULT_CORPUS; }
const char* default_corpus_tok() { return DEFAULT_CORPUS_TOK; }
const char* default_tokenizer()  { return DEFAULT_TOKENIZER; }

std::size_t trainable_floats() { return PARAM_FLOATS; }
float*      params_ptr()       { return g_param_data.data(); }
float*      grad_ptr()         { return g_param_grad.data(); }   // reduced grad the optimizer reads
float*      adam_m_ptr()       { return g_param_m.data(); }
float*      adam_v_ptr()       { return g_param_vel.data(); }

// ============================================================================
//  Runtime BPE tokenizer (loaded from tokenizer.bin)
// ============================================================================

namespace {

struct PairHash {
    std::size_t operator()(const std::pair<int, int>& p) const noexcept {
        return (static_cast<std::size_t>(static_cast<std::uint32_t>(p.first)) << 32) ^
               static_cast<std::uint32_t>(p.second);
    }
};

struct Tokenizer {
    bool loaded = false;
    int  vocab  = 0;
    int  n_base = 0;
    std::array<int, 256> byte_base{};  // byte value -> base id (-1 if unused)
    int  cap_id = -1, up_id = -1;      // base ids of the case markers
    std::vector<std::vector<int>> expansion;  // id -> base symbol codes (0-255, 256, 257)
    std::unordered_map<std::pair<int, int>, int, PairHash> merge_rank;  // (left,right) -> merge index
    std::unordered_set<std::string> attested;  // lowercase words eligible for case collapse

    int sym_to_base(int code) const {
        if (code == casing::TOK_CAP) return cap_id;
        if (code == casing::TOK_UP)  return up_id;
        return byte_base[static_cast<unsigned char>(code)];
    }
};

Tokenizer g_tok;

template <class T> T read_pod(std::ifstream& is) {
    T v{};
    is.read(reinterpret_cast<char*>(&v), sizeof v);
    return v;
}

// Apply learned merges to one pre-token word (sequence of base ids), lowest rank
// first, then append the resulting ids to `out`. This reproduces the corpus
// tokenization for the same word, keeping prompts in-distribution.
void bpe_encode_word(std::vector<int>& seq, std::vector<int>& out) {
    while (seq.size() >= 2) {
        int best_rank = std::numeric_limits<int>::max(), best_pos = -1;
        for (std::size_t k = 0; k + 1 < seq.size(); ++k) {
            auto it = g_tok.merge_rank.find({seq[k], seq[k + 1]});
            if (it != g_tok.merge_rank.end() && it->second < best_rank) {
                best_rank = it->second;
                best_pos  = static_cast<int>(k);
            }
        }
        if (best_pos < 0) break;
        seq[static_cast<std::size_t>(best_pos)] = g_tok.n_base + best_rank;
        seq.erase(seq.begin() + best_pos + 1);
    }
    for (int id : seq) out.push_back(id);
}

}  // namespace

bool load_tokenizer(const char* path) {
    std::ifstream is(path, std::ios::binary);
    if (!is) { std::println(stderr, "tokenizer: cannot open '{}'", path); return false; }
    if (read_pod<std::uint32_t>(is) != 0x5A543053u) {  // "S0TZ"
        std::println(stderr, "tokenizer: bad magic in '{}'", path);
        return false;
    }
    Tokenizer t;
    t.vocab  = static_cast<int>(read_pod<std::uint32_t>(is));
    t.n_base = static_cast<int>(read_pod<std::uint32_t>(is));
    t.byte_base.fill(-1);
    t.expansion.resize(static_cast<std::size_t>(t.n_base));
    for (int i = 0; i < t.n_base; ++i) {
        const int code = static_cast<int>(read_pod<std::uint16_t>(is));
        t.expansion[static_cast<std::size_t>(i)] = {code};
        if (code == casing::TOK_CAP)     t.cap_id = i;
        else if (code == casing::TOK_UP) t.up_id  = i;
        else                             t.byte_base[static_cast<unsigned char>(code)] = i;
    }
    const int n_merges = static_cast<int>(read_pod<std::uint32_t>(is));
    t.expansion.reserve(static_cast<std::size_t>(t.n_base + n_merges));
    for (int i = 0; i < n_merges; ++i) {
        const int a = static_cast<int>(read_pod<std::uint32_t>(is));
        const int b = static_cast<int>(read_pod<std::uint32_t>(is));
        t.merge_rank.emplace(std::pair{a, b}, i);
        std::vector<int> exp = t.expansion[static_cast<std::size_t>(a)];
        exp.insert(exp.end(), t.expansion[static_cast<std::size_t>(b)].begin(),
                   t.expansion[static_cast<std::size_t>(b)].end());
        t.expansion.push_back(std::move(exp));
    }
    const int n_words = static_cast<int>(read_pod<std::uint32_t>(is));
    for (int i = 0; i < n_words; ++i) {
        const int len = static_cast<int>(read_pod<std::uint16_t>(is));
        std::string w(static_cast<std::size_t>(len), '\0');
        is.read(w.data(), len);
        t.attested.insert(std::move(w));
    }
    if (!is) { std::println(stderr, "tokenizer: truncated '{}'", path); return false; }
    if (t.vocab != VOCAB)
        std::println(stderr, "tokenizer: vocab {} != built-in VOCAB {} (stale artifact?)", t.vocab, VOCAB);
    t.loaded = true;
    g_tok = std::move(t);
    return true;
}

std::vector<int> encode(const std::string& text) {
    std::vector<int> out;
    if (!g_tok.loaded) return out;
    long replaced = 0;
    const std::string norm = casing::normalize_text(text, replaced);
    const std::vector<int> stream = casing::truecase_tokenize(norm, g_tok.attested, nullptr);

    // Mirror the configurator's pre-tokenization exactly (casing::word_unit_end):
    // word units are letter runs incl. accented UTF-8 and interior apostrophes, the
    // case markers stay atomic. A prompt's "They" thus encodes as <|cap|> + the
    // shared `they` token, in-distribution with training.
    out.reserve(stream.size());
    for (std::size_t i = 0, n = stream.size(); i < n;) {
        const std::size_t end = casing::word_unit_end(stream, i);
        if (end == i) {
            const int id = g_tok.sym_to_base(stream[i]);
            if (id >= 0) out.push_back(id);
            ++i;
            continue;
        }
        std::vector<int> seq;
        for (std::size_t k = i; k < end; ++k) {
            const int id = g_tok.sym_to_base(stream[k]);
            if (id >= 0) seq.push_back(id);
        }
        bpe_encode_word(seq, out);
        i = end;
    }
    return out;
}

std::string detokenize(const std::vector<int>& ids) {
    std::vector<int> stream;
    for (int id : ids) {
        if (id < 0 || id >= static_cast<int>(g_tok.expansion.size())) continue;
        const std::vector<int>& e = g_tok.expansion[static_cast<std::size_t>(id)];
        stream.insert(stream.end(), e.begin(), e.end());
    }
    return casing::detokenize(stream);
}

namespace {
// Render a token's expansion (base-symbol codes) into a printable string: case
// markers become <|cap|>/<|up|>, control and high bytes are escaped, so the table
// stays one line per token regardless of the bytes a merge happens to contain.
std::string render_expansion(const std::vector<int>& codes) {
    std::string s;
    for (int code : codes) {
        if (code == casing::TOK_CAP)      { s += "<|cap|>"; continue; }
        if (code == casing::TOK_UP)       { s += "<|up|>";  continue; }
        const unsigned char c = static_cast<unsigned char>(code);
        switch (c) {
            case '\n': s += "\\n"; break;
            case '\t': s += "\\t"; break;
            case '\r': s += "\\r"; break;
            default:
                if (c >= 0x20 && c < 0x7F) s += static_cast<char>(c);
                else { char b[5]; std::snprintf(b, sizeof b, "\\x%02X", c); s += b; }
        }
    }
    return s;
}
}  // namespace

std::vector<TokenEntry> vocab_entries() {
    std::vector<TokenEntry> rows;
    if (!g_tok.loaded) return rows;
    rows.reserve(g_tok.expansion.size());
    for (std::size_t id = 0; id < g_tok.expansion.size(); ++id) {
        const std::vector<int>& exp = g_tok.expansion[id];
        TokenEntry e;
        e.id = static_cast<int>(id);
        if (static_cast<int>(id) >= g_tok.n_base)        e.kind = TokenEntry::Kind::Merge;
        else if (!exp.empty() && exp[0] == casing::TOK_CAP) e.kind = TokenEntry::Kind::CapMarker;
        else if (!exp.empty() && exp[0] == casing::TOK_UP)  e.kind = TokenEntry::Kind::UpMarker;
        else                                                e.kind = TokenEntry::Kind::Byte;
        e.expansion_len = static_cast<int>(exp.size());
        e.text = render_expansion(exp);
        rows.push_back(std::move(e));
    }
    return rows;
}

void graph_reset() { W->pool_used = 0; W->act_used = 0; }

Node* forward(const int* ids, int T) { ensure_thread_built(); return g_model.forward(ids, T); }
Node* cross_entropy(Node* logits, const int* targets) { return op_cross_entropy(logits, targets); }

void backward(Node* loss, float seed) {
    loss->grad[0] = seed;
    for (Node& n : W->pool | std::views::take(W->pool_used) | std::views::reverse) backward_node(n);
}

// Single-window reduction: publish this thread's accumulator as the shared gradient
// the optimizer consumes. (train_batch does the parallel multi-thread reduction.)
void reduce_gradients() { std::ranges::copy(W->grad, g_param_grad.begin()); }

// Data-parallel minibatch: each window's full forward+backward runs on its own
// thread into a private gradient accumulator, then the accumulators are summed into
// the shared gradient. Returns the mean loss; call AdamW::step() afterwards.
float train_batch(const int* data, const std::size_t* starts, int batch, int T) {
    double total = 0.0;
    #pragma omp parallel num_threads(std::min(omp_get_max_threads(), MAX_WORKERS))
    {
        ensure_thread_built();
        std::ranges::fill(W->grad, 0.f);
        #pragma omp for reduction(+ : total) schedule(static)
        for (int b = 0; b < batch; ++b) {
            graph_reset();
            Node* logits = g_model.forward(data + starts[b], T);
            Node* loss   = op_cross_entropy(logits, data + starts[b] + 1);
            total += loss->data[0];
            backward(loss, 1.f / static_cast<float>(batch));
        }
        // (implicit barrier above: every thread's grad slot is complete)
        const int nthreads = omp_get_num_threads();
        #pragma omp for schedule(static)
        for (std::size_t i = 0; i < PARAM_FLOATS; ++i) {
            float s = 0.f;
            for (int t = 0; t < nthreads; ++t)
                s += g_workers[t].grad[i];
            g_param_grad[i] = s;
        }
    }
    return static_cast<float>(total / batch);
}

// --- AdamW ------------------------------------------------------------------

AdamW::AdamW(float lr) : lr_(lr) {}

void AdamW::zero_grad() { ensure_thread_built(); std::ranges::fill(W->grad, 0.f); }

void AdamW::step() {
    double sq = 0.0;
    #pragma omp simd reduction(+ : sq)
    for (size_t i = 0; i < PARAM_FLOATS; ++i) { double g = g_param_grad[i]; sq += g * g; }
    float norm = (float)std::sqrt(sq);
    float gs = (norm > clip_) ? clip_ / (norm + 1e-6f) : 1.f;

    ++t_;
    float bc1 = 1.f - std::pow(b1_, (float)t_);
    float bc2 = 1.f - std::pow(b2_, (float)t_);
    for (size_t pi = 0; pi < W->pcount; ++pi) {
        const ParamView& pv = W->views[pi];
        const float wd = pv.decay ? wd_ : 0.f;   // hoist the invariant branch so the loop vectorizes
        #pragma omp simd
        for (size_t i = pv.off; i < pv.off + pv.n; ++i) {
            float g = g_param_grad[i] * gs;
            g_param_m[i]   = b1_ * g_param_m[i]   + (1 - b1_) * g;
            g_param_vel[i] = b2_ * g_param_vel[i] + (1 - b2_) * g * g;
            float mhat = g_param_m[i] / bc1;
            float vhat = g_param_vel[i] / bc2;
            g_param_data[i] -= lr_ * mhat / (std::sqrt(vhat) + eps_);
            g_param_data[i] -= lr_ * wd * g_param_data[i];
        }
    }
}

}  // namespace sub0
