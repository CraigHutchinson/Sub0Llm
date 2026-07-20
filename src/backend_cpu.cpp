// backend_cpu.cpp — CPU compute backend for the engine core (part of libsub0_core).
//
// Implements the differentiable math: the autograd ops, reverse-mode backward, the
// forward pass, the data-parallel minibatch (train_batch) and the AdamW optimizer,
// plus the statically allocated parameter/activation/worker arenas. The backend-
// agnostic parts (tokenizer, serialization, sampler, config paths) live in
// engine_core.cpp. The seam is at the step level (forward / backward / train_batch /
// AdamW + the host/device parameter-sync hooks), so a GPU backend can replace this
// translation unit wholesale behind the same include/sub0/core.hpp API.
//
// Precision: the CPU backend computes in FP32 only. The per-section Dtype config
// (GEMM_DTYPE/ACT_DTYPE in sub0_config.hpp) selects reduced precision (BF16) on the GPU
// backend; on CPU every section is FP32 regardless, which is the F32 baseline those
// dtypes accumulate against. Future CPU reduced-precision support would key off the
// same Dtype enum.
//
// Statically allocated: every model dimension is a compile-time constant, so all
// parameter and activation memory lives in fixed-size std::array buffers in BSS,
// reused with no per-step heap allocation. The parameter layout is the shared
// constexpr table in include/sub0/layout.hpp.

#include "sub0/core.hpp"
#include "sub0/cpu_affinity.hpp"    // P-core-first thread pinning (hybrid Intel CPUs) -- see its own header comment
#include "sub0/layout.hpp"
#include "sub0/muon.hpp"
#include "sub0/scratch_slots.hpp"   // content-derived scratch-slot embeddings (range + ScratchBindings + encoders)

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <mdspan>
#include <memory>
#include <mutex>
#include <print>
#include <random>
#include <ranges>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

// OpenMP gate. The data-parallel training path is the whole point of the worker pool;
// a build that silently loses OpenMP (e.g. find_package's flaky libomp probe on
// clang/Windows) would compile fine and then run every "thread" serially -- a silent
// perf regression that is painful to diagnose after the fact. So unless the build
// explicitly opts out (-DSUB0_REQUIRE_OPENMP=OFF), a missing _OPENMP is a hard compile
// error here rather than a quiet fallback. Do NOT relax this to an unconditional stub.
#if defined(_OPENMP)
#include <omp.h>
#elif defined(SUB0_REQUIRE_OPENMP)
#error "OpenMP required but _OPENMP is undefined: this translation unit was compiled without OpenMP, so the data-parallel path would silently run single-threaded. Reconfigure with OpenMP available, or pass -DSUB0_REQUIRE_OPENMP=OFF to build single-threaded on purpose."
#else
// Intentional single-threaded fallback (configured with -DSUB0_REQUIRE_OPENMP=OFF).
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

// NUM_PARAMS and PARAM_FLOATS come from the shared constexpr parameter layout
// (include/sub0/layout.hpp). The on-disk checkpoint and every parameter arena are
// PARAM_FLOATS long, in the order PARAM_LAYOUT defines.

consteval size_t calc_act_cap() {
    const size_t T = SEQ_LEN, C = D_MODEL, F = D_FF, H = N_HEADS, V = VOCAB;
    size_t base = 3 * T * C;
    // FFN activation nodes: plain (W1-out, gelu-out) = 2*T*F; gated (gate-out, up-out, swiglu-out)
    // = 3*T*F (one extra T*F node -- see op_swiglu in the forward pass).
    size_t per  = 10 * T * C + (USE_GATED_FFN ? 3 : 2) * T * F + H * T * T + 2 * T
                + (USE_TERNARY ? (size_t)4 * C * C + (USE_GATED_FFN ? 3 : 2) * C * F : 0)
                + (POS_ENCODING == PosEncoding::Rope ? (size_t)2 * T * C : 0)   // op_rope(q), op_rope(k)
                + (USE_QK_NORM ? (size_t)2 * T * C + 2 * T * H : 0);  // op_qknorm(q), op_qknorm(k) + rinv scratch
    size_t fin  = T * C + 2 * T * V + 64;
    return base + (size_t)N_LAYERS * per + fin;
}
constexpr size_t ACT_CAP   = calc_act_cap() * 3 / 2 + 8192;
// The gated FFN adds one extra per-layer node (gate-linear + up-linear + swiglu vs. W1-linear +
// gelu) -- a generous flat per-layer headroom either way, not a tight count. QK-norm adds 2 more
// per-layer nodes (op_qknorm(q), op_qknorm(k)).
constexpr size_t MAX_NODES = 16 + (USE_GATED_FFN ? 18 : 16) * (size_t)N_LAYERS
                                + (USE_QK_NORM ? 2 * (size_t)N_LAYERS : 0);

// ============================================================================
//  Static storage
// ============================================================================

// Shared across all worker threads: the weights (read-only during fwd/bwd), the optimizer moments
// (touched only by AdamW::step on the main thread), and the REDUCED gradient that AdamW::step consumes.
// HEAP-allocated (not static std::array): at a large config these are 4*PARAM_FLOATS floats -- e.g.
// d768 ~= 2.6 GB -- and as zero-init BSS they push the DLL's SizeOfImage past what the Windows loader
// will map, so the image fails to load with STATUS_INVALID_IMAGE_FORMAT (0xC000007B). This is the same
// reason the per-thread Worker arrays below are heap-allocated. ensure_shared_params() allocates them
// once (zeroed) before any parameter node references them; unique_ptr<float[]> keeps [] and .get().
static std::unique_ptr<float[]> g_param_data;
static std::unique_ptr<float[]> g_param_grad;
static std::unique_ptr<float[]> g_param_m;
static std::unique_ptr<float[]> g_param_vel;
static std::once_flag           g_shared_params_once;
static void ensure_shared_params() {
    std::call_once(g_shared_params_once, [] {
        g_param_data = std::make_unique<float[]>(PARAM_FLOATS);   // value-initialized -> zeroed
        g_param_grad = std::make_unique<float[]>(PARAM_FLOATS);
        g_param_m    = std::make_unique<float[]>(PARAM_FLOATS);
        g_param_vel  = std::make_unique<float[]>(PARAM_FLOATS);
    });
}

struct ParamView { size_t off, n; bool decay; };

// All per-thread state for a data-parallel window lives in one Worker. Workers are a pool indexed
// by OpenMP thread id -- not thread_local (a multi-MB thread_local in a DLL overruns Windows'
// static-TLS block and faults). Each Worker is LAZILY HEAP-ALLOCATED on first use: only the slots
// a run actually touches allocate (a GPU run or gen/eval/bench uses just slot 0), and -- critically
// for large models -- the per-worker arrays (a PARAM_FLOATS gradient + activation arenas) stay OFF
// the DLL's static image, which a static pool would otherwise bloat past the PE 4GB SizeOfImage
// limit (the image fails to load with STATUS_INVALID_IMAGE_FORMAT). Each thread binds its slot once
// into the thread_local handle `W` (see ensure_thread_built), keeping the hot paths pointer-direct.
// The pool is sized to MAX_WORKERS, a hardware-scaled constant cached into the config; only the
// MAX_WORKERS unique_ptr handles (not the Workers) live in BSS.
struct Worker {
    std::array<float, PARAM_FLOATS> grad{};        // gradient accumulator (this slot)
    std::array<float, ACT_CAP>      act_data{};    // activation arena: values
    std::array<float, ACT_CAP>      act_grad{};    // activation arena: grads
    std::array<Node, NUM_PARAMS>    param_nodes{}; // parameter leaves (data->shared, grad->this)
    std::array<Node, MAX_NODES>     pool{};        // forward-graph node pool
    std::array<ParamView, NUM_PARAMS> views{};     // optimizer parameter spans
    size_t act_used = 0, pool_used = 0, pcount = 0, pused = 0;
};
static std::array<std::unique_ptr<Worker>, MAX_WORKERS> g_workers{};
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
    nd.data = std::span<float>(g_param_data.get() + off, n);    // shared weights
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
static inline float fast_sigmoid(float x) { return 1.f / (1.f + fast_exp(-x)); }
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
// SiLU/Swish: silu(v) = v * sigmoid(v); value and derivative kept mutually consistent (same fast_exp
// basis as GELU above) so the gradient check passes in fast mode. Used by the SwiGLU-gated FFN
// (USE_GATED_FFN): silu(gate) * up, the GGUF/Llama-family FFN convention this variant exists to import.
static inline float silu_fast(float v) { return v * fast_sigmoid(v); }
static inline float dsilu_fast(float v) {
    const float s = fast_sigmoid(v);
    return s * (1.f + v * (1.f - s));
}

// ============================================================================
//  Differentiable ops (forward builders)
// ============================================================================

// Per-context scratch-slot bindings for content-derived slot embeddings: when set, a BOUND scratch-slot
// token embeds as a function of its fragments (encode_slot) instead of a fixed reserved-id row. null =>
// today's plain tok_emb lookup, byte-for-byte. thread_local so each data-parallel train_batch worker
// carries its own window's bindings; set via set_scratch_bindings before forward/forward_one. Backward
// re-consults it (same thread, same step), so no per-node storage is needed.
thread_local const ScratchBindings* g_scratch_binds = nullptr;

// Persistent (unbounded) slot range -- SPIKE, see scratch_slots.hpp's PersistentBindings comment.
// thread_local since 2026-07-17 (originally a plain process-global under a "set once, immutable" design):
// real spike usage swaps a DIFFERENT per-document table per training window, and train_batch's
// win_persist path installs per-window tables on every worker thread -- the same per-thread lifetime
// g_scratch_binds always had. A caller wanting the original whole-process behavior just sets it once on
// each thread that forwards (or once on the only thread, the common case).
thread_local const PersistentBindings* g_persistent_binds = nullptr;

// Sentinel-PAIR bindings -- SPIKE, see scratch_slots.hpp's SentinelBindings comment: the token AFTER the
// sentinel embeds from the binding table keyed by that token. Same per-window thread_local lifetime as
// g_scratch_binds above.
thread_local const SentinelBindings* g_sentinel_binds = nullptr;

// The CALLING THREAD's token-embedding table node, registered by Model::build_layout (each worker thread
// lays out its own Node arena, so this must be thread_local, not a single global). The binding dispatches
// in op_embed/backward gate on it: a scratch-slot id (~282-287, just above the byte range) IS a valid
// POSITION id once SEQ_LEN >= 283 (e.g. the production d448 fineweb config's SEQ_LEN=512), so without
// this gate an Absolute-positional-encoding model training with content-embed would silently compose
// position rows 282-287 from POS-TABLE "fragments" -- a latent bug found in the Phase-2 review
// (2026-07-17; latent only because every production config uses RoPE, which never embeds positions
// through op_embed). The old comment here claimed positions "never satisfy is_scratch_slot" -- false
// past SEQ_LEN 282.
thread_local const Node* g_tok_emb_node = nullptr;

static Node* op_embed(Node* table, const int* ids, int T) {
    const int C = table->cols;
    Node* out = mk_node(Op::Embed, T, C);
    out->w = table; out->ids = ids;
    Mat o = mat(out->data, T, C), tab = mat(table->data, table->rows, C);
    const ScratchBindings*  binds = g_scratch_binds;    // null (the common case) => plain lookup, unchanged
    const SentinelBindings* sb    = g_sentinel_binds;
    const bool tok_table = (table == g_tok_emb_node);   // binding dispatches are TOKEN-table-only (above)
    for (int t = 0; t < T; ++t) {
        // Sentinel PAIR first (most specific): the token AFTER the sigil embeds from its handle's
        // binding. t==0 can't be a pair tail (a pair split across a window boundary degrades to the
        // plain rows -- benign: the reference simply isn't content-composed in that window).
        if (sb && tok_table && t > 0 && ids[t - 1] == sb->sigil && sb->bound(ids[t])) {
            encode_slot(table->data.data(), C, sb->fragments(ids[t]), sb->encoding,
                        out->data.data() + static_cast<std::size_t>(t) * C, sb->enc_w);
        } else if (binds && tok_table && is_scratch_slot(ids[t]) && binds->bound(ids[t])) {
            encode_slot(table->data.data(), C, binds->fragments(ids[t]), binds->encoding,
                        out->data.data() + static_cast<std::size_t>(t) * C, binds->enc_w);
        } else if (is_persistent_slot(ids[t], VOCAB)) {
            // UNCONDITIONAL for any id >= VOCAB -- never falls through to tab[ids[t],j] below, which
            // would read out of the table's [VOCAB,C] bounds. persistent_fragments is null/unbound-safe
            // (empty -> encode_slot's zero-row contract), so this is correct whether or not a real
            // persistent table is installed yet. See PersistentBindings' own comment, scratch_slots.hpp.
            const SlotEncoding enc = g_persistent_binds ? g_persistent_binds->encoding : SlotEncoding::MeanPool;
            encode_slot(table->data.data(), C, persistent_fragments(g_persistent_binds, ids[t]), enc,
                        out->data.data() + static_cast<std::size_t>(t) * C,
                        g_persistent_binds ? g_persistent_binds->enc_w : nullptr);
        } else {
            for (int j = 0; j < C; ++j) o[t, j] = tab[ids[t], j];
        }
    }
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

// QK-norm: RMSNorm applied independently to EACH head's D_HEAD-length slice of Q/K (a single
// [1,D_HEAD] gamma shared across heads, same convention as Gemma2's query/key norm), applied right
// after the Q/K projection and before RoPE. This is NOT a generalization of op_rmsnorm's math onto
// groups -- mixing every head into one norm statistic (what calling op_rmsnorm on the whole row
// would do) defeats the point of a PER-HEAD stabilizer, so this is a separate op with its own
// per-(t,head) rinv scratch, mirroring op_rmsnorm's structure at the per-head granularity instead.
static Node* op_qknorm(Node* x, Node* gamma, int H) {
    const int T = x->rows, C = x->cols, d = C / H;
    const float eps = 1e-5f;
    Node* y = mk_node(Op::QKNorm, T, C);
    y->a = x; y->w = gamma; y->heads = H;
    auto [rinv, rinv_g] = arena_alloc((size_t)T * H);
    y->scratch = rinv;
    const float* __restrict G = gamma->data.data();
    for (int t = 0; t < T; ++t) {
        const float* __restrict xr = x->data.data() + (size_t)t * C;
        float* __restrict yr       = y->data.data() + (size_t)t * C;
        for (int h = 0; h < H; ++h) {
            const int off = h * d;
            float ms = 0.f;
            #pragma omp simd reduction(+ : ms)
            for (int j = 0; j < d; ++j) ms += xr[off + j] * xr[off + j];
            ms /= d;
            const float r = 1.f / std::sqrt(ms + eps);
            rinv[(size_t)t * H + h] = r;
            for (int j = 0; j < d; ++j) yr[off + j] = xr[off + j] * r * G[j];
        }
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

// SwiGLU gate: y = silu(gate_pre) * up_pre, elementwise (gate_pre/up_pre are two separate linear
// projections of the same input -- the caller builds `op_linear(x, Wgate, ...)` and
// `op_linear(x, Wup, ...)` and passes both here, same two-input shape as op_add). The GGUF/Llama-family
// gated FFN: h = SwiGLU(x@Wgate, x@Wup); out = h@Wdown (no FFN bias in that convention).
static Node* op_swiglu(Node* gate_pre, Node* up_pre) {
    Node* y = mk_node(Op::SwiGLU, gate_pre->rows, gate_pre->cols);
    y->a = gate_pre; y->b = up_pre;
    const float* __restrict gd = gate_pre->data.data();
    const float* __restrict ud = up_pre->data.data();
    float* __restrict yd       = y->data.data();
    const size_t n = y->data.size();
    if constexpr (FAST_MATH) {
        #pragma omp simd
        for (size_t i = 0; i < n; ++i) yd[i] = silu_fast(gd[i]) * ud[i];
    } else {
        for (size_t i = 0; i < n; ++i) {
            const float g = gd[i];
            const float silu = g / (1.f + std::exp(-g));
            yd[i] = silu * ud[i];
        }
    }
    return y;
}

// Tied-embedding LM head: logits[t,v] = dot(x[t,:], table[v,:]), no bias (the common tied-embedding
// convention drops the head bias too). `table` is the SAME [VOCAB,D_MODEL] tok_emb tensor op_embed
// reads for lookup -- there is no separate head weight when USE_TIED_EMBEDDINGS.
//
// This is INHERENTLY a dot-product-per-output-column pattern, not op_linear's axpy style: op_embed
// needs tok_emb row-major-by-vocab for efficient row lookup, but an axpy-style head GEMM would want
// D_MODEL as the contiguous/outer axis instead -- one physical layout cannot be efficient for both
// uses, so the forward here pays a genuinely slower access pattern than an untied head's op_linear
// call (same total FLOPs, worse vectorization per output). This is inherent to weight tying, not a
// bug -- see the weight-tying memory note for the measured cost. Both BACKWARD passes stay
// axpy-efficient (they reduce to exactly op_linear's own forward/dW shapes), so only the forward
// pays this cost.
static Node* op_tied_head(Node* x, Node* table) {
    const int T = x->rows, C = x->cols, V = table->rows;   // table (tok_emb): [V, C], V=VOCAB
    Node* y = mk_node(Op::TiedHead, T, V);
    y->a = x; y->w = table;
    const float* __restrict X  = x->data.data();
    const float* __restrict Tb = table->data.data();
    for (int t = 0; t < T; ++t) {
        const float* __restrict xt = X + static_cast<size_t>(t) * C;
        float* __restrict yr = y->data.data() + static_cast<size_t>(t) * V;
        for (int v = 0; v < V; ++v) {
            const float* __restrict tv = Tb + static_cast<size_t>(v) * C;
            double s = 0.0;
            #pragma omp simd reduction(+ : s)
            for (int c = 0; c < C; ++c) s += static_cast<double>(xt[c]) * tv[c];
            yr[v] = static_cast<float>(s);
        }
    }
    return y;
}

// RoPE (rotary positional embedding): rotate each head's d-dimensional sub-vector of x by a
// position-dependent angle, in interleaved pairs (x[2m], x[2m+1]). Applied to the Q and K
// projections before attention, so the score q_i . k_j ends up depending only on the RELATIVE
// offset (i - j) -- no learned position table, and the rotation is an orthogonal map whose
// backward is the inverse rotation. The position of row t is t (position within the window).
// inv_freq[m] = ROPE_THETA^(-2m/d); a head dimension d is assumed even (an odd tail is left
// unrotated, which is the identity for that lone component).
static Node* op_rope(Node* x, int H) {
    const int T = x->rows, C = x->cols, d = C / H, half = d / 2;
    Node* y = mk_node(Op::Rope, T, C);
    y->a = x; y->heads = H;
    const float* __restrict xd = x->data.data();
    float* __restrict yd       = y->data.data();
    for (int t = 0; t < T; ++t) {
        const float* __restrict xr = xd + (size_t)t * C;
        float* __restrict yr       = yd + (size_t)t * C;
        for (int h = 0; h < H; ++h) {
            const int off = h * d;
            for (int m = 0; m < half; ++m) {
                const float ang = t * std::pow(ROPE_THETA, -2.f * m / d);
                const float cs = std::cos(ang), sn = std::sin(ang);
                const float x0 = xr[off + 2 * m], x1 = xr[off + 2 * m + 1];
                yr[off + 2 * m]     = x0 * cs - x1 * sn;
                yr[off + 2 * m + 1] = x0 * sn + x1 * cs;
            }
            if (2 * half < d) yr[off + d - 1] = xr[off + d - 1];   // odd-d tail: identity
        }
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

// Count of ACTIVE (non-ignored) target positions -- the normalizer both the forward loss and the
// CrossEnt backward divide by, so they must agree. A target < 0 (LOSS_IGNORE_INDEX) is masked out.
// With no masking this is just T (identical to the pre-masking behavior).
static int ce_active(const int* targets, int T) {
    int a = 0;
    for (int t = 0; t < T; ++t) a += (targets[t] >= 0);
    return a;
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
        // A masked position still gets its softmax computed (so backward's scratch is populated
        // uniformly) but contributes no loss; backward likewise skips it.
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
        if (targets[t] >= 0)                                   // skip LOSS_IGNORE_INDEX positions
            total += -std::log(std::max(1e-9f, probs[(size_t)t * V + targets[t]]));
    }
    loss->data[0] = total / static_cast<float>(std::max(1, ce_active(targets, T)));
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
        const ScratchBindings*  binds = g_scratch_binds;
        const SentinelBindings* sb    = g_sentinel_binds;
        const bool tok_table = (n.w == g_tok_emb_node);   // same token-table-only gate as the forward
        for (int t = 0; t < T; ++t) {
            // Adjoint of op_embed's content-derived branches: a composed position's row grad flows to its
            // fragment rows (encode_slot_bwd); else the plain scatter into the token's own row. The
            // dispatch conditions mirror the forward EXACTLY (same pair/slot/persistent precedence).
            if (sb && tok_table && t > 0 && n.ids[t - 1] == sb->sigil && sb->bound(n.ids[t])) {
                encode_slot_bwd(n.grad.data() + static_cast<std::size_t>(t) * C, C,
                                sb->fragments(n.ids[t]), sb->encoding, n.w->grad.data(),
                                n.w->data.data(), sb->enc_w, sb->enc_w_grad);
            } else if (binds && tok_table && is_scratch_slot(n.ids[t]) && binds->bound(n.ids[t])) {
                encode_slot_bwd(n.grad.data() + static_cast<std::size_t>(t) * C, C,
                                binds->fragments(n.ids[t]), binds->encoding, n.w->grad.data(),
                                n.w->data.data(), binds->enc_w, binds->enc_w_grad);
            } else if (is_persistent_slot(n.ids[t], VOCAB)) {
                // Same unconditional guard as op_embed's forward branch -- see its comment. An
                // unbound/table-absent persistent id has empty fragments -> encode_slot_bwd early-
                // returns (no gradient scattered anywhere), which is correct: nothing to update.
                const SlotEncoding enc = g_persistent_binds ? g_persistent_binds->encoding : SlotEncoding::MeanPool;
                encode_slot_bwd(n.grad.data() + static_cast<std::size_t>(t) * C, C,
                                persistent_fragments(g_persistent_binds, n.ids[t]), enc,
                                n.w->grad.data(), n.w->data.data(),
                                g_persistent_binds ? g_persistent_binds->enc_w : nullptr,
                                g_persistent_binds ? g_persistent_binds->enc_w_grad : nullptr);
            } else {
                for (int j = 0; j < C; ++j) wg[n.ids[t], j] += ng[t, j];
            }
        }
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
    case Op::QKNorm: {
        // Same math as Op::RMSNorm, applied independently per (t,head) group of width d=C/H
        // instead of once per whole row -- see op_qknorm's comment for why this isn't just
        // op_rmsnorm called with a reshaped view.
        Node* x = n.a; Node* g = n.w;
        const int T = x->rows, C = x->cols, H = n.heads, d = C / H;
        std::span<float> rinv = n.scratch;
        Mat gy = mat(n.grad, T, C), xd = mat(x->data, T, C), xg = mat(x->grad, T, C);
        for (int t = 0; t < T; ++t) {
            for (int h = 0; h < H; ++h) {
                const int off = h * d;
                float S = 0.f;
                #pragma omp simd reduction(+ : S)
                for (int j = 0; j < d; ++j) S += gy[t, off + j] * g->data[j] * xd[t, off + j];
                float r = rinv[(size_t)t * H + h], r3 = r * r * r;
                #pragma omp simd
                for (int j = 0; j < d; ++j) {
                    float xj = xd[t, off + j], dy = gy[t, off + j], gj = g->data[j];
                    xg[t, off + j] += r * dy * gj - (xj * r3 / d) * S;
                    g->grad[j] += dy * xj * r;
                }
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
    case Op::SwiGLU: {
        // y = silu(gate) * up  ->  d(gate) = dy * up * dsilu(gate), d(up) = dy * silu(gate).
        Node* gate = n.a; Node* up = n.b;
        const float* __restrict gd = gate->data.data();
        const float* __restrict ud = up->data.data();
        const float* __restrict gy = n.grad.data();
        float* __restrict gg       = gate->grad.data();
        float* __restrict ug       = up->grad.data();
        const size_t n_el = n.data.size();
        if constexpr (FAST_MATH) {
            #pragma omp simd
            for (size_t i = 0; i < n_el; ++i) {
                const float g = gd[i], dy = gy[i];
                gg[i] += dy * ud[i] * dsilu_fast(g);
                ug[i] += dy * silu_fast(g);
            }
        } else {
            for (size_t i = 0; i < n_el; ++i) {
                const float g = gd[i], dy = gy[i];
                const float s = 1.f / (1.f + std::exp(-g));
                const float silu = g * s;
                const float dsilu = s * (1.f + g * (1.f - s));
                gg[i] += dy * ud[i] * dsilu;
                ug[i] += dy * silu;
            }
        }
        break;
    }
    case Op::TiedHead: {
        // y[t,v] = dot(x[t,:], table[v,:])  ->  dx = dY @ table (op_linear's forward shape,
        // in=V out=C), dtable[v,:] += dY[t,v]*x[t,:] (op_linear's dW shape). Both axpy-efficient --
        // only the forward pays the transposed-access cost (see op_tied_head's comment). Looping t
        // outer/v inner for BOTH accumulations (combined into one pass) keeps every row access
        // sequential; the reverse order would stride through dY with stride V (V is VOCAB, large).
        Node* x = n.a; Node* table = n.w;
        const int T = n.rows, V = n.cols, C = x->cols;
        const float* __restrict dY = n.grad.data();
        const float* __restrict X  = x->data.data();
        const float* __restrict Tb = table->data.data();
        for (int t = 0; t < T; ++t) {
            const float* __restrict dYr = dY + static_cast<size_t>(t) * V;
            const float* __restrict xt  = X  + static_cast<size_t>(t) * C;
            float* __restrict xg        = x->grad.data() + static_cast<size_t>(t) * C;
            for (int v = 0; v < V; ++v) {
                const float dyv = dYr[v];
                if (dyv == 0.f) continue;
                const float* __restrict tv = Tb + static_cast<size_t>(v) * C;
                float* __restrict tg       = table->grad.data() + static_cast<size_t>(v) * C;
                #pragma omp simd
                for (int c = 0; c < C; ++c) { xg[c] += dyv * tv[c]; tg[c] += dyv * xt[c]; }
            }
        }
        break;
    }
    case Op::Rope: {
        // Inverse rotation (R^T): grad of the un-rotated input from the grad of the rotated
        // output. Mirrors op_rope exactly; accumulates into x->grad.
        Node* x = n.a;
        const int T = x->rows, C = x->cols, H = n.heads, d = C / H, half = d / 2;
        const float* __restrict gy = n.grad.data();
        float* __restrict gx       = x->grad.data();
        for (int t = 0; t < T; ++t) {
            const float* __restrict gyr = gy + (size_t)t * C;
            float* __restrict gxr       = gx + (size_t)t * C;
            for (int h = 0; h < H; ++h) {
                const int off = h * d;
                for (int m = 0; m < half; ++m) {
                    const float ang = t * std::pow(ROPE_THETA, -2.f * m / d);
                    const float cs = std::cos(ang), sn = std::sin(ang);
                    const float g0 = gyr[off + 2 * m], g1 = gyr[off + 2 * m + 1];
                    gxr[off + 2 * m]     +=  g0 * cs + g1 * sn;
                    gxr[off + 2 * m + 1] += -g0 * sn + g1 * cs;
                }
                if (2 * half < d) gxr[off + d - 1] += gyr[off + d - 1];
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
        // Divide by the SAME active count the forward used (recomputed from ids, O(T)), so a masked
        // position dilutes neither the loss nor the gradient. A masked row (n.ids[t] < 0) is skipped
        // entirely, leaving its logit-grad at the arena's zero (graph_reset zeroes grads) -- no signal.
        const float g = n.grad[0] / static_cast<float>(std::max(1, ce_active(n.ids, T)));
        Mat lg = mat(logits->grad, T, V);
        for (int t = 0; t < T; ++t) {
            if (n.ids[t] < 0) continue;                       // LOSS_IGNORE_INDEX: zero gradient row
            for (int j = 0; j < V; ++j) {
                float p = probs[(size_t)t * V + j];
                lg[t, j] += g * (p - (j == n.ids[t] ? 1.f : 0.f));
            }
        }
        break;
    }
    }
}

// ============================================================================
//  Model (internal): parameter layout + forward pass
// ============================================================================

namespace {
// --- Incremental single-token inference (KV-cache) --------------------------
// forward() recomputes the WHOLE context every token, so autoregressive gen is O(n*T^2). forward_one
// keeps a per-layer K/V cache and, given the new token + its position, runs the network over just
// that ONE row -- appending its K/V and attending the single new query against the cache. Per token
// this is O(T) attention + O(1) rows through the projections/FFN/head, instead of O(T) rows through
// everything. The row helpers below replicate the dense op_* math EXACTLY (same accumulation order),
// so forward_one's logits match forward()'s last row to fast-math tolerance (gated by a unit test).
// Dense weights only (the sparse-ternary linear path is not mirrored here); a ternary build keeps
// the full-forward gen path. Positions must stay < SEQ_LEN (gen falls back past the window).
struct KVCache {
    std::vector<float> k, v;                                    // [N_LAYERS][SEQ_LEN][D_MODEL], flat
    void reset() {
        const size_t n = static_cast<size_t>(N_LAYERS) * SEQ_LEN * D_MODEL;
        if (k.size() != n) { k.assign(n, 0.f); v.assign(n, 0.f); }
    }
    float* krow(int l, int pos) { return k.data() + ((static_cast<size_t>(l) * SEQ_LEN) + pos) * D_MODEL; }
    float* vrow(int l, int pos) { return v.data() + ((static_cast<size_t>(l) * SEQ_LEN) + pos) * D_MODEL; }
};
thread_local KVCache g_kv;                                      // gen is single-threaded; lazily sized

// y[out] = x[in] . W[in,out]  (+ bias); dense, same order as op_linear's non-ternary path.
static inline void linear_row(const float* __restrict x, const Node* W, const Node* bias,
                              float* __restrict y, int in, int out) {
    for (int o = 0; o < out; ++o) y[o] = 0.f;
    const float* __restrict Wf = W->data.data();
    for (int p = 0; p < in; ++p) {
        const float xp = x[p];
        if (xp == 0.f) continue;
        const float* __restrict Wr = Wf + static_cast<size_t>(p) * out;
        for (int o = 0; o < out; ++o) y[o] += xp * Wr[o];
    }
    if (bias) for (int o = 0; o < out; ++o) y[o] += bias->data[o];
}
// Tied-embedding head, single-row (generation) form: y[v] = dot(x[:], table[v,:]), no bias -- see
// op_tied_head's comment for why this is a dot-product pattern rather than linear_row's axpy one.
static inline void tied_head_row(const float* __restrict x, const Node* table,
                                 float* __restrict y, int C, int V) {
    const float* __restrict Tb = table->data.data();
    for (int v = 0; v < V; ++v) {
        const float* __restrict tv = Tb + static_cast<size_t>(v) * C;
        float s = 0.f;
        #pragma omp simd reduction(+ : s)
        for (int c = 0; c < C; ++c) s += x[c] * tv[c];
        y[v] = s;
    }
}
static inline void rmsnorm_row(const float* __restrict x, const Node* gamma, float* __restrict y, int C) {
    float ms = 0.f; for (int j = 0; j < C; ++j) ms += x[j] * x[j]; ms /= C;
    const float r = 1.f / std::sqrt(ms + 1e-5f);
    const float* __restrict G = gamma->data.data();
    for (int j = 0; j < C; ++j) y[j] = x[j] * r * G[j];
}
static inline void qknorm_row(float* __restrict x, const Node* gamma, int H, int C) {   // in place, mirrors rope_row
    const int d = C / H;
    const float* __restrict G = gamma->data.data();
    for (int h = 0; h < H; ++h) {
        const int off = h * d;
        float ms = 0.f; for (int j = 0; j < d; ++j) ms += x[off + j] * x[off + j]; ms /= d;
        const float r = 1.f / std::sqrt(ms + 1e-5f);
        for (int j = 0; j < d; ++j) x[off + j] *= r * G[j];
    }
}
static inline void rope_row(float* __restrict x, int pos, int H, int C) {   // rotate Q/K in place (t = pos)
    const int d = C / H, half = d / 2;
    for (int h = 0; h < H; ++h) {
        const int off = h * d;
        for (int m = 0; m < half; ++m) {
            const float ang = pos * std::pow(ROPE_THETA, -2.f * m / d);
            const float cs = std::cos(ang), sn = std::sin(ang);
            const float x0 = x[off + 2 * m], x1 = x[off + 2 * m + 1];
            x[off + 2 * m]     = x0 * cs - x1 * sn;
            x[off + 2 * m + 1] = x0 * sn + x1 * cs;
        }
    }
}
static inline float gelu_row(float v) {
    if constexpr (FAST_MATH) return gelu_fast(v);
    else return 0.5f * v * (1.f + std::erf(v * 0.70710678f));
}
static inline float silu_row(float v) {
    if constexpr (FAST_MATH) return silu_fast(v);
    else return v / (1.f + std::exp(-v));
}

// Wg (gate matrix) is only used when USE_GATED_FFN; b1/b2 (FFN biases) only when !USE_GATED_FFN;
// q_norm/k_norm only when USE_QK_NORM -- the always-present fields keep this struct's shape
// independent of the compile-time choice, cheaper than conditionally compiling the struct itself,
// and unused pointers just stay nullptr.
struct Layer { Node *ln1, *ln2, *Wq, *Wk, *Wv, *Wo, *W1, *b1, *W2, *b2, *Wg, *q_norm, *k_norm; };

struct Model {
    Node* tok_emb;
    Node* pos_emb;
    std::array<Layer, N_LAYERS> layers;
    Node* ln_f;
    Node* lm_head;   // nullptr when USE_TIED_EMBEDDINGS -- the head reads tok_emb directly instead
    Node* lm_bias;   // nullptr when USE_TIED_EMBEDDINGS -- tied models drop the head bias too

    // Diagnostic-only: forward_one's residual-stream hidden state at its last call's position, right
    // before ln_f/the head projection -- i.e. the fully-processed, pre-readout representation. Written
    // unconditionally (one D_MODEL-length copy, negligible next to forward_one's own cost) rather than
    // behind an optional out-param, to avoid touching forward_one's signature (shared with the CUDA
    // backend's device_backend.hpp interface -- see last_hidden_ptr() in core.hpp for why this stays
    // CPU-only and out-of-band instead).
    std::array<float, D_MODEL> last_hidden{};

    // Lay out the parameter nodes for the CALLING thread: data spans into the shared
    // weights, grad spans into this thread's accumulator. Deterministic offsets, so
    // every thread agrees on the layout. No weight initialization here.
    void build_layout() {
        W->pused = 0; W->pcount = 0;
        tok_emb = mk_param(VOCAB, D_MODEL, false);
        g_tok_emb_node = tok_emb;   // register THIS thread's token table for op_embed's binding gate
        pos_emb = mk_param(SEQ_LEN, D_MODEL, false);
        for (auto& L : layers) {
            L.ln1 = mk_param(1, D_MODEL, false);
            L.ln2 = mk_param(1, D_MODEL, false);
            L.Wq = mk_param(D_MODEL, D_MODEL, true);
            L.Wk = mk_param(D_MODEL, D_MODEL, true);
            L.Wv = mk_param(D_MODEL, D_MODEL, true);
            L.Wo = mk_param(D_MODEL, D_MODEL, true);
            // Order here MUST match layout.hpp's make_param_layout() exactly -- it IS the
            // serialization order (see that file's header comment).
            if constexpr (USE_QK_NORM) {
                L.q_norm = mk_param(1, D_HEAD, false);
                L.k_norm = mk_param(1, D_HEAD, false);
            }
            if constexpr (USE_GATED_FFN) {
                L.Wg = mk_param(D_MODEL, D_FF, true);   // gate
                L.W1 = mk_param(D_MODEL, D_FF, true);   // up
                L.W2 = mk_param(D_FF, D_MODEL, true);   // down (no bias)
            } else {
                L.W1 = mk_param(D_MODEL, D_FF, true);
                L.b1 = mk_param(1, D_FF, false);
                L.W2 = mk_param(D_FF, D_MODEL, true);
                L.b2 = mk_param(1, D_MODEL, false);
            }
        }
        ln_f = mk_param(1, D_MODEL, false);
        if constexpr (!USE_TIED_EMBEDDINGS) {
            lm_head = mk_param(D_MODEL, VOCAB, true);
            lm_bias = mk_param(1, VOCAB, false);
        }
    }

    // Randomly initialize the SHARED weights through this thread's node layout, from a fixed seed
    // (deterministic, so every caller -- production and tests alike -- gets the identical baseline
    // model). Biases stay zero from the static arena.
    void init_weights() {
        std::mt19937 rng(1234);
        auto randn = [&](Node* t, float std) {
            std::normal_distribution<float> nd(0.f, std);
            for (auto& x : t->data) x = nd(rng);
        };
        auto ones = [](Node* t) { std::fill(t->data.begin(), t->data.end(), 1.f); };
        randn(tok_emb, 0.02f);
        // Under RoPE the position table is unused (kept in the layout for format stability);
        // zero it so the checkpoint is deterministic and it never perturbs anything.
        if constexpr (POS_ENCODING == PosEncoding::Absolute) randn(pos_emb, 0.02f);
        else                                                 std::fill(pos_emb->data.begin(), pos_emb->data.end(), 0.f);
        for (auto& L : layers) {
            ones(L.ln1); ones(L.ln2);
            randn(L.Wq, 0.02f); randn(L.Wk, 0.02f); randn(L.Wv, 0.02f); randn(L.Wo, 0.02f);
            if constexpr (USE_QK_NORM) { ones(L.q_norm); ones(L.k_norm); }
            if constexpr (USE_GATED_FFN) randn(L.Wg, 0.02f);
            randn(L.W1, 0.02f); randn(L.W2, 0.02f);
        }
        ones(ln_f);
        if constexpr (!USE_TIED_EMBEDDINGS) randn(lm_head, 0.02f);
    }

    Node* forward(const int* ids, int T) {
        // Persistent per-thread: op_embed stores this pointer and backward reads it
        // after forward returns, so it must outlive the call (a local would dangle).
        static thread_local int pos_ids[SEQ_LEN];
        constexpr bool q = USE_TERNARY;
        Node* h;
        if constexpr (POS_ENCODING == PosEncoding::Absolute) {
            for (int t = 0; t < T; ++t) pos_ids[t] = t;
            h = op_add(op_embed(tok_emb, ids, T), op_embed(pos_emb, pos_ids, T));
        } else {
            h = op_embed(tok_emb, ids, T);   // RoPE injects position inside attention instead
        }
        for (auto& L : layers) {
            Node* a = op_rmsnorm(h, L.ln1);
            Node* qn = op_linear(a, L.Wq, nullptr, q);
            Node* kn = op_linear(a, L.Wk, nullptr, q);
            Node* vn = op_linear(a, L.Wv, nullptr, q);
            if constexpr (USE_QK_NORM) {
                qn = op_qknorm(qn, L.q_norm, N_HEADS);
                kn = op_qknorm(kn, L.k_norm, N_HEADS);
            }
            if constexpr (POS_ENCODING == PosEncoding::Rope) {
                qn = op_rope(qn, N_HEADS);
                kn = op_rope(kn, N_HEADS);
            }
            Node* att = op_attn(qn, kn, vn, N_HEADS);
            h = op_add(h, op_linear(att, L.Wo, nullptr, q));
            Node* f = op_rmsnorm(h, L.ln2);
            if constexpr (USE_GATED_FFN) {
                Node* gate_pre = op_linear(f, L.Wg, nullptr, q);
                Node* up_pre   = op_linear(f, L.W1, nullptr, q);
                f = op_linear(op_swiglu(gate_pre, up_pre), L.W2, nullptr, q);
            } else {
                f = op_linear(op_gelu(op_linear(f, L.W1, L.b1, q)), L.W2, L.b2, q);
            }
            h = op_add(h, f);
        }
        h = op_rmsnorm(h, ln_f);
        if constexpr (USE_TIED_EMBEDDINGS) return op_tied_head(h, tok_emb);
        else                               return op_linear(h, lm_head, lm_bias, false);  // head stays full precision
    }

    // Incremental single-token forward using the KV-cache (see KVCache above). Runs token `id` at
    // window position `pos` through the network over one row, updates the cache, and returns its
    // logits [VOCAB] (thread-local). Requires g_kv.reset() at the start of a generation and pos in
    // [0, SEQ_LEN). Dense weights only -- callers gate on !USE_TERNARY.
    const float* forward_one(int id, int pos) {
        static thread_local std::array<float, VOCAB> logits;
        // Sentinel-pair detection needs the PREVIOUS fed token: decode feeds positions strictly
        // sequentially (prefill 0..n-1, then one per generated token), so "previous" is simply the last
        // (id,pos) this thread fed -- valid only when pos follows it directly. No explicit reset needed:
        // a NEW generation starts at pos 0, and 0 == prev_pos+1 is unsatisfiable for any real prev_pos
        // (>= 0), so a previous generation's tail can never leak in as this one's sigil.
        static thread_local int prev_id = -1, prev_pos = -2;
        const int prev = (pos == prev_pos + 1) ? prev_id : -1;
        prev_id = id; prev_pos = pos;
        constexpr int C = D_MODEL, H = N_HEADS, d = C / H;
        const float scale = 1.f / std::sqrt(static_cast<float>(d));
        float h[C], a[C], qn[C], kn[C], vn[C], att[C], proj[C], f1[D_FF];
        [[maybe_unused]] float g1[D_FF];   // gate branch, gated FFN only

        if (g_sentinel_binds && prev == g_sentinel_binds->sigil && g_sentinel_binds->bound(id)) {
            encode_slot(tok_emb->data.data(), C, g_sentinel_binds->fragments(id), g_sentinel_binds->encoding,
                        h, g_sentinel_binds->enc_w);
        } else if (g_scratch_binds && is_scratch_slot(id) && g_scratch_binds->bound(id)) {
            encode_slot(tok_emb->data.data(), C, g_scratch_binds->fragments(id), g_scratch_binds->encoding, h,
                        g_scratch_binds->enc_w);
        } else if (is_persistent_slot(id, VOCAB)) {
            // Same unconditional guard as op_embed's forward branch (backend_cpu.cpp) -- see its
            // comment. Decode never runs backward, so this path only needs the forward compose (enc_w,
            // never enc_w_grad).
            const SlotEncoding enc = g_persistent_binds ? g_persistent_binds->encoding : SlotEncoding::MeanPool;
            encode_slot(tok_emb->data.data(), C, persistent_fragments(g_persistent_binds, id), enc, h,
                        g_persistent_binds ? g_persistent_binds->enc_w : nullptr);
        } else {
            const float* emb = tok_emb->data.data() + static_cast<size_t>(id) * C;
            for (int j = 0; j < C; ++j) h[j] = emb[j];
        }
        if constexpr (POS_ENCODING == PosEncoding::Absolute) {
            const float* pe = pos_emb->data.data() + static_cast<size_t>(pos) * C;
            for (int j = 0; j < C; ++j) h[j] += pe[j];
        }
        for (int l = 0; l < N_LAYERS; ++l) {
            Layer& L = layers[l];
            rmsnorm_row(h, L.ln1, a, C);
            linear_row(a, L.Wq, nullptr, qn, C, C);
            linear_row(a, L.Wk, nullptr, kn, C, C);
            linear_row(a, L.Wv, nullptr, vn, C, C);
            if constexpr (USE_QK_NORM) { qknorm_row(qn, L.q_norm, H, C); qknorm_row(kn, L.k_norm, H, C); }
            if constexpr (POS_ENCODING == PosEncoding::Rope) { rope_row(qn, pos, H, C); rope_row(kn, pos, H, C); }
            float* kc = g_kv.krow(l, pos); float* vc = g_kv.vrow(l, pos);        // append this token's K/V
            for (int j = 0; j < C; ++j) { kc[j] = kn[j]; vc[j] = vn[j]; }
            for (int hd = 0; hd < H; ++hd) {                                     // attend query pos over j<=pos
                const int off = hd * d;
                std::array<float, SEQ_LEN> sc{};
                float mx = -1e30f;
                for (int j = 0; j <= pos; ++j) {
                    const float* kj = g_kv.krow(l, j) + off;
                    float s = 0.f; for (int aa = 0; aa < d; ++aa) s += qn[off + aa] * kj[aa];
                    s *= scale; sc[j] = s; mx = std::max(mx, s);
                }
                float Z = 0.f;
                for (int j = 0; j <= pos; ++j) { sc[j] = FAST_MATH ? fast_exp(sc[j] - mx) : std::exp(sc[j] - mx); Z += sc[j]; }
                for (int aa = 0; aa < d; ++aa) att[off + aa] = 0.f;
                for (int j = 0; j <= pos; ++j) {
                    const float p = sc[j] / Z; const float* vj = g_kv.vrow(l, j) + off;
                    for (int aa = 0; aa < d; ++aa) att[off + aa] += p * vj[aa];
                }
            }
            linear_row(att, L.Wo, nullptr, proj, C, C);
            for (int j = 0; j < C; ++j) h[j] += proj[j];                         // residual
            rmsnorm_row(h, L.ln2, a, C);
            if constexpr (USE_GATED_FFN) {
                linear_row(a, L.Wg, nullptr, g1, C, D_FF);
                linear_row(a, L.W1, nullptr, f1, C, D_FF);
                for (int j = 0; j < D_FF; ++j) f1[j] = silu_row(g1[j]) * f1[j];
                linear_row(f1, L.W2, nullptr, proj, D_FF, C);
            } else {
                linear_row(a, L.W1, L.b1, f1, C, D_FF);
                for (int j = 0; j < D_FF; ++j) f1[j] = gelu_row(f1[j]);
                linear_row(f1, L.W2, L.b2, proj, D_FF, C);
            }
            for (int j = 0; j < C; ++j) h[j] += proj[j];                         // residual
        }
        for (int j = 0; j < C; ++j) last_hidden[static_cast<std::size_t>(j)] = h[j];   // diagnostic capture
        rmsnorm_row(h, ln_f, a, C);
        if constexpr (USE_TIED_EMBEDDINGS) tied_head_row(a, tok_emb, logits.data(), C, VOCAB);
        else                               linear_row(a, lm_head, lm_bias, logits.data(), C, VOCAB);
        return logits.data();
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
// runtime. The shared arenas and each Worker are heap-allocated (see above); the
// reduction reads each slot's grad directly, so there is nothing to register.
static void ensure_thread_built() {
    if (W) return;
    ensure_shared_params();                               // heap-alloc the shared weight/grad/moment arenas once
    set_flush_denormals();                                // FTZ/DAZ for this thread's MXCSR
    const int tid = omp_get_thread_num() % MAX_WORKERS;   // clamp; train_batch caps the width
    sub0::pin_current_thread_p_first(tid);                // P-cores first (see cpu_affinity.hpp); a hint,
                                                            // never required -- silently no-ops if it fails
    // Each thread owns its slot (unique tid within the team), so this lazy alloc needs no lock.
    if (!g_workers[tid]) g_workers[tid] = std::make_unique<Worker>();
    W = g_workers[tid].get();                             // grad spans now reference this slot
    g_model.build_layout();
}
}  // anonymous namespace

// ============================================================================
//  Exposed API
// ============================================================================

void build_model() {
    ensure_thread_built();      // this (main) thread's node layout
    // Unconditional, not "once": every real caller (each production stage -- train/gen/report/...)
    // invokes this exactly once per process, immediately followed by load_model()/load_checkpoint()
    // whenever a prior model exists, so re-running the deterministic init here is a no-op in
    // practice for them. Test suites are the one place that calls build_model() many times in one
    // process (once per TEST_CASE) EXPECTING a fresh baseline model each time -- a guard that only
    // randomized on the process's first call silently left every later test's "fresh" model as
    // whatever gradient/optimizer steps earlier tests had left in the shared param arena, which
    // surfaced as order-dependent CPU/GPU parity failures once params drifted far enough from
    // init-scale (e.g. after Muon/AdamW step tests or the saturating-activation test ran first).
    g_model.init_weights();
}

// Install (or clear, with nullptr) the per-context scratch-slot bindings the next forward/forward_one on
// THIS thread will use for content-derived slot embeddings. The pointee must outlive the forward+backward
// it drives. Null restores the plain tok_emb lookup exactly. See g_scratch_binds above.
void set_scratch_bindings(const ScratchBindings* b) { g_scratch_binds = b; }

// Install (or clear) the persistent-slot table (declared near g_scratch_binds above). Unlike the
// ephemeral table (rebound per window/context), this is read-only and immutable for the process once
// set. The pointee must outlive every subsequent forward/forward_one/backward until cleared or replaced.
void set_persistent_bindings(const PersistentBindings* b) { g_persistent_binds = b; }

// Install (or clear) the sentinel-pair table for THIS thread (declared near g_scratch_binds above --
// same per-context/per-window lifetime as set_scratch_bindings, unlike the process-global persistent one).
void set_sentinel_bindings(const SentinelBindings* b) { g_sentinel_binds = b; }

// save_model / load_model live in engine_core.cpp: serialization is backend-agnostic
// and goes through params_ptr() + the host/device sync hooks.

void print_config() {
    std::println("model: d={} L={} H={} ff={} seq={} vocab={}{} | params: {:.2f}M | "
                 "heap mem: params {:.1f}MB acts {:.1f}MB | math: {}",
                 D_MODEL, N_LAYERS, N_HEADS, D_FF, SEQ_LEN, VOCAB,
                 USE_TERNARY ? " (ternary)" : "", PARAM_FLOATS / 1e6,
                 4 * PARAM_FLOATS * sizeof(float) / 1e6, 2 * ACT_CAP * sizeof(float) / 1e6,
                 FAST_MATH ? "fast" : "exact");
    // Compute backend + the host GPU detected at configure time (constexpr facts from
    // sub0_config.hpp). COMPUTE_MODE is the backend actually compiled in; HAS_CUDA flags
    // that a Phase-2 GPU build is possible on this host.
    constexpr const char* backend =
        COMPUTE_MODE == ComputeBackend::Gpu    ? "GPU"    :
        COMPUTE_MODE == ComputeBackend::Hybrid ? "HYBRID" : "CPU";
    if constexpr (HAS_CUDA)
        std::println("compute: {} | CUDA available: sm_{} ({} MB VRAM + {} MB shared overflow)",
                     backend, CUDA_ARCH, GPU_VRAM_MB, GPU_SHARED_MEM_MB);
    else
        std::println("compute: {} | CUDA: none", backend);
}

bool fast_math() { return FAST_MATH; }

std::size_t trainable_floats() { return PARAM_FLOATS; }
float*      params_ptr()       { ensure_shared_params(); return g_param_data.get(); }
float*      grad_ptr()         { ensure_shared_params(); return g_param_grad.get(); }  // reduced grad the optimizer reads
float*      adam_m_ptr()       { ensure_shared_params(); return g_param_m.get(); }
float*      adam_v_ptr()       { ensure_shared_params(); return g_param_vel.get(); }

// CPU backend: parameters already live in host memory, so the host/device sync hooks
// are no-ops. A device backend overrides these to copy the params_ptr()/adam_*_ptr()
// staging buffers across the PCIe boundary around serialization (see core.hpp).
void sync_params_to_host()   {}
void sync_params_to_device() {}

// The runtime BPE tokenizer, the logits sampler (sample_token) and model
// serialization (save_model / load_model) are backend-agnostic and live in
// engine_core.cpp.

// ensure_thread_built() matches forward()/forward_one() below: a caller running on a thread that has
// never touched the engine before (e.g. a freshly-spawned OpenMP worker) must not dereference a null
// thread_local W. Idempotent -- cheap to call even when W is already built.
void graph_reset() { ensure_thread_built(); W->pool_used = 0; W->act_used = 0; }

Node* forward(const int* ids, int T) { ensure_thread_built(); return g_model.forward(ids, T); }
Node* cross_entropy(Node* logits, const int* targets) { return op_cross_entropy(logits, targets); }

// Incremental single-token inference (KV-cache). kv_reset() clears/sizes the cache at the start of a
// generation; forward_one(id, pos) returns the logits [VOCAB] for the next token. See KVCache above.
void kv_reset() { g_kv.reset(); }
const float* forward_one(int id, int pos) { ensure_thread_built(); return g_model.forward_one(id, pos); }
const float* last_hidden_ptr() { return g_model.last_hidden.data(); }   // see Model::last_hidden's comment

void backward(Node* loss, float seed) {
    loss->grad[0] = seed;
    for (Node& n : W->pool | std::views::take(W->pool_used) | std::views::reverse) backward_node(n);
}

// Single-window reduction: publish this thread's accumulator as the shared gradient
// the optimizer consumes. (train_batch does the parallel multi-thread reduction.)
void reduce_gradients() { std::ranges::copy(W->grad, g_param_grad.get()); }

// Data-parallel minibatch: each window's full forward+backward runs on its own
// thread into a private gradient accumulator, then the accumulators are summed into
// the shared gradient. Returns the mean loss; call AdamW::step() afterwards. When
// `lengths` is given, window b trains at its own length lengths[b] (<= T) -- so a short
// document trains on exactly its tokens with no padding (an END mask). `loss_mask`, if given,
// is an INTERIOR mask: a per-token 0/1 array parallel to `data` where predicting token p counts
// only if loss_mask[p] != 0 -- masked target positions become LOSS_IGNORE_INDEX, which
// op_cross_entropy skips (no loss, no grad) and normalizes around. null = today's behavior.
float train_batch(const int* data, const std::size_t* starts, int batch, int T,
                  const int* lengths, const std::uint8_t* loss_mask,
                  const ScratchBindings* const* win_binds,
                  const SentinelBindings* const* win_sentinel,
                  const PersistentBindings* const* win_persist) {
    double total = 0.0;
    #pragma omp parallel num_threads(DEFAULT_THREADS)   // tuned worker count (<= MAX_WORKERS)
    {
        ensure_thread_built();
        std::ranges::fill(W->grad, 0.f);
        // Per-window targets buffer (only when masking): reused across this thread's windows, and
        // read by op_cross_entropy/backward within the SAME iteration before it is next resized.
        thread_local std::vector<int> masked_tgt;
        #pragma omp for reduction(+ : total) schedule(static)
        for (int b = 0; b < batch; ++b) {
            const int Tb = lengths ? lengths[b] : T;
            const int* tgt = data + starts[b] + 1;
            if (loss_mask) {
                masked_tgt.resize(static_cast<std::size_t>(Tb));
                for (int i = 0; i < Tb; ++i) {
                    const std::size_t p = starts[b] + static_cast<std::size_t>(i) + 1;
                    masked_tgt[static_cast<std::size_t>(i)] = loss_mask[p] ? data[p] : LOSS_IGNORE_INDEX;
                }
                tgt = masked_tgt.data();
            }
            // Install this window's binding views (any combination of the three) for its
            // forward+backward on this worker thread; cleared right after so nothing leaks to the next
            // window or beyond this call. All three setters write thread_local state, so per-window
            // per-worker installation is race-free by construction.
            if (win_binds)    set_scratch_bindings(win_binds[b]);
            if (win_sentinel) set_sentinel_bindings(win_sentinel[b]);
            if (win_persist)  set_persistent_bindings(win_persist[b]);
            graph_reset();
            Node* logits = g_model.forward(data + starts[b], Tb);
            Node* loss   = op_cross_entropy(logits, tgt);
            total += loss->data[0];
            backward(loss, 1.f / static_cast<float>(batch));
            if (win_binds)    set_scratch_bindings(nullptr);
            if (win_sentinel) set_sentinel_bindings(nullptr);
            if (win_persist)  set_persistent_bindings(nullptr);
        }
        // (implicit barrier above: every thread's grad slot is complete)
        const int nthreads = omp_get_num_threads();
        // OpenMP's `for` worksharing construct requires a SIGNED loop variable (unlike `simd`, which
        // MSVC accepts size_t for, just ignoring reduction clauses on it) -- ptrdiff_t matches size_t's
        // width so PARAM_FLOATS (a size_t) still fits, just signed to satisfy the spec.
        #pragma omp for schedule(static)
        for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(PARAM_FLOATS); ++i) {
            float s = 0.f;
            for (int t = 0; t < nthreads; ++t)
                s += g_workers[t]->grad[static_cast<std::size_t>(i)];
            g_param_grad[static_cast<std::size_t>(i)] = s;
        }
    }
    return static_cast<float>(total / batch);
}

// --- AdamW (optionally hybrid with Muon) -------------------------------------

AdamW::AdamW(float lr, bool use_muon) : lr_(lr), use_muon_(use_muon) {}

void AdamW::zero_grad() { ensure_thread_built(); std::ranges::fill(W->grad, 0.f); }

// One Muon-routed weight matrix's update: momentum EMA -> Nesterov lookahead -> Newton-Schulz
// orthogonalization (sub0/muon.hpp) -> fan-ratio scale -> decoupled weight decay -> apply. Reuses
// AdamW's own g_param_m arena as Muon's momentum buffer (g_param_vel goes UNUSED for these params,
// left at zero) so the checkpoint format needs no new field for this -- see the design note on
// AdamW in include/sub0/core.hpp. `gs` is the same global gradient-clip scale the AdamW path uses,
// applied here too so clipping stays uniform across the whole model regardless of routing.
static void muon_step_one(std::size_t off, int rows, int cols, float lr, float beta, float wd, float gs) {
    const std::size_t n = static_cast<std::size_t>(rows) * static_cast<std::size_t>(cols);
    std::vector<float> upd(n);
    for (std::size_t i = 0; i < n; ++i) {
        const float g = g_param_grad[off + i] * gs;
        float& m = g_param_m[off + i];
        m = beta * m + (1.f - beta) * g;             // momentum EMA (muon_update's momentum.lerp_)
        upd[i] = (1.f - beta) * g + beta * m;         // Nesterov lookahead (grad.lerp_(momentum, beta))
    }
    sub0::muon::newton_schulz5(upd.data(), rows, cols, upd.data(), 5);
    const float scale = sub0::muon::scale_factor(rows, cols);
    for (std::size_t i = 0; i < n; ++i) {
        float& p = g_param_data[off + i];
        p -= lr * wd * p;             // decoupled weight decay, same convention as the AdamW path
        p -= lr * scale * upd[i];
    }
}

void AdamW::step() {
    double sq = 0.0;
    #pragma omp simd reduction(+ : sq)
    for (size_t i = 0; i < PARAM_FLOATS; ++i) { double g = g_param_grad[i]; sq += g * g; }
    float norm = (float)std::sqrt(sq);
    float gs = (norm > clip_) ? clip_ / (norm + 1e-6f) : 1.f;

    ++t_;
    float bc1 = 1.f - std::pow(b1_, (float)t_);
    float bc2 = 1.f - std::pow(b2_, (float)t_);
    // PARAM_LAYOUT (layout.hpp) walks the SAME sequential param slots build_layout()'s mk_param()
    // calls created (that lock-step is this table's whole reason to exist), so it doubles here as
    // the shape+kind lookup W->views alone doesn't carry.
    //
    // Data-parallel across param TENSORS -- this was entirely single-threaded before (confirmed
    // empirically: a d448 CPU training run showed a repeating ~10s-busy / ~30s-near-idle-at-~1-core
    // cycle on EVERY step, not just the periodic eval). Each PARAM_LAYOUT entry owns a DISJOINT
    // [off, off+n) slice of every param/grad/moment array (that IS what "layout" means), so different
    // entries never touch the same memory -- safe to run concurrently with no synchronization.
    // muon_step_one's `upd` scratch buffer is a local std::vector per call, so concurrent calls for
    // different matrices don't share it either. newton_schulz5 (muon.hpp) only uses `#pragma omp simd`
    // internally, never `#pragma omp parallel` -- no nested-parallelism thread-explosion risk here.
    // schedule(dynamic): Muon-eligible entries (newton_schulz5, several matrix multiplies) cost far
    // more than plain-AdamW entries, so a static chunking would load-balance badly.
    const int n_layout = static_cast<int>(PARAM_LAYOUT.size());
    #pragma omp parallel for num_threads(DEFAULT_THREADS) schedule(dynamic)
    for (int pi = 0; pi < n_layout; ++pi) {
        const ParamDesc& pd = PARAM_LAYOUT[static_cast<std::size_t>(pi)];
        const bool muon_eligible = use_muon_ && is_muon_kind(pd.kind);   // shared set, layout.hpp
        if (muon_eligible) {
            muon_step_one(pd.off, pd.rows, pd.cols, muon_lr_, muon_beta_, wd_, gs);
            continue;
        }
        const float wd = pd.decay ? wd_ : 0.f;   // hoist the invariant branch so the loop vectorizes
        #pragma omp simd
        for (size_t i = pd.off; i < pd.off + pd.n(); ++i) {
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
