// backend.cpp — CPU compute backend for the engine core (part of libsub0_core).
//
// Implements the differentiable math: the autograd ops, reverse-mode backward, the
// batched forward pass, the data-parallel minibatch (train_batch) and the AdamW
// optimizer, plus the statically allocated parameter/activation/worker arenas. The
// backend-agnostic parts (tokenizer, serialization, sampler, config paths) live in
// engine_core.cpp. The seam is at the step level (forward / backward / train_batch /
// AdamW + the host/device parameter-sync hooks), so a GPU backend can replace this
// translation unit wholesale behind the same include/sub0/core.hpp API.
//
// The INCREMENTAL single-token (KV-cache) path -- Model::forward_one, its row kernels,
// the three decode-persistent caches and the kv_* entry points -- lives beside this file
// in decode.cpp, and the process/thread state both share (arenas, the Worker pool, `Model`
// itself) is declared in internal.hpp and defined here. That split is file organization
// only: the two translation units are one backend, and nothing outside src/backends/cpu/
// can see either private header.
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
#include "sub0/gdn_math.hpp"        // Gated DeltaNet Stage 1 forward math (sub0::gdn::forward/recurrence_step)
#include "sub0/gated_residual_math.hpp"  // Gated Residual Stage 1 forward math (sub0::gr::hc_norm/mix/gate/combine/tile)
#include "sub0/moe_math.hpp"         // Mixture of Experts Stage 1 forward math (sub0::moe::forward_row/forward)
#include "sub0/moe_quant.hpp"        // WP4e: quantized-resident routed experts + the dequant-on-demand pool
#include "sub0/qsa_math.hpp"         // QSA Stage 1 forward math (sub0::qsa::forward/indexer_*/attn_*)
#include "sub0/layout.hpp"
#include "sub0/muon.hpp"
#include "sub0/scratch_slots.hpp"   // content-derived scratch-slot embeddings (range + ScratchBindings + encoders)
#include "internal.hpp"          // backend-private shared detail: arenas, Worker/W, Layer, Model

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

// (The x86 FTZ/DAZ control-register helper this file used to define locally now lives in internal.hpp:
// decode.cpp's own compute threads must set the same MXCSR bits, and a thread that missed them would
// decode to different last bits than one that did not -- see set_flush_denormals' comment there.)

namespace sub0 {

// ============================================================================
//  Static storage
// ============================================================================
// Two kinds of thing live here. The `extern`-declared ones (g_moe_quant, g_workers, W, the three
// binding tables, g_qsa_rope, g_model) are declared in internal.hpp and defined HERE, once, so both
// CPU-backend translation units address the same instance -- see that header for the reasoning
// attached to each. Everything else on this page is `static` because only this TU needs it: the
// parameter arenas and the helpers that write a Worker (arena_alloc/mk_param/mk_node, and the `Mat`
// view) belong to the Node-graph path alone. decode.cpp allocates nothing.

// Shared across all worker threads: the weights (read-only during fwd/bwd), the optimizer moments
// (touched only by AdamW::step on the main thread), and the REDUCED gradient that AdamW::step consumes.
// HEAP-allocated (not static std::array): at a large config these are 4*PARAM_FLOATS floats -- e.g.
// d768 ~= 2.6 GB -- and as zero-init BSS they push the DLL's SizeOfImage past what the Windows loader
// will map, so the image fails to load with STATUS_INVALID_IMAGE_FORMAT (0xC000007B). This is the same
// reason internal.hpp's per-thread Worker arrays are heap-allocated. ensure_shared_params() allocates
// them once (zeroed) before any parameter node references them; unique_ptr<float[]> keeps [] and .get().
// See internal.hpp's FORWARD_ONLY for why a build that cannot train allocates only the first of them.
static std::unique_ptr<float[]> g_param_data;
static std::unique_ptr<float[]> g_param_grad;
static std::unique_ptr<float[]> g_param_m;
static std::unique_ptr<float[]> g_param_vel;
static std::once_flag           g_shared_params_once;
static void ensure_shared_params() {
    std::call_once(g_shared_params_once, [] {
        g_param_data = std::make_unique<float[]>(PARAM_FLOATS);   // value-initialized -> zeroed
        if constexpr (!FORWARD_ONLY) {
            g_param_grad = std::make_unique<float[]>(PARAM_FLOATS);
            g_param_m    = std::make_unique<float[]>(PARAM_FLOATS);
            g_param_vel  = std::make_unique<float[]>(PARAM_FLOATS);
        }
    });
}

// The one seam every training consumer of the three arenas comes through. Refusing HERE rather than
// letting a null pointer reach memcpy is this project's "put the refusal at the lowest callable seam"
// rule (memory: caps-bit-nothing-reads-is-not-a-guard): the backward pass already aborts, so reaching
// this is a bug in a caller that skipped it, and a null deref would name the wrong thing.
[[noreturn]] static void refuse_training_arena(const char* what) {
    std::println(stderr,
                 "fatal: {} does not exist in this build -- Gated Residual / MoE / QSA are CPU-forward-"
                 "only (backward_node aborts), so the gradient and AdamW-moment arenas are not "
                 "allocated at all ({} floats each would be {:.1f} GiB apiece here). Nothing but a "
                 "training path can want this pointer.",
                 what, PARAM_FLOATS,
                 static_cast<double>(PARAM_FLOATS) * 4.0 / (1024.0 * 1024.0 * 1024.0));
    std::abort();
}

moeq::Store g_moe_quant;

std::array<std::unique_ptr<Worker>, MAX_WORKERS> g_workers{};
thread_local Worker* W = nullptr;

thread_local const ScratchBindings* g_scratch_binds = nullptr;
thread_local const PersistentBindings* g_persistent_binds = nullptr;
thread_local const SentinelBindings* g_sentinel_binds = nullptr;
thread_local int   g_scratch_reinject_stride = 0;
thread_local float g_scratch_reinject_scale  = 1.0f;

// The Worker-writing helpers. `static`, and deliberately so: the activation arena and the node pool
// are the Node-graph path's own storage, and the Node-graph path is entirely in this file. forward_one
// (decode.cpp) runs one row through stack buffers and allocates nothing, so it has no business naming
// any of these -- keeping them private is what makes "decode allocates nothing" checkable by grep
// rather than by reading it.
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

static Node* mk_param(int r, int c, bool decay) {
    size_t n = (size_t)r * c, off = W->pused;
    W->pused += n;
    Node& nd = W->param_nodes[W->pcount];
    nd = Node{};
    nd.op = Op::Leaf; nd.rows = r; nd.cols = c;
    nd.data = std::span<float>(g_param_data.get() + off, n);    // shared weights
    // FORWARD_ONLY: no per-thread gradient accumulator exists, so a parameter leaf carries an EMPTY
    // grad span rather than a span into a buffer that was never allocated. Nothing in the forward path
    // reads a parameter's grad, and the backward path aborts before it could.
    if constexpr (FORWARD_ONLY) nd.grad = std::span<float>{};
    else nd.grad = std::span<float>(W->grad.data() + off, n);   // this thread's grad accumulator
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

// When a model was loaded with weights already in their final (ternary) form,
// the linear op must not re-quantize them (absmean re-quantization is not
// idempotent). Training keeps this false so the straight-through estimator runs.
static bool g_packed_inference = false;

// ============================================================================
//  Differentiable ops (forward builders)
// ============================================================================


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
// inv_freq[m] = ROPE_THETA^(-2m/ROTARY_DIM).
//
// PARTIAL ROTARY (WP4b blocker C): only each head's first ROTARY_DIM channels are rotated; channels
// [ROTARY_DIM, d) are copied through unchanged. ROTARY_DIM == D_HEAD (the default, --rotary-dim 0) is
// the full-width rotation every build before this axis existed performed, and at that setting the
// pass-through loop below has zero trips -- so the neutral path is bit-identical. Note the inverse
// frequency's denominator is ROTARY_DIM, not the head width: that matches the reference's own
// inv_freq, which is built over rotary_dim (see layout.hpp's ROTARY_DIM comment for the convention
// re-derivation, and for why the interleaved-vs-half-split pairing difference is orthogonal to this).
static Node* op_rope(Node* x, int H) {
    const int T = x->rows, C = x->cols, d = C / H;
    constexpr int rd = ROTARY_DIM, half = ROTARY_HALF;
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
                const float ang = (static_cast<float>(t) * ROPE_POS_SCALE) * std::pow(ROPE_THETA, -2.f * m / rd);
                const float cs = std::cos(ang), sn = std::sin(ang);
                const float x0 = xr[off + 2 * m], x1 = xr[off + 2 * m + 1];
                yr[off + 2 * m]     = x0 * cs - x1 * sn;
                yr[off + 2 * m + 1] = x0 * sn + x1 * cs;
            }
            // Un-rotated remainder: the partial-rotary tail, plus (at rd == d) the odd-d lone component
            // the previous full-width form already passed through as the identity.
            for (int j = 2 * half; j < d; ++j) yr[off + j] = xr[off + j];
        }
    }
    return y;
}

// --- Depth attention (see layout.hpp's USE_DEPTH_ATTN, docs/DEPTH_ATTENTION.md) ---------------------
//
// Op::DepthAttn consumes S+1 (K, V) pairs, where S grows with the execution index. A `Node` has a FIXED
// fanout (a/b/w/bias) and cannot express that, and the backward needs those exact nodes to accumulate
// its cross-execution dK/dV into -- so the variable-length part lives in a side table keyed by cache
// slot, and each node records how many slots existed when it ran (Node::depth_s).
//
// thread_local for the same reason the arena is: every worker thread owns its own graph. Storing NODES
// (not copies of their data) is what makes the backward possible at all. The lifetime this relies on is
// exactly the one op_embed's `ids` pointer already relies on -- backward runs on the same thread before
// the next graph_reset(), so these arena nodes are still live when it reads them. Model::forward()
// clears `n` at its top, so a second forward never mixes the first one's cache into itself.
//
// Sized LOOP_EXEC_COUNT rather than DEPTH_CACHE_MAX (which is <= it, and is 0 when depth attention is
// off) so the array is never zero-length and the indices below need no separate bound.
struct DepthCache {
    std::array<Node*, LOOP_EXEC_COUNT> k{}, v{};
    int n = 0;
    void push(Node* kn, Node* vn) { k[(size_t)n] = kn; v[(size_t)n] = vn; ++n; }
};
thread_local DepthCache g_depth;

// --- GDN Node-linkage side table (Stage 2; docs/GATED_DELTANET.md S6, resolving S4's deferred question)
//
// op_gdn (defined later, once `Layer` exists) takes `Layer&` directly rather than routing its 9 weight
// tensors through Node's generic a/b/w/bias fanout -- Stage 1 could get away with that because it had no
// backward to preserve node-graph linkage for (see op_gdn's own comment). Stage 2 needs backward_node's
// Op::GDN case to reach all 9 of a GDN layer's parameter Nodes (to write their gradients) plus the input
// node `a` (already carried via Node::a, unchanged from Stage 1) -- 10 pointers total, and Node's
// a/b/w/bias fanout only has 4 slots. This is the SAME wall docs/DEPTH_ATTENTION.md S5a hit (a Node
// cannot express more inputs than its fixed pointer fanout), so it gets the SAME fix already established
// here for exactly that situation (DepthCache, just above): a thread_local side table of full Node*
// bundles, keyed by a small int stored on the Node (Node::gdn_link, mirroring Node::depth_s's own
// pattern) -- populated once per op_gdn call (1:1 with GDN Node creation, unlike DepthCache's own
// variable-length "how many entries so far" bookkeeping, which this doesn't need). Declared here, ahead
// of backward_node (which needs it) and ahead of `Layer` (which op_gdn needs but this table does not --
// push() takes 9 plain Node* so this struct has no dependency on Layer's definition).
//
// Sized LOOP_EXEC_COUNT for the same reason DepthCache is: every execution that runs op_gdn gets its own
// slot, even though most entries alias the SAME underlying Layer (LoopSplit reruns the SAME middle
// layer, i.e. the SAME 9 parameter Nodes, several times) -- storing the bundle redundantly per execution
// is simpler than deduplicating, and correctness does not depend on deduplication since every
// parameter-gradient write is `+=`.
struct GdnLink {
    Node *in_qkv, *in_z, *in_b, *in_a, *conv, *a_log, *dt_bias, *norm, *out_proj;
};
struct GdnLinkCache {
    std::array<GdnLink, LOOP_EXEC_COUNT> links{};
    int n = 0;
    int push(Node* in_qkv, Node* in_z, Node* in_b, Node* in_a, Node* conv, Node* a_log, Node* dt_bias,
              Node* norm, Node* out_proj) {
        links[static_cast<std::size_t>(n)] = {in_qkv, in_z, in_b, in_a, conv, a_log, dt_bias, norm, out_proj};
        return n++;
    }
};
thread_local GdnLinkCache g_gdn_link;   // only ever populated/consulted when USE_GATED_DELTANET

// Gated DeltaNet's Stage 2 backward-time recompute scratch (docs/GATED_DELTANET.md S4's closing
// paragraph / gdn_math.hpp's own header comment on sub0::gdn::backward): a single, reused,
// thread_local, heap-backed buffer -- NOT an arena_alloc allocation, deliberately. backward_node walks
// the node pool in reverse ONE NODE AT A TIME, so at most one Op::GDN node's backward is ever in flight
// per thread; routing this through arena_alloc instead would require ACT_CAP headroom for as many
// simultaneous copies as there are GDN layers (arena_alloc never reclaims within one graph's lifetime),
// which would badly inflate the activation arena for a buffer that is, by construction, never live more
// than once at a time. Sized once, lazily, to gdn::bwd_scratch_floats(GDN_DIMS, SEQ_LEN) -- the worst
// case over every T <= SEQ_LEN a real forward() call could have produced -- the same lazy-lifetime
// pattern already established by decode.cpp's KVCache/GdnCache (a std::vector, not a raw thread_local array:
// Worker's own comment on why multi-MB thread_local statics are unsafe on Windows applies here too, and
// this buffer is comparably large at production dims).
struct GdnBwdScratch {
    std::vector<float> buf;
    float* ensure() {
        const std::size_t n = gdn::bwd_scratch_floats(GDN_DIMS, SEQ_LEN);
        if (buf.size() != n) buf.assign(n, 0.f);
        return buf.data();
    }
};
thread_local GdnBwdScratch g_gdn_bwd;

// Per (position t, KV head hd): build a softmax over the DEPTH axis from the query's affinity to each
// depth entry's key, and return that convex mixture of the entries' VALUES. K is untouched -- sequence
// attention then runs normally on (unchanged K, mixed V).
//
// Entries are the S cached ones plus this execution's own (k, v) as entry S, so S == 0 makes the
// softmax a single 1.0 and the op the exact identity on v (the first participating execution is a
// numeric no-op, by construction). The cached V is itself already MIXED -- the reference reassigns
// value_states before appending -- so the mixture is recursive across participating executions.
//
// Under GQA the query is reduced to KV groups by the MEAN over the GQA_GROUP query heads of the group
// (the reference's own reduction); at GQA_GROUP == 1 that is the identity. The scale is D_HEAD^(-1/2),
// as in sequence attention.
//
// NOTE on RoPE: the depth logit <q_t, k_d,t> compares vectors at the SAME position t, and RoPE applies
// the same rotation R_t to both, so <R_t q, R_t k> == <q, k>. The logits are therefore invariant to
// whether this runs before or after op_rope, and passing the post-RoPE tensors (as forward() does)
// costs nothing and keeps the op adjacent to op_attn. QK-norm is NOT position-common, so its effect
// does carry through -- deliberately, since it is a per-head rescaling of exactly these vectors.
static Node* op_depth_attn(Node* q, Node* k, Node* v, int H) {
    const int T = v->rows, Ckv = v->cols, d = Ckv / H;   // H == N_KV_HEADS, d == D_HEAD
    const int C = q->cols;                               // query rows are H * GQA_GROUP * d wide
    const int S = g_depth.n;
    const float scale = 1.f / std::sqrt((float)d);
    Node* out = mk_node(Op::DepthAttn, T, Ckv);
    out->a = q; out->b = k; out->bias = v; out->heads = H; out->depth_s = S;
    auto [P, Pg] = arena_alloc((size_t)H * T * (S + 1));
    out->scratch = P;
    auto Pidx = [T, S](int hd, int t, int dd) { return ((size_t)hd * T + t) * (S + 1) + dd; };
    for (int hd = 0; hd < H; ++hd) {
        const int off = hd * d;
        for (int t = 0; t < T; ++t) {
            std::array<float, D_HEAD> qb{};              // q-bar: the GQA mean-reduced query
            const float* __restrict qr = q->data.data() + (size_t)t * C;
            for (int g = 0; g < GQA_GROUP; ++g) {
                const float* __restrict qh = qr + (size_t)(hd * GQA_GROUP + g) * d;
                for (int a = 0; a < d; ++a) qb[a] += qh[a];
            }
            for (int a = 0; a < d; ++a) qb[a] *= 1.f / GQA_GROUP;
            std::array<float, LOOP_EXEC_COUNT + 1> lg{};
            float mx = -1e30f;
            for (int dd = 0; dd <= S; ++dd) {
                const Node* src = (dd < S) ? g_depth.k[(size_t)dd] : k;
                const float* __restrict kd = src->data.data() + (size_t)t * Ckv + off;
                float s = 0.f;
                #pragma omp simd reduction(+ : s)
                for (int a = 0; a < d; ++a) s += qb[a] * kd[a];
                s *= scale; lg[(size_t)dd] = s; mx = std::max(mx, s);
            }
            float Z = 0.f;
            if constexpr (FAST_MATH) for (int dd = 0; dd <= S; ++dd) { lg[(size_t)dd] = fast_exp(lg[(size_t)dd] - mx); Z += lg[(size_t)dd]; }
            else                     for (int dd = 0; dd <= S; ++dd) { lg[(size_t)dd] = std::exp(lg[(size_t)dd] - mx);  Z += lg[(size_t)dd]; }
            float* __restrict o = out->data.data() + (size_t)t * Ckv + off;
            for (int dd = 0; dd <= S; ++dd) {
                const float p = lg[(size_t)dd] / Z;
                P[Pidx(hd, t, dd)] = p;
                const Node* src = (dd < S) ? g_depth.v[(size_t)dd] : v;
                const float* __restrict vd = src->data.data() + (size_t)t * Ckv + off;
                for (int a = 0; a < d; ++a) o[a] += p * vd[a];
            }
        }
    }
    return out;
}

// Under GQA the K/V rows are D_KV wide (N_KV_HEADS heads) while Q/out stay D_MODEL wide (N_HEADS
// heads): query head h reads KV head h / GQA_GROUP. With N_KV_HEADS == N_HEADS this is GQA_GROUP == 1,
// off_kv == off and Ckv == C -- byte-identical to the pre-GQA path.
static Node* op_attn(Node* q, Node* k, Node* v, int H) {
    const int T = q->rows, C = q->cols, d = C / H;
    const int Ckv = k->cols;                       // D_KV; == C when not using GQA
    const float scale = 1.f / std::sqrt((float)d);
    Node* out = mk_node(Op::Attn, T, C);
    out->a = q; out->b = k; out->bias = v; out->heads = H;
    auto [P, Pg] = arena_alloc((size_t)H * T * T);
    out->scratch = P;
    auto Pidx = [T](int h, int i, int j) { return ((size_t)h * T + i) * T + j; };
    for (int h = 0; h < H; ++h) {
        int off    = h * d;                        // query/output head offset (stride C)
        int off_kv = (h / GQA_GROUP) * d;          // shared KV head offset (stride Ckv)
        for (int i = 0; i < T; ++i) {
            const float* __restrict qi = q->data.data() + (size_t)i * C + off;
            float* __restrict oi       = out->data.data() + (size_t)i * C + off;
            float mx = -1e30f;
            std::array<float, SEQ_LEN> sc{};
            for (int j = 0; j <= i; ++j) {
                const float* __restrict kj = k->data.data() + (size_t)j * Ckv + off_kv;
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
                const float* __restrict vj = v->data.data() + (size_t)j * Ckv + off_kv;
                for (int a = 0; a < d; ++a) oi[a] += p * vj[a];      // contiguous axpy
            }
        }
    }
    return out;
}

// op_gdn (Gated DeltaNet, Stage 1) is defined further below, in the whole-sublayer op section -- see
// that definition for the full comment. It sits apart from the ops here because it takes a `Layer&`
// (internal.hpp) rather than op_attn/op_depth_attn's plain Node* args, which is the same reason
// op_moe/op_qsa/op_gr_* are down there with it.

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
        const int T = x->rows, C = x->cols, H = n.heads, d = C / H;
        constexpr int rd = ROTARY_DIM, half = ROTARY_HALF;   // partial rotary -- mirrors op_rope exactly
        const float* __restrict gy = n.grad.data();
        float* __restrict gx       = x->grad.data();
        for (int t = 0; t < T; ++t) {
            const float* __restrict gyr = gy + (size_t)t * C;
            float* __restrict gxr       = gx + (size_t)t * C;
            for (int h = 0; h < H; ++h) {
                const int off = h * d;
                for (int m = 0; m < half; ++m) {
                    const float ang = (static_cast<float>(t) * ROPE_POS_SCALE) * std::pow(ROPE_THETA, -2.f * m / rd);
                    const float cs = std::cos(ang), sn = std::sin(ang);
                    const float g0 = gyr[off + 2 * m], g1 = gyr[off + 2 * m + 1];
                    gxr[off + 2 * m]     +=  g0 * cs + g1 * sn;
                    gxr[off + 2 * m + 1] += -g0 * sn + g1 * cs;
                }
                // The un-rotated remainder is the identity, so its gradient passes straight through.
                for (int j = 2 * half; j < d; ++j) gxr[off + j] += gyr[off + j];
            }
        }
        break;
    }
    case Op::Attn: {
        Node* q = n.a; Node* k = n.b; Node* v = n.bias;
        const int T = q->rows, C = q->cols, H = n.heads, d = C / H;
        const int Ckv = k->cols;                   // D_KV; == C when not using GQA
        const float scale = 1.f / std::sqrt((float)d);
        std::span<float> P = n.scratch;
        auto Pidx = [T](int h, int i, int j) { return ((size_t)h * T + i) * T + j; };
        // K/V (and their grads) are Ckv-wide, Q/out are C-wide -- distinct mdspan shapes under GQA.
        Mat ng = mat(n.grad, T, C), vg = mat(v->grad, T, Ckv), vd = mat(v->data, T, Ckv);
        Mat qg = mat(q->grad, T, C), kg = mat(k->grad, T, Ckv);
        Mat qd = mat(q->data, T, C), kd = mat(k->data, T, Ckv);
        for (int h = 0; h < H; ++h) {
            int off    = h * d;
            int off_kv = (h / GQA_GROUP) * d;
            // Under GQA the GQA_GROUP query heads sharing this KV head all accumulate into the SAME
            // kg/vg columns. That is already correct here because every write below is `+=` and the
            // h loop is serial -- no structural change was needed, but it IS the load-bearing reason
            // this backward stays correct (the CUDA path has the opposite default; see its own note).
            for (int i = 0; i < T; ++i) {
                std::array<float, SEQ_LEN> dP{};
                for (int j = 0; j <= i; ++j) {
                    float p = P[Pidx(h, i, j)], dp = 0.f;
                    #pragma omp simd reduction(+ : dp)
                    for (int a = 0; a < d; ++a) {
                        float dout = ng[i, off + a];
                        vg[j, off_kv + a] += p * dout;
                        dp += dout * vd[j, off_kv + a];
                    }
                    dP[j] = dp;
                }
                float dot = 0.f;
                #pragma omp simd reduction(+ : dot)
                for (int j = 0; j <= i; ++j) dot += P[Pidx(h, i, j)] * dP[j];
                for (int j = 0; j <= i; ++j) {
                    float ds = P[Pidx(h, i, j)] * (dP[j] - dot) * scale;
                    for (int a = 0; a < d; ++a) {
                        qg[i, off + a]     += ds * kd[j, off_kv + a];
                        kg[j, off_kv + a]  += ds * qd[i, off + a];
                    }
                }
            }
        }
        break;
    }
    case Op::DepthAttn: {
        // Adjoint of op_depth_attn. Derived in docs/DEPTH_ATTENTION.md 4; with g = dL/dv_out,
        //   dL/dv_d = p_d . g                                   <- CROSS-EXECUTION for d < S
        //   dL/dp_d = <g, v_d>
        //   dL/dl_d = p_d . (dL/dp_d - sum_e p_e . dL/dp_e)     softmax Jacobian, same shape as Op::Attn
        //   dL/dk_d = scale . dL/dl_d . q-bar                   <- CROSS-EXECUTION for d < S
        //   dL/dq-bar = scale . sum_d dL/dl_d . k_d, scattered back over the group as 1/G each
        //
        // The two CROSS-EXECUTION terms write into nodes from EARLIER executions. That is safe here
        // purely because backward() walks the pool in REVERSE: this node sits after every node it
        // reads, so an earlier execution's K/V node has already received every contribution by the
        // time the walk reaches it and propagates onward. Every write is `+=`, as everywhere on this
        // backend -- an earlier V node is read by several later depth mixes and must accumulate.
        if constexpr (USE_DEPTH_ATTN) {
            Node* q = n.a; Node* k = n.b; Node* v = n.bias;
            const int T = n.rows, Ckv = n.cols, H = n.heads, d = Ckv / H, S = n.depth_s;
            const int C = q->cols;
            const float scale = 1.f / std::sqrt((float)d);
            const float inv_g = 1.f / GQA_GROUP;
            std::span<float> P = n.scratch;
            auto Pidx = [T, S](int hd, int t, int dd) { return ((size_t)hd * T + t) * (S + 1) + dd; };
            for (int hd = 0; hd < H; ++hd) {
                const int off = hd * d;
                for (int t = 0; t < T; ++t) {
                    // q-bar is recomputed rather than kept in scratch: it is GQA_GROUP * D_HEAD adds
                    // against a [T, D_KV] node's worth of arena, and the forward's own expression is
                    // right here to compare against.
                    std::array<float, D_HEAD> qb{};
                    const float* __restrict qr = q->data.data() + (size_t)t * C;
                    for (int g = 0; g < GQA_GROUP; ++g) {
                        const float* __restrict qh = qr + (size_t)(hd * GQA_GROUP + g) * d;
                        for (int a = 0; a < d; ++a) qb[a] += qh[a];
                    }
                    for (int a = 0; a < d; ++a) qb[a] *= inv_g;

                    const float* __restrict go = n.grad.data() + (size_t)t * Ckv + off;
                    std::array<float, LOOP_EXEC_COUNT + 1> dP{};
                    for (int dd = 0; dd <= S; ++dd) {
                        Node* src = (dd < S) ? g_depth.v[(size_t)dd] : v;
                        const float p = P[Pidx(hd, t, dd)];
                        float* __restrict vg       = src->grad.data() + (size_t)t * Ckv + off;
                        const float* __restrict vd = src->data.data() + (size_t)t * Ckv + off;
                        float dp = 0.f;
                        #pragma omp simd reduction(+ : dp)
                        for (int a = 0; a < d; ++a) { vg[a] += p * go[a]; dp += go[a] * vd[a]; }
                        dP[(size_t)dd] = dp;
                    }
                    float dot = 0.f;
                    for (int dd = 0; dd <= S; ++dd) dot += P[Pidx(hd, t, dd)] * dP[(size_t)dd];
                    std::array<float, D_HEAD> dqb{};
                    for (int dd = 0; dd <= S; ++dd) {
                        const float dl = P[Pidx(hd, t, dd)] * (dP[(size_t)dd] - dot) * scale;
                        Node* src = (dd < S) ? g_depth.k[(size_t)dd] : k;
                        float* __restrict kg       = src->grad.data() + (size_t)t * Ckv + off;
                        const float* __restrict kd = src->data.data() + (size_t)t * Ckv + off;
                        for (int a = 0; a < d; ++a) { kg[a] += dl * qb[a]; dqb[a] += dl * kd[a]; }
                    }
                    float* __restrict qg = q->grad.data() + (size_t)t * C;
                    for (int g = 0; g < GQA_GROUP; ++g) {
                        float* __restrict qh = qg + (size_t)(hd * GQA_GROUP + g) * d;
                        for (int a = 0; a < d; ++a) qh[a] += inv_g * dqb[a];
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
    case Op::GDN: {
        // Stage 2 (docs/GATED_DELTANET.md S4/S6): recompute-based backward, delegated to
        // sub0::gdn::backward (include/sub0/gdn_math.hpp) -- see that function's own header comment for
        // the full derivation and its two-oracle verification. `n.gdn_link` indexes the thread_local
        // GdnLinkCache (this file, just above DepthCache) that op_gdn populated at forward time with
        // this node's Layer's 9 parameter Nodes -- see that struct's own comment for why this side
        // table exists (Node's a/b/w/bias fanout has only 4 slots; this op needs 10 pointers: the input
        // `n.a`, already carried the normal way, plus 9 weight tensors). Every one of the 10 gradient
        // writes below is `+=` into the SAME arena-owned grad spans every other op writes into, so this
        // needs no special-casing for LoopSplit (a repeated middle layer's several GDN executions all
        // point at the same 9 underlying parameter Nodes and correctly accumulate into them).
        const GdnLink& L = g_gdn_link.links[static_cast<std::size_t>(n.gdn_link)];
        gdn::backward(GDN_DIMS, n.rows, n.a->data.data(),
                      L.in_qkv->data.data(), L.in_z->data.data(), L.in_b->data.data(), L.in_a->data.data(),
                      L.conv->data.data(), L.dt_bias->data.data(), L.a_log->data.data(), L.norm->data.data(),
                      L.out_proj->data.data(),
                      n.grad.data(),
                      n.a->grad.data(),
                      L.in_qkv->grad.data(), L.in_z->grad.data(), L.in_b->grad.data(), L.in_a->grad.data(),
                      L.conv->grad.data(), L.dt_bias->grad.data(), L.a_log->grad.data(), L.norm->grad.data(),
                      L.out_proj->grad.data(),
                      g_gdn_bwd.ensure());
        break;
    }
    case Op::GrTile: case Op::GrMix: case Op::GrGate: case Op::GrCombine: {
        // Stage 1's own, deliberate scope boundary (docs/GATED_RESIDUAL.md S6): no backward exists for
        // these ops yet. A GR CPU forward binary compiles and runs fine (gen/eval/report's forward-only
        // uses, and this correctness-gate suite, all work) -- but reaching backward() on a graph that
        // contains one would otherwise silently leave this node's upstream gradient at the arena's zero
        // (graph_reset zeroes grads), which train_batch would then silently treat as "this layer
        // contributed nothing," training a DIFFERENT, wrong architecture with no diagnostic at all. Same
        // "guard at the lowest callable seam" refusal, moved to the exact same seam, as the Op::GDN
        // Stage 1 placeholder this mirrors (see project history, commit bac8bfd, before GDN's own
        // Stage 2 replaced it). Do not relax this until a future stage lands a real backward.
        std::println(stderr, "fatal: Gated Residual has no backward pass yet (Stage 1 is CPU forward "
                              "only, see docs/GATED_RESIDUAL.md) -- refusing to silently train a "
                              "different architecture than the one requested.");
        std::abort();
    }
    case Op::Moe: {
        // Stage 1's own, deliberate scope boundary (docs/MOE.md S6): no backward exists for this op yet
        // -- the exact same "guard at the lowest callable seam" refusal Op::GDN's own Stage 1 placeholder
        // established (commit bac8bfd, before GDN's own Stage 2 replaced it) and Op::GrTile/GrMix/GrGate/
        // GrCombine just above still use. docs/MOE.md S6 also names the real subtlety a future Stage 2
        // backward needs to get right (the router's own softmax gradient is NOT sparse to the selected
        // experts the way the expert FFN weights' gradient is) -- not implemented here, only documented.
        std::println(stderr, "fatal: Mixture of Experts has no backward pass yet (Stage 1 is CPU forward "
                              "only, see docs/MOE.md) -- refusing to silently train a different "
                              "architecture than the one requested.");
        std::abort();
    }
    case Op::Qsa: {
        // Stage 1's own, deliberate scope boundary (docs/QSA.md S6): no backward exists for this op yet
        // -- the exact same "guard at the lowest callable seam" refusal Op::GDN's Stage 1 placeholder
        // established and Op::GrTile/GrMix/GrGate/GrCombine/Op::Moe still use. docs/QSA.md S6 also names
        // the three real subtleties a future Stage 2 must not get wrong (no softmax couples the
        // unselected blocks -- the OPPOSITE of MoE's router; gradient does not flow through a mask, so a
        // naive implementation would train index_qk_proj not at all; and a block key's gradient must be
        // split 1/compress_ratio across the tokens pooled into it).
        std::println(stderr, "fatal: Qwen Sparse Attention has no backward pass yet (Stage 1 is CPU "
                              "forward only, see docs/QSA.md) -- refusing to silently train a different "
                              "architecture than the one requested.");
        std::abort();
    }
    }
}

// ============================================================================
//  Whole-sublayer ops, then Model: parameter layout + the batched forward pass
// ============================================================================
// Each op below is one WHOLE mixer/FFN sublayer behind a single Node, taking `Layer&` directly
// (see op_gdn's own comment for why). Model::forward_one's row-at-a-time counterparts to these
// live in decode.cpp; `Layer` and `Model` themselves are in internal.hpp, shared by both.

namespace {
// --- Gated DeltaNet (Stage 1: CPU forward only; docs/GATED_DELTANET.md, include/sub0/gdn_math.hpp) ---
//
// One call spans the WHOLE window (all T positions) for one GDN layer's one execution -- the recurrent
// state and the causal conv's history are TRAINING SCRATCH (docs/GATED_DELTANET.md S2): allocated fresh
// (zeroed by arena_alloc, matching every other scratch allocation in this file) at the top of this call
// and fully consumed within it, the direct analogue of op_attn's own `[H,T,T]` probability scratch. This
// is NOT the decode-persistent form (forward_one's GDN branch + GdnCache thread state ACROSS calls the
// way the KV-cache does) -- a batched forward() call always represents a fresh window starting at
// position 0, so a zero initial state is the correct value here, not a simplification of one.
//
// Backward (Stage 2, docs/GATED_DELTANET.md S6): backward_node's Op::GDN case delegates to
// sub0::gdn::backward, a recompute-based design per S4's finding (recomputes the state trajectory
// during backward from x + the retained weights, rather than retaining it from forward). See that
// case's own comment for the Node-linkage mechanism (`out->gdn_link` below, GdnLinkCache just above
// DepthCache) that makes this op's 9 weight tensors reachable from backward_node at all.
static Node* op_gdn(Node* a, Layer& L) {
    const int T = a->rows;
    Node* out = mk_node(Op::GDN, T, D_MODEL);
    out->a = a;   // the input node -- Stage 2's backward writes its gradient via n.a->grad, as usual
    out->gdn_link = g_gdn_link.push(L.gdn_in_qkv, L.gdn_in_z, L.gdn_in_b, L.gdn_in_a, L.gdn_conv,
                                     L.gdn_a_log, L.gdn_dt_bias, L.gdn_norm, L.gdn_out_proj);
    auto [state, state_g]     = arena_alloc(gdn::state_floats(GDN_DIMS));
    auto [convh, convh_g]     = arena_alloc(gdn::conv_hist_floats(GDN_DIMS));
    auto [scratch, scratch_g] = arena_alloc(gdn::scratch_floats(GDN_DIMS, T));
    // NOTE (found by Stage 3's independent CUDA re-derivation, not a Stage 3 change of its own): the
    // real gdn::forward() signature order is `(..., conv_w, dt_bias, a_log, norm_w, ...)` -- dt_bias
    // BEFORE a_log (see that function's own declaration, gdn_math.hpp). This call previously passed
    // L.gdn_a_log into the dt_bias slot and L.gdn_dt_bias into the a_log slot (swapped), even though
    // both g_gdn_link.push() just above and backward_node's Op::GDN case (gdn::backward's own call)
    // already had the two in the CORRECT, name-matching order -- so forward computed a genuinely
    // different function (dt_bias's real values exponentiated where A_log should be, and vice versa)
    // than backward differentiated through its own recompute. Neither the fixture test (which calls
    // gdn::forward directly, with its own correctly-ordered variables, never through op_gdn) nor the
    // whole-model finite-difference check (which perturbs a NAMED tensor and compares against that
    // SAME name's analytic grad, so a consistent relabeling of "which tensor plays which role" does
    // not by itself fail it) could catch this from the CPU side alone -- Stage 3's CUDA port, built
    // independently from the verified gdn_math.hpp reference rather than copied from this call site,
    // disagreed with the CPU engine's real forward output at a real mixed-layer model and exposed it.
    gdn::forward(GDN_DIMS, T, a->data.data(),
                 L.gdn_in_qkv->data.data(), L.gdn_in_z->data.data(),
                 L.gdn_in_b->data.data(), L.gdn_in_a->data.data(),
                 L.gdn_conv->data.data(), L.gdn_dt_bias->data.data(), L.gdn_a_log->data.data(),
                 L.gdn_norm->data.data(), L.gdn_out_proj->data.data(),
                 state.data(), convh.data(), out->data.data(), scratch.data());
    return out;
}

// --- Gated Residual (Stage 1: CPU forward only; docs/GATED_RESIDUAL.md, include/sub0/
// gated_residual_math.hpp) ---
//
// Four small ops, each fitting Node's native a/b/w/bias fanout with NO side table (docs/GATED_RESIDUAL.md
// S5: Stage 1 has no backward walking the node pool yet, mirroring op_gdn's own Stage 1 form before its
// Stage 2 needed GdnLinkCache). Each stores its wide-stream input on `out->a` (matching op_gdn's own
// `out->a = a`) -- enough for backward_node's abort-placeholder case (below) to at least name it.
//
// op_gr_mix/op_gr_gate each call gr::hc_norm() independently on their OWN scratch rather than sharing one
// precomputed buffer -- a deliberate Stage 1 simplification, docs/GATED_RESIDUAL.md S4c.
static Node* op_gr_tile(Node* h) {
    const int T = h->rows;
    Node* out = mk_node(Op::GrTile, T, HC_WIDE);
    out->a = h;
    gr::tile(GR_DIMS, T, h->data.data(), out->data.data());
    return out;
}

static Node* op_gr_mix(Node* wide, Node* norm_w, Node* down_w, Node* up_w) {
    const int T = wide->rows;
    Node* out = mk_node(Op::GrMix, T, D_MODEL);
    out->a = wide; out->b = norm_w; out->w = down_w; out->bias = up_w;
    auto [normed, normed_g] = arena_alloc(gr::normed_scratch_floats(GR_DIMS, T));
    gr::hc_norm(GR_DIMS, T, wide->data.data(), norm_w->data.data(), normed.data());
    auto [dscr, dscr_g] = arena_alloc(gr::mix_scratch_floats(GR_DIMS, T));
    gr::mix(GR_DIMS, T, normed.data(), down_w->data.data(), up_w->data.data(), out->data.data(), dscr.data());
    return out;
}

static Node* op_gr_gate(Node* wide, Node* norm_w, Node* block_inject_w) {
    const int T = wide->rows;
    Node* out = mk_node(Op::GrGate, T, HC_COUNT);
    out->a = wide; out->b = norm_w; out->w = block_inject_w;
    auto [normed, normed_g] = arena_alloc(gr::normed_scratch_floats(GR_DIMS, T));
    gr::hc_norm(GR_DIMS, T, wide->data.data(), norm_w->data.data(), normed.data());
    gr::gate(GR_DIMS, T, normed.data(), block_inject_w->data.data(), out->data.data());
    return out;
}

// The WRITE step (docs/GATED_RESIDUAL.md S1b): no weight tensors at all, a plain 3-activation combine
// (wide, mixer_out, injection weights) -- fits Node's native fanout trivially, no Layer& needed.
static Node* op_gr_combine(Node* wide, Node* mixer_out, Node* inj) {
    const int T = wide->rows;
    Node* out = mk_node(Op::GrCombine, T, HC_WIDE);
    out->a = wide; out->b = mixer_out; out->w = inj;
    gr::combine(GR_DIMS, T, wide->data.data(), mixer_out->data.data(), inj->data.data(), out->data.data());
    return out;
}

// --- Mixture of Experts (Stage 1: CPU forward only; docs/MOE.md, include/sub0/moe_math.hpp) ---
//
// ONE op does routing + all NUM_EXPERTS-worth of per-token expert selection + the shared expert
// internally, taking `Layer&` directly and reading its router/expert/shared-expert tensors off it --
// the exact `op_gdn(Node* a, Layer& L)` precedent (docs/MOE.md S4c/S5: top-k selection is host-side
// scalar code inside this op's own forward body, not a separate differentiable Node). `out->a = x`
// mirrors op_gdn's own single-input-node bookkeeping -- enough for backward_node's abort-placeholder
// case to at least name the right input node. No side table: Stage 1 has no backward walking the node
// pool yet (docs/MOE.md S6), so nothing needs to recover the router/expert/shared-expert tensors from a
// bare `Node*`.
static Node* op_moe(Node* x, Layer& L, int layer_index) {
    const int T = x->rows;
    Node* out = mk_node(Op::Moe, T, D_MODEL);
    out->a = x;
    if constexpr (USE_MOE_QUANT) W->moe_cache.allocate();
    auto [scratch, scratch_g] = arena_alloc(moe::scratch_floats(MOE_DIMS));
    // The batched path stays SERIAL and stays on this Worker's own 8-slot pool: it is already inside
    // train_batch's parallel team when training, and its cache has a real cross-row hit rate that
    // decode's does not (see MOE_DECODE_SLOTS in internal.hpp). B20 part 2 changed decode, not this.
    moe::forward_via(MOE_DIMS, T, x->data.data(), L.moe_router->data.data(),
                     [&](int e) { return moe_resolve(L, layer_index, e, W->moe_cache); },
                     L.moe_shared_gate->data.data(), L.moe_shared_up->data.data(),
                     L.moe_shared_down->data.data(), L.moe_shared_gate_proj->data.data(),
                     out->data.data(), scratch.data());
    return out;
}

// --- Qwen Sparse Attention (Stage 1: CPU forward only; docs/QSA.md, include/sub0/qsa_math.hpp) ---
//
// ONE op is the WHOLE mixer sublayer for a QSA layer -- the lightning indexer, the q/gate/k/v
// projections, the per-head (1+w) RMSNorms, the half-split partial RoPE, the per-query MASKED softmax
// attention, the sigmoid output gate and o_proj -- taking `Layer&` directly and reading its ten tensors
// off it, the exact `op_gdn(Node* a, Layer& L)` / `op_moe(Node* x, Layer& L)` precedent. It deliberately
// does NOT reuse op_attn/op_rope/op_qknorm: op_attn has no mask input at all, and QSA's norm/rotary
// conventions genuinely differ from this engine's (docs/QSA.md S2a). `out->a = a` mirrors op_gdn's own
// single-input-node bookkeeping -- enough for backward_node's abort-placeholder case to name the right
// input node. No side table: Stage 1 has no backward walking the node pool yet (docs/QSA.md S6).
static Node* op_qsa(Node* a, Layer& L) {
    const int T = a->rows;
    Node* out = mk_node(Op::Qsa, T, D_MODEL);
    out->a = a;
    auto [scratch, scratch_g] = arena_alloc(qsa::scratch_floats(QSA_DIMS_BUF, T));
    qsa::forward(QSA_DIMS, T, a->data.data(),
                 L.qsa_idx_qk->data.data(), L.qsa_idx_qnorm->data.data(), L.qsa_idx_knorm->data.data(),
                 L.qsa_q->data.data(), L.qsa_gate->data.data(), L.qsa_k->data.data(),
                 L.qsa_v->data.data(), L.qsa_qnorm->data.data(), L.qsa_knorm->data.data(),
                 L.qsa_o->data.data(),
                 g_qsa_rope.cos.data(), g_qsa_rope.sin.data(), qsa::RMS_EPS,
                 out->data.data(), scratch.data());
    return out;
}
}  // anonymous namespace

// QSA's cos/sin tables (declared in internal.hpp). Defined in this TU because op_qsa just above
// is their batched consumer; forward_one (decode.cpp) reads the same single instance. The extern
// declaration in the header is what gives this const object external linkage.
const QsaRopeTables g_qsa_rope;

// Model's own members. Declared in internal.hpp, where each one's contract comment lives.
void Model::build_layout() {
    W->pused = 0; W->pcount = 0;
    tok_emb = mk_param(VOCAB, D_MODEL, false);
    g_tok_emb_node = tok_emb;   // register THIS thread's token table for op_embed's binding gate
    // No position table under RoPE -- see layout.hpp's HAS_POS_EMB. Must stay in lock-step with
    // make_param_layout(): the order there IS the serialization order.
    if constexpr (HAS_POS_EMB) pos_emb = mk_param(SEQ_LEN, D_MODEL, false);
    else                       pos_emb = nullptr;
    for (int li = 0; li < N_LAYERS; ++li) {
        Layer& L = layers[static_cast<std::size_t>(li)];
        // WP4b blocker D: no Ln1/Ln2 under Gated Residual -- the real Qwen4ExpTextDecoderLayer
        // has no input_layernorm / post_attention_layernorm; GR's own grouped hc_norm is the
        // pre-block norm and the mixer reads mixed_input directly. MUST match make_param_layout().
        if constexpr (!USE_GATED_RESIDUAL) {
            L.ln1 = mk_param(1, D_MODEL, false);
            L.ln2 = mk_param(1, D_MODEL, false);
        }
        // Gated Residual (docs/GATED_RESIDUAL.md S3b/S4a): the attn_hyper_connection instance,
        // MUST match layout.hpp's own placement (right here, before the sub-block's own weights).
        if constexpr (USE_GATED_RESIDUAL) {
            L.gr_attn_norm   = mk_param(1, HC_WIDE, false);
            L.gr_attn_down   = mk_param(HC_WIDE, HC_LOWRANK, true);
            L.gr_attn_up     = mk_param(HC_LOWRANK, HC_WIDE, true);
            L.gr_attn_inject = mk_param(HC_WIDE, HC_COUNT, true);
        }
        // Order here MUST match layout.hpp's make_param_layout() exactly -- it IS the
        // serialization order (see that file's header comment), INCLUDING which of the two
        // branches below a given layer takes (GDN_SCHEDULE.full_attn[li] must agree with
        // make_param_layout()'s own read of it -- both read the same compile-time array).
        if (MIXER_SCHEDULE[static_cast<std::size_t>(li)] == LayerMixer::Qsa) {
            // QSA layer (docs/QSA.md S3b/S4) -- see layout.hpp's make_param_layout() for the same
            // shapes with the reasoning attached. Order MUST match it exactly.
            L.qsa_q     = mk_param(D_MODEL, sub0::D_Q, true);   // D_Q, not D_MODEL (blocker A)
            L.qsa_gate  = mk_param(D_MODEL, sub0::D_Q, true);
            L.qsa_k     = mk_param(D_MODEL, D_KV, true);
            L.qsa_v     = mk_param(D_MODEL, D_KV, true);
            L.qsa_o     = mk_param(sub0::D_Q, D_MODEL, true);   // no longer square
            L.qsa_qnorm = mk_param(1, D_HEAD, false);
            L.qsa_knorm = mk_param(1, D_HEAD, false);
            L.qsa_idx_qk    = mk_param(D_MODEL, QSA_IDX_QK_OUT, true);
            L.qsa_idx_qnorm = mk_param(1, QSA_INDEXER_HEAD_DIM, false);
            L.qsa_idx_knorm = mk_param(1, QSA_INDEXER_HEAD_DIM, false);
        } else if (GDN_SCHEDULE.full_attn[static_cast<std::size_t>(li)]) {
            // Wq [D_MODEL, D_Q] and Wo [D_Q, D_MODEL] -- not square once --head-dim makes
            // N_HEADS*D_HEAD independent of D_MODEL (WP4b blocker A). MUST match make_param_layout().
            L.Wq = mk_param(D_MODEL, sub0::D_Q, true);
            L.Wk = mk_param(D_MODEL, D_KV, true);      // GQA: narrower than Wq when N_KV_HEADS < N_HEADS
            L.Wv = mk_param(D_MODEL, D_KV, true);
            L.Wo = mk_param(sub0::D_Q, D_MODEL, true);
            if constexpr (USE_QK_NORM) {
                L.q_norm = mk_param(1, D_HEAD, false);
                L.k_norm = mk_param(1, D_HEAD, false);
            }
        } else {
            // Gated DeltaNet layer (docs/GATED_DELTANET.md S3a/S3b) -- see layout.hpp's
            // make_param_layout() for the same shapes with the reasoning attached.
            L.gdn_in_qkv   = mk_param(D_MODEL, GDN_CONV_DIM, true);
            L.gdn_in_z     = mk_param(D_MODEL, GDN_VALUE_DIM, true);
            L.gdn_in_b     = mk_param(D_MODEL, GDN_V_HEADS, true);
            L.gdn_in_a     = mk_param(D_MODEL, GDN_V_HEADS, true);
            L.gdn_conv     = mk_param(GDN_CONV_DIM, GDN_CONV_KERNEL, false);
            L.gdn_a_log    = mk_param(1, GDN_V_HEADS, false);
            L.gdn_dt_bias  = mk_param(1, GDN_V_HEADS, false);
            L.gdn_norm     = mk_param(1, GDN_V_HEAD_DIM, false);
            L.gdn_out_proj = mk_param(GDN_VALUE_DIM, D_MODEL, true);
        }
        // Gated Residual: the mlp_hyper_connection instance, right before the FFN's own weights.
        if constexpr (USE_GATED_RESIDUAL) {
            L.gr_mlp_norm   = mk_param(1, HC_WIDE, false);
            L.gr_mlp_down   = mk_param(HC_WIDE, HC_LOWRANK, true);
            L.gr_mlp_up     = mk_param(HC_LOWRANK, HC_WIDE, true);
            L.gr_mlp_inject = mk_param(HC_WIDE, HC_COUNT, true);
        }
        // Mixture of Experts (docs/MOE.md S3b/S4): REPLACES the FFN's own weights for EVERY layer
        // when on -- MUST match layout.hpp's make_param_layout() exactly (router, then NUM_EXPERTS
        // routed-expert SwiGLU triples, then the shared expert's own triple + gate projection).
        if constexpr (USE_MOE) {
            L.moe_router = mk_param(D_MODEL, NUM_EXPERTS, true);
            // WP4e: under MOE_QUANT_EXPERTS make_param_layout() emits no routed-expert tensors, so
            // this mirror must not claim any either -- mk_param walks the same cursor PARAM_LAYOUT
            // does, and one extra call here would silently shift every subsequent tensor's offset.
            if constexpr (!USE_MOE_QUANT) {
                for (int e = 0; e < NUM_EXPERTS; ++e) {
                    L.moe_gate[static_cast<std::size_t>(e)] = mk_param(D_MODEL, D_FF, true);
                    L.moe_up[static_cast<std::size_t>(e)]   = mk_param(D_MODEL, D_FF, true);
                    L.moe_down[static_cast<std::size_t>(e)] = mk_param(D_FF, D_MODEL, true);
                }
            }
            L.moe_shared_gate      = mk_param(D_MODEL, D_FF, true);
            L.moe_shared_up        = mk_param(D_MODEL, D_FF, true);
            L.moe_shared_down      = mk_param(D_FF, D_MODEL, true);
            L.moe_shared_gate_proj = mk_param(D_MODEL, 1, true);
        } else if constexpr (USE_GATED_FFN) {
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
    // Gated Residual's model-level exit collapse, right after the per-layer loop, before ln_f --
    // MUST match layout.hpp's own placement. No block_inject (use_combine=False, S1c).
    if constexpr (USE_GATED_RESIDUAL) {
        gr_top_norm = mk_param(1, HC_WIDE, false);
        gr_top_down = mk_param(HC_WIDE, HC_LOWRANK, true);
        gr_top_up   = mk_param(HC_LOWRANK, HC_WIDE, true);
    }
    // No LnF under Gated Residual: the real Qwen4ExpTextModel has no separate final RMSNorm -- the
    // GR exit collapse's own grouped hc_norm IS it (docs/GATED_RESIDUAL.md S1c). MUST match
    // make_param_layout(), which stops emitting the tensor under the same condition.
    if constexpr (!USE_GATED_RESIDUAL) ln_f = mk_param(1, D_MODEL, false);
    else                               ln_f = nullptr;
    if constexpr (!USE_TIED_EMBEDDINGS) {
        lm_head = mk_param(D_MODEL, VOCAB, true);
        lm_bias = mk_param(1, VOCAB, false);
    }
    // N-gram embeddings: appended at the very end -- MUST match layout.hpp's make_param_layout()
    // append order exactly (that file's header comment: "the order here IS the serialization order").
    if constexpr (NGRAM_EMBED) {
        for (int e = 0; e < NGRAM_NUM_EMBEDDERS; ++e)
            ngram_tab[static_cast<std::size_t>(e)] =
                mk_param(NGRAM_VOCAB_DIMS[static_cast<std::size_t>(e)], NGRAM_EMB_DIM, false);
        ngram_proj = mk_param(D_MODEL, D_MODEL, true);
        for (int e = 0; e < NGRAM_NUM_EMBEDDERS; ++e) {
            Node& v = ngram_wblock[static_cast<std::size_t>(e)];
            v = Node{};
            v.op = Op::Leaf; v.rows = NGRAM_EMB_DIM; v.cols = D_MODEL;
            const std::size_t off = static_cast<std::size_t>(e) * NGRAM_EMB_DIM * D_MODEL;
            const std::size_t n   = static_cast<std::size_t>(NGRAM_EMB_DIM) * D_MODEL;
            v.data = ngram_proj->data.subspan(off, n);   // ALIASES ngram_proj -- not a separate param
            // FORWARD_ONLY leaves every parameter leaf's grad span EMPTY (mk_param), so there is
            // nothing to slice a row-block out of -- the alias stays empty too.
            v.grad = ngram_proj->grad.empty() ? std::span<float>{} : ngram_proj->grad.subspan(off, n);
        }
    }
}

void Model::init_weights() {
    std::mt19937 rng(1234);
    auto randn = [&](Node* t, float std) {
        std::normal_distribution<float> nd(0.f, std);
        for (auto& x : t->data) x = nd(rng);
    };
    auto ones = [](Node* t) { std::fill(t->data.begin(), t->data.end(), 1.f); };
    randn(tok_emb, 0.02f);
    if constexpr (HAS_POS_EMB) randn(pos_emb, 0.02f);   // absent entirely under RoPE
    std::uniform_real_distribution<float> gdn_a_init(0.01f, 16.f);   // real model's own A_log init range, S1a
    for (int li = 0; li < N_LAYERS; ++li) {
        Layer& L = layers[static_cast<std::size_t>(li)];
        if constexpr (!USE_GATED_RESIDUAL) { ones(L.ln1); ones(L.ln2); }   // absent under GR (blocker D)
        // Gated Residual init: GrHcNorm is left at the arena's own zero -- the real model's own
        // `torch.zeros(dim)` convention (gain = 1 + w, so w=0 is the identity RMS-norm, S1a), NOT
        // this engine's usual ones()-initialized gain. down/up/block_inject are ordinary GEMM
        // weights with no special reference init documented, so this project's standard 0.02-std
        // normal is used, matching every other GEMM weight here.
        if constexpr (USE_GATED_RESIDUAL) {
            randn(L.gr_attn_down, 0.02f); randn(L.gr_attn_up, 0.02f); randn(L.gr_attn_inject, 0.02f);
            randn(L.gr_mlp_down, 0.02f);  randn(L.gr_mlp_up, 0.02f);  randn(L.gr_mlp_inject, 0.02f);
        }
        if (MIXER_SCHEDULE[static_cast<std::size_t>(li)] == LayerMixer::Qsa) {
            // QSA init: the projections get this project's standard 0.02-std normal (the real
            // module's __init__ documents no scheme beyond "a Linear layer"), same reasoning GDN's/
            // GR's/MoE's own projections use. The four RMSNorm gains are left at the arena's own
            // ZERO -- NOT ones() -- because Qwen4ExpTextRMSNorm's gain is (1 + w), so w == 0 IS the
            // identity norm (docs/QSA.md S1b; the same divergence GrHcNorm's own init already has).
            randn(L.qsa_q, 0.02f);  randn(L.qsa_gate, 0.02f); randn(L.qsa_k, 0.02f);
            randn(L.qsa_v, 0.02f);  randn(L.qsa_o, 0.02f);    randn(L.qsa_idx_qk, 0.02f);
        } else if (GDN_SCHEDULE.full_attn[static_cast<std::size_t>(li)]) {
            randn(L.Wq, 0.02f); randn(L.Wk, 0.02f); randn(L.Wv, 0.02f); randn(L.Wo, 0.02f);
            if constexpr (USE_QK_NORM) { ones(L.q_norm); ones(L.k_norm); }
        } else {
            // Gated DeltaNet init, matching the real model's own scheme (S1a's __init__, quoted in
            // docs/GATED_DELTANET.md): dt_bias = ones(num_v_heads); A_log = log(uniform(0.01,16));
            // norm (RMSNormGated) weight = ones(head_v_dim). The projections have no special
            // reference init documented beyond "a Linear layer", so this project's own standard
            // 0.02-std normal is used, matching every other GEMM weight here.
            randn(L.gdn_in_qkv, 0.02f); randn(L.gdn_in_z, 0.02f);
            randn(L.gdn_in_b, 0.02f);   randn(L.gdn_in_a, 0.02f);
            randn(L.gdn_conv, 0.02f);
            ones(L.gdn_dt_bias);
            for (auto& x : L.gdn_a_log->data) x = std::log(gdn_a_init(rng));
            ones(L.gdn_norm);
            randn(L.gdn_out_proj, 0.02f);
        }
        // Mixture of Experts init: no special reference init beyond "a Linear layer" for the router/
        // expert/shared-expert weights (S1a's __init__ shows only `nn.Parameter(torch.empty(...))`,
        // i.e. PyTorch's own default uninitialized-then-caller-inits convention, not a documented
        // scheme this project can port) -- this project's own standard 0.02-std normal is used,
        // matching every other GEMM weight here, same reasoning GDN's/GR's own projections use.
        if constexpr (USE_MOE) {
            randn(L.moe_router, 0.02f);
            // WP4e: a quantized-resident build has no routed-expert parameter nodes to initialize.
            // Its experts come from the S0Q1 sidecar or they do not exist at all -- there is no
            // random-init path for them, and that is deliberate: a MOE_QUANT_EXPERTS build is a
            // real-weight-import build by construction (docs/WP4_SCOPE.md WP4e), not a training one.
            if constexpr (!USE_MOE_QUANT) {
                for (int e = 0; e < NUM_EXPERTS; ++e) {
                    randn(L.moe_gate[static_cast<std::size_t>(e)], 0.02f);
                    randn(L.moe_up[static_cast<std::size_t>(e)], 0.02f);
                    randn(L.moe_down[static_cast<std::size_t>(e)], 0.02f);
                }
            }
            randn(L.moe_shared_gate, 0.02f); randn(L.moe_shared_up, 0.02f);
            randn(L.moe_shared_down, 0.02f); randn(L.moe_shared_gate_proj, 0.02f);
        } else {
            if constexpr (USE_GATED_FFN) randn(L.Wg, 0.02f);
            randn(L.W1, 0.02f); randn(L.W2, 0.02f);
        }
    }
    // Gated Residual's model-level exit collapse -- same init convention as the per-layer instances
    // above (GrHcNorm left at the arena's own zero, down/up randn(0.02)).
    if constexpr (USE_GATED_RESIDUAL) { randn(gr_top_down, 0.02f); randn(gr_top_up, 0.02f); }
    if constexpr (!USE_GATED_RESIDUAL) ones(ln_f);   // absent under GR (see build_layout)
    if constexpr (!USE_TIED_EMBEDDINGS) randn(lm_head, 0.02f);
    if constexpr (NGRAM_EMBED) {
        for (int e = 0; e < NGRAM_NUM_EMBEDDERS; ++e) randn(ngram_tab[static_cast<std::size_t>(e)], 0.02f);
        randn(ngram_proj, 0.02f);   // also initializes every ngram_wblock[e] view (same underlying data)
    }
}

Node* Model::forward(const int* ids, int T) {
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
    // N-gram embeddings: input-embedding injection only (Stage 1 scope -- see
    // docs/NGRAM_EMBEDDING.md's "deferred" section for the multi-layer NanbeigeNgramLayerFusion
    // this does NOT implement yet). ids_e[t] is looked up in table e; each embed output is
    // projected by that table's ROW-BLOCK of concat_proj and the results summed -- the block-matmul
    // identity for "concatenate then one linear" (see ngram_wblock's own comment above). Persistent
    // per-thread for the same reason pos_ids is: op_embed stores the pointer for backward.
    if constexpr (NGRAM_EMBED) {
        static thread_local int ngram_ids[NGRAM_TABLES_BUF][SEQ_LEN];
        // A persistent-slot id (>= VOCAB -- op_embed's `is_persistent_slot` gate) is not a real,
        // recurring vocabulary token: its embedding is dynamically COMPOSED per context, so it has
        // no fixed hash identity to be consistent with across occurrences, and its raw integer
        // value is UNBOUNDED (unlike an ordinary token id, already < VOCAB and so already
        // "in-distribution" for this hash). Feeding it through the polynomial hash unchanged would
        // make the ngram contribution at this position, AND at every later position that reads it
        // as context, depend on an essentially arbitrary large integer -- caught by
        // `persistent_slots_engine_tests.cpp`'s forward differential (a persistent-slot sequence
        // must match a plain-token reference sequence bit-for-bit outside the composed column,
        // which a raw-id hash breaks). Treat it as "no signal" (id 0), the same convention
        // `_shift_right_ignore_eos` already uses for "no real token here".
        const auto ngram_tok = [&](int t) { return ids[t] < VOCAB ? ids[t] : 0; };
        for (int e = 0; e < NGRAM_NUM_EMBEDDERS; ++e) {
            const int order      = NGRAM_ORDERS[static_cast<std::size_t>(e)];
            const int vocab_dim  = NGRAM_VOCAB_DIMS[static_cast<std::size_t>(e)];
            for (int t = 0; t < T; ++t) {
                // The reference's `_shift_right_ignore_eos`: a context token before the start of
                // the current document is ZERO (not "no term" -- literally token id 0). Every
                // training window lives inside exactly one document (window.hpp), so "before the
                // window's own start" IS "before this document's start" here -- a deliberate
                // simplification of the reference's mid-corpus doc-boundary scan (see
                // docs/NGRAM_EMBEDDING.md).
                std::int64_t acc = ngram_tok(t);
                for (int k = 2; k <= order; ++k) {
                    const int shift = k - 1;
                    const int tpos  = t - shift;
                    const int prev  = (tpos >= 0) ? ngram_tok(tpos) : 0;
                    acc += static_cast<std::int64_t>(prev) * NGRAM_VOCAB_MODS[static_cast<std::size_t>(e)][static_cast<std::size_t>(k - 2)];
                }
                ngram_ids[e][t] = static_cast<int>(((acc % vocab_dim) + vocab_dim) % vocab_dim);
            }
        }
        Node* ng = op_linear(op_embed(ngram_tab[0], ngram_ids[0], T), &ngram_wblock[0], nullptr, false);
        for (int e = 1; e < NGRAM_NUM_EMBEDDERS; ++e)
            ng = op_add(ng, op_linear(op_embed(ngram_tab[static_cast<std::size_t>(e)], ngram_ids[e], T),
                                       &ngram_wblock[static_cast<std::size_t>(e)], nullptr, false));
        h = op_add(h, ng);
    }
    // Gated Residual's model-level ENTRY tile (docs/GATED_RESIDUAL.md S1c): seed all HC_COUNT
    // streams identically by literal duplication, right before the per-layer loop -- everything
    // above (embed, absolute pos, n-gram injection) stays D_MODEL-wide and completely untouched.
    emit(h, "tok_embd");            // [T, D_MODEL] -- the last D_MODEL-wide state before GR widens it
    if constexpr (USE_GATED_RESIDUAL) h = op_gr_tile(h);
    if constexpr (USE_GATED_RESIDUAL) emit(h, "gr_tile");   // [T, HC_COUNT * D_MODEL]
    // Gated Residual READ/WRITE helpers (docs/GATED_RESIDUAL.md S2): wrap one sub-block's entry/exit
    // without touching the sub-block's own code -- ln1/ln2/the mixer/the FFN below are ALL literally
    // unchanged from the GR-off form. `wide` is the residual stream BEFORE this sub-block's read
    // step (== `h`, passed explicitly rather than captured, since gr_write needs the SAME pre-read
    // value and a caller passing it twice by name is clearer than relying on `h` not having moved).
    auto gr_read = [&](Node* wide, Node* norm_w, Node* down_w, Node* up_w, Node* inject_w, Node* ln,
                        Node** out_inj) -> Node* {
        if constexpr (USE_GATED_RESIDUAL) {
            // WP4b blocker D: the mixer reads GR's mixed_input DIRECTLY, un-normed. The real
            // Qwen4ExpTextDecoderLayer has no input_layernorm / post_attention_layernorm at all --
            // GR's own grouped hc_norm (already applied inside op_gr_mix, at the real model's
            // rms_norm_eps = 1e-6) IS the pre-block norm. Routing mixed_input through op_rmsnorm
            // applied a norm the real model does not, at the WRONG eps (1e-5), so no amount of
            // correct weight transplanting could have reproduced the real output.
            // `ln` is nullptr here -- Ln1/Ln2 are not emitted under GR (make_param_layout).
            (void)ln;
            Node* mixed = op_gr_mix(wide, norm_w, down_w, up_w);
            *out_inj = op_gr_gate(wide, norm_w, inject_w);
            return mixed;
        } else {
            return op_rmsnorm(wide, ln);
        }
    };
    auto gr_write = [&](Node* wide, Node* mixer_out, Node* inj) -> Node* {
        if constexpr (USE_GATED_RESIDUAL) return op_gr_combine(wide, mixer_out, inj);
        else                              return op_add(wide, mixer_out);
    };
    // LAYER_EXEC_ORDER, not `layers` directly: under LoopSplit the middle block's indices repeat,
    // re-running the SAME Layer (same parameter Nodes) several times. The backward needs no change
    // for that -- every CPU parameter-gradient write is `+=` (see backward_node's Op::Linear dW
    // block), which the codebase already relies on for tied embeddings. Off, this is 0..N_LAYERS-1.
    int exec_i = 0;
    g_depth.n = 0;   // see DepthCache: a second forward must not mix the first one's entries in
    g_gdn_link.n = 0;   // see GdnLinkCache: ditto, for a second forward's GDN Node-linkage entries
    for (const int li : LAYER_EXEC_ORDER) {
        Layer& L = layers[static_cast<std::size_t>(li)];
        const Node* const h_in = h;   // diagnostic only; arena nodes outlive the iteration
        Node* h_before_attn = h;   // Gated Residual's write step needs the PRE-read wide stream
        Node* gr_inj_attn = nullptr;
        Node* a = gr_read(h, L.gr_attn_norm, L.gr_attn_down, L.gr_attn_up, L.gr_attn_inject, L.ln1,
                           &gr_inj_attn);
        emit_layer(h, li, "res_in");   // [T, HC_WIDE] residual stream entering the layer
        emit_layer(a, li, "attn_in");   // [T, D_MODEL] what the mixer actually reads
        // GDN_SCHEDULE.full_attn[li] decides softmax attention vs. Gated DeltaNet for THIS layer
        // (per-LAYER, not per-execution -- a layer's weight identity fixes its type, so every
        // execution of a repeated LoopSplit middle layer inherits it automatically via LAYER_EXEC_
        // ORDER's existing indirection; see layout.hpp's GDN_SCHEDULE comment). At the only
        // buildable-before-Stage-1 setting (stride 0) full_attn is true for every layer, so this
        // `if constexpr` wrapper is the ONLY new code default builds compile at all -- the runtime
        // `if` inside it never has a false branch to take there.
        if constexpr (USE_GATED_DELTANET) {
            if (!GDN_SCHEDULE.full_attn[static_cast<std::size_t>(li)]) {
                // Gated DeltaNet: op_gdn is the WHOLE mixer sublayer (in_proj*, conv, recurrence,
                // gated-norm, out_proj all inside one op -- see its own comment) -- its output is
                // already D_MODEL-wide and ready to add straight into the residual stream, unlike
                // softmax attention's separate op_attn + Wo. No RoPE (GDN has none, S1b), no
                // QK-norm (GDN does its own internal, ungained L2-norm), no depth-attention (that
                // mechanism is defined in terms of softmax attention's own K/V, S1/S2 of
                // docs/DEPTH_ATTENTION.md -- out of scope for a GDN layer, and neither doc discusses
                // the combination, so this is a deliberate Stage-1 simplification, not an oversight).
                Node* gdn_out = op_gdn(a, L);
                emit_layer(gdn_out, li, "attn_out");   // [T, D_MODEL] the GDN mixer's own output
                h = gr_write(h_before_attn, gdn_out, gr_inj_attn);
                emit_layer(h, li, "attn_res");         // [T, HC_WIDE] stream after the attn combine
                Node* h_before_mlp = h;
                Node* gr_inj_mlp = nullptr;
                Node* f = gr_read(h, L.gr_mlp_norm, L.gr_mlp_down, L.gr_mlp_up, L.gr_mlp_inject,
                                   L.ln2, &gr_inj_mlp);
                emit_layer(f, li, "ffn_in");           // [T, D_MODEL] what MoE/the FFN reads
                if constexpr (USE_MOE) {
                    f = op_moe(f, L, li);
                } else if constexpr (USE_GATED_FFN) {
                    Node* gate_pre = op_linear(f, L.Wg, nullptr, q);
                    Node* up_pre   = op_linear(f, L.W1, nullptr, q);
                    f = op_linear(op_swiglu(gate_pre, up_pre), L.W2, nullptr, q);
                } else {
                    f = op_linear(op_gelu(op_linear(f, L.W1, L.b1, q)), L.W2, L.b2, q);
                }
                emit_layer(f, li, "ffn_out");          // [T, D_MODEL] the MoE/FFN block's output
                h = gr_write(h_before_mlp, f, gr_inj_mlp);
                emit_layer(h, li, "out");              // [T, HC_WIDE] stream leaving the layer
                if (pass_delta || pass_hnorm) {
                    const std::size_t nn = static_cast<std::size_t>(h->rows) * h->cols;
                    double d2 = 0.0, i2 = 0.0;
                    for (std::size_t j = 0; j < nn; ++j) {
                        const double before = h_in->data[j], after = h->data[j];
                        d2 += (after - before) * (after - before);
                        i2 += before * before;
                    }
                    if (pass_delta) pass_delta[exec_i] = static_cast<float>(std::sqrt(d2));
                    if (pass_hnorm) pass_hnorm[exec_i] = static_cast<float>(std::sqrt(i2));
                }
                ++exec_i;
                continue;
            }
        }
        // QSA (docs/QSA.md S2/S2a): a full-attention layer becomes a QSA layer when the mechanism is
        // on -- op_qsa IS the whole mixer sublayer (indexer, projections, norms, rotary, masked
        // attention, output gate and o_proj all inside one op), so its output goes straight into the
        // residual write the same way op_gdn's does, with no separate Wo. The `else` branch below is
        // the pre-QSA softmax-attention path, byte-for-byte unchanged; at USE_QSA == false it is the
        // ONLY branch compiled at all (L.Wq/Wk/Wv/Wo are never allocated on a QSA layer, so reaching
        // the else branch there would be a null dereference -- the exact bug the MoE stage hit by
        // wiring its own replacement into only one of the two FFN sites, docs/MOE.md S9).
        Node* mixer_out = nullptr;
        if constexpr (USE_QSA) {
        mixer_out = op_qsa(a, L);
        } else {
        Node* qn = op_linear(a, L.Wq, nullptr, q);
        Node* kn = op_linear(a, L.Wk, nullptr, q);
        Node* vn = op_linear(a, L.Wv, nullptr, q);
        // K carries N_KV_HEADS heads over a D_KV-wide row, Q carries N_HEADS over D_MODEL. Both
        // ops derive their per-head width as (cols / heads), so each still sees D_HEAD -- but the
        // HEAD COUNT must match the tensor, or the per-head split silently straddles head
        // boundaries (caught by the forward_one-vs-forward parity test, not by the gradient check).
        if constexpr (USE_QK_NORM) {
            qn = op_qknorm(qn, L.q_norm, N_HEADS);
            kn = op_qknorm(kn, L.k_norm, N_KV_HEADS);
        }
        if constexpr (POS_ENCODING == PosEncoding::Rope) {
            qn = op_rope(qn, N_HEADS);
            kn = op_rope(kn, N_KV_HEADS);
        }
        // Depth attention rewrites V only, then (on a participating execution) appends this
        // execution's key and its ALREADY-MIXED value to the depth cache -- the append order is
        // load-bearing, see op_depth_attn. The gate is on the EXECUTION index, so a looped middle
        // layer appends once per PASS and later passes attend over earlier ones: that is exactly
        // the cross-pass channel arm D exists to measure (docs/DEPTH_ATTENTION.md 2).
        if constexpr (USE_DEPTH_ATTN) {
            vn = op_depth_attn(qn, kn, vn, N_KV_HEADS);
            if (DEPTH_SCHEDULE.own[static_cast<std::size_t>(exec_i)] >= 0) g_depth.push(kn, vn);
        }
        Node* att = op_attn(qn, kn, vn, N_HEADS);
        mixer_out = op_linear(att, L.Wo, nullptr, q);
        }
        emit_layer(mixer_out, li, "attn_out");   // [T, D_MODEL] QSA's (or Wo's) own output
        h = gr_write(h_before_attn, mixer_out, gr_inj_attn);
        emit_layer(h, li, "attn_res");
        Node* h_before_mlp = h;
        Node* gr_inj_mlp = nullptr;
        Node* f = gr_read(h, L.gr_mlp_norm, L.gr_mlp_down, L.gr_mlp_up, L.gr_mlp_inject, L.ln2,
                           &gr_inj_mlp);
        emit_layer(f, li, "ffn_in");
        if constexpr (USE_MOE) {
            f = op_moe(f, L, li);
        } else if constexpr (USE_GATED_FFN) {
            Node* gate_pre = op_linear(f, L.Wg, nullptr, q);
            Node* up_pre   = op_linear(f, L.W1, nullptr, q);
            f = op_linear(op_swiglu(gate_pre, up_pre), L.W2, nullptr, q);
        } else {
            f = op_linear(op_gelu(op_linear(f, L.W1, L.b1, q)), L.W2, L.b2, q);
        }
        emit_layer(f, li, "ffn_out");
        h = gr_write(h_before_mlp, f, gr_inj_mlp);
        emit_layer(h, li, "out");
        if (pass_delta || pass_hnorm) {
            const std::size_t n = static_cast<std::size_t>(h->rows) * h->cols;
            double d2 = 0.0, i2 = 0.0;
            for (std::size_t j = 0; j < n; ++j) {
                const double before = h_in->data[j], after = h->data[j];
                d2 += (after - before) * (after - before);
                i2 += before * before;
            }
            if (pass_delta) pass_delta[exec_i] = static_cast<float>(std::sqrt(d2));
            if (pass_hnorm) pass_hnorm[exec_i] = static_cast<float>(std::sqrt(i2));
        }
        ++exec_i;
    }
    // Gated Residual's model-level EXIT collapse (docs/GATED_RESIDUAL.md S1c): use_combine=False,
    // so just op_gr_mix -- no gate/combine call, mirroring the real model's own use_combine=False
    // branch exactly (block_inject_weight genuinely does not exist for this instance).
    // ...and that collapse's own output goes STRAIGHT to the head: the real model applies no final
    // RMSNorm after it (no `Qwen4ExpTextModel.norm` exists -- confirmed absent from the real GGUF
    // and from the real safetensors index, whose only model-level norm is the exit instance's own
    // hc_norm). Feeding op_rmsnorm(h, ln_f) here normalized a second time, which WP4d measured as a
    // ~27%-of-scale shift in the real model's logits -- the LnF counterpart of blocker D's Ln1/Ln2.
    if constexpr (USE_GATED_RESIDUAL) h = op_gr_mix(h, gr_top_norm, gr_top_down, gr_top_up);
    else                              h = op_rmsnorm(h, ln_f);
    // [T, D_MODEL] -- what lm_head reads. Under GR this is the exit collapse's output with NO final
    // norm after it (the LnF removal, docs/WP4_SCOPE.md "WP4d's LnF item -- CLOSED"), which is the
    // single most useful tensor in this dump: it is the deepest point the engine reaches before the
    // vocabulary projection, and it is D_MODEL-wide, so it is directly comparable to llama.cpp's own
    // final hidden state without any hyper-connection-layout reconciliation.
    emit(h, "final_hidden");
    Node* logits;
    if constexpr (USE_TIED_EMBEDDINGS) logits = op_tied_head(h, tok_emb);
    else                               logits = op_linear(h, lm_head, lm_bias, false);  // head stays full precision
    emit(logits, "logits");
    return logits;
}

thread_local Model g_model;

// Declared in internal.hpp, where its contract comment lives.
void ensure_thread_built() {
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

// ============================================================================
//  Backend-private API implementation
// ============================================================================

namespace cpu_detail {

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
void set_scratch_reinject(int stride, float scale) { g_scratch_reinject_stride = stride; g_scratch_reinject_scale = scale; }

// Install (or clear) the persistent-slot table (declared near g_scratch_binds above). Unlike the
// ephemeral table (rebound per window/context), this is read-only and immutable for the process once
// set. The pointee must outlive every subsequent forward/forward_one/backward until cleared or replaced.
void set_persistent_bindings(const PersistentBindings* b) { g_persistent_binds = b; }

// Install (or clear) the sentinel-pair table for THIS thread (declared near g_scratch_binds above --
// same per-context/per-window lifetime as set_scratch_bindings, unlike the process-global persistent one).
void set_sentinel_bindings(const SentinelBindings* b) { g_sentinel_binds = b; }

// save_model / load_model live in engine_core.cpp: serialization is backend-agnostic
// and goes through params_ptr() + the host/device sync hooks.

// The HOST half of the memory plan, reported by `sub0llm memplan` alongside the device half. It lives
// here, in the backend that owns the allocations, because the per-thread cost is sizeof(Worker) -- a type
// private to the CPU backend (internal.hpp is not a public header, and nothing outside
// src/backends/cpu/ may include it), and deliberately so: a member added to Worker must not be able to
// escape this figure. Until this existed, `sub0llm memplan` reported a DEVICE plan unconditionally, which
// on a CPU-only build described memory that is never allocated (see docs/MEMORY_AUDIT.md 5).
void print_host_memplan() {
    constexpr double kMiB = 1024.0 * 1024.0;
    // FORWARD_ONLY builds allocate the weights and NOTHING else shared (see that constant), so the
    // multiplier is 1, not 4 -- reporting 4 here would over-state the footprint by 3x PARAM_FLOATS,
    // which at the real Qwen4-preview axes is 130 GiB of memory that is never asked for.
    constexpr int    shared_copies = FORWARD_ONLY ? 1 : 4;                  // data (+ grad + m + vel)
    constexpr double shared_mb = shared_copies * PARAM_FLOATS * sizeof(float) / kMiB;
    constexpr double worker_mb = sizeof(Worker) / kMiB;
    constexpr double wgrad_mb  = WORKER_GRAD_FLOATS * sizeof(float) / kMiB; // the per-worker gradient
    constexpr double arena_mb  = 2 * ACT_CAP * sizeof(float) / kMiB;        // act_data + act_grad
    constexpr int    workers   = COMPUTE_MODE == ComputeBackend::Gpu ? 1 : DEFAULT_THREADS;
    std::println("host (CPU) plan: shared {:.0f} MiB + {} x worker {:.0f} MiB = {:.0f} MiB",
                 shared_mb, workers, worker_mb, shared_mb + workers * worker_mb);
    if constexpr (FORWARD_ONLY)
        std::println("  shared: params only ({} floats x 1) -- no grad/m/vel: this build cannot train",
                     PARAM_FLOATS);
    else
        std::println("  shared: params + grad + m + vel ({} floats x 4)", PARAM_FLOATS);
    std::println("  worker: gradient {:.0f} MiB + activation arenas {:.0f} MiB + graph nodes {:.0f} MiB",
                 wgrad_mb, arena_mb, worker_mb - wgrad_mb - arena_mb);
    if constexpr (COMPUTE_MODE == ComputeBackend::Gpu)
        std::println("  (GPU build: one worker slot is touched -- CPU/Hybrid fans out to {})", DEFAULT_THREADS);
    else
        std::println("  scales with DEFAULT_THREADS={}; the arenas are sized by SEQ_LEN, NOT by batch --"
                     " CPU parallelises over WINDOWS, so batch costs worker slots, not arena bytes", DEFAULT_THREADS);
}

void print_config() {
    // Host footprint, reported as it is actually paid: a SHARED parameter set plus one whole Worker per
    // compute thread. This used to print `2 * ACT_CAP * sizeof(float)` as "acts", which was wrong twice
    // over -- it counted ONE worker's two activation arenas (real cost scales with DEFAULT_THREADS) and it
    // omitted the per-worker PARAM_FLOATS gradient accumulator entirely, which at production dims EXCEEDS
    // the arenas it was standing in for. sizeof(Worker) is used rather than a term-by-term sum so a member
    // added to Worker cannot silently escape the figure -- the CPU analogue of the device side's
    // measured-vs-predicted footprint check (see docs/MEMORY_AUDIT.md 5).
    // Workers are lazily heap-allocated, so only the slots a run actually TOUCHES cost anything: a GPU
    // run drives the engine from one thread (slot 0) while CPU and Hybrid fan out to DEFAULT_THREADS.
    // Reporting the full count unconditionally would replace an under-report with an over-report.
    constexpr double kMB = 1e6;
    // x1, not x4, when this build cannot train -- see FORWARD_ONLY and print_host_memplan's own note.
    constexpr double shared_mb = (FORWARD_ONLY ? 1 : 4) * PARAM_FLOATS * sizeof(float) / kMB;
    constexpr double worker_mb = sizeof(Worker) / kMB;                     // grad + arenas + nodes + views
    constexpr int    workers   = COMPUTE_MODE == ComputeBackend::Gpu ? 1 : DEFAULT_THREADS;
    std::println("model: d={} L={} H={} ff={} seq={} vocab={}{} | params: {:.2f}M | "
                 "heap mem: shared {:.1f}MB + {}x worker {:.1f}MB = {:.1f}MB | math: {}",
                 D_MODEL, N_LAYERS, N_HEADS, D_FF, SEQ_LEN, VOCAB,
                 USE_TERNARY ? " (ternary)" : "", PARAM_FLOATS / 1e6,
                 shared_mb, workers, worker_mb, shared_mb + workers * worker_mb,
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
// WP4e (docs/WP4_SCOPE.md WP4e): open `<model>.moeq`, the quantized-resident routed-expert sidecar, and
// check it belongs beside THIS blob. Declared in core.hpp; called only by load_model, right after the
// blob's own header/fingerprint checks pass, so the two files are accepted or refused together.
//
// The pairing check is `model_param_floats`, which is the same job ModelHeader::param_floats already
// does for the blob and for exactly the same reason: a sidecar written for a different build has the
// right magic and the right shape fields and is still the wrong weights.
bool load_moe_quant_sidecar(const char* model_path) {
    if constexpr (!USE_MOE_QUANT) {
        (void)model_path;
        return true;    // no sidecar exists, and none is wanted -- every build today takes this branch
    } else {
        const std::string path = std::string(model_path) + ".moeq";
        std::string err;
        if (!g_moe_quant.open(path, err)) {
            std::println(stderr,
                         "error: this build keeps its routed experts quantized-resident "
                         "(MOE_QUANT_EXPERTS), so it needs the S0Q1 sidecar beside the model: {}",
                         err);
            return false;
        }
        const moeq::Header& h = g_moe_quant.header();
        if (h.n_layers != N_LAYERS || h.num_experts != NUM_EXPERTS || h.d_model != D_MODEL ||
            h.d_ff != D_FF || h.model_param_floats != PARAM_FLOATS) {
            std::println(stderr,
                         "error: {} does not belong to this model: sidecar says layers {} experts {} "
                         "d_model {} d_ff {} param_floats {}; this build is {} / {} / {} / {} / {}",
                         path, h.n_layers, h.num_experts, h.d_model, h.d_ff, h.model_param_floats,
                         N_LAYERS, NUM_EXPERTS, D_MODEL, D_FF, PARAM_FLOATS);
            return false;
        }
        std::println("routed experts: {} tensors resident in their native GGUF encoding, {} bytes "
                     "({:.2f} GiB) -- vs {:.2f} GiB the same experts would occupy as f32; resolve pool "
                     "{} slots x {} floats ({:.1f} MiB)",
                     h.n_tensors, h.data_bytes,
                     static_cast<double>(h.data_bytes) / (1024.0 * 1024.0 * 1024.0),
                     static_cast<double>(h.n_tensors) * MOE_EXPERT_SLOT_FLOATS * 4.0 /
                         (1024.0 * 1024.0 * 1024.0),
                     MOE_RESOLVE_SLOTS, MOE_EXPERT_SLOT_FLOATS,
                     static_cast<double>(MoeExpertCache::pool_bytes()) / (1024.0 * 1024.0));
        return true;
    }
}

float*      params_ptr()       { ensure_shared_params(); return g_param_data.get(); }
// reduced grad the optimizer reads / the two AdamW moments -- all three absent under FORWARD_ONLY.
float*      grad_ptr()         { ensure_shared_params(); if constexpr (FORWARD_ONLY) refuse_training_arena("the parameter-gradient arena"); else return g_param_grad.get(); }
float*      adam_m_ptr()       { ensure_shared_params(); if constexpr (FORWARD_ONLY) refuse_training_arena("the AdamW first-moment arena"); else return g_param_m.get(); }
float*      adam_v_ptr()       { ensure_shared_params(); if constexpr (FORWARD_ONLY) refuse_training_arena("the AdamW second-moment arena"); else return g_param_vel.get(); }

// CPU backend: parameters already live in host memory, so the host/device sync hooks
// are no-ops. A device backend overrides these to copy the params_ptr()/adam_*_ptr()
// staging buffers across the PCIe boundary around serialization (see core.hpp).
void sync_params_to_host()   {}
void sync_params_to_device() {}

// The runtime BPE tokenizer, the logits sampler (sample_token) and model
// serialization (save_model / load_model) are backend-agnostic and live in
// engine_core.cpp.

// ensure_thread_built() matches forward() below and decode.cpp's forward_one(): a caller running on a thread that has
// never touched the engine before (e.g. a freshly-spawned OpenMP worker) must not dereference a null
// thread_local W. Idempotent -- cheap to call even when W is already built.
void graph_reset() { ensure_thread_built(); W->pool_used = 0; W->act_used = 0; }

Node* forward(const int* ids, int T) { ensure_thread_built(); return g_model.forward(ids, T); }
// Per-execution residual-stream diagnostic -- see Model::pass_delta. Both outputs are
// [LOOP_EXEC_COUNT] and either may be null. Runs ONE forward over the given window; the caller owns
// graph_reset() around it, exactly like a plain forward().
void loop_pass_stats(const int* ids, int T, float* out_delta, float* out_hnorm) {
    ensure_thread_built();
    g_model.pass_delta = out_delta;
    g_model.pass_hnorm = out_hnorm;
    (void)g_model.forward(ids, T);
    g_model.pass_delta = nullptr;   // disarm: every other forward() must stay on the untouched path
    g_model.pass_hnorm = nullptr;
}
// WP4f's named-intermediate capture -- see core.hpp. Armed for exactly ONE forward and disarmed after,
// the same discipline loop_pass_stats above uses, so no other forward() can ever be on this path. The
// caller owns graph_reset() around it exactly like a plain forward(); the returned node is the ordinary
// logits node, so a caller can use this INSTEAD of forward() rather than paying for a second pass.
Node* forward_capture(const int* ids, int T, HiddenSink sink, void* ctx) {
    ensure_thread_built();
    g_model.cap_sink = sink;
    g_model.cap_ctx  = ctx;
    Node* out = g_model.forward(ids, T);
    g_model.cap_sink = nullptr;
    g_model.cap_ctx  = nullptr;
    return out;
}
Node* cross_entropy(Node* logits, const int* targets) { return op_cross_entropy(logits, targets); }

void backward(Node* loss, float seed) {
    loss->grad[0] = seed;
    for (Node& n : W->pool | std::views::take(W->pool_used) | std::views::reverse) backward_node(n);
}

// Single-window reduction: publish this thread's accumulator as the shared gradient
// the optimizer consumes. (train_batch does the parallel multi-thread reduction.)
void reduce_gradients() {
    // FORWARD_ONLY: neither side of this copy exists (see that constant). Refuse at the seam rather
    // than copying one float into a null pointer.
    if constexpr (FORWARD_ONLY) refuse_training_arena("the parameter-gradient arena");
    else std::ranges::copy(W->grad, g_param_grad.get());
}

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
            // Qualified explicitly: these arguments' own types live in namespace sub0, so
            // unqualified lookup here would let ADL pull in the SUB0_API sub0:: declarations
            // (core.hpp) alongside these cpu_detail:: ones and make every call ambiguous.
            if (win_binds)    cpu_detail::set_scratch_bindings(win_binds[b]);
            if (win_sentinel) cpu_detail::set_sentinel_bindings(win_sentinel[b]);
            if (win_persist)  cpu_detail::set_persistent_bindings(win_persist[b]);
            graph_reset();
            Node* logits = g_model.forward(data + starts[b], Tb);
            Node* loss   = op_cross_entropy(logits, tgt);
            total += loss->data[0];
            cpu_detail::backward(loss, 1.f / static_cast<float>(batch));
            if (win_binds)    cpu_detail::set_scratch_bindings(nullptr);
            if (win_sentinel) cpu_detail::set_sentinel_bindings(nullptr);
            if (win_persist)  cpu_detail::set_persistent_bindings(nullptr);
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

}  // namespace cpu_detail

// --- AdamW (optionally hybrid with Muon) -------------------------------------

// Sized from this binary's immutable layout, like the shared parameter arenas. Keep one buffer per
// optimizer-team slot, not per OS thread: a later OpenMP team can reuse it without lazy allocation.
static constexpr std::size_t MUON_SCRATCH_FLOATS =
    MUON_MAX_MN + muon::scratch_floats(MUON_MAX_MN, MUON_MAX_MM);
static std::array<std::unique_ptr<float[]>, DEFAULT_THREADS> g_muon_scratch{};

AdamW::AdamW(float lr, bool use_muon) : lr_(lr), use_muon_(use_muon) {
    if constexpr (!FORWARD_ONLY) {
        if (use_muon_)
            for (auto& scratch : g_muon_scratch)
                if (!scratch) scratch = std::make_unique<float[]>(MUON_SCRATCH_FLOATS);
    }
}

void AdamW::zero_grad() { ensure_thread_built(); std::ranges::fill(W->grad, 0.f); }

// One Muon-routed weight matrix's update: momentum EMA -> Nesterov lookahead -> Newton-Schulz
// orthogonalization (sub0/muon.hpp) -> fan-ratio scale -> decoupled weight decay -> apply. Reuses
// AdamW's own g_param_m arena as Muon's momentum buffer (g_param_vel goes UNUSED for these params,
// left at zero) so the checkpoint format needs no new field for this -- see the design note on
// AdamW in include/sub0/core.hpp. `gs` is the same global gradient-clip scale the AdamW path uses,
// applied here too so clipping stays uniform across the whole model regardless of routing.
static void muon_step_one(std::size_t off, int rows, int cols, float lr, float beta, float wd, float gs) {
    const std::size_t n = static_cast<std::size_t>(rows) * static_cast<std::size_t>(cols);
    float* const upd = g_muon_scratch[static_cast<std::size_t>(omp_get_thread_num())].get();
    for (std::size_t i = 0; i < n; ++i) {
        const float g = g_param_grad[off + i] * gs;
        float& m = g_param_m[off + i];
        m = beta * m + (1.f - beta) * g;             // momentum EMA (muon_update's momentum.lerp_)
        upd[i] = (1.f - beta) * g + beta * m;         // Nesterov lookahead (grad.lerp_(momentum, beta))
    }
    sub0::muon::newton_schulz5(upd, rows, cols, upd,
        std::span<float>(upd + MUON_MAX_MN, MUON_SCRATCH_FLOATS - MUON_MAX_MN), 5);
    const float scale = sub0::muon::scale_factor(rows, cols);
    for (std::size_t i = 0; i < n; ++i) {
        float& p = g_param_data[off + i];
        p -= lr * wd * p;             // decoupled weight decay, same convention as the AdamW path
        p -= lr * scale * upd[i];
    }
}

void AdamW::step() {
    // FORWARD_ONLY: the gradient and both moment arenas were never allocated (see that constant).
    if constexpr (FORWARD_ONLY) refuse_training_arena("the AdamW optimizer state");
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
    // muon_step_one's scratch is private to each team slot and prepared by the constructor, so
    // dynamically assigned matrices never share scratch. newton_schulz5 only uses `#pragma omp simd`
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
