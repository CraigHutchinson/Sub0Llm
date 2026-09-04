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
#include "sub0/gdn_math.hpp"        // Gated DeltaNet Stage 1 forward math (sub0::gdn::forward/recurrence_step)
#include "sub0/gated_residual_math.hpp"  // Gated Residual Stage 1 forward math (sub0::gr::hc_norm/mix/gate/combine/tile)
#include "sub0/moe_math.hpp"         // Mixture of Experts Stage 1 forward math (sub0::moe::forward_row/forward)
#include "sub0/qsa_math.hpp"         // QSA Stage 1 forward math (sub0::qsa::forward/indexer_*/attn_*)
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
    // N-gram embeddings run ONCE per forward (input-embedding injection only, not per execution):
    // NGRAM_NUM_EMBEDDERS op_embed nodes ([T, NGRAM_EMB_DIM] each, summing to [T, D_MODEL]) + that many
    // op_linear nodes ([T, D_MODEL] each) + that many op_add nodes (the accumulator chain plus the
    // final residual add) -- see forward()'s ngram block in backend_cpu.cpp.
    // Gated Residual (Stage 1): the model-level entry tile ([T, HC_WIDE], once) and exit collapse
    // (op_gr_mix only -- [T,D_MODEL] output + its [T,HC_WIDE]+[T,HC_LOWRANK] scratch, once). The two
    // PER-SUB-BLOCK-WRAP instances (attn-wrapping, mlp-wrapping, once per EXECUTION) are costed in
    // `per` below, not here. Zero when GR is off.
    size_t base = 3 * T * C + (NGRAM_EMBED ? (size_t)(1 + 2 * NGRAM_NUM_EMBEDDERS) * T * C : 0)
                + (USE_GATED_RESIDUAL ? (size_t)T * HC_WIDE
                                        + T * C + T * HC_WIDE + T * HC_LOWRANK
                                      : 0);
    // FFN activation nodes: plain (W1-out, gelu-out) = 2*T*F; gated (gate-out, up-out, swiglu-out)
    // = 3*T*F (one extra T*F node -- see op_swiglu in the forward pass).
    // WP4b blocker A: every attention-side node budgeted at T*C above (the Q projection, its QK-norm
    // and RoPE copies, and op_attn's output) is really D_Q = N_HEADS*D_HEAD wide, which is no longer
    // necessarily D_MODEL. Budget the EXCESS explicitly rather than re-spelling the terms, so the
    // expression is exactly 0 -- hence ACT_CAP is bit-identical -- at the derived head width, and
    // generously over-provisioned (12 nodes' worth, more than the ~5 that are really D_Q-sided) when a
    // --head-dim build widens them. Under-sizing this arena is a silent overwrite of a live node.
    const size_t DQ_EXCESS = sub0::D_Q > (int)C ? (size_t)(sub0::D_Q - (int)C) : 0;
    size_t per  = 10 * T * C + 12 * T * DQ_EXCESS + (USE_GATED_FFN ? 3 : 2) * T * F + H * T * T + 2 * T
                + (USE_TERNARY ? (size_t)4 * C * C + (USE_GATED_FFN ? 3 : 2) * C * F : 0)
                + (POS_ENCODING == PosEncoding::Rope ? (size_t)2 * T * C : 0)   // op_rope(q), op_rope(k)
                + (USE_QK_NORM ? (size_t)2 * T * C + 2 * T * H : 0)   // op_qknorm(q), op_qknorm(k) + rinv scratch
                // op_depth_attn: one [T, D_KV] mixed-V node, plus its depth-softmax scratch. The
                // scratch is [N_KV_HEADS, T, S+1] where S is the cache depth THIS execution saw, so
                // DEPTH_CACHE_MAX + 1 is the worst case (the last execution, which sees every entry).
                + (USE_DEPTH_ATTN ? (size_t)T * D_KV
                                  + (size_t)N_KV_HEADS * T * (DEPTH_CACHE_MAX + 1) : 0)
                // op_gdn (Stage 1): a GDN layer does not ALSO pay rope/qknorm/depth-attn/op_attn's own
                // scratch above, but this formula sums a flat "per" cost assuming every execution could
                // be EITHER kind -- a safe, simple over-provision (this stays exact rather than tracking
                // GDN_SCHEDULE's per-execution mix here) -- so this adds GDN's OWN scratch on top: the
                // op's [T,D_MODEL] output node plus its recurrent state / conv history / gdn_math
                // internal scratch (state/conv_hist are technically call-scoped training-scratch here,
                // not persistent, but still arena-allocated once per op_gdn call -- see that op's
                // comment). Zero when GDN is off.
                + (USE_GATED_DELTANET ? (size_t)T * C + gdn::state_floats(GDN_DIMS)
                                        + gdn::conv_hist_floats(GDN_DIMS) + gdn::scratch_floats(GDN_DIMS, T)
                                      : 0)
                // Gated Residual (Stage 1): TWO per-execution sub-block wraps (attn-wrapping,
                // mlp-wrapping), each costing op_gr_mix's [T,D_MODEL] output + its
                // [T,HC_WIDE]+[T,HC_LOWRANK] scratch, op_gr_gate's [T,HC_COUNT] output + its
                // [T,HC_WIDE] scratch, and op_gr_combine's [T,HC_WIDE] output -- see
                // gated_residual_math.hpp's own *_scratch_floats(). Zero when GR is off.
                + (USE_GATED_RESIDUAL ? (size_t)2 * T * (C + 3 * HC_WIDE + HC_LOWRANK + HC_COUNT) : 0)
                // op_moe (Stage 1): same over-provisioning idiom as op_gdn's own term above -- every
                // execution is budgeted as if it could be MoE, added ON TOP of the plain/gated FFN term
                // already in `per` rather than replacing it (MoE genuinely does replace the FFN computed
                // at runtime, but this stays a simple, safe upper bound rather than tracking that). The
                // op's own [T,D_MODEL] output node plus moe_math.hpp's own scratch_floats() (NOT scaled
                // by T -- see that file's own header comment on row-independence). Zero when MoE is off.
                + (USE_MOE ? (size_t)T * C + moe::scratch_floats(MOE_DIMS) : 0)
                // op_qsa (Stage 1): same over-provisioning idiom as op_gdn's own term above -- every
                // execution is budgeted as if it could be QSA, ON TOP of the softmax-attention terms
                // already in `per` rather than replacing them (QSA genuinely does replace them at
                // runtime, but this stays a simple, safe upper bound). The op's own [T,D_MODEL] output
                // node plus qsa_math.hpp's own scratch_floats(). Zero when QSA is off.
                + (USE_QSA ? (size_t)T * C + qsa::scratch_floats(QSA_DIMS_BUF, T) : 0);
    size_t fin  = T * C + 2 * T * V + 64;
    // LOOP_EXEC_COUNT, not N_LAYERS: a LoopSplit middle block re-executed R times allocates its
    // activation nodes R times (the weights are shared; the activations are not).
    return base + (size_t)LOOP_EXEC_COUNT * per + fin;
}
constexpr size_t ACT_CAP   = calc_act_cap() * 3 / 2 + 8192;
// The gated FFN adds one extra per-layer node (gate-linear + up-linear + swiglu vs. W1-linear +
// gelu) -- a generous flat per-layer headroom either way, not a tight count. QK-norm adds 2 more
// per-layer nodes (op_qknorm(q), op_qknorm(k)).
// Depth attention adds exactly one node per EXECUTION (the mixed-V node) -- every execution runs the
// op, whether or not it also appends to the cache. Under-sizing this array is a silent overwrite of a
// live node, not an allocation failure, so it has to be counted rather than absorbed by the headroom.
// N-gram embeddings add 3 nodes per embedder ONCE (op_embed + op_linear + op_add each), not per
// execution -- see calc_act_cap()'s matching comment.
// Gated DeltaNet adds exactly one node per EXECUTION too (op_gdn's single output node), same
// over-provisioning reasoning as calc_act_cap()'s own GDN term above -- every execution is budgeted as
// if it could be GDN, rather than tracking GDN_SCHEDULE's actual per-execution mix here.
// Gated Residual adds 6 nodes per EXECUTION (op_gr_mix+op_gr_gate+op_gr_combine, x2 sub-block wraps)
// plus 2 more ONCE (the model-level entry op_gr_tile and exit op_gr_mix) -- see calc_act_cap()'s
// matching comment for the per-instance shape.
constexpr size_t MAX_NODES = 16 + (USE_GATED_FFN ? 18 : 16) * (size_t)LOOP_EXEC_COUNT
                                + (USE_QK_NORM ? 2 * (size_t)LOOP_EXEC_COUNT : 0)
                                + (USE_DEPTH_ATTN ? (size_t)LOOP_EXEC_COUNT : 0)
                                + (USE_GATED_DELTANET ? (size_t)LOOP_EXEC_COUNT : 0)
                                + (USE_GATED_RESIDUAL ? 6 * (size_t)LOOP_EXEC_COUNT + 2 : 0)
                                + (NGRAM_EMBED ? 3 * (size_t)NGRAM_NUM_EMBEDDERS : 0)
                                // op_moe adds exactly one node per EXECUTION (its single output node),
                                // same over-provisioning reasoning as GDN's own term above.
                                + (USE_MOE ? (size_t)LOOP_EXEC_COUNT : 0)
                                // op_qsa adds exactly one node per EXECUTION (its single output node),
                                // same over-provisioning reasoning as GDN's/MoE's own terms above.
                                + (USE_QSA ? (size_t)LOOP_EXEC_COUNT : 0);

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
//
// WP4d -- THE TRAINING ARENAS ARE NOT ALLOCATED IN A BUILD THAT PROVABLY CANNOT TRAIN. There are four
// of these, each PARAM_FLOATS long, plus a FIFTH per Worker (its own gradient accumulator, below). At
// this project's own training scales that is a few GB and nobody notices. At the real Qwen4-preview
// axes PARAM_FLOATS is 11,647,617,440 -- 43.4 GiB apiece -- so the eager form asks for ~217 GiB before
// a single token is embedded, on a 63 GiB machine. A forward pass at the real axes could not start.
//
// The saving grace is that they are DEAD, not merely large, in exactly those builds: Gated Residual,
// Mixture of Experts and Qwen Sparse Attention each abort() in backward_node (Stage 1 is CPU-forward-
// only, see those three sites), so a build with any of them on cannot reach one gradient write. This
// is a DERIVED compile-time fact, not a new user knob (AGENTS.md S8): it reads the same three USE_*
// constants the refusals themselves are written against, so it cannot drift out of agreement with them.
//
// Gated DeltaNet is deliberately NOT in this set. Its Stage 2 CPU backward is real and gradient-checked,
// so a GDN-only build still trains and still allocates everything it always did -- which is also why
// every existing build in this repo is bit-identical under this change: FORWARD_ONLY is false unless
// one of the three forward-only mechanisms is on, and no default build turns any of them on.
constexpr bool FORWARD_ONLY = USE_GATED_RESIDUAL || USE_MOE || USE_QSA;
// Sized 1 (not 0 -- a zero-length std::array has no data()) when the gradient path is dead. Only the
// Worker's array can be conditionally sized like this; the shared three are heap handles that simply
// stay null, and every accessor that would hand one out refuses loudly instead (see grad_ptr below).
constexpr size_t WORKER_GRAD_FLOATS = FORWARD_ONLY ? 1 : PARAM_FLOATS;
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
    std::array<float, WORKER_GRAD_FLOATS> grad{};  // gradient accumulator (this slot); 1 float when
                                                    // FORWARD_ONLY -- see that constant's comment
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

// Periodic packed-content re-injection -- SPIKE, see core.hpp's set_scratch_reinject doc comment for the
// full design. stride=0 (default) leaves forward_one byte-for-byte unchanged; stride>0 re-adds a bound
// scratch slot's own packed vector (computed once at layer 0) into its hidden state every `stride`
// layers. EVAL-ONLY (forward_one), never consulted by the training graph.
thread_local int   g_scratch_reinject_stride = 0;
thread_local float g_scratch_reinject_scale  = 1.0f;

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
// pattern already established by KVCache/GdnCache below (a std::vector, not a raw thread_local array:
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

// op_gdn (Gated DeltaNet, Stage 1) is defined further below, right after the `Layer` struct it needs --
// see that definition for the full comment. It cannot live here: it takes a `Layer&`, and `Layer` is
// not declared until the "Model (internal)" section, unlike op_attn/op_depth_attn's plain Node* args.

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
    // Indexed by EXECUTION, not layer: under LoopSplit a repeated middle layer runs several times per
    // token and each execution attends over its own K/V history, so it needs its own slot. Equal to
    // N_LAYERS when looping is off.
    std::vector<float> k, v;                                    // [LOOP_EXEC_COUNT][SEQ_LEN][D_KV], flat
    void reset() {
        const size_t n = static_cast<size_t>(LOOP_EXEC_COUNT) * SEQ_LEN * D_KV;
        if (k.size() != n) { k.assign(n, 0.f); v.assign(n, 0.f); }
    }
    // Rows are D_KV wide, not D_MODEL: under GQA the cache holds N_KV_HEADS heads, which is the whole
    // point (the cache shrinks by N_HEADS/N_KV_HEADS). Identical to D_MODEL when not using GQA.
    float* krow(int l, int pos) { return k.data() + ((static_cast<size_t>(l) * SEQ_LEN) + pos) * D_KV; }
    float* vrow(int l, int pos) { return v.data() + ((static_cast<size_t>(l) * SEQ_LEN) + pos) * D_KV; }
};
thread_local KVCache g_kv;                                      // gen is single-threaded; lazily sized

// Gated DeltaNet's decode-persistent state (Stage 1; docs/GATED_DELTANET.md S2's "decode" bullet) --
// structurally NOT a bigger/smaller KVCache: it does not grow with position, it is READ AND WRITTEN IN
// PLACE by every forward_one call (an accumulator, not a per-position row store), and per-execution
// (LOOP_EXEC_COUNT slots) for the identical LoopSplit reason KVCache is. Because it is an accumulator,
// reset() must UNCONDITIONALLY re-zero it every time (unlike KVCache's reset(), which only assigns on a
// SIZE change -- safe there because every row it will ever READ was already WRITTEN earlier in the SAME
// generation; a GDN state left over from a PREVIOUS generation would otherwise silently leak into a new
// one, since nothing about starting a new generation naturally overwrites an accumulator's stale value).
struct GdnCache {
    std::vector<float> state, conv_hist;   // [LOOP_EXEC_COUNT][state/conv_hist floats-per-exec], flat
    void reset() {
        state.assign(static_cast<size_t>(LOOP_EXEC_COUNT) * gdn::state_floats(GDN_DIMS), 0.f);
        conv_hist.assign(static_cast<size_t>(LOOP_EXEC_COUNT) * gdn::conv_hist_floats(GDN_DIMS), 0.f);
    }
    float* state_of(int e) { return state.data() + static_cast<size_t>(e) * gdn::state_floats(GDN_DIMS); }
    float* conv_of(int e)  { return conv_hist.data() + static_cast<size_t>(e) * gdn::conv_hist_floats(GDN_DIMS); }
};
thread_local GdnCache g_gdn_cache;   // only ever populated/consulted when USE_GATED_DELTANET

// QSA's decode-persistent indexer key cache (Stage 1; docs/QSA.md S6's own note that this is QSA's ONLY
// cross-call state). Structurally a KVCache, not a GdnCache: it IS a per-position row store that grows
// with position (the indexer must be able to pool a block out of ANY earlier token's raw key), and it is
// per-EXECUTION for the identical LoopSplit reason KVCache is. Rows are the RAW, unnormed, unrotated
// indexer keys -- k_layernorm and RoPE happen later, on the POOLED block key, at the block's own start
// position (docs/QSA.md S1a). reset() follows KVCache's own assign-on-size-change rule, safe for the same
// reason: every row ever READ was WRITTEN earlier in the SAME generation.
//
// It ALSO carries the POOLED block-key cache (docs/QSA.md S11). A block's pooled+k_layernorm'd+RoPE'd
// key is a function of the block alone -- its fixed start position and the compress_ratio raw keys in
// it -- never of the querying position, so recomputing it on every decode step was O(T^2/ratio) work
// across a generation for an O(T/ratio) quantity. `n_cached[e]` is how many leading blocks of slot e
// are valid; qsa::indexer_select_row extends it by the (at most one) block that completed this step.
// UNLIKE raw_k, `n_cached` MUST be zeroed on every reset, not only when the size changes: a stale
// non-zero count would make a fresh generation trust the PREVIOUS generation's block keys. (raw_k keeps
// the assign-on-size-change rule, which is still safe for the same reason as before -- every row read
// was written earlier in the same generation -- and block_k needs no clearing at all, since only its
// first n_cached[e] entries are ever read.)
struct QsaCache {
    std::vector<float> raw_k;   // [LOOP_EXEC_COUNT][SEQ_LEN][QSA_IDX_HEAD_DIM_BUF], flat
    std::vector<float> block_k; // [LOOP_EXEC_COUNT][block_key_cache_floats(QSA_DIMS_BUF, SEQ_LEN)], flat
    std::vector<int>   n_cached;// [LOOP_EXEC_COUNT] -- valid leading block count per execution slot
    static constexpr size_t BLOCK_K_PER_EXEC = qsa::block_key_cache_floats(QSA_DIMS_BUF, SEQ_LEN);
    void reset() {
        const size_t n = static_cast<size_t>(LOOP_EXEC_COUNT) * SEQ_LEN * QSA_IDX_HEAD_DIM_BUF;
        if (raw_k.size() != n) raw_k.assign(n, 0.f);
        const size_t bn = static_cast<size_t>(LOOP_EXEC_COUNT) * BLOCK_K_PER_EXEC;
        if (block_k.size() != bn) block_k.assign(bn, 0.f);
        n_cached.assign(static_cast<size_t>(LOOP_EXEC_COUNT), 0);   // unconditional -- see above
    }
    float* base(int e) {
        return raw_k.data() + static_cast<size_t>(e) * SEQ_LEN * QSA_IDX_HEAD_DIM_BUF;
    }
    float* block_base(int e) { return block_k.data() + static_cast<size_t>(e) * BLOCK_K_PER_EXEC; }
    int*   n_cached_of(int e) { return n_cached.data() + e; }
};
thread_local QsaCache g_qsa_cache;   // only ever populated/consulted when USE_QSA

// QSA's cos/sin tables. Built ONCE at static-init (never inside a forward -- AGENTS.md S1), because
// qsa_math.hpp deliberately takes cos/sin as caller-supplied data rather than deriving them from
// ROPE_THETA: the real model's convention is HALF-SPLIT over a rotary PREFIX, while this engine's own
// op_rope is INTERLEAVED-pair and full-width (docs/QSA.md S1b/S2a). Feeding the real convention here is
// what makes a QSA layer numerically the real model's layer, and taking the tables as data is what lets
// the fixture test feed the real model's own PARTIAL-rotary cos/sin unchanged.
// Layout matches HF's own `emb = cat(freqs, freqs)`: cos[pos][m] == cos[pos][m + rotary_dim/2].
// rotary_dim is ROTARY_DIM -- a real axis since WP4b blocker C (--rotary-dim, layout.hpp's own
// ROTARY_DIM comment); it was pinned to D_HEAD before that, which is still what --rotary-dim 0 gives.
struct QsaRopeTables {
    std::vector<float> cos, sin;   // [SEQ_LEN][ROTARY_DIM]
    QsaRopeTables() {
        const size_t n = static_cast<size_t>(SEQ_LEN) * ROTARY_DIM;
        cos.assign(n, 1.f); sin.assign(n, 0.f);
        if constexpr (USE_QSA) {
            constexpr int rd = ROTARY_DIM, half = ROTARY_HALF;
            for (int p = 0; p < SEQ_LEN; ++p) {
                for (int m = 0; m < half; ++m) {
                    const float ang = static_cast<float>(p) * ROPE_POS_SCALE *
                                      std::pow(ROPE_THETA, -2.f * static_cast<float>(m) / static_cast<float>(rd));
                    const size_t base = static_cast<size_t>(p) * rd;
                    cos[base + m] = cos[base + m + half] = std::cos(ang);
                    sin[base + m] = sin[base + m + half] = std::sin(ang);
                }
            }
        }
    }
};
static const QsaRopeTables g_qsa_rope;

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
    const int d = C / H;
    constexpr int rd = ROTARY_DIM, half = ROTARY_HALF;   // partial rotary -- mirrors op_rope; the
                                                          // un-rotated remainder needs no work in place
    for (int h = 0; h < H; ++h) {
        const int off = h * d;
        for (int m = 0; m < half; ++m) {
            const float ang = (static_cast<float>(pos) * ROPE_POS_SCALE) * std::pow(ROPE_THETA, -2.f * m / rd);
            const float cs = std::cos(ang), sn = std::sin(ang);
            const float x0 = x[off + 2 * m], x1 = x[off + 2 * m + 1];
            x[off + 2 * m]     = x0 * cs - x1 * sn;
            x[off + 2 * m + 1] = x0 * sn + x1 * cs;
        }
    }
}
// Depth attention for ONE position: forward_one's counterpart to op_depth_attn, which it must agree
// with numerically (the forward_one-vs-forward parity test is the gate). `dk`/`dv` are this token's
// depth cache, `n` its live entry count, and `v` is REWRITTEN IN PLACE with the mixture -- via a temp,
// since v is also entry `n` of the mix. Cheap because the softmax is over depth alone: a single
// position needs no sequence history, so decode needs no extra cache beyond the current token's rows.
static inline void depth_mix_row(const float* __restrict q, const float* __restrict k,
                                 float* __restrict v, const float* dk, const float* dv, int n) {
    constexpr int d = D_HEAD;
    const float scale = 1.f / std::sqrt((float)d);
    float mixed[D_KV];
    for (int hd = 0; hd < N_KV_HEADS; ++hd) {
        const int off = hd * d;
        float qb[D_HEAD] = {};
        for (int g = 0; g < GQA_GROUP; ++g) {
            const float* __restrict qh = q + (size_t)(hd * GQA_GROUP + g) * d;
            for (int a = 0; a < d; ++a) qb[a] += qh[a];
        }
        for (int a = 0; a < d; ++a) qb[a] *= 1.f / GQA_GROUP;
        float lg[DEPTH_CACHE_MAX + 1];
        float mx = -1e30f;
        for (int dd = 0; dd <= n; ++dd) {
            const float* __restrict kd = (dd < n) ? dk + (size_t)dd * D_KV + off : k + off;
            float s = 0.f;
            for (int a = 0; a < d; ++a) s += qb[a] * kd[a];
            s *= scale; lg[dd] = s; mx = std::max(mx, s);
        }
        float Z = 0.f;
        for (int dd = 0; dd <= n; ++dd) { lg[dd] = FAST_MATH ? fast_exp(lg[dd] - mx) : std::exp(lg[dd] - mx); Z += lg[dd]; }
        for (int a = 0; a < d; ++a) mixed[off + a] = 0.f;
        for (int dd = 0; dd <= n; ++dd) {
            const float p = lg[dd] / Z;
            const float* __restrict vd = (dd < n) ? dv + (size_t)dd * D_KV + off : v + off;
            for (int a = 0; a < d; ++a) mixed[off + a] += p * vd[a];
        }
    }
    for (int j = 0; j < D_KV; ++j) v[j] = mixed[j];
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
//
// gdn_* (Stage 1): present only on a layer GDN_SCHEDULE.full_attn[l] marks GDN (nullptr on an
// attention layer, and vice versa for Wq/Wk/Wv/Wo/q_norm/k_norm) -- see layout.hpp's PKind comment for
// what each one is. A layer is one or the other, never both, at every GDN_FULL_ATTN_STRIDE value.
struct Layer {
    // ln1/ln2 are nullptr under USE_GATED_RESIDUAL (WP4b blocker D: the real decoder layer has neither,
    // and GR's own hc_norm is the pre-block norm) -- the same "unused pointers just stay nullptr" idiom
    // q_norm/k_norm and the gdn_* set already use, so the struct's shape stays config-independent.
    Node *ln1 = nullptr, *ln2 = nullptr;
    Node *Wq, *Wk, *Wv, *Wo, *W1, *b1, *W2, *b2, *Wg, *q_norm, *k_norm;
    Node *gdn_in_qkv = nullptr, *gdn_in_z = nullptr, *gdn_in_b = nullptr, *gdn_in_a = nullptr,
         *gdn_conv = nullptr, *gdn_a_log = nullptr, *gdn_dt_bias = nullptr, *gdn_norm = nullptr,
         *gdn_out_proj = nullptr;
    // Gated Residual (Stage 1, docs/GATED_RESIDUAL.md S3b/S4): two independent GatedResidual instances
    // per layer -- gr_attn_* wraps the attention/GDN sub-block, gr_mlp_* wraps the FFN sub-block, each
    // with its own hc_norm/down/up/block_inject tensors. nullptr on every layer when !USE_GATED_RESIDUAL.
    Node *gr_attn_norm = nullptr, *gr_attn_down = nullptr, *gr_attn_up = nullptr, *gr_attn_inject = nullptr,
         *gr_mlp_norm  = nullptr, *gr_mlp_down  = nullptr, *gr_mlp_up  = nullptr, *gr_mlp_inject  = nullptr;
    // Mixture of Experts (Stage 1, docs/MOE.md S3b/S4): present only when USE_MOE (replaces Wg/W1/W2 or
    // W1/b1/W2/b2 for THIS layer, per make_param_layout()'s own if constexpr branch -- every layer, no
    // per-layer schedule, unlike GDN). moe_gate/moe_up/moe_down are NUM_EXPERTS_BUF-sized arrays, one
    // slot per routed expert's own SwiGLU triple; only the first NUM_EXPERTS entries are ever populated.
    Node *moe_router = nullptr;
    std::array<Node*, NUM_EXPERTS_BUF> moe_gate{}, moe_up{}, moe_down{};
    Node *moe_shared_gate = nullptr, *moe_shared_up = nullptr, *moe_shared_down = nullptr,
         *moe_shared_gate_proj = nullptr;
    // QSA (Stage 1, docs/QSA.md S3b/S4): present only on a layer MIXER_SCHEDULE marks Qsa (nullptr on
    // every layer when !USE_QSA). REPLACES Wq/Wk/Wv/Wo (+q_norm/k_norm) for that layer, exactly as the
    // gdn_* set does -- qsa_q/qsa_gate are the two halves of the real model's one DOUBLE-WIDTH q_proj.
    Node *qsa_q = nullptr, *qsa_gate = nullptr, *qsa_k = nullptr, *qsa_v = nullptr, *qsa_o = nullptr,
         *qsa_qnorm = nullptr, *qsa_knorm = nullptr,
         *qsa_idx_qk = nullptr, *qsa_idx_qnorm = nullptr, *qsa_idx_knorm = nullptr;
};

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
static Node* op_moe(Node* x, Layer& L) {
    const int T = x->rows;
    Node* out = mk_node(Op::Moe, T, D_MODEL);
    out->a = x;
    std::array<const float*, NUM_EXPERTS_BUF> gate_w{}, up_w{}, down_w{};
    for (int e = 0; e < NUM_EXPERTS; ++e) {
        gate_w[static_cast<std::size_t>(e)] = L.moe_gate[static_cast<std::size_t>(e)]->data.data();
        up_w[static_cast<std::size_t>(e)]   = L.moe_up[static_cast<std::size_t>(e)]->data.data();
        down_w[static_cast<std::size_t>(e)] = L.moe_down[static_cast<std::size_t>(e)]->data.data();
    }
    auto [scratch, scratch_g] = arena_alloc(moe::scratch_floats(MOE_DIMS));
    moe::forward(MOE_DIMS, T, x->data.data(), L.moe_router->data.data(),
                 gate_w.data(), up_w.data(), down_w.data(),
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

struct Model {
    Node* tok_emb;
    Node* pos_emb;
    std::array<Layer, N_LAYERS> layers;
    Node* ln_f;
    Node* lm_head;   // nullptr when USE_TIED_EMBEDDINGS -- the head reads tok_emb directly instead
    Node* lm_bias;   // nullptr when USE_TIED_EMBEDDINGS -- tied models drop the head bias too

    // Gated Residual (Stage 1, docs/GATED_RESIDUAL.md S1c/S3b): the model-level EXIT collapse, ONE
    // instance shared across the whole stack (not per-layer, unlike Layer's own gr_attn_*/gr_mlp_*),
    // WITHOUT a block_inject tensor (the real model's hyper_connection_mixer is use_combine=False).
    // nullptr when !USE_GATED_RESIDUAL.
    Node *gr_top_norm = nullptr, *gr_top_down = nullptr, *gr_top_up = nullptr;

    // N-gram embeddings (see docs/NGRAM_EMBEDDING.md). ngram_tab[e]: the e-th hashed n-gram embedding
    // table (a real PARAM_LAYOUT leaf, like tok_emb). ngram_proj: the single learned concat_proj GEMM
    // weight, [D_MODEL, D_MODEL]. ngram_wblock[e]: a NON-owning VIEW into ngram_proj's own data/grad --
    // rows [e*NGRAM_EMB_DIM, (e+1)*NGRAM_EMB_DIM), i.e. the row-block that table e's slice of the
    // concatenated vector would multiply against. This is the block-matmul identity
    // ([x0|x1|..|xk] @ W == sum_e x_e @ W[e*d:(e+1)*d, :]) applied so "concat then one linear" needs no
    // separate Concat op or Node-fanout widening (contrast docs/DEPTH_ATTENTION.md 5a, which genuinely
    // needed a side table because ITS fanout was variable and cross-execution; here it is a FIXED,
    // compile-time partition of one static parameter, so a plain array of aliasing Leaf-shaped Nodes
    // suffices). AdamW steps ngram_proj exactly ONCE, over its whole PARAM_LAYOUT range, as normal --
    // the views are read-only aliases used only to route Linear's backward into the right row-range of
    // that ONE grad buffer; the partition is disjoint, so backward's per-view `+=` writes never collide.
    std::array<Node*, NGRAM_TABLES_BUF> ngram_tab{};
    Node*                               ngram_proj = nullptr;
    std::array<Node, NGRAM_TABLES_BUF>  ngram_wblock{};

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
                for (int e = 0; e < NUM_EXPERTS; ++e) {
                    L.moe_gate[static_cast<std::size_t>(e)] = mk_param(D_MODEL, D_FF, true);
                    L.moe_up[static_cast<std::size_t>(e)]   = mk_param(D_MODEL, D_FF, true);
                    L.moe_down[static_cast<std::size_t>(e)] = mk_param(D_FF, D_MODEL, true);
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
        ln_f = mk_param(1, D_MODEL, false);
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
                for (int e = 0; e < NUM_EXPERTS; ++e) {
                    randn(L.moe_gate[static_cast<std::size_t>(e)], 0.02f);
                    randn(L.moe_up[static_cast<std::size_t>(e)], 0.02f);
                    randn(L.moe_down[static_cast<std::size_t>(e)], 0.02f);
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
        ones(ln_f);
        if constexpr (!USE_TIED_EMBEDDINGS) randn(lm_head, 0.02f);
        if constexpr (NGRAM_EMBED) {
            for (int e = 0; e < NGRAM_NUM_EMBEDDERS; ++e) randn(ngram_tab[static_cast<std::size_t>(e)], 0.02f);
            randn(ngram_proj, 0.02f);   // also initializes every ngram_wblock[e] view (same underlying data)
        }
    }

    // Per-execution residual-stream diagnostic (nullptr = off, the production path). When armed by
    // sub0::loop_pass_stats, forward() records for each EXECUTION e: the norm of the residual stream
    // entering the block, and the norm of what the block ADDED to it. Under LoopSplit the same layer
    // runs several times, so comparing pass 1's contribution against pass 2's for the SAME layers is a
    // direct test of the fixed-point prediction -- if repeated passes perturb the stream less and less,
    // the extra compute is buying nothing and no amount of tuning will change that.
    //
    // Deliberately a hook on the REAL forward rather than a reimplementation: a separate diagnostic
    // copy of the layer loop would drift from the one that actually trains, and then measure the wrong
    // thing convincingly.
    float* pass_delta = nullptr;   // [LOOP_EXEC_COUNT] ||h_out - h_in||
    float* pass_hnorm = nullptr;   // [LOOP_EXEC_COUNT] ||h_in||

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
        if constexpr (USE_GATED_RESIDUAL) h = op_gr_tile(h);
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
                    h = gr_write(h_before_attn, op_gdn(a, L), gr_inj_attn);
                    Node* h_before_mlp = h;
                    Node* gr_inj_mlp = nullptr;
                    Node* f = gr_read(h, L.gr_mlp_norm, L.gr_mlp_down, L.gr_mlp_up, L.gr_mlp_inject,
                                       L.ln2, &gr_inj_mlp);
                    if constexpr (USE_MOE) {
                        f = op_moe(f, L);
                    } else if constexpr (USE_GATED_FFN) {
                        Node* gate_pre = op_linear(f, L.Wg, nullptr, q);
                        Node* up_pre   = op_linear(f, L.W1, nullptr, q);
                        f = op_linear(op_swiglu(gate_pre, up_pre), L.W2, nullptr, q);
                    } else {
                        f = op_linear(op_gelu(op_linear(f, L.W1, L.b1, q)), L.W2, L.b2, q);
                    }
                    h = gr_write(h_before_mlp, f, gr_inj_mlp);
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
            h = gr_write(h_before_attn, mixer_out, gr_inj_attn);
            Node* h_before_mlp = h;
            Node* gr_inj_mlp = nullptr;
            Node* f = gr_read(h, L.gr_mlp_norm, L.gr_mlp_down, L.gr_mlp_up, L.gr_mlp_inject, L.ln2,
                               &gr_inj_mlp);
            if constexpr (USE_MOE) {
                f = op_moe(f, L);
            } else if constexpr (USE_GATED_FFN) {
                Node* gate_pre = op_linear(f, L.Wg, nullptr, q);
                Node* up_pre   = op_linear(f, L.W1, nullptr, q);
                f = op_linear(op_swiglu(gate_pre, up_pre), L.W2, nullptr, q);
            } else {
                f = op_linear(op_gelu(op_linear(f, L.W1, L.b1, q)), L.W2, L.b2, q);
            }
            h = gr_write(h_before_mlp, f, gr_inj_mlp);
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
        if constexpr (USE_GATED_RESIDUAL) h = op_gr_mix(h, gr_top_norm, gr_top_down, gr_top_up);
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
        // `d` is D_HEAD, NOT C / H: WP4b blocker A made the head width its own axis, so the two differ
        // the moment --head-dim is set (and the decode path's per-head strides are all in D_HEAD).
        // Spelling it C / H here was a real bug -- the forward-vs-forward_one parity check caught it at
        // 1.71 relative error on the first --head-dim build, exactly the AGENTS.md S10 class of defect
        // (a derived width re-spelled locally instead of read from the one source of truth).
        constexpr int C = D_MODEL, H = N_HEADS, d = D_HEAD;
        const float scale = 1.f / std::sqrt(static_cast<float>(d));
        // Gated Residual (Stage 1): `h` is HC_WIDE wide when GR is on (== D_MODEL, i.e. today's exact
        // size, when off -- layout.hpp's own collapse). Every OTHER per-layer temporary below (a, qn,
        // kn, vn, att, proj, f1, g1) stays exactly D_MODEL/D_FF-wide -- GR only ever widens the
        // persistent residual itself, never the sub-block's own internal working set (S2).
        // qn/att are D_Q = N_HEADS*D_HEAD wide and kn/vn are D_KV wide -- NOT D_MODEL, since WP4b
        // blocker A made D_HEAD its own axis. All four are sized to the widest of the three so one
        // constant covers them (D_KV <= D_Q by construction, see layout.hpp's static_assert); at the
        // derived head width this is exactly C, so these arrays are unchanged for every existing build.
        constexpr int ATT_W = sub0::D_Q > C ? sub0::D_Q : C;
        float h[HC_WIDE], a[C], qn[ATT_W], kn[ATT_W], vn[ATT_W], att[ATT_W], proj[C], f1[D_FF];
        [[maybe_unused]] float g1[D_FF];   // gate branch, gated FFN only
        [[maybe_unused]] float packed_copy[C];             // periodic-reinject spike, see below
        // Gated Residual scratch/output buffers -- sized 1 (never zero-length) when off, same idiom as
        // every other never-degenerate buffer in this function. gr_normed is shared by gr_read_row's
        // mix AND gate steps (both read-only consumers of the SAME hc_norm(wide) result) -- a cheaper,
        // equally correct alternative to op_gr_mix/op_gr_gate's own independent-recompute form
        // (docs/GATED_RESIDUAL.md S4c), available here because this is plain imperative code, not two
        // separate Node-graph ops that would each need their own self-contained scratch.
        [[maybe_unused]] float gr_normed[HC_WIDE];
        [[maybe_unused]] float gr_mixscr[GR_MIX_SCRATCH1 ? GR_MIX_SCRATCH1 : 1];
        [[maybe_unused]] float gr_mixed[C];
        [[maybe_unused]] float gr_inj[HC_COUNT_BUF];
        // Mixture of Experts (Stage 1, docs/MOE.md S4b) decode-path scratch -- MOE_SCRATCH1 is never
        // zero (see its own comment), unlike GR_MIX_SCRATCH1 above, so no ternary-guard is needed here.
        [[maybe_unused]] float moe_scratch[MOE_SCRATCH1];
        [[maybe_unused]] std::array<const float*, NUM_EXPERTS_BUF> moe_gate_w{}, moe_up_w{}, moe_down_w{};
        // QSA (Stage 1, docs/QSA.md S4b) decode-path scratch. Every bound uses QSA_DIMS_BUF, whose widths
        // are never zero even when QSA is off, so these stay valid array bounds in a QSA-off build
        // (where nothing ever reads them) -- the same never-degenerate idiom GR_MIX_SCRATCH1 needed.
        // Sized for the WHOLE window (SEQ_LEN), not the current position: forward_one's kv window grows
        // with `pos`, and AGENTS.md S1 forbids sizing per call.
        [[maybe_unused]] float qsa_idx_q[QSA_DIMS_BUF.idx_q_width()];
        [[maybe_unused]] float qsa_gate_row[QSA_DIMS_BUF.q_width()];
        [[maybe_unused]] float qsa_mask[SEQ_LEN];
        [[maybe_unused]] float qsa_sel_scr[qsa::select_scratch_floats(QSA_DIMS_BUF, SEQ_LEN)];
        [[maybe_unused]] float qsa_att_scr[qsa::attn_scratch_floats(QSA_DIMS_BUF, SEQ_LEN)];
        // GR READ (mix+gate, into `out_a`) / WRITE (combine, in place on `wide`) row-helpers, T==1 --
        // the exact decode-path counterpart of forward()'s gr_read/gr_write lambdas (docs/GATED_RESIDUAL.md
        // S2). Safe to write `wide` in place in gr_write_row: combine()'s per-(stream,channel) output
        // depends only on that SAME index's own input (plus mixer_out, a disjoint buffer), never on any
        // other index, so there is no read-after-write hazard.
        auto gr_read_row = [&](const float* wide, Node* norm_w, Node* down_w, Node* up_w, Node* inject_w,
                                Node* ln, float* out_a) {
            if constexpr (USE_GATED_RESIDUAL) {
                gr::hc_norm(GR_DIMS, 1, wide, norm_w->data.data(), gr_normed);
                gr::mix(GR_DIMS, 1, gr_normed, down_w->data.data(), up_w->data.data(), gr_mixed, gr_mixscr);
                gr::gate(GR_DIMS, 1, gr_normed, inject_w->data.data(), gr_inj);
                // WP4b blocker D: the mixer reads mixed_input DIRECTLY -- no Ln1/Ln2 exists under GR
                // (`ln` is nullptr), and gr::hc_norm above already applied the real model's own
                // pre-block norm at its own 1e-6 eps. Mirrors forward()'s gr_read exactly; the
                // forward-vs-forward_one parity test is what gates that they stay mirrored.
                (void)ln;
                std::copy_n(gr_mixed, C, out_a);
            } else {
                rmsnorm_row(wide, ln, out_a, C);
            }
        };
        auto gr_write_row = [&](float* wide, const float* mixer_out) {
            if constexpr (USE_GATED_RESIDUAL) gr::combine(GR_DIMS, 1, wide, mixer_out, gr_inj, wide);
            else                              for (int j = 0; j < C; ++j) wide[j] += mixer_out[j];
        };
        // This token's depth cache: rows only, since the depth softmax never crosses positions (see
        // depth_mix_row). Scoped to one forward_one call -- unlike g_kv, nothing here spans tokens.
        [[maybe_unused]] float dep_k[(DEPTH_CACHE_MAX ? DEPTH_CACHE_MAX : 1) * D_KV];
        [[maybe_unused]] float dep_v[(DEPTH_CACHE_MAX ? DEPTH_CACHE_MAX : 1) * D_KV];
        [[maybe_unused]] int   dep_n = 0;
        bool do_reinject = false;

        if (g_sentinel_binds && prev == g_sentinel_binds->sigil && g_sentinel_binds->bound(id)) {
            encode_slot(tok_emb->data.data(), C, g_sentinel_binds->fragments(id), g_sentinel_binds->encoding,
                        h, g_sentinel_binds->enc_w);
        } else if (g_scratch_binds && is_scratch_slot(id) && g_scratch_binds->bound(id)) {
            encode_slot(tok_emb->data.data(), C, g_scratch_binds->fragments(id), g_scratch_binds->encoding, h,
                        g_scratch_binds->enc_w);
            if (g_scratch_reinject_stride > 0) {   // save the layer-0 packed vector for periodic re-injection
                for (int j = 0; j < C; ++j) packed_copy[j] = h[j];
                do_reinject = true;
            }
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
        // N-gram embeddings, decode path: mirrors forward()'s block exactly (same hashing, same
        // block-matmul-via-row-slice composition), but the context comes from a small rolling history
        // of the last NGRAM_MAX_N-1 FED ids instead of indexing back into a batched window -- this
        // engine's analogue of the reference's `NgramCache.update_ngram_context`. `prev < 0` (computed
        // above, BEFORE it was overwritten) is the same "fresh generation or non-sequential jump"
        // signal forward_one's sentinel-pair detection already relies on, so history resets exactly
        // when that context would otherwise be wrong.
        if constexpr (NGRAM_EMBED) {
            static thread_local int hist[NGRAM_MAX_SHIFT_BUF];
            static thread_local int hist_len = 0;
            if (prev < 0) hist_len = 0;
            // Same guard as forward()'s ngram_tok: a persistent-slot id (>= VOCAB) is not a real
            // recurring vocabulary token and its raw integer value is unbounded, so it hashes as "no
            // signal" (id 0) -- see forward()'s comment for the full reasoning and the differential
            // test (persistent_slots_engine_tests.cpp) that caught this without the guard.
            const int id_tok = (id < VOCAB) ? id : 0;
            for (int e = 0; e < NGRAM_NUM_EMBEDDERS; ++e) {
                const int order     = NGRAM_ORDERS[static_cast<std::size_t>(e)];
                const int vocab_dim = NGRAM_VOCAB_DIMS[static_cast<std::size_t>(e)];
                std::int64_t acc = id_tok;
                for (int k = 2; k <= order; ++k) {
                    const int shift = k - 1;
                    const int ptok  = (shift - 1 < hist_len) ? hist[shift - 1] : 0;
                    acc += static_cast<std::int64_t>(ptok) * NGRAM_VOCAB_MODS[static_cast<std::size_t>(e)][static_cast<std::size_t>(k - 2)];
                }
                const int tid = static_cast<int>(((acc % vocab_dim) + vocab_dim) % vocab_dim);
                const float* row = ngram_tab[static_cast<std::size_t>(e)]->data.data()
                                 + static_cast<std::size_t>(tid) * NGRAM_EMB_DIM;
                const float* Wb = ngram_wblock[static_cast<std::size_t>(e)].data.data();
                for (int p = 0; p < NGRAM_EMB_DIM; ++p) {
                    const float xp = row[p];
                    const float* __restrict Wr = Wb + static_cast<std::size_t>(p) * C;
                    for (int o = 0; o < C; ++o) h[o] += xp * Wr[o];
                }
            }
            for (int s = NGRAM_MAX_SHIFT_BUF - 1; s > 0; --s) hist[s] = hist[s - 1];
            hist[0] = id_tok;   // history stores the GUARDED value, so it stays consistent later too
            if (hist_len < NGRAM_MAX_SHIFT_BUF) ++hist_len;
        }
        // Gated Residual's model-level ENTRY tile (docs/GATED_RESIDUAL.md S1c): everything above (embed,
        // absolute pos, n-gram injection) wrote into h's first D_MODEL elements exactly as before --
        // this duplicates that across the other HC_COUNT-1 streams. Safe in place (out_wide aliases the
        // same buffer tile() reads from): stream 0's write is `h[j] = h[j]` (a no-op), and every later
        // stream's write target lies entirely outside the [0,D_MODEL) range tile() ever reads from.
        if constexpr (USE_GATED_RESIDUAL) gr::tile(GR_DIMS, 1, h, h);
        // `e` is the EXECUTION index (the KV-cache slot); `li` is which layer's weights run there.
        // They differ only under LoopSplit -- see LAYER_EXEC_ORDER and KVCache's own comments.
        for (int e = 0; e < LOOP_EXEC_COUNT; ++e) {
            const int l = LAYER_EXEC_ORDER[static_cast<std::size_t>(e)];
            Layer& L = layers[static_cast<std::size_t>(l)];
            // `h` is mutated IN PLACE by gr_write_row below (unlike forward()'s Node graph, where a new
            // Node replaces `h`), so gr_read_row's read and gr_write_row's later write on the SAME `h`
            // need no separate "before" snapshot -- nothing between them mutates it.
            gr_read_row(h, L.gr_attn_norm, L.gr_attn_down, L.gr_attn_up, L.gr_attn_inject, L.ln1, a);
            // The softmax-attention mixer sublayer, factored into a lambda (rather than duplicated
            // verbatim in both branches below) so the GDN_SCHEDULE dispatch reads as a single small
            // if/else rather than two copies of this block drifting apart over time.
            auto do_attention_mixer = [&] {
                // Wq is [C, D_Q] -- the OUT width is D_Q = N_HEADS*D_HEAD, not D_MODEL, since WP4b
                // blocker A. Identical at the derived head width.
                linear_row(a, L.Wq, nullptr, qn, C, sub0::D_Q);
                // K/V project to D_KV (N_KV_HEADS heads), Q to D_Q (N_HEADS heads). qknorm_row/rope_row
                // both derive their per-head width as (width / heads), so passing the KV pair yields
                // the same D_HEAD they always did -- the rotation and norm are per-head, so nothing
                // else changes.
                linear_row(a, L.Wk, nullptr, kn, C, D_KV);
                linear_row(a, L.Wv, nullptr, vn, C, D_KV);
                if constexpr (USE_QK_NORM) { qknorm_row(qn, L.q_norm, H, sub0::D_Q); qknorm_row(kn, L.k_norm, N_KV_HEADS, D_KV); }
                if constexpr (POS_ENCODING == PosEncoding::Rope) { rope_row(qn, pos, H, sub0::D_Q); rope_row(kn, pos, N_KV_HEADS, D_KV); }
                // Depth attention, before the KV-cache append: the sequence cache must hold the MIXED
                // V, because that is what op_attn receives in the batched forward. Mirrors forward()'s
                // block.
                if constexpr (USE_DEPTH_ATTN) {
                    depth_mix_row(qn, kn, vn, dep_k, dep_v, dep_n);
                    if (DEPTH_SCHEDULE.own[static_cast<std::size_t>(e)] >= 0) {
                        for (int j = 0; j < D_KV; ++j) {
                            dep_k[(size_t)dep_n * D_KV + j] = kn[j];
                            dep_v[(size_t)dep_n * D_KV + j] = vn[j];
                        }
                        ++dep_n;
                    }
                }
                float* kc = g_kv.krow(e, pos); float* vc = g_kv.vrow(e, pos);    // append this token's K/V
                for (int j = 0; j < D_KV; ++j) { kc[j] = kn[j]; vc[j] = vn[j]; }
                for (int hd = 0; hd < H; ++hd) {                                 // attend query pos over j<=pos
                    const int off    = hd * d;
                    const int off_kv = (hd / GQA_GROUP) * d;    // shared KV head for this query group
                    std::array<float, SEQ_LEN> sc{};
                    float mx = -1e30f;
                    for (int j = 0; j <= pos; ++j) {
                        const float* kj = g_kv.krow(e, j) + off_kv;
                        float s = 0.f; for (int aa = 0; aa < d; ++aa) s += qn[off + aa] * kj[aa];
                        s *= scale; sc[j] = s; mx = std::max(mx, s);
                    }
                    float Z = 0.f;
                    for (int j = 0; j <= pos; ++j) { sc[j] = FAST_MATH ? fast_exp(sc[j] - mx) : std::exp(sc[j] - mx); Z += sc[j]; }
                    for (int aa = 0; aa < d; ++aa) att[off + aa] = 0.f;
                    for (int j = 0; j <= pos; ++j) {
                        const float p = sc[j] / Z; const float* vj = g_kv.vrow(e, j) + off_kv;
                        for (int aa = 0; aa < d; ++aa) att[off + aa] += p * vj[aa];
                    }
                }
                linear_row(att, L.Wo, nullptr, proj, sub0::D_Q, C);   // Wo is [D_Q, D_MODEL] (blocker A)
                gr_write_row(h, proj);                                           // residual (write step)
            };
            // The QSA decode counterpart of the same sublayer -- the T==1 form of op_qsa's own batched
            // call, built from the SAME qsa_math.hpp row helpers op_qsa's qsa::forward() loops over, so
            // the two provably compute identical arithmetic (the forward-vs-forward_one parity test is
            // what gates that -- docs/QSA.md S9). Reuses g_kv for K/V exactly as the softmax path does
            // (kv_width() == D_KV), and g_qsa_cache for the indexer's own raw keys, which g_kv cannot
            // hold (a different width and a different -- unnormed, unrotated -- content).
            [[maybe_unused]] auto do_qsa_mixer = [&] {
                float* raw_k_base = g_qsa_cache.base(e);
                // Row stride is ROTARY_DIM (the cos/sin tables are [SEQ_LEN][ROTARY_DIM]) -- see
                // QsaRopeTables above; it was D_HEAD before --rotary-dim became an axis.
                const float* cos_pos = g_qsa_rope.cos.data() + static_cast<size_t>(pos) * ROTARY_DIM;
                const float* sin_pos = g_qsa_rope.sin.data() + static_cast<size_t>(pos) * ROTARY_DIM;
                qsa::indexer_project_row(QSA_DIMS, a, L.qsa_idx_qk->data.data(),
                                          L.qsa_idx_qnorm->data.data(), cos_pos, sin_pos, qsa::RMS_EPS,
                                          qsa_idx_q,
                                          raw_k_base + static_cast<size_t>(pos) * QSA_INDEXER_HEAD_DIM);
                qsa::attn_project_row(QSA_DIMS, a, L.qsa_q->data.data(), L.qsa_gate->data.data(),
                                       L.qsa_k->data.data(), L.qsa_v->data.data(),
                                       L.qsa_qnorm->data.data(), L.qsa_knorm->data.data(),
                                       cos_pos, sin_pos, qsa::RMS_EPS,
                                       qn, qsa_gate_row, g_kv.krow(e, pos), g_kv.vrow(e, pos));
                // The pooled block keys persist across decode steps in this execution's own QsaCache
                // slot, exactly as the raw keys above already do -- the decode counterpart of the
                // batched path's in-scratch cache, filled by the SAME primitive (docs/QSA.md S11).
                qsa::indexer_select_row(QSA_DIMS, qsa_idx_q, raw_k_base, pos + 1,
                                         L.qsa_idx_knorm->data.data(), g_qsa_rope.cos.data(),
                                         g_qsa_rope.sin.data(), qsa::RMS_EPS,
                                         g_qsa_cache.block_base(e), g_qsa_cache.n_cached_of(e),
                                         qsa_mask, qsa_sel_scr);
                qsa::attn_row(QSA_DIMS, qn, qsa_gate_row, g_kv.krow(e, 0), g_kv.vrow(e, 0), pos + 1,
                               qsa_mask, L.qsa_o->data.data(), proj, qsa_att_scr);
                gr_write_row(h, proj);                                           // residual (write step)
            };
            // Which of the two full-attention forms this build uses, decided ONCE at compile time. This
            // is the seam the MoE stage's own SIGSEGV came through (docs/MOE.md S9: a replacement wired
            // into only ONE of two call sites), so it is expressed as a single named lambda that BOTH
            // the GDN-on and GDN-off dispatch paths below call, rather than duplicated in each.
            auto do_full_attn_mixer = [&] {
                if constexpr (USE_QSA) do_qsa_mixer();
                else                   do_attention_mixer();
            };
            // GDN_SCHEDULE.full_attn[l] mirrors Model::forward()'s own dispatch exactly (per-LAYER, not
            // per-execution) -- see that function's comment. `if constexpr` keeps a GDN-off build
            // exactly the original single call, no branch at all.
            if constexpr (USE_GATED_DELTANET) {
                if (GDN_SCHEDULE.full_attn[static_cast<std::size_t>(l)]) {
                    do_full_attn_mixer();
                } else {
                    // Gated DeltaNet decode: T=1, persistent per-execution state/conv history
                    // (GdnCache, reset by kv_reset() at the start of each generation). op_gdn's whole
                    // output (in_proj*, conv, recurrence, gated-norm, out_proj) IS the mixer sublayer,
                    // so it goes straight into the residual the same way `proj` does above -- no
                    // separate Wo, no RoPE, no QK-norm, no depth-attention (see Model::forward()'s
                    // matching comment for why those are out of scope for a GDN layer).
                    // Same dt_bias/a_log ARGUMENT-ORDER fix as op_gdn's batched forward above (this
                    // decode path had the identical swap, independently) -- see that call site's comment.
                    float gdn_scratch[GDN_SCRATCH1];
                    gdn::forward(GDN_DIMS, 1, a,
                                 L.gdn_in_qkv->data.data(), L.gdn_in_z->data.data(),
                                 L.gdn_in_b->data.data(), L.gdn_in_a->data.data(),
                                 L.gdn_conv->data.data(), L.gdn_dt_bias->data.data(),
                                 L.gdn_a_log->data.data(), L.gdn_norm->data.data(),
                                 L.gdn_out_proj->data.data(),
                                 g_gdn_cache.state_of(e), g_gdn_cache.conv_of(e), proj, gdn_scratch);
                    gr_write_row(h, proj);                                       // residual (write step)
                }
            } else {
                do_full_attn_mixer();
            }
            gr_read_row(h, L.gr_mlp_norm, L.gr_mlp_down, L.gr_mlp_up, L.gr_mlp_inject, L.ln2, a);
            if constexpr (USE_MOE) {
                for (int e = 0; e < NUM_EXPERTS; ++e) {
                    moe_gate_w[static_cast<std::size_t>(e)] = L.moe_gate[static_cast<std::size_t>(e)]->data.data();
                    moe_up_w[static_cast<std::size_t>(e)]   = L.moe_up[static_cast<std::size_t>(e)]->data.data();
                    moe_down_w[static_cast<std::size_t>(e)] = L.moe_down[static_cast<std::size_t>(e)]->data.data();
                }
                moe::forward_row(MOE_DIMS, a, L.moe_router->data.data(),
                                  moe_gate_w.data(), moe_up_w.data(), moe_down_w.data(),
                                  L.moe_shared_gate->data.data(), L.moe_shared_up->data.data(),
                                  L.moe_shared_down->data.data(), L.moe_shared_gate_proj->data.data(),
                                  proj, moe_scratch);
            } else if constexpr (USE_GATED_FFN) {
                linear_row(a, L.Wg, nullptr, g1, C, D_FF);
                linear_row(a, L.W1, nullptr, f1, C, D_FF);
                for (int j = 0; j < D_FF; ++j) f1[j] = silu_row(g1[j]) * f1[j];
                linear_row(f1, L.W2, nullptr, proj, D_FF, C);
            } else {
                linear_row(a, L.W1, L.b1, f1, C, D_FF);
                for (int j = 0; j < D_FF; ++j) f1[j] = gelu_row(f1[j]);
                linear_row(f1, L.W2, L.b2, proj, D_FF, C);
            }
            gr_write_row(h, proj);                                              // residual (write step)
            // Periodic packed-content re-injection spike (Nanbeige-inspired, see core.hpp's
            // set_scratch_reinject doc comment): every `stride` layers, add the SAME layer-0 packed
            // vector back into this position's hidden state -- tests whether reinforcing the signal
            // partway through the stack counters the dilution a single upfront injection leaves (axis 9).
            // SCALE-ADAPTIVE: the residual stream's own RMS norm grows across depth (standard pre-norm
            // behavior -- it's why ln_f exists before the head at all), while `packed_copy` sits at fixed
            // ordinary-embedding-row scale (scratch_slots.hpp's own amplitude convention) -- a first,
            // fixed-scale version of this spike was measurably a near no-op even at full strength every
            // layer, because the addition was dwarfed by h's already-larger accumulated norm by mid-stack.
            // Rescaling packed_copy to match h's CURRENT norm before applying `scale` (now a fraction of
            // h's own magnitude, not an absolute embedding-scale constant) is what makes this a real test.
            //
            // NOT Gated-Residual-aware: this spike only ever touches h's first D_MODEL elements (stream 0
            // when GR is wide), same deliberate-scope-gap shape as GDN+depth-attention's own "neither doc
            // discusses the combination" (docs/GATED_DELTANET.md) -- this spike and GR have not been
            // designed to interact, and this pass does not attempt it.
            if (do_reinject && ((l + 1) % g_scratch_reinject_stride == 0)) {
                float h_ms = 0.f; for (int j = 0; j < C; ++j) h_ms += h[j] * h[j]; h_ms /= C;
                float pk_ms = 0.f; for (int j = 0; j < C; ++j) pk_ms += packed_copy[j] * packed_copy[j]; pk_ms /= C;
                const float norm_scale = g_scratch_reinject_scale * std::sqrt((h_ms + 1e-8f) / (pk_ms + 1e-8f));
                for (int j = 0; j < C; ++j) h[j] += norm_scale * packed_copy[j];
            }
        }
        // Gated Residual's model-level EXIT collapse (docs/GATED_RESIDUAL.md S1c), mirroring forward()'s
        // own exit-collapse placement: use_combine=False, so just the mix half (no gate/combine) --
        // last_hidden captures the FULLY-COLLAPSED, D_MODEL-wide representation, same as the GR-off path.
        if constexpr (USE_GATED_RESIDUAL) {
            gr::hc_norm(GR_DIMS, 1, h, gr_top_norm->data.data(), gr_normed);
            gr::mix(GR_DIMS, 1, gr_normed, gr_top_down->data.data(), gr_top_up->data.data(), gr_mixed, gr_mixscr);
            for (int j = 0; j < C; ++j) last_hidden[static_cast<std::size_t>(j)] = gr_mixed[j];
            rmsnorm_row(gr_mixed, ln_f, a, C);
        } else {
            for (int j = 0; j < C; ++j) last_hidden[static_cast<std::size_t>(j)] = h[j];   // diagnostic capture
            rmsnorm_row(h, ln_f, a, C);
        }
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
// private to this translation unit, and deliberately so: a member added to Worker must not be able to
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

// ensure_thread_built() matches forward()/forward_one() below: a caller running on a thread that has
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
Node* cross_entropy(Node* logits, const int* targets) { return op_cross_entropy(logits, targets); }

// Incremental single-token inference (KV-cache). kv_reset() clears/sizes the cache at the start of a
// generation; forward_one(id, pos) returns the logits [VOCAB] for the next token. See KVCache above.
// Also resets GdnCache's decode-persistent recurrent state -- `if constexpr` so this costs nothing
// (not even the vector-emptiness check) on a build with GDN off, per AGENTS.md S4.
void kv_reset() {
    g_kv.reset();
    if constexpr (USE_GATED_DELTANET) g_gdn_cache.reset();
    if constexpr (USE_QSA) g_qsa_cache.reset();   // the indexer's own raw-key store -- docs/QSA.md S6
}
const float* forward_one(int id, int pos) { ensure_thread_built(); return g_model.forward_one(id, pos); }
const float* last_hidden_ptr() { return g_model.last_hidden.data(); }   // see Model::last_hidden's comment

// KV-trace memoization primitives (spike, see core.hpp's declarations for the full design comment).
// Caller-responsibility contract matches forward_one's own: valid only after kv_reset() has sized g_kv
// on this thread (no redundant guard here, same discipline as g_kv's other two entry points above).
const float* kv_krow_ptr(int layer, int pos) { return g_kv.krow(layer, pos); }
const float* kv_vrow_ptr(int layer, int pos) { return g_kv.vrow(layer, pos); }
// KV-cache rows are D_KV wide with N_KV_HEADS heads (== D_MODEL/N_HEADS when not using GQA); the
// per-head rotation width is D_HEAD either way.
void kv_rope_rotate(float* row, int pos) { rope_row(row, pos, N_KV_HEADS, D_KV); }
void kv_splice_row(int layer, int pos, const float* k_canonical, const float* v) {
    float k[D_KV];
    for (int j = 0; j < D_KV; ++j) k[j] = k_canonical[j];
    rope_row(k, pos, N_KV_HEADS, D_KV);          // rotate the de-rotated canonical row to its splice position
    float* kc = g_kv.krow(layer, pos);
    float* vc = g_kv.vrow(layer, pos);
    for (int j = 0; j < D_KV; ++j) { kc[j] = k[j]; vc[j] = v[j]; }      // V is position-invariant, no rotation
}

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
