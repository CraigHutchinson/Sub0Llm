// internal.hpp -- backend-private shared detail for the CPU backend's translation units.
//
// This header exists for exactly one reason: `backend.cpp` (the Node-graph ops, reverse-mode
// backward, the batched forward, train_batch and AdamW) and `decode.cpp` (the KV-cache
// single-token path) are two translation units that must share ONE copy of the process/thread
// state -- the shared parameter arenas, the Worker pool and its thread_local handle `W`, the
// per-thread `Model` node layout, and the per-context binding tables. Duplicating any of those
// per TU would silently create a SECOND set of thread_locals and a second arena owner, which is
// the one thing this split must not do.
//
// It is NOT an abstraction layer, and deliberately not one (docs/INTEL_IGPU_WORK_PACKAGES.md
// I08: "do not introduce a common polymorphic tensor class"). Everything here is verbatim the
// code that used to sit at the top of backend.cpp, moved rather than rewritten: the only edits
// are `static` -> `inline` on the leaf helpers that must keep inlining into BOTH TUs, and
// definition -> `extern` declaration on the objects that must stay single-instance. Anything
// only one TU needs stayed in that TU.
//
// Include it only from src/backends/cpu/*.cpp. It is not a public header and nothing outside
// the CPU backend may depend on it.
#pragma once

#include "sub0/core.hpp"
#include "sub0/gdn_math.hpp"        // Gated DeltaNet dims/scratch sizing (calc_act_cap)
#include "sub0/moe_math.hpp"        // moe::ExpertWeights (moe_resolve) + scratch sizing
#include "sub0/moe_quant.hpp"       // WP4e: the quantized-resident routed-expert store + pool
#include "sub0/qsa_math.hpp"        // QSA scratch sizing (calc_act_cap) + the rotary tables' own math
#include "sub0/layout.hpp"
#include "sub0/scratch_slots.hpp"   // ScratchBindings / PersistentBindings / SentinelBindings

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <mdspan>
#include <memory>
#include <print>
#include <span>
#include <utility>
#include <vector>

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
    // final residual add) -- see Model::forward()'s ngram block in backend.cpp.
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
// Defined in backend.cpp -- ONE set of arenas for the whole process, shared by both CPU-backend
// translation units. That single ownership is the reason these are declared here rather than
// duplicated: a second definition would be a second arena, not a second view of the same one.
extern std::unique_ptr<float[]> g_param_data;
extern std::unique_ptr<float[]> g_param_grad;
extern std::unique_ptr<float[]> g_param_m;
extern std::unique_ptr<float[]> g_param_vel;
void ensure_shared_params();

// --- WP4e: the quantized-resident routed experts, and the pool they are resolved through ----------
//
// docs/WP4_SCOPE.md WP4e. Under MOE_QUANT_EXPERTS the 512 routed experts per layer are NOT in
// g_param_data at all -- they sit in an S0Q1 sidecar in their own native GGUF encoding, and the ten
// EXPERTS_PER_TOK a token actually selects are dequantized into this pool immediately before
// moe::expert_ffn_row consumes them.
//
// WHY A POOL AND NOT A PER-CALL BUFFER (AGENTS.md S1). This runs per selected expert per token, so it
// is as hot as anything in this backend. One expert is 3 * D_MODEL * D_FF floats -- 18.75 MiB at the real
// axes -- which is neither a stack array nor something that may live in the DLL's static image (see
// g_param_data's own comment on SizeOfImage), so it is a lazily heap-allocated pool sized by a
// COMPILE-TIME slot count, allocated once and reused for the life of the process.
//
// WHY 8 SLOTS. Correctness needs exactly ONE: expert_ffn_row consumes a resolved expert before the next
// resolve happens, and nothing holds a pointer across calls. Every further slot is a pure cache, and it
// is a cache with a real hit rate rather than a speculative one: op_moe runs all T rows of a window
// through the SAME layer, each picking 10 of 512, so rows that agree on an expert pay for one dequant
// instead of T. 8 is chosen against EXPERTS_PER_TOK's own real value (10) -- comfortably more than the
// 1 correctness requires, and small enough (150 MiB at the real axes) to stay a rounding error against
// the 5.9 GiB of f32 the same build no longer has to hold. It is a constant here, not a CLI knob,
// because nothing consumes a knob (AGENTS.md S8) and it cannot change the answer.
inline constexpr int    MOE_RESOLVE_SLOTS      = USE_MOE_QUANT ? 8 : 1;
// gate/up are [D_MODEL, D_FF] and down is [D_FF, D_MODEL] -- the same element count either way, so one
// slot width serves all three planes.
inline constexpr size_t MOE_EXPERT_SLOT_FLOATS = static_cast<size_t>(D_MODEL) * D_FF;
using MoeExpertCache = moeq::ExpertCache<MOE_RESOLVE_SLOTS, MOE_EXPERT_SLOT_FLOATS>;
// The sidecar itself: read once by load_model, immutable thereafter (this is a forward-only build by
// construction -- FORWARD_ONLY below -- so nothing can write a routed expert), hence shared across
// threads without synchronization. The CACHE is per-Worker, because it is mutable scratch.
extern moeq::Store g_moe_quant;

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
    // WP4e: this thread's dequantize-on-demand expert pool. A member (not a global) because it is
    // MUTABLE scratch and Workers are the per-thread scratch owner here; its own storage is lazily
    // heap-allocated on first use, so a build with MOE_QUANT_EXPERTS off pays a few dozen bytes and
    // never allocates. See MOE_RESOLVE_SLOTS above for the sizing argument.
    MoeExpertCache                  moe_cache{};
    size_t act_used = 0, pool_used = 0, pcount = 0, pused = 0;
};
extern std::array<std::unique_ptr<Worker>, MAX_WORKERS> g_workers;
// nullptr until this thread runs ensure_thread_built(); a non-null W is the single
// per-thread "bound and laid out" flag (its own TLS lifetime tracks the OS thread,
// so each thread builds its g_model exactly once even if OpenMP reuses slots).
extern thread_local Worker* W;

inline std::pair<std::span<float>, std::span<float>> arena_alloc(size_t n) {
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

inline Node* mk_param(int r, int c, bool decay) {
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

inline Node* mk_node(Op op, int r, int c) {
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
inline Mat mat(std::span<float> s, int rows, int cols) {
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

// Per-context scratch-slot bindings for content-derived slot embeddings: when set, a BOUND scratch-slot
// token embeds as a function of its fragments (encode_slot) instead of a fixed reserved-id row. null =>
// today's plain tok_emb lookup, byte-for-byte. thread_local so each data-parallel train_batch worker
// carries its own window's bindings; set via set_scratch_bindings before forward/forward_one. Backward
// re-consults it (same thread, same step), so no per-node storage is needed.
extern thread_local const ScratchBindings* g_scratch_binds;

// Persistent (unbounded) slot range -- SPIKE, see scratch_slots.hpp's PersistentBindings comment.
// thread_local since 2026-07-17 (originally a plain process-global under a "set once, immutable" design):
// real spike usage swaps a DIFFERENT per-document table per training window, and train_batch's
// win_persist path installs per-window tables on every worker thread -- the same per-thread lifetime
// g_scratch_binds always had. A caller wanting the original whole-process behavior just sets it once on
// each thread that forwards (or once on the only thread, the common case).
extern thread_local const PersistentBindings* g_persistent_binds;

// Sentinel-PAIR bindings -- SPIKE, see scratch_slots.hpp's SentinelBindings comment: the token AFTER the
// sentinel embeds from the binding table keyed by that token. Same per-window thread_local lifetime as
// g_scratch_binds above.
extern thread_local const SentinelBindings* g_sentinel_binds;

// Periodic packed-content re-injection -- SPIKE, see core.hpp's set_scratch_reinject doc comment for the
// full design. stride=0 (default) leaves forward_one byte-for-byte unchanged; stride>0 re-adds a bound
// scratch slot's own packed vector (computed once at layer 0) into its hidden state every `stride`
// layers. EVAL-ONLY (forward_one), never consulted by the training graph.
extern thread_local int   g_scratch_reinject_stride;
extern thread_local float g_scratch_reinject_scale;

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
extern const QsaRopeTables g_qsa_rope;   // defined in backend.cpp, op_qsa's own TU

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

// --- Mixture of Experts: the expert resolver BOTH call sites share (docs/MOE.md) ---------------
//
// It lives in this header, and stays `inline`, precisely BECAUSE it has two call sites in two
// different translation units -- op_moe's batched loop (backend.cpp) and forward_one's decode row
// (decode.cpp). One definition, inlined into both, is what keeps WP4e's bitwise forward/forward_one
// parity a property of the code rather than of two copies staying in step.
//
// WP4e: the one resolver both MoE call sites (op_moe and forward_one's decode path) share. Under
// MOE_QUANT_EXPERTS it dequantizes expert `e` out of the sidecar into this Worker's pool; otherwise it
// simply hands back the f32 parameter nodes' own pointers. Returning a moe::ExpertWeights either way is
// what lets ONE moe::forward_row_via serve both residency strategies, which is what makes WP4e's
// bitwise gate structural (see moe_math.hpp's own comment there).
//
// The refusal is at the lowest callable seam, per this project's own rule: a decode failure means an
// unsupported GGML type or a corrupt payload, and continuing would compute with whatever the slot last
// held -- a plausible answer from the wrong weights, which is the failure mode this whole work package
// exists to avoid.
inline moe::ExpertWeights moe_resolve(Layer& L, int layer_index, int e) {
    if constexpr (USE_MOE_QUANT) {
        const auto r = W->moe_cache.resolve(g_moe_quant, layer_index, e);
        if (r.gate == nullptr) {
            std::println(stderr,
                         "fatal: could not dequantize routed expert {} of layer {} from the S0Q1 "
                         "sidecar (unsupported GGML type or corrupt payload)", e, layer_index);
            std::abort();
        }
        return {r.gate, r.up, r.down};
    } else {
        return {L.moe_gate[static_cast<std::size_t>(e)]->data.data(),
                L.moe_up[static_cast<std::size_t>(e)]->data.data(),
                L.moe_down[static_cast<std::size_t>(e)]->data.data()};
    }
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
    void build_layout();

    // Randomly initialize the SHARED weights through this thread's node layout, from a fixed seed
    // (deterministic, so every caller -- production and tests alike -- gets the identical baseline
    // model). Biases stay zero from the static arena.
    void init_weights();

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

    // WP4f's named-intermediate capture (sub0::forward_capture -- see core.hpp for why it is a sink on
    // the real loop rather than a second diagnostic copy of it). nullptr on every production path, so
    // each emit() below is one predictable never-taken branch.
    HiddenSink cap_sink = nullptr;
    void*      cap_ctx  = nullptr;
    // Model-level tensors keep their literal name; per-layer ones are formatted into a thread_local
    // buffer rather than a std::string, because this runs inside forward() and AGENTS.md S1 carves out
    // no exception for a path that only executes when a diagnostic is armed. The format string is a
    // literal at the snprintf call (not a forwarded parameter) so -Wformat-nonliteral stays quiet; 64 is
    // comfortably past the longest name emitted ("blk.<int>.attn_out").
    void emit(const Node* n, const char* name) {
        if (!cap_sink || !n) return;
        cap_sink(cap_ctx, name, n->rows, n->cols, n->data.data());
    }
    void emit_layer(const Node* n, int li, const char* suffix) {
        if (!cap_sink || !n) return;
        static thread_local char nm[64];
        std::snprintf(nm, sizeof nm, "blk.%d.%s", li, suffix);
        cap_sink(cap_ctx, nm, n->rows, n->cols, n->data.data());
    }

    Node* forward(const int* ids, int T);

    // Incremental single-token forward using the KV-cache (defined in decode.cpp, alongside the
    // KVCache it drives). Runs token `id` at window position `pos` through the network over one row,
    // updates the cache, and returns its logits [VOCAB] (thread-local). Requires kv_reset() at the
    // start of a generation and pos in [0, SEQ_LEN). Dense weights only -- callers gate on
    // !USE_TERNARY.
    const float* forward_one(int id, int pos);
};

extern thread_local Model g_model;

// Bind the calling thread to its Worker slot and lay out its parameter nodes once.
// `W` (thread_local) doubles as the guard: once non-null the thread is bound and
// g_model is populated, so the hot path early-outs without touching the OpenMP
// runtime. The shared arenas and each Worker are heap-allocated (see above); the
// reduction reads each slot's grad directly, so there is nothing to register.
void ensure_thread_built();

}  // namespace sub0
