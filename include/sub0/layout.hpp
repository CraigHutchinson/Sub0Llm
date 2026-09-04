// sub0/layout.hpp — constexpr parameter-layout descriptor for the model weights.
//
// The model's parameter tensors are laid out back-to-back in one flat float blob
// (g_param_data). Their order, shapes and offsets are 100% determined by the
// compile-time config, so we express them ONCE here as a constexpr table rather
// than re-deriving them in three places (the size calc, the node layout, and the
// future GPU backend that must slice the device blob into per-GEMM operands).
//
// This is the backend-agnostic "shape of the weights" truth. The CPU backend binds
// autograd nodes in this exact order (Model::build_layout); the on-disk checkpoint
// is this same blob; a GPU backend will mirror this blob on the device and use the
// descriptors to address each weight. Keep this table and Model::build_layout in
// lock-step: the order here IS the serialization order.

#pragma once

#include "sub0_config.hpp"  // D_MODEL, N_LAYERS, N_HEADS, SEQ_LEN, D_FF, VOCAB, D_HEAD, USE_TERNARY, USE_GATED_FFN, USE_TIED_EMBEDDINGS, USE_QK_NORM
#include "memplan.hpp"      // memplan::Dims for current_build_dims() below. memplan is
                            // dependency-free (no sub0_config.hpp, no layout.hpp), so this is one-way.
#include "gdn_math.hpp"     // sub0::gdn::Dims for GDN_DIMS below. Also dependency-free (see its own
                            // header comment) -- included here purely so GDN_DIMS can be a derived
                            // constant like GQA_GROUP/D_KV, not because gdn_math.hpp itself needs
                            // anything from this file.
#include "gated_residual_math.hpp"  // sub0::gr::Dims for GR_DIMS below -- same reasoning as gdn_math.hpp
                            // just above; dependency-free (see its own header comment).
#include "moe_math.hpp"     // sub0::moe::Dims for MOE_DIMS below -- same reasoning again; dependency-free.
#include "qsa_math.hpp"     // sub0::qsa::Dims for QSA_DIMS below -- same reasoning again; dependency-free.

#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>

namespace sub0 {

// Grouped-query attention: N_KV_HEADS key/value heads shared across N_HEADS query heads, so each KV
// head serves a group of GQA_GROUP queries. N_KV_HEADS == N_HEADS is plain MHA (GQA_GROUP == 1), the
// default -- at that setting every expression below collapses to the pre-GQA form exactly.
// D_KV is the width of the K and V projections, i.e. what replaces D_MODEL in Wk/Wv and in the KV cache.
static_assert(N_KV_HEADS >= 1, "N_KV_HEADS must be at least 1");
static_assert(N_HEADS % N_KV_HEADS == 0, "N_HEADS must be divisible by N_KV_HEADS (equal query groups)");
inline constexpr int GQA_GROUP = N_HEADS / N_KV_HEADS;
inline constexpr int D_KV      = N_KV_HEADS * D_HEAD;
static_assert(D_KV <= D_MODEL, "D_KV cannot exceed D_MODEL");

// Column layout of the CUDA backend's FUSED QKV activation buffer [rows, QKV_STRIDE], which packs the
// three projections side by side so they collapse into one GEMM. Under GQA the three sub-blocks are no
// longer equal widths -- Q is D_MODEL, K and V are D_KV -- so the old "3*D_MODEL, sub-block s at s*C"
// arithmetic no longer holds and is replaced by these named offsets. At N_KV_HEADS == N_HEADS they
// evaluate to exactly the previous 3*D_MODEL / C / 2*C, so the MHA path is unchanged.
//
// Defined HERE rather than in backend_cuda.cu because memplan.hpp must size the same buffer and cannot
// see the CUDA sources. memplan is deliberately dependency-free -- it takes EXPLICIT dims so the tuner
// and configurator can call it before a device exists -- so it cannot use these constants and carries
// runtime `Dims`-based mirrors instead (its qkv/dqkv/dwqkv/qk_pre terms). Keep the two in lock-step.
// The CPU backend does not fuse and so has no use for them.
inline constexpr int QKV_STRIDE = D_MODEL + 2 * D_KV;   // total fused row width
inline constexpr int QKV_K_OFF  = D_MODEL;              // column where the K sub-block starts
inline constexpr int QKV_V_OFF  = D_MODEL + D_KV;       // column where the V sub-block starts

// Companion stash for QK-norm's backward (the pre-norm Q/K values). Same Q|K packing as above, minus
// the V sub-block: Q at 0 (D_MODEL wide), K at D_MODEL (D_KV wide). Only the STRIDE is named here --
// the kernels that address the K sub-block are templated on their own head counts so the toy-shape
// self-test can instantiate them, and derive the offset locally; a named K offset had no consumer.
inline constexpr int QK_PRE_STRIDE = D_MODEL + D_KV;

// Under RoPE there is no position table at all: RoPE injects position inside attention, so a
// [SEQ_LEN, D_MODEL] pos_emb would be allocated, zeroed, serialized and never read. Omitting it is
// not just a saving -- it DECOUPLES SEQ_LEN from the checkpoint. With no pos_emb, no parameter
// tensor's shape depends on SEQ_LEN, so a model trained at one window loads into a binary built
// with a larger one, which is what makes long-context inference possible at all here (the window is
// otherwise compile-time everywhere: KV cache, attention score buffers, memplan). Same conditional
// shape as USE_TIED_EMBEDDINGS' lm_head/lm_bias below.
inline constexpr bool HAS_POS_EMB = (POS_ENCODING == PosEncoding::Absolute);

// --- RoPE position scaling (context extension) --------------------------------------------------
// Linear scaling divides the POSITION before the angle is formed, stretching the same rotation over a
// longer window so positions beyond the trained one stay in-distribution. Taken verbatim from the
// reference implementation rather than from recall (AGENTS.md 5) -- Nanbeige's own
// NanbeigeLinearScalingRotaryEmbedding, whose entire body is:
//     position_ids = position_ids.float() / self.scaling_factor
// Convention re-derivation against THIS engine (also AGENTS.md 5, which requires it explicitly):
//   * The reference's `dim` is the head dimension; ours is D_HEAD. Its inv_freq exponent
//     `arange(0,dim,2)/dim` equals our `-2*m/d` for m in [0, d/2). Same frequencies.
//   * The reference rotates HALF-SPLIT pairs, ours rotates INTERLEAVED pairs. That difference is real
//     but ORTHOGONAL here: scaling changes only the angle's position term, never which components
//     pair up, so it transfers unchanged.
//   * Our position variable is `t` in op_rope, `pos` in rope_row, and `m % T` in the CUDA kernels --
//     all the same quantity, so all of them scale identically.
// ROPE_POS_SCALE is the single multiplier every angle site applies. At RopeScaling::None it is
// exactly 1.0f, and IEEE-754 multiplication by 1.0 is exact, so the disabled path stays bit-identical.
enum class RopeScaling { None = 0, Linear = 1 };
inline constexpr RopeScaling ROPE_SCALING_MODE = static_cast<RopeScaling>(ROPE_SCALING);
static_assert(ROPE_SCALE_FACTOR > 0.0f, "ROPE_SCALE_FACTOR must be positive");
static_assert(ROPE_SCALING_MODE == RopeScaling::None || ROPE_SCALE_FACTOR >= 1.0f,
              "linear RoPE scaling expects a factor >= 1 (it stretches positions over a longer window)");
inline constexpr float ROPE_POS_SCALE =
    (ROPE_SCALING_MODE == RopeScaling::Linear) ? (1.0f / ROPE_SCALE_FACTOR) : 1.0f;

// --- PARTIAL ROTARY (WP4b blocker C, docs/WP4_SCOPE.md S2) ---------------------------------------
// The rotary PREFIX width: only a head vector's first ROTARY_DIM channels are rotated, the rest pass
// through untouched. The real Qwen4-preview model has partial_rotary_factor = 0.25 against head_dim
// 256, i.e. it rotates 64 of 256 channels; every build this engine has produced rotated the FULL head
// (ROTARY_DIM == D_HEAD), which is what --rotary-dim 0 still resolves to (the configurator emits the
// RESOLVED value, exactly as it does for N_KV_HEADS and GDN's own head axes).
//
// Convention note, and it is a REAL divergence that is deliberately preserved: the reference rotates
// HALF-SPLIT pairs (i, i + rotary_dim/2) while this engine's op_rope rotates INTERLEAVED pairs
// (2m, 2m+1). Partial rotary is orthogonal to that difference -- it changes only WHICH CHANNELS are
// rotated (the first ROTARY_DIM of each head, in both conventions) and the inverse-frequency
// denominator (ROTARY_DIM, matching the reference's inv_freq over rotary_dim, not head_dim). qsa_math.hpp
// takes rotary_dim as an explicit Dims field and uses the real half-split convention on the caller's
// own cos/sin tables, so a QSA layer stays the real model's layer either way.
//
// CLASSIFICATION (layout.hpp's own three-way rule): rule #2 -- computation-changing, SHAPE-NEUTRAL. It
// changes NO tensor shape at all, so PARAM_FLOATS cannot discriminate it and a checkpoint would load
// silently and compute the wrong attention. It therefore joins ARCH_FINGERPRINT2 below, in the 16 bits
// at [63:48] (D_HEAD can exceed 255 -- the real model's is 256 -- so 8 bits would not do). The field is
// canonicalized to 0 at ROTARY_DIM == D_HEAD, which is what keeps every fingerprint this engine has
// ever written bit-identical while still distinguishing every partial-rotary build.
// ROTARY_HALF is the number of interleaved PAIRS actually rotated. The truncating division is load-
// bearing and NOT a rounding accident: this engine has always supported an ODD head width (d132 H4 gives
// D_HEAD = 33, one of this project's own standard test shapes), where op_rope rotated floor(d/2) pairs
// and passed the lone trailing component through as the identity. floor(ROTARY_DIM/2) reproduces that
// exactly at ROTARY_DIM == D_HEAD, which is why ROTARY_DIM is NOT required to be even in general --
// only under QSA, whose half-split convention genuinely needs an even prefix (see USE_QSA's own
// static_asserts, which already require an even D_HEAD for the same reason).
inline constexpr int ROTARY_HALF = ROTARY_DIM / 2;
static_assert(ROTARY_DIM >= 2, "ROTARY_DIM must be at least 2 (RoPE rotates channel PAIRS)");
static_assert(ROTARY_DIM <= D_HEAD,
              "ROTARY_DIM cannot exceed D_HEAD -- the rotary prefix must fit inside the head vector "
              "it rotates (--rotary-dim 0 means the full head width)");

// --- LoopSplit: a weight-shared repeated middle block -------------------------------------------
// An un-looped head block, a MIDDLE block executed LOOP_REPEATS times reusing the SAME weights, then
// an un-looped tail block -- a refinement of "Looped Transformers" that loops only the reasoning core
// rather than the whole stack, so early feature extraction and late output layers stay single-pass.
// Buys effective depth at a fixed parameter count. LOOP_MIDDLE_LAYERS == 0 (or LOOP_REPEATS == 1) is
// OFF, and at that setting every expression below collapses to the plain 0..N_LAYERS-1 sequence.
//
// Deliberately NOT a parameter-layout change: the loop re-executes EXISTING layer indices, and
// make_param_layout() is keyed on layer index, so no new tensors and no checkpoint change. What DOES
// grow is per-EXECUTION state -- activation arena, node pool and KV-cache slots (see LOOP_EXEC_COUNT's
// use in backend_cpu.cpp).
inline constexpr bool LOOP_SPLIT_ON   = (LOOP_MIDDLE_LAYERS > 0 && LOOP_REPEATS > 1);

static_assert(LOOP_MIDDLE_LAYERS >= 0 && LOOP_MIDDLE_LAYERS <= N_LAYERS,
              "LOOP_MIDDLE_LAYERS must be within [0, N_LAYERS]");
static_assert(LOOP_REPEATS >= 1, "LOOP_REPEATS must be at least 1");
// Only meaningful when looping is actually ON: with no middle block there is no head/tail split to
// keep symmetric, and requiring it unconditionally would reject every ODD-layer model (N_LAYERS - 0
// is odd) even with the feature disabled -- which would break the "zero effect when off" contract.
static_assert(!LOOP_SPLIT_ON || (N_LAYERS - LOOP_MIDDLE_LAYERS) % 2 == 0,
              "N_LAYERS - LOOP_MIDDLE_LAYERS must be even so the head and tail blocks are symmetric");
inline constexpr int  LOOP_MID_START  = (N_LAYERS - LOOP_MIDDLE_LAYERS) / 2;
inline constexpr int  LOOP_MID_END    = LOOP_MID_START + LOOP_MIDDLE_LAYERS;
// Total layer EXECUTIONS per forward pass (== N_LAYERS when off).
inline constexpr int  LOOP_EXEC_COUNT = N_LAYERS + LOOP_MIDDLE_LAYERS * (LOOP_REPEATS - 1);

// The execution order, resolved at compile time. consteval + std::array rather than building a
// std::vector per forward() call: this is the engine's hottest path and AGENTS.md 1 forbids heap
// allocation there. Same shape as make_param_layout() below.
consteval std::array<int, LOOP_EXEC_COUNT> make_layer_execution_order() {
    std::array<int, LOOP_EXEC_COUNT> order{};
    int i = 0;
    for (int l = 0; l < LOOP_MID_START; ++l) order[i++] = l;                    // head, once
    for (int r = 0; r < LOOP_REPEATS; ++r)                                      // middle, repeated
        for (int l = LOOP_MID_START; l < LOOP_MID_END; ++l) order[i++] = l;
    for (int l = LOOP_MID_END; l < N_LAYERS; ++l) order[i++] = l;               // tail, once
    return order;
}
inline constexpr std::array<int, LOOP_EXEC_COUNT> LAYER_EXEC_ORDER = make_layer_execution_order();

// DEPTH ATTENTION -- a per-position convex re-mixing of the VALUE vector across depth, adding no
// parameters at all. Every DEPTH_ATTN_STRIDE-th execution appends its (K, V) to a per-window depth
// cache; EVERY execution then replaces its own V with a softmax-weighted mixture over the cached
// entries plus its own, the softmax running over the DEPTH axis (per batch item, KV head and position).
// K is passed through untouched, so sequence attention afterwards is unchanged. Stride 0 is off, and at
// stride 0 every expression below is inert.
//
// Because it adds no parameters, PARAM_FLOATS cannot discriminate it -- which is exactly why
// DEPTH_ATTN_STRIDE is folded into ARCH_FINGERPRINT below. See docs/DEPTH_ATTENTION.md.
inline constexpr bool USE_DEPTH_ATTN = (DEPTH_ATTN_STRIDE > 0);
// Live cache entries at the END of a forward: one per PARTICIPATING execution. The gate is on the
// EXECUTION index, not the layer index as the reference's is -- a deliberate, documented divergence
// (docs/DEPTH_ATTENTION.md 3): it makes the entry count exactly this expression and spreads the entries
// evenly across passes, and at stride 1 -- the setting the arm-D measurement runs -- the two are
// identical anyway, since then every execution participates either way.
inline constexpr int DEPTH_CACHE_MAX = USE_DEPTH_ATTN
    ? (LOOP_EXEC_COUNT + DEPTH_ATTN_STRIDE - 1) / DEPTH_ATTN_STRIDE : 0;
static_assert(DEPTH_CACHE_MAX <= LOOP_EXEC_COUNT, "depth cache cannot outnumber the executions");

// Per-execution depth-cache bookkeeping, resolved at compile time so the CPU tape, the CUDA forward and
// the CUDA backward cannot disagree about it -- they each need the SAME two numbers per execution, and
// deriving them independently at three sites is how a cross-execution index bug would get in.
//   live[e] = how many cache entries were already present when execution e ran; its mix reads exactly
//             those, plus its own (K,V) as entry live[e].
//   own[e]  = the slot execution e appends to, or -1 if it does not participate.
// Note own[e] == live[e] whenever e participates: an execution appends AFTER mixing, so it never reads
// its own slot. The backward relies on that -- see docs/DEPTH_ATTENTION.md 5b.
// Parameterised on (executions, stride) rather than reading the build's constants directly, so the
// schedule for ANY configuration -- including one this binary was not built for -- can be computed and
// asserted in a test. Without that, the only exercisable schedule is the current build's, and the shape
// a long training run actually uses goes unverified. `stride <= 0` means depth attention is off.
template <int EXECS>
struct DepthScheduleT {
    std::array<int, EXECS> live{};
    std::array<int, EXECS> own{};
    int slots = 0;               // how many cache entries exist at the END of a forward
};
template <int EXECS>
consteval DepthScheduleT<EXECS> depth_schedule_for(int stride) {
    DepthScheduleT<EXECS> d{};
    int n = 0;
    for (int e = 0; e < EXECS; ++e) {
        d.live[static_cast<std::size_t>(e)] = n;
        // `&&` short-circuits during constant evaluation, so the modulo never runs at stride 0.
        if (stride > 0 && e % stride == 0) { d.own[static_cast<std::size_t>(e)] = n; ++n; }
        else                               { d.own[static_cast<std::size_t>(e)] = -1; }
    }
    d.slots = n;
    return d;
}
using DepthSchedule = DepthScheduleT<LOOP_EXEC_COUNT>;
inline constexpr DepthSchedule DEPTH_SCHEDULE = depth_schedule_for<LOOP_EXEC_COUNT>(DEPTH_ATTN_STRIDE);
// The schedule and the sizing constant must agree, or a buffer is short by one slot -- which on the
// device is a silent out-of-bounds write, not an allocation failure.
static_assert(DEPTH_SCHEDULE.slots == DEPTH_CACHE_MAX,
              "DEPTH_CACHE_MAX must equal the number of slots depth_schedule_for() fills");
// The device kernels size their per-depth shared arrays DEPTH_CACHE_MAX + 1, and index them [0, live[e]].
// This is the invariant that makes that exactly sufficient rather than merely usually sufficient.
static_assert(DEPTH_SCHEDULE.live[LOOP_EXEC_COUNT - 1] <= DEPTH_SCHEDULE.slots,
              "a depth mix can never read more entries than the cache holds");

// GATED DELTANET -- Stage 1: CPU forward exists (op_gdn / gdn_forward_one, backend_cpu.cpp; the shared
// math core is include/sub0/gdn_math.hpp, correctness-gated against the real Qwen4-preview fixtures at
// tests/fixtures/qwen4_preview/gdn_layer0_small_* -- see tests/gdn_qwen4_fixture_tests.cpp). No backward
// pass and no CUDA implementation yet -- that is this stage's own, deliberate scope boundary (S5/S6 of
// docs/GATED_DELTANET.md), not an oversight; backend_cuda.cu keeps its own static_assert refusing to
// build a GDN CUDA binary until Stage 3 lands.
//
// GDN_FULL_ATTN_STRIDE mirrors DEPTH_ATTN_STRIDE's shape exactly: every Nth LAYER (0-indexed, N =
// stride) keeps ordinary softmax attention; every other layer is a Gated DeltaNet (linear attention,
// O(1) per token via a fixed-size recurrent state) layer. 0 = off, i.e. every layer is softmax
// attention -- the ONLY value every build before this stage could take, and still the default.
inline constexpr bool USE_GATED_DELTANET = (GDN_FULL_ATTN_STRIDE > 0);
static_assert(GDN_FULL_ATTN_STRIDE >= 0, "GDN_FULL_ATTN_STRIDE must be non-negative (0 = off)");

// The real model's own verified conv kernel size (S1a's table: linear_conv_kernel_dim = 4, both at
// full scale and in the small fixture -- not a placeholder). Deliberately NOT a CLI flag yet, per
// AGENTS.md S8 and docs/GATED_DELTANET.md S3a's own reasoning: making it sweepable is a decision to
// make WITH a working op to measure against, not before one exists -- and Stage 1 is that working op,
// but has not yet run the sweep that would justify exposing this. GDN_CONV_DIM is the depthwise conv's
// channel count -- key_dim (Q) + key_dim (K) + value_dim (V), this project's D_KV/D_KV/D_MODEL mapping
// (layout.hpp S3a's table) -- computed even when GDN is off so it stays a valid array bound (never
// zero-length -- same idiom as DEPTH_CACHE_MAX/NGRAM_TABLES_BUF above), though its VALUE is unused then.
inline constexpr int GDN_CONV_KERNEL = 4;
static_assert(GDN_CONV_KERNEL >= 2, "a causal depthwise conv needs at least a 2-tap kernel");

// --- GDN's OWN head geometry (WP4b blocker B, docs/WP4_SCOPE.md S2) ------------------------------
// Until this pass GDN_DIMS aliased GDN's linear-attention key/value head counts and head dims onto the
// ORDINARY attention axes (N_KV_HEADS / N_HEADS / D_HEAD / D_HEAD). That is a real constraint, not a
// notational one: the real Qwen4-preview model has linear_num_key_heads=16, linear_num_value_heads=48,
// linear_key_head_dim=128, linear_value_head_dim=128 against attention's own num_key_value_heads=2,
// num_attention_heads=24, head_dim=256 -- four genuinely independent axes the alias cannot express, so
// a real GDN weight transplant was impossible (the destination tensors were literally the wrong shape).
//
// Each axis is 0 = "alias to the corresponding attention axis", the same "0 means the old behaviour"
// idiom N_KV_HEADS / DEPTH_ATTN_STRIDE already use -- so every existing build resolves to exactly the
// previous constants and stays byte-identical. gdn_math.hpp's `Dims` already takes all four fields
// explicitly (its own header comment says so), so NO math changes: this is purely the binding.
//
// CLASSIFICATION (layout.hpp's own three-way rule, worked through rather than assumed):
//   * GDN_V_HEADS / GDN_V_HEAD_DIM / GDN_K_HEAD_DIM are rule #1 (shape-changing). Value heads size
//     GdnInProjB/GdnInProjA's cols and GdnALog/GdnDtBias; the value head dim sizes GdnNorm; the key
//     head dim widens key_dim, hence GdnInProjQkv and GdnConv. PARAM_FLOATS discriminates all three.
//   * GDN_K_HEADS is rule #2 (computation-changing, SHAPE-NEUTRAL) and is the one that would have been
//     mis-classified by assumption: it enters the tensor shapes ONLY through key_dim = k_heads *
//     k_head_dim, so two builds with the same PRODUCT (e.g. 2x64 and 4x32) have byte-identical
//     parameter blobs while computing a different recurrence -- rep() = v_heads / k_heads changes which
//     value heads share a key head. So it joins ARCH_FINGERPRINT2 below. (The value side does NOT need
//     to: GdnInProjB's cols are v_heads alone, so no product collision exists there.)
inline constexpr int GDN_K_HEADS    = GDN_KEY_HEADS       > 0 ? GDN_KEY_HEADS       : N_KV_HEADS;
inline constexpr int GDN_V_HEADS    = GDN_VALUE_HEADS     > 0 ? GDN_VALUE_HEADS     : N_HEADS;
inline constexpr int GDN_K_HEAD_DIM = GDN_KEY_HEAD_DIM    > 0 ? GDN_KEY_HEAD_DIM    : D_HEAD;
inline constexpr int GDN_V_HEAD_DIM = GDN_VALUE_HEAD_DIM  > 0 ? GDN_VALUE_HEAD_DIM  : D_HEAD;
static_assert(GDN_KEY_HEADS >= 0 && GDN_VALUE_HEADS >= 0 && GDN_KEY_HEAD_DIM >= 0 &&
              GDN_VALUE_HEAD_DIM >= 0, "every GDN head axis must be non-negative (0 = alias)");
static_assert(GDN_V_HEADS % GDN_K_HEADS == 0,
              "GDN's value/gate head count must be a multiple of its key head count -- gdn_math.hpp's "
              "rep() = num_v_heads / num_k_heads is the reference's repeat_interleave factor");
// GDN's key/value projection widths and the depthwise conv's channel count, now derived from GDN's OWN
// axes rather than from D_KV/D_MODEL. At the neutral (all-zero) setting GDN_KEY_DIM == D_KV and
// GDN_VALUE_DIM == N_HEADS * D_HEAD, so GDN_CONV_DIM reproduces the previous `2 * D_KV + D_MODEL`
// exactly for every build this engine has ever produced (N_HEADS * D_HEAD == D_MODEL held then).
// Computed even when GDN is off so they stay valid array bounds (same idiom as DEPTH_CACHE_MAX).
inline constexpr int GDN_KEY_DIM   = GDN_K_HEADS * GDN_K_HEAD_DIM;
inline constexpr int GDN_VALUE_DIM = GDN_V_HEADS * GDN_V_HEAD_DIM;
inline constexpr int GDN_CONV_DIM  = 2 * GDN_KEY_DIM + GDN_VALUE_DIM;

// This build's GDN dims, per S3a's mapping: hidden_size=D_MODEL (in AND out -- see gdn_math.hpp's own
// note on hidden_size playing double duty), with GDN's own four head axes above. Valid (never divides
// by zero, etc.) even when USE_GATED_DELTANET is false -- it just describes a shape nothing builds.
inline constexpr gdn::Dims GDN_DIMS{D_MODEL, GDN_K_HEADS, GDN_V_HEADS,
                                    GDN_K_HEAD_DIM, GDN_V_HEAD_DIM, GDN_CONV_KERNEL};
static_assert(GDN_DIMS.conv_dim() == GDN_CONV_DIM,
              "GDN_CONV_DIM must equal gdn::Dims::conv_dim() -- two derivations of one width");

// Per-EXECUTION single-token scratch sizes for forward_one's decode path (T == 1 always there, unlike
// op_gdn's batched, per-call arena allocation). Precomputed here (not re-derived at every forward_one
// call) so the decode row helper can size a fixed local array the same way forward_one already does for
// its other per-layer temporaries (e.g. `f1[D_FF]`) -- no heap allocation in this hot path (AGENTS.md S1).
inline constexpr std::size_t GDN_SCRATCH1 = gdn::scratch_floats(GDN_DIMS, 1);

// Per-LAYER (not per-execution) classification: a layer's weight IDENTITY decides whether it is softmax
// attention or GDN, so under LoopSplit every repeated execution of a given layer index inherits that
// layer's type automatically via LAYER_EXEC_ORDER's existing layer-index indirection -- no separate
// per-execution bookkeeping needed here, unlike DEPTH_SCHEDULE (which genuinely is per-execution because
// the depth cache is cross-EXECUTION state, not a per-layer weight property). Rule, matching the
// Qwen3-Next/Qwen3.5 hybrid pattern this is modeled on ("3x GDN -> 1x full attention", stride 4): layer
// `l` is full attention iff `l % stride == stride - 1`; every other layer is GDN. Stride 0 (off) makes
// every layer full attention, reproducing today's only architecture exactly.
// Parameterised on (layers, stride) rather than reading this build's own constants, so the schedule for
// ANY configuration -- not just the one this binary happens to be built for -- can be asserted in a test
// (the same reasoning depth_schedule_for<EXECS> above was given, and the same two-scale lesson AGENTS.md
// S7 exists to enforce: a single compiled shape cannot show whether the modulo rule behaves at an odd
// layer count or a stride that does not divide it evenly).
template <int LAYERS>
struct GdnScheduleT {
    std::array<bool, LAYERS> full_attn{};   // true = ordinary softmax attention, false = GDN
    int gdn_layers = 0;                      // how many layers are GDN (0 when stride <= 0)
};
template <int LAYERS>
consteval GdnScheduleT<LAYERS> gdn_schedule_for(int stride) {
    GdnScheduleT<LAYERS> s{};
    for (int l = 0; l < LAYERS; ++l) {
        // `&&` short-circuits during constant evaluation, so the modulo never runs at stride <= 0.
        const bool is_gdn = stride > 0 && (l % stride != stride - 1);
        s.full_attn[static_cast<std::size_t>(l)] = !is_gdn;
        if (is_gdn) ++s.gdn_layers;
    }
    return s;
}
using GdnSchedule = GdnScheduleT<N_LAYERS>;
// Consumed by Model::forward/forward_one's per-execution dispatch (backend_cpu.cpp:
// GDN_SCHEDULE.full_attn[LAYER_EXEC_ORDER[e]]) and by make_param_layout() below (which weight tensors a
// given layer gets). At stride 0 every entry is `full_attn = true`, i.e. this evaluates to nothing
// changing -- asserted below so the default build's identity is a proven invariant, not an observation.
inline constexpr GdnSchedule GDN_SCHEDULE = gdn_schedule_for<N_LAYERS>(GDN_FULL_ATTN_STRIDE);
static_assert(GDN_FULL_ATTN_STRIDE > 0 || GDN_SCHEDULE.gdn_layers == 0,
              "GDN_FULL_ATTN_STRIDE == 0 must yield zero GDN layers");

// GATED RESIDUAL (hyper-connections) -- Stage 1: CPU forward exists (op_gr_tile/op_gr_mix/op_gr_gate/
// op_gr_combine, backend_cpu.cpp; the shared math core is include/sub0/gated_residual_math.hpp,
// correctness-gated against the real Qwen4-preview fixtures at
// tests/fixtures/qwen4_preview/gated_residual_layer0_small_* -- see
// tests/gated_residual_qwen4_fixture_tests.cpp). No backward pass yet -- this stage's own, deliberate
// scope boundary (docs/GATED_RESIDUAL.md S6), not an oversight.
// HC_COUNT parallel residual streams read/write-gated per sub-block (Qwen4-preview's own
// Qwen4ExpTextGatedResidual, "Gated Residual (GR)", tech_report.pdf S2.2). 0 = off, the only value every
// build before this stage could take, and still the default.
inline constexpr bool USE_GATED_RESIDUAL = (HC_COUNT >= 2);
static_assert(HC_COUNT == 0 || HC_COUNT >= 2,
              "HC_COUNT must be 0 (off) or a real hyper-connection stream count >= 2 -- see docs/GATED_RESIDUAL.md");
static_assert((HC_COUNT >= 2) == (HC_LOWRANK >= 1),
              "HC_COUNT and HC_LOWRANK must be on or off together (HC_COUNT >= 2 iff HC_LOWRANK >= 1) "
              "-- see docs/GATED_RESIDUAL.md");

// The width `h` (the residual stream) actually has. At the neutral setting this IS D_MODEL, so every
// existing [T,D_MODEL]-shaped expression in Model::forward/forward_one that is not explicitly wrapped in
// a GR branch continues to typecheck and compute identically -- see docs/GATED_RESIDUAL.md S2.
inline constexpr int HC_WIDE = USE_GATED_RESIDUAL ? HC_COUNT * D_MODEL : D_MODEL;

// This build's GR dims, per docs/GATED_RESIDUAL.md S4a's mapping. Valid (never divides by zero) even
// when USE_GATED_RESIDUAL is false -- same "describes a shape nothing builds" idiom GDN_DIMS/
// DEPTH_CACHE_MAX already use above. HC_COUNT_BUF/HC_LOWRANK_BUF are the never-zero array-bound forms
// (same idiom as NGRAM_TABLES_BUF) for scratch sizing below, since a zero-sized quantity would make
// *_scratch_floats() degenerate even though nothing ever reads that scratch when GR is off.
inline constexpr int HC_COUNT_BUF   = USE_GATED_RESIDUAL ? HC_COUNT   : 1;
inline constexpr int HC_LOWRANK_BUF = USE_GATED_RESIDUAL ? HC_LOWRANK : 1;
inline constexpr gr::Dims GR_DIMS{D_MODEL, HC_COUNT, HC_LOWRANK};
// Same shape as GR_DIMS but never degenerate (hc_count/hc_lowrank >= 1) -- used only to size the
// per-execution scratch arrays below, which must be valid array bounds even when GR is off.
inline constexpr gr::Dims GR_DIMS_BUF{D_MODEL, HC_COUNT_BUF, HC_LOWRANK_BUF};

// Per-call scratch sizes for forward_one's decode-path (T==1) op_gr_mix/op_gr_gate counterparts,
// precomputed here rather than re-derived at every call -- same reasoning as GDN_SCRATCH1.
// GR_NORMED_SCRATCH1: the hc_norm() output buffer both op_gr_mix and op_gr_gate need (each calls
// hc_norm() independently on their own weights, docs/GATED_RESIDUAL.md S4c). GR_MIX_SCRATCH1: mix()'s
// OWN scratch need on top of that (the down-projection's pre-activation) -- see gated_residual_math.hpp.
inline constexpr std::size_t GR_NORMED_SCRATCH1 = gr::normed_scratch_floats(GR_DIMS_BUF, 1);
inline constexpr std::size_t GR_MIX_SCRATCH1    = gr::mix_scratch_floats(GR_DIMS_BUF, 1);

// Exact PARAM_FLOATS delta a GR-on build adds over an otherwise-identical GR-off build, per
// docs/GATED_RESIDUAL.md S3b -- a pure function of explicit parameters (not closed over this build's own
// constants), same reasoning as ngram_num_embedders()/gdn_schedule_for<LAYERS>() above: lets a test
// exercise hypothetical (n_layers, d_model, hc_count, hc_lowrank) combinations regardless of what this
// binary happens to be compiled for, and (once Stage 1 wires PARAM_LAYOUT) lets make_param_layout()'s
// own per-layer append logic be checked against this SAME closed form rather than trusting it by
// construction. Two full per-layer GatedResidual instances (attn_hyper_connection, mlp_hyper_connection,
// each WITH block_inject) plus one top-level instance (hyper_connection_mixer, WITHOUT block_inject).
// Zero at hc_count < 2 (matching USE_GATED_RESIDUAL's own gate) and strictly positive at hc_count >= 2,
// hc_lowrank >= 1, d_model >= 1, n_layers >= 1 -- every term is a PRODUCT of positive quantities, so no
// cancellation with any other axis is possible (contrast GDN's own mixed-sign Delta, S3c).
inline constexpr long long gr_param_delta(int n_layers, int d_model, int hc_count, int hc_lowrank) {
    if (hc_count < 2) return 0;
    const long long wide = static_cast<long long>(hc_count) * d_model;
    const long long per_instance_with_inject = wide + 2 * wide * hc_lowrank + wide * hc_count;
    const long long top_instance             = wide + 2 * wide * hc_lowrank;
    return static_cast<long long>(n_layers) * 2 * per_instance_with_inject + top_instance;
}

// MIXTURE OF EXPERTS -- Stage 0: config skeleton, hard-gated off (docs/MOE.md). NUM_EXPERTS routed
// experts + 1 always-on shared expert per layer, EXPERTS_PER_TOK selected per token (Qwen4-preview's own
// Qwen4ExpTextSparseMoeBlock). Unlike Gated DeltaNet (a per-LAYER schedule, some layers softmax attention
// some GDN) this mechanism replaces the FFN block for EVERY layer uniformly when on -- there is no
// per-layer schedule to derive, only a single on/off gate. 0 = off, the only value every build before
// this stage could take, and still the default.
inline constexpr bool USE_MOE = (NUM_EXPERTS >= 2);
static_assert(NUM_EXPERTS == 0 || NUM_EXPERTS >= 2,
              "NUM_EXPERTS must be 0 (off) or a real routed-expert count >= 2 -- see docs/MOE.md");
static_assert((NUM_EXPERTS >= 2) == (EXPERTS_PER_TOK >= 1),
              "NUM_EXPERTS and EXPERTS_PER_TOK must be on or off together (NUM_EXPERTS >= 2 iff "
              "EXPERTS_PER_TOK >= 1) -- see docs/MOE.md");
static_assert(EXPERTS_PER_TOK <= (NUM_EXPERTS == 0 ? 0 : NUM_EXPERTS),
              "EXPERTS_PER_TOK cannot exceed NUM_EXPERTS -- see docs/MOE.md");
static_assert(EXPERTS_PER_TOK <= moe::TOPK_MAX,
              "EXPERTS_PER_TOK exceeds moe_math.hpp's own TOPK_MAX stack-buffer cap -- see its header comment");

// Never-zero array-bound forms (same idiom as HC_COUNT_BUF/NGRAM_TABLES_BUF above) -- valid, non-
// degenerate shapes even when USE_MOE is false, so downstream code never needs a separate guard just to
// declare an array of this width.
inline constexpr int NUM_EXPERTS_BUF = USE_MOE ? NUM_EXPERTS : 1;
// MoE's per-expert (and shared-expert) SwiGLU intermediate width. Reuses D_FF -- this project's own
// existing FFN-width knob -- rather than introducing a separate CLI axis: the real model's own
// moe_intermediate_size (640) and shared_expert_intermediate_size (640) are already equal
// (docs/QWEN4_MEMORY_ORCHESTRATION.md S2a), so one shared width is real-model-faithful, not an
// approximation, and AGENTS.md S8 disfavors a speculative extra knob nothing yet needs.
inline constexpr moe::Dims MOE_DIMS{D_MODEL, D_FF, NUM_EXPERTS, EXPERTS_PER_TOK};
// forward_one's decode-path scratch need, precomputed here (not re-derived at every call) -- same
// reasoning as GDN_SCRATCH1/GR_NORMED_SCRATCH1. Not scaled by T (moe_math.hpp's own scratch_floats() is
// row-independent, docs/MOE.md S2) and never zero even when USE_MOE is false (D_FF/D_MODEL are always
// >= 1), so no separate "_BUF" never-degenerate form is needed the way HC_COUNT_BUF/NUM_EXPERTS_BUF are.
inline constexpr std::size_t MOE_SCRATCH1 = moe::scratch_floats(MOE_DIMS);

// Exact PARAM_FLOATS delta a MoE-on build adds over an otherwise-identical MoE-off build, per
// docs/MOE.md S3b -- a pure function of explicit parameters (not closed over this build's own constants),
// same reasoning as gr_param_delta()/gdn_schedule_for<LAYERS>() above. UNLIKE Gated Residual's own
// gr_param_delta (a pure ADDITION on top of an unchanged existing tensor set), MoE REPLACES the FFN
// block's existing tensors -- so this is `moe_layer_floats - dense_ffn_floats`, not merely `+something`.
// `d_ff` is the FFN width whichever variant is active at those dims (this project's own d_ff_for()); the
// dense side must reflect whether the config being replaced was gated (3 tensors) or plain (4 tensors,
// including 2 bias vectors) SO THE CALLER PASSES THE RIGHT ONE -- see layout_tests.cpp's own call site.
// Strictly positive whenever num_experts >= 2 (this function's own "on" precondition): even the cheapest
// possible replacement (num_experts=2) needs 2*3 expert tensors + 3 shared-expert tensors + 1 router +
// 1 shared-gate tensor, each sized D_MODEL*d_ff or larger, which already exceeds the gated dense FFN's own
// 3 D_MODEL*d_ff tensors -- worked through explicitly in the final report's own arithmetic, not assumed.
inline constexpr long long moe_param_delta(int n_layers, int d_model, int d_ff, int num_experts,
                                            int experts_per_tok, bool was_gated_ffn) {
    if (num_experts < 2) return 0;
    (void)experts_per_tok;   // EXPERTS_PER_TOK never changes a tensor SHAPE -- see ARCH_FINGERPRINT2 below
    const long long router       = static_cast<long long>(d_model) * num_experts;
    const long long per_expert   = 2LL * d_model * d_ff + static_cast<long long>(d_ff) * d_model;  // gate+up+down
    const long long shared       = 2LL * d_model * d_ff + static_cast<long long>(d_ff) * d_model;  // shared FFN
    const long long shared_gate  = d_model;                                                        // [d_model,1]
    const long long moe_layer_floats = router + static_cast<long long>(num_experts) * per_expert + shared + shared_gate;
    const long long dense_ffn_floats = was_gated_ffn
        ? 3LL * d_model * d_ff                                    // Wg, W1, W2 (SwiGLU, no bias)
        : 2LL * static_cast<long long>(d_model) * d_ff + d_ff + d_model;  // W1,B1,W2,B2 (plain GELU+bias)
    return static_cast<long long>(n_layers) * (moe_layer_floats - dense_ffn_floats);
}

// QWEN SPARSE ATTENTION (QSA) + lightning indexer -- Stage 0: config skeleton, hard-gated off
// (docs/QSA.md). A QSA layer is the real model's `full_attention` layer type: an ordinary gated softmax
// attention sublayer whose visible-token set is first NARROWED, per query, by a small "lightning indexer"
// that scores mean-pooled key BLOCKS and keeps only the top `indexer_budget / indexer_compress_ratio` of
// them (plus the always-visible incomplete tail) -- Qwen4-preview's own Qwen4ExpTextQSAIndexer +
// Qwen4ExpTextAttention.
//
// Five new axes, all on or off TOGETHER. USE_QSA is the single gate every `if constexpr` keys off.
// NOTE what is deliberately NOT here: num_attention_heads / head_dim / num_key_value_heads are this
// project's EXISTING N_HEADS / D_HEAD / N_KV_HEADS (docs/QSA.md S3a/S4a) -- adding duplicates would be
// unconsumed surface (AGENTS.md S8) and a second source of truth for one axis, the exact hazard
// current_build_dims() exists to prevent. Same call docs/MOE.md S3a made for D_FF.
inline constexpr bool USE_QSA = (QSA_INDEXER_N_HEADS >= 1 && QSA_INDEXER_BUDGET >= 1);
// All five axes are on or off TOGETHER -- a half-configured QSA build cannot compile (the same
// "guard at the lowest callable seam" pattern HC_COUNT/HC_LOWRANK and NUM_EXPERTS/EXPERTS_PER_TOK
// already use, generalized to five). Stage 1 relaxed Stage 0's hard clamp to these real ranges.
static_assert(((QSA_INDEXER_N_HEADS != 0) + (QSA_INDEXER_KV_HEADS != 0) + (QSA_INDEXER_HEAD_DIM != 0) +
               (QSA_INDEXER_BUDGET != 0) + (QSA_INDEXER_COMPRESS_RATIO != 0)) % 5 == 0,
              "every QSA_INDEXER_* axis must be set together or all left 0 -- see docs/QSA.md S4a");
static_assert(!USE_QSA || QSA_INDEXER_KV_HEADS == 1,
              "QSA_INDEXER_KV_HEADS must be exactly 1 when QSA is on -- the reference's own "
              "token_k.reshape(...).squeeze(2) requires it (docs/QSA.md S1a)");
static_assert(!USE_QSA || (QSA_INDEXER_HEAD_DIM >= 2 && QSA_INDEXER_HEAD_DIM % 2 == 0),
              "QSA_INDEXER_HEAD_DIM must be even and >= 2 (the half-split rotary needs an even prefix)");
static_assert(!USE_QSA || QSA_INDEXER_BUDGET >= QSA_INDEXER_COMPRESS_RATIO,
              "QSA_INDEXER_BUDGET must be >= QSA_INDEXER_COMPRESS_RATIO, or block_topk would be 0");
static_assert(!USE_QSA || D_HEAD % 2 == 0,
              "QSA needs an even D_HEAD (the half-split rotary rotates D_HEAD/2 channel pairs)");
static_assert(!USE_QSA || ROTARY_DIM % 2 == 0,
              "QSA needs an even ROTARY_DIM: the half-split rotary pairs channel i with i + rotary_dim/2, "
              "so an odd prefix would leave one channel unpaired -- see qsa_math.hpp's rope_apply_row");
// The rotary prefix cannot be wider than the vector it rotates. The SAME cos/sin rotate BOTH the
// attention heads and the indexer's OWN idx_head_dim-wide query and pooled block keys (the reference
// reuses one cos/sin for both -- docs/QSA.md S1b), so the indexer's head width must be at least the
// rotary prefix. Violating it is a silent OUT-OF-BOUNDS WRITE past the end of a per-head slice -- found
// exactly that way in WP3 (see qsa_math.hpp's rope_apply_row precondition comment and docs/QSA.md S10).
//
// This used to read `>= D_HEAD`, because the engine's rotary prefix WAS D_HEAD (docs/QSA.md S2b.2's
// documented simplification). WP4b blocker C made the prefix its own axis, so the real relationship is
// against ROTARY_DIM -- and the change is not cosmetic: the real model has rotary_dim 64 <=
// indexer_head_dim 128 <= head_dim 256, so the OLD form would REJECT the real configuration outright
// (128 >= 256 is false). Relaxing to ROTARY_DIM is exactly what makes the real axes expressible, and it
// is still the tight bound -- ROTARY_DIM is what rope_apply_row actually indexes.
static_assert(!USE_QSA || QSA_INDEXER_HEAD_DIM >= ROTARY_DIM,
              "QSA_INDEXER_HEAD_DIM must be >= ROTARY_DIM: the engine's rotary prefix is ROTARY_DIM wide "
              "and the SAME cos/sin rotate the indexer's own idx_head_dim-wide vectors -- docs/QSA.md S2b.2");

// Never-zero array-bound / never-divide-by-zero forms (same idiom as HC_COUNT_BUF/NUM_EXPERTS_BUF above).
inline constexpr int QSA_IDX_N_HEADS_BUF = USE_QSA ? QSA_INDEXER_N_HEADS       : 1;
inline constexpr int QSA_IDX_KV_HEADS_BUF = USE_QSA ? QSA_INDEXER_KV_HEADS     : 1;
inline constexpr int QSA_IDX_HEAD_DIM_BUF = USE_QSA ? QSA_INDEXER_HEAD_DIM     : 1;
inline constexpr int QSA_BUDGET_BUF       = USE_QSA ? QSA_INDEXER_BUDGET       : 1;
inline constexpr int QSA_RATIO_BUF        = USE_QSA ? QSA_INDEXER_COMPRESS_RATIO : 1;
// Total output width of the indexer's single fused index_qk_proj: (n_heads + kv_heads) * head_dim.
inline constexpr int QSA_IDX_QK_OUT = (QSA_INDEXER_N_HEADS + QSA_INDEXER_KV_HEADS) * QSA_INDEXER_HEAD_DIM;

// This build's QSA dims, per docs/QSA.md S2b/S4a's mapping. head_dim = D_HEAD was WP3's one remaining
// engine-side simplification and rotary_dim = D_HEAD was the other; blocker C above turns the second
// into a real axis (ROTARY_DIM), and blocker A does the same for the first. qsa_math.hpp always took
// both as independent Dims fields, which is why its fixture already ran at the real model's non-dividing
// head_dim=256 and partial rotary_dim=64. Valid (never divides by zero) even when USE_QSA is false --
// same "describes a shape nothing builds" idiom GDN_DIMS/GR_DIMS/MOE_DIMS already use.
inline constexpr qsa::Dims QSA_DIMS{D_MODEL, N_HEADS, D_HEAD, N_KV_HEADS,
                                    QSA_INDEXER_N_HEADS, QSA_INDEXER_KV_HEADS, QSA_INDEXER_HEAD_DIM,
                                    QSA_INDEXER_BUDGET, QSA_INDEXER_COMPRESS_RATIO, ROTARY_DIM};
// Same shape but never degenerate (every indexer width >= 1) -- used ONLY to size the fixed decode-path
// scratch arrays below, which must be valid array bounds even when QSA is off.
inline constexpr qsa::Dims QSA_DIMS_BUF{D_MODEL, N_HEADS, D_HEAD, N_KV_HEADS,
                                        QSA_IDX_N_HEADS_BUF, QSA_IDX_KV_HEADS_BUF, QSA_IDX_HEAD_DIM_BUF,
                                        QSA_BUDGET_BUF, QSA_RATIO_BUF, ROTARY_DIM};

// Per-LAYER three-way mixer classification (docs/QSA.md S2). QSA is NOT a new schedule axis: in the real
// model every `full_attention` layer IS a QSA layer (Qwen4ExpTextAttention unconditionally constructs and
// calls its own indexer -- there is no full-attention layer without one), and the real config.json's own
// `layer_types` array is bit-for-bit gdn_schedule_for(4). So the three-way choice is DERIVED from the two
// existing axes rather than given a redundant third one that could express layer sets the real model
// cannot have. Parameterised on (layers, gdn_stride, qsa_on) rather than reading this build's constants,
// for the same reason gdn_schedule_for<LAYERS>/depth_schedule_for<EXECS> are: AGENTS.md S7's odd-layer-
// count/non-dividing-stride cases must be assertable without compiling those builds.
enum class LayerMixer : unsigned char { Attn = 0, Gdn = 1, Qsa = 2 };
template <int LAYERS>
consteval std::array<LayerMixer, LAYERS> qsa_schedule_for(int gdn_stride, bool qsa_on) {
    std::array<LayerMixer, LAYERS> s{};
    for (int l = 0; l < LAYERS; ++l) {
        const bool is_gdn = gdn_stride > 0 && (l % gdn_stride != gdn_stride - 1);
        s[static_cast<std::size_t>(l)] = is_gdn ? LayerMixer::Gdn
                                                : (qsa_on ? LayerMixer::Qsa : LayerMixer::Attn);
    }
    return s;
}
inline constexpr std::array<LayerMixer, N_LAYERS> MIXER_SCHEDULE =
    qsa_schedule_for<N_LAYERS>(GDN_FULL_ATTN_STRIDE, USE_QSA);
// The invariant that keeps MIXER_SCHEDULE and the pre-existing GDN_SCHEDULE from ever drifting apart:
// Qsa is definitionally "full attention AND QSA on", Gdn is definitionally "not full attention".
consteval bool mixer_schedule_agrees_with_gdn() {
    for (int l = 0; l < N_LAYERS; ++l) {
        const bool full = GDN_SCHEDULE.full_attn[static_cast<std::size_t>(l)];
        const LayerMixer m = MIXER_SCHEDULE[static_cast<std::size_t>(l)];
        if (full != (m != LayerMixer::Gdn)) return false;
        if (full && (m == LayerMixer::Qsa) != USE_QSA) return false;
    }
    return true;
}
static_assert(mixer_schedule_agrees_with_gdn(),
              "MIXER_SCHEDULE must agree with GDN_SCHEDULE.full_attn on every layer -- see docs/QSA.md S2");

// Exact PARAM_FLOATS delta a QSA-on build adds over an otherwise-identical QSA-off build, per
// docs/QSA.md S3b -- a pure function of explicit parameters (not closed over this build's own constants),
// same reasoning as gr_param_delta()/moe_param_delta()/gdn_schedule_for<LAYERS>() above. UNLIKE Gated
// Residual's own delta (a pure addition), this is a REPLACEMENT of the softmax-attention layer's own
// Wq/Wk/Wv/Wo[+QNorm/KNorm] -- and it applies only to the FULL-ATTENTION layers, so the GDN stride is an
// input. Strictly positive whenever qsa_on and n_layers >= 1: the D_MODEL*D_MODEL gate projection alone
// (which plain attention does not have at all) already dominates, and no term can cancel.
inline constexpr long long qsa_param_delta(int n_layers, int gdn_stride, int d_model, int d_kv,
                                            int d_head, bool qk_norm, int idx_n_heads, int idx_kv_heads,
                                            int idx_head_dim, bool qsa_on) {
    if (!qsa_on) return 0;
    int full_attn_layers = 0;
    for (int l = 0; l < n_layers; ++l)
        if (!(gdn_stride > 0 && (l % gdn_stride != gdn_stride - 1))) ++full_attn_layers;
    const long long idx_qk_out = static_cast<long long>(idx_n_heads + idx_kv_heads) * idx_head_dim;
    const long long attn_layer = 2LL * d_model * d_model + 2LL * d_model * d_kv
                               + (qk_norm ? 2LL * d_head : 0);
    const long long qsa_layer  = 3LL * d_model * d_model + 2LL * d_model * d_kv + 2LL * d_head
                               + static_cast<long long>(d_model) * idx_qk_out + 2LL * idx_head_dim;
    return static_cast<long long>(full_attn_layers) * (qsa_layer - attn_layer);
}

// --- N-GRAM EMBEDDINGS -- additional per-position features from rolling polynomial-hash token n-grams,
// added into the input embedding (Nanbeige's `NanbeigeNgramEmbedding`, "concat" fusion mode -- see
// docs/NGRAM_EMBEDDING.md). NGRAM_MAX_N is the highest n-gram order used (bigrams..NGRAM_MAX_N-grams);
// 0 (or < 2) is off, and at that setting every expression below is inert and contributes ZERO new
// parameters. Unlike depth attention, this axis DOES change PARAM_FLOATS (it adds real embedding-table
// and projection weights) -- classified under layout.hpp's own ARCH_FINGERPRINT rule #1 ("changes a
// tensor shape -> PARAM_FLOATS already discriminates it, nothing to do"), the same precedent as GQA's
// D_KV narrowing. So NGRAM_MAX_N/NGRAM_TABLES_PER_ORDER/NGRAM_TABLE_SIZE do NOT join ARCH_FINGERPRINT --
// but they DO join MODEL_ARCH_ID below, which (per its own doc) covers every axis, shape-changing and
// computation-changing alike, unconditionally (the same way DEPTH_ATTN_STRIDE's mix there already
// changes MODEL_ARCH_ID for every build regardless of its value -- an accepted, precedented cost:
// MODEL_ARCH_ID is a diagnostic/directory-naming identity, not the checkpoint-format gate).
inline constexpr bool NGRAM_EMBED = (NGRAM_MAX_N >= 2);
static_assert(NGRAM_MAX_N == 0 || NGRAM_MAX_N >= 2,
              "NGRAM_MAX_N must be 0 (off) or a real highest order >= 2 (bigrams)");
static_assert(NGRAM_TABLES_PER_ORDER >= 1, "NGRAM_TABLES_PER_ORDER must be at least 1");

// num_embedders = k tables x (n-1) orders (bigram..n-gram) -- the reference's `k * (n - 1)`. A free
// function of explicit parameters (not folded straight into a consteval array-builder closed over this
// build's own constants) so a test can exercise arbitrary hypothetical (max_n, tables_per_order) pairs
// regardless of what THIS build was configured with -- the same reason DepthScheduleT's
// depth_schedule_for() above takes its parameters explicitly instead of reading the build's constants.
inline constexpr int ngram_num_embedders(int max_n, int tables_per_order) {
    return max_n >= 2 ? tables_per_order * (max_n - 1) : 0;
}
inline constexpr int NGRAM_NUM_EMBEDDERS = ngram_num_embedders(NGRAM_MAX_N, NGRAM_TABLES_PER_ORDER);
static_assert(!NGRAM_EMBED || D_MODEL % NGRAM_NUM_EMBEDDERS == 0,
              "D_MODEL must be divisible by NGRAM_TABLES_PER_ORDER * (NGRAM_MAX_N - 1) so each "
              "table's embedding slice (D_MODEL / num_embedders) is an integer width -- the concat-mode "
              "invariant emb_dim * num_embedders == ngram_hidden_size (here pinned to D_MODEL)");
inline constexpr int NGRAM_EMB_DIM = NGRAM_EMBED ? D_MODEL / NGRAM_NUM_EMBEDDERS : 0;
// Never zero-length: a 0-sized std::array is invalid, and both buffer this even when the feature is off
// (same idiom as DEPTH_CACHE_MAX / forward_one's dep_k/dep_v arrays above).
inline constexpr int NGRAM_TABLES_BUF    = NGRAM_NUM_EMBEDDERS ? NGRAM_NUM_EMBEDDERS : 1;
inline constexpr int NGRAM_MAX_SHIFT_BUF = NGRAM_EMBED ? NGRAM_MAX_N - 1 : 1;   // shifts the HIGHEST order needs

// Per-table vocab size: `m + index*2 + 1` -- the reference's `_ngram_embedding_vocab_sizes` NON-prime-
// forced branch (config default `ngram_mod_force_prime=False`; the prime-forcing branch is not
// implemented here -- deferred, see docs/NGRAM_EMBEDDING.md's scope note).
inline constexpr int ngram_vocab_dim(int table_size, int index) { return table_size + index * 2 + 1; }
// Which highest n-gram order (2..max_n) embedder `index` belongs to -- the reference's
// `index = (i - 2) * k + j`, inverted.
inline constexpr int ngram_order_of(int index, int tables_per_order) { return 2 + index / tables_per_order; }
// The polynomial-hash base -- the reference's `_ngram_hash_base(vocab_size, force_prime=False)`, whose
// non-force-prime branch is simply the token vocabulary size.
inline constexpr int NGRAM_HASH_BASE = VOCAB;
// The multiplier for a context token `shift` positions back (shift = 1..order-1): `power_mod =
// (power_mod * hash_base) % vocab_dim`, accumulated -- the reference's `_precompute_vocab_mods`.
// int64_t: hash_base * vocab_dim can briefly exceed int32 range before the mod reduces it.
inline constexpr std::int64_t ngram_vocab_mod(int table_size, int index, int hash_base, int shift) {
    const int vocab_dim = ngram_vocab_dim(table_size, index);
    std::int64_t power = 1;
    for (int s = 0; s < shift; ++s) power = (power * hash_base) % vocab_dim;
    return power;
}

consteval std::array<int, NGRAM_TABLES_BUF> make_ngram_vocab_dims() {
    std::array<int, NGRAM_TABLES_BUF> out{};
    for (int e = 0; e < NGRAM_NUM_EMBEDDERS; ++e) out[e] = ngram_vocab_dim(NGRAM_TABLE_SIZE, e);
    return out;
}
inline constexpr std::array<int, NGRAM_TABLES_BUF> NGRAM_VOCAB_DIMS = make_ngram_vocab_dims();

consteval std::array<int, NGRAM_TABLES_BUF> make_ngram_orders() {
    std::array<int, NGRAM_TABLES_BUF> out{};
    for (int e = 0; e < NGRAM_NUM_EMBEDDERS; ++e) out[e] = ngram_order_of(e, NGRAM_TABLES_PER_ORDER);
    return out;
}
inline constexpr std::array<int, NGRAM_TABLES_BUF> NGRAM_ORDERS = make_ngram_orders();

// vocab_mods[e][s], s = 0..(order(e)-2): the multiplier for the context token `s+1` positions back.
// Precomputed once at compile time (order/vocab_dim/hash_base are all constexpr for this build) rather
// than recomputed every forward() call.
consteval std::array<std::array<std::int64_t, NGRAM_MAX_SHIFT_BUF>, NGRAM_TABLES_BUF> make_ngram_vocab_mods() {
    std::array<std::array<std::int64_t, NGRAM_MAX_SHIFT_BUF>, NGRAM_TABLES_BUF> out{};
    for (int e = 0; e < NGRAM_NUM_EMBEDDERS; ++e) {
        const int order = NGRAM_ORDERS[static_cast<std::size_t>(e)];
        for (int s = 0; s < order - 1; ++s)
            out[e][s] = ngram_vocab_mod(NGRAM_TABLE_SIZE, e, NGRAM_HASH_BASE, s + 1);
    }
    return out;
}
inline constexpr std::array<std::array<std::int64_t, NGRAM_MAX_SHIFT_BUF>, NGRAM_TABLES_BUF>
    NGRAM_VOCAB_MODS = make_ngram_vocab_mods();

// ARCHITECTURE FINGERPRINT -- checkpoint identity for axes that PARAM_FLOATS cannot see.
//
// The Header and PARAM_FLOATS between them discriminate every axis that changes a tensor SHAPE. They
// discriminate nothing that changes what the model COMPUTES while leaving shapes alone, and weights are
// meaningless under the wrong computation. Two such axes exist today and both were silently loadable
// before this fingerprint:
//   * LoopSplit (LOOP_MIDDLE_LAYERS / LOOP_REPEATS) -- re-executes EXISTING layer indices, so a looped
//     and un-looped build have byte-identical parameter blobs.
//   * ROPE_THETA -- changes every rotation angle, adds no parameter. A model trained at 10000 loaded
//     into a 500000 build produced garbage with no diagnostic.
//
// RULE FOR ANY NEW COMPILE-TIME AXIS -- classify it into exactly one of three, and say which:
//   1. Changes a tensor shape        -> PARAM_FLOATS already discriminates it. Nothing to do. (GQA is
//      here: D_KV narrows Wk/Wv, strictly monotonically in N_KV_HEADS, so no collision is possible.)
//   2. Changes computation, not shape -> ADD IT HERE, or it loads silently and computes the wrong thing.
//   3. Deliberately variable between train and inference -> EXCLUDE, and record why. ROPE_SCALING /
//      ROPE_SCALE_FACTOR are the only members: training short and running long WITH scaling on is the
//      intended workflow, so guarding them would break the feature that motivated them.
//
// Packed rather than hashed so a mismatch can name the actual values -- a fingerprint you cannot decode
// makes for a diagnostic that says only "different", which is what sent the LoopSplit case unnoticed in
// the first place. float is bit_cast, so the comparison is exact and no tolerance question arises.
// Field layout, high to low:  [63:56] depth_attn_stride | [55:48] middle_layers | [47:32] repeats |
// [31:0] rope_theta. depth_attn_stride was added (2026-07-29) into the byte middle_layers was never
// able to reach: it occupies [55:48] in practice because a model with 256+ layers does not exist and
// is now asserted against. That placement is deliberate -- with the stride at its default 0, every
// fingerprint this function has ever produced is reproduced BIT-IDENTICALLY, so no existing checkpoint
// is invalidated by gaining the new axis. Any non-zero stride yields a distinct fingerprint, which is
// the point: depth attention changes computation while leaving PARAM_FLOATS untouched (it adds no
// parameters at all), so nfloat cannot catch a cross-load and this is the only thing that can.
static_assert(LOOP_MIDDLE_LAYERS >= 0 && LOOP_MIDDLE_LAYERS <= 0xff, "LOOP_MIDDLE_LAYERS must fit 8 bits");
static_assert(LOOP_REPEATS       >= 0 && LOOP_REPEATS       <= 0xffff, "LOOP_REPEATS must fit 16 bits");
static_assert(DEPTH_ATTN_STRIDE  >= 0 && DEPTH_ATTN_STRIDE  <= 0xff, "DEPTH_ATTN_STRIDE must fit 8 bits");
inline constexpr std::uint64_t arch_fingerprint(int middle_layers, int repeats, float rope_theta,
                                                int depth_attn_stride = 0) {
    return (static_cast<std::uint64_t>(static_cast<unsigned>(depth_attn_stride) & 0xffu)   << 56)
         | (static_cast<std::uint64_t>(static_cast<unsigned>(middle_layers)     & 0xffu)   << 48)
         | (static_cast<std::uint64_t>(static_cast<unsigned>(repeats)           & 0xffffu) << 32)
         |  static_cast<std::uint64_t>(std::bit_cast<std::uint32_t>(rope_theta));
}
struct ArchAxes { int middle_layers; int repeats; float rope_theta; int depth_attn_stride; };
inline constexpr ArchAxes arch_axes_of(std::uint64_t fp) {
    return ArchAxes{ static_cast<int>((fp >> 48) & 0xffu),
                     static_cast<int>((fp >> 32) & 0xffffu),
                     std::bit_cast<float>(static_cast<std::uint32_t>(fp & 0xffffffffu)),
                     static_cast<int>((fp >> 56) & 0xffu) };
}
inline constexpr std::uint64_t ARCH_FINGERPRINT =
    arch_fingerprint(LOOP_MIDDLE_LAYERS, LOOP_REPEATS, ROPE_THETA, DEPTH_ATTN_STRIDE);
// What a file carrying NO fingerprint must be assumed to have been built with. Such a file predates the
// record, so it is necessarily un-looped -- that part is a sound inference. ROPE_THETA is NOT inferable
// the same way (it was settable long before the record existed), so the legacy default is this build's
// own theta: it cannot catch a pre-record theta mismatch, and pretending otherwise would reject every
// legacy model on a non-default-theta build. Guarding theta starts from files written from here on.
inline constexpr std::uint64_t ARCH_FINGERPRINT_LEGACY = arch_fingerprint(0, 1, ROPE_THETA);

// ARCH_FINGERPRINT2 -- a SECOND fingerprint word, because ARCH_FINGERPRINT above has ZERO spare bits
// left (8 + 8 + 16 + 32 = 64, worked through explicitly here rather than assumed, per AGENTS.md S3
// rule 1): depth_attn_stride already took the one byte middle_layers could never reach, and no further
// axis fits without shrinking an existing field, which would either lose ROPE_THETA's exact bit_cast
// recovery (needs all 32 bits) or cap LOOP_REPEATS below a real future value (16 bits already tight).
// Per AGENTS.md S3 rule 2 ("prefer an ADDITIVE... format... over reshuffling existing fields"), this is
// a NEW additive word, not a repack of the first one -- and it is deliberately given 56 SPARE bits from
// day one (only byte 0 is assigned) so the next computation-changing, shape-neutral axis after this one
// does not repeat the exact scramble that produced ARCH_FINGERPRINT's own comment about a byte
// "middle_layers was never able to reach".
//   Field layout, high to low: [63:48] partial_rotary_dim | [47:40] gdn_key_heads | [39:32] qsa_compress_ratio |
//   [31:16] qsa_indexer_budget | [15:8] experts_per_tok | [7:0] gdn_full_attn_stride.
// At GDN_FULL_ATTN_STRIDE == 0 (the only value this build can currently take -- see the static_assert
// above) this is always 0, which is also what every checkpoint that predates this field must be
// ASSUMED to have been built with (Gated DeltaNet does not exist as a computable option anywhere yet,
// so "no GDN" is not a guess the way ARCH_FINGERPRINT_LEGACY's "un-looped" inference was -- it is the
// only architecture that has ever existed). See docs/GATED_DELTANET.md and engine_core.cpp/
// train_stage.cpp's third trailing checkpoint record for the on-disk (additive, gracefully-degrading)
// side of this.
//
// EXPERTS_PER_TOK (docs/MOE.md S3c): a MoE-on build's routed-expert TENSORS (NUM_EXPERTS of them) are
// already discriminated by PARAM_FLOATS (moe_param_delta() above) -- but EXPERTS_PER_TOK itself changes
// nothing about their SHAPE, only how many of them a forward pass actually reads per token. Two builds at
// identical NUM_EXPERTS/D_MODEL/D_FF but different EXPERTS_PER_TOK produce byte-identical checkpoints
// that compute different things -- exactly ARCH_FINGERPRINT2's rule #2 ("changes computation, not shape
// -> ADD IT HERE, or it loads silently and computes the wrong thing"), the same reasoning
// GDN_FULL_ATTN_STRIDE's own byte here already establishes, not NUM_EXPERTS's (which stays a pure
// PARAM_FLOATS/moe_param_delta concern and does NOT join this fingerprint, mirroring HC_COUNT/HC_LOWRANK's
// own MODEL_ARCH_ID-only treatment). Placed in byte 1 -- the next of the 56 spare bits this word was
// deliberately given from day one specifically so a future axis like this one would not need to repeat
// ARCH_FINGERPRINT's own cramped-bit-budget scramble (see that struct's own comment).
static_assert(GDN_FULL_ATTN_STRIDE >= 0 && GDN_FULL_ATTN_STRIDE <= 0xff,
              "GDN_FULL_ATTN_STRIDE must fit 8 bits");
static_assert(EXPERTS_PER_TOK >= 0 && EXPERTS_PER_TOK <= 0xff, "EXPERTS_PER_TOK must fit 8 bits");
//
// QSA (docs/QSA.md S3a): of QSA's five new axes, the three INDEXER SHAPE axes (n_heads/kv_heads/head_dim)
// widen index_qk_proj and the two indexer norms strictly monotonically, so PARAM_FLOATS/qsa_param_delta()
// already discriminate them (rule #1) and they do NOT belong here. The other two do:
//   * QSA_INDEXER_BUDGET and QSA_INDEXER_COMPRESS_RATIO change NO tensor shape at all -- checked
//     exhaustively against both real classes' __init__ (the indexer's only parameters are index_qk_proj,
//     q_layernorm and k_layernorm, none of which mentions either) -- while changing which tokens every
//     query may attend to, via block_topk = budget / compress_ratio and the pooling arity. Two builds
//     identical but for them produce BYTE-IDENTICAL checkpoints that compute different attention:
//     exactly rule #2, the same hazard GDN_FULL_ATTN_STRIDE and EXPERTS_PER_TOK already sit here for.
// Placed in the next of the 56 spare bits this word was deliberately given from day one:
//   budget gets 16 bits at [31:16] (the real value is 2048, which does not fit 8 -- worked through, not
//   assumed), compress_ratio 8 bits at [39:32]. ([63:40] was reserved then; byte 5 has since gone to
//   GDN's key-head count, see just below.) At the neutral setting (both 0)
//   this word is bit-identical to every value it has ever produced, so no existing checkpoint is
//   invalidated -- the same additive, gracefully-degrading property AGENTS.md S3 rule 2 requires.
static_assert(QSA_INDEXER_BUDGET >= 0 && QSA_INDEXER_BUDGET <= 0xffff,
              "QSA_INDEXER_BUDGET must fit 16 bits");
static_assert(QSA_INDEXER_COMPRESS_RATIO >= 0 && QSA_INDEXER_COMPRESS_RATIO <= 0xff,
              "QSA_INDEXER_COMPRESS_RATIO must fit 8 bits");
//
// GDN_K_HEADS (WP4b blocker B, docs/WP4_SCOPE.md S2): GDN's key-head COUNT enters every GDN tensor
// shape only through key_dim = k_heads * k_head_dim, so two builds with the same product (2x64 vs
// 4x32) have byte-identical parameter blobs while computing a different recurrence -- rep() =
// v_heads / k_heads decides which value heads share a key head. Rule #2, so it belongs here; the other
// three GDN axes are rule #1 (see their own comment in the GATED DELTANET section above).
// Placed in byte 5, [47:40] -- the next of the spare bits this word was given from day one.
// The field carries 0 whenever GDN is OFF (GDN_FULL_ATTN_STRIDE == 0 ⇒ no GDN layer exists ⇒ the axis
// is inert), which is what keeps every fingerprint this function has EVER produced bit-identical: the
// resolved key-head count is non-zero even in a default build, so folding it in unconditionally would
// have invalidated every existing checkpoint for an axis those checkpoints cannot possibly depend on.
static_assert(GDN_K_HEADS >= 0 && GDN_K_HEADS <= 0xff, "GDN_K_HEADS must fit 8 bits");
//
// ROTARY_DIM (WP4b blocker C): the rotary PREFIX width. Rule #2 -- it rotates fewer channels of every
// head while changing no tensor shape whatsoever, so PARAM_FLOATS cannot see it and a checkpoint would
// load silently and compute different attention. Exactly the ROPE_THETA / LOOP_MIDDLE_LAYERS hazard
// ARCH_FINGERPRINT exists for. It takes the LAST 16 bits, [63:48] -- 16 and not 8 because it is bounded
// by D_HEAD, which is 256 in the real model and no longer bounded by D_MODEL/N_HEADS after blocker A.
// The field is CANONICALIZED to 0 at ROTARY_DIM == D_HEAD (full-width rotary, the only behaviour this
// engine has ever had), so every fingerprint ever written stays bit-identical, and the canonical form
// also means an explicit `--rotary-dim <D_HEAD>` and the default `0` cannot disagree.
// This exhausts ARCH_FINGERPRINT2: the next computation-changing, shape-neutral axis needs a THIRD
// additive word, built the same way (see this word's own header comment for why additive, not repacked).
static_assert(ROTARY_DIM >= 0 && ROTARY_DIM <= 0xffff, "ROTARY_DIM must fit 16 bits");
inline constexpr std::uint64_t arch_fingerprint2(int gdn_full_attn_stride, int experts_per_tok = 0,
                                                  int qsa_indexer_budget = 0,
                                                  int qsa_indexer_compress_ratio = 0,
                                                  int gdn_key_heads = 0,
                                                  int partial_rotary_dim = 0) {
    return (static_cast<std::uint64_t>(static_cast<unsigned>(partial_rotary_dim)     & 0xffffu) << 48)
         | (static_cast<std::uint64_t>(static_cast<unsigned>(gdn_key_heads)          & 0xffu)   << 40)
         | (static_cast<std::uint64_t>(static_cast<unsigned>(qsa_indexer_compress_ratio) & 0xffu)   << 32)
         | (static_cast<std::uint64_t>(static_cast<unsigned>(qsa_indexer_budget)         & 0xffffu) << 16)
         | (static_cast<std::uint64_t>(static_cast<unsigned>(experts_per_tok)            & 0xffu)   << 8)
         |  static_cast<std::uint64_t>(static_cast<unsigned>(gdn_full_attn_stride)       & 0xffu);
}
struct ArchAxes2 { int gdn_full_attn_stride; int experts_per_tok; int qsa_indexer_budget;
                   int qsa_indexer_compress_ratio; int gdn_key_heads; int partial_rotary_dim; };
inline constexpr ArchAxes2 arch_axes2_of(std::uint64_t fp2) {
    return ArchAxes2{ static_cast<int>(fp2 & 0xffu), static_cast<int>((fp2 >> 8) & 0xffu),
                      static_cast<int>((fp2 >> 16) & 0xffffu), static_cast<int>((fp2 >> 32) & 0xffu),
                      static_cast<int>((fp2 >> 40) & 0xffu), static_cast<int>((fp2 >> 48) & 0xffffu) };
}
inline constexpr std::uint64_t ARCH_FINGERPRINT2 =
    arch_fingerprint2(GDN_FULL_ATTN_STRIDE, EXPERTS_PER_TOK, QSA_INDEXER_BUDGET,
                      QSA_INDEXER_COMPRESS_RATIO, USE_GATED_DELTANET ? GDN_K_HEADS : 0,
                      ROTARY_DIM == D_HEAD ? 0 : ROTARY_DIM);
inline constexpr std::uint64_t ARCH_FINGERPRINT2_LEGACY = arch_fingerprint2(0, 0, 0, 0, 0, 0);

// MODEL_ARCH_ID -- the FULL architecture identity: every axis that makes two models different things,
// shape-changing and computation-changing alike. ARCH_FINGERPRINT above deliberately covers only the
// shape-NEUTRAL axes (PARAM_FLOATS discriminates the rest at load time); this one has a different job,
// so it covers everything.
//
// It exists because the model REGISTRY had the same omission the config banner and the checkpoint each
// had: registry::model_dir() built a directory name from a HAND-LISTED subset (d/l/h/seq/vocab plus
// four flags) and registry::compatible() compared that same subset. Neither knew about n_kv_heads,
// loop_middle, loop_repeats, rope_scaling or rope_theta. Two consequences, both real:
//   * a GQA model and an MHA model at identical dims got the SAME directory name, so the naming scheme
//     stopped encoding identity -- the one thing it exists to do;
//   * compatible() answered YES for architectures load_model then REJECTS, so `train` would offer to
//     resume a checkpoint the loader refuses.
// Deriving both from one id means a new axis is added HERE, once, and naming plus compatibility follow.
// That is the same lesson as AGENTS.md 10's consumer sweep, applied structurally instead of by habit.
consteval std::uint64_t make_model_arch_id() {
    std::uint64_t h = 1469598103934665603ull;                  // FNV-1a
    auto mix = [&h](std::uint64_t v) {
        for (int i = 0; i < 8; ++i) { h ^= (v >> (8 * i)) & 0xffull; h *= 1099511628211ull; }
    };
    mix(static_cast<std::uint64_t>(D_MODEL));
    mix(static_cast<std::uint64_t>(N_LAYERS));
    mix(static_cast<std::uint64_t>(N_HEADS));
    mix(static_cast<std::uint64_t>(N_KV_HEADS));
    mix(static_cast<std::uint64_t>(D_FF));
    mix(static_cast<std::uint64_t>(SEQ_LEN));
    mix(static_cast<std::uint64_t>(VOCAB));
    mix(static_cast<std::uint64_t>(USE_TERNARY));
    mix(static_cast<std::uint64_t>(static_cast<int>(POS_ENCODING)));
    mix(static_cast<std::uint64_t>(USE_GATED_FFN));
    mix(static_cast<std::uint64_t>(USE_TIED_EMBEDDINGS));
    mix(static_cast<std::uint64_t>(USE_QK_NORM));
    mix(ARCH_FINGERPRINT);      // folds in LoopSplit's schedule and ROPE_THETA
    mix(static_cast<std::uint64_t>(DEPTH_ATTN_STRIDE));   // adds no parameters; changes computation
    mix(static_cast<std::uint64_t>(ROPE_SCALING));
    mix(static_cast<std::uint64_t>(std::bit_cast<std::uint32_t>(ROPE_SCALE_FACTOR)));
    mix(ARCH_FINGERPRINT2);                               // folds in Gated DeltaNet's layer schedule AND
                                                            // MoE's experts_per_tok (docs/MOE.md S3c)
    mix(static_cast<std::uint64_t>(GDN_FULL_ATTN_STRIDE)); // (currently always 0 -- see the static_assert
                                                            //  in the GATED DELTANET section above)
    // GDN's own four head axes (WP4b blocker B). Three of them are SHAPE-changing (PARAM_FLOATS
    // discriminates them for the checkpoint-load gate, so they do NOT join ARCH_FINGERPRINT2) and the
    // fourth, GDN_K_HEADS, already rides the mix(ARCH_FINGERPRINT2) call above -- but MODEL_ARCH_ID
    // covers every axis unconditionally, exactly as DEPTH_ATTN_STRIDE's own mix above already does.
    // Mixed as the RESOLVED values, not the raw 0-means-alias flags, so two spellings of one
    // architecture (--gdn-value-heads 0 vs --gdn-value-heads <N_HEADS>) share one identity -- and
    // gated on USE_GATED_DELTANET for the same reason ARCH_FINGERPRINT2's own gdn_key_heads field is:
    // with GDN off there is no GDN layer, so the axes are inert and every pre-existing build must keep
    // the MODEL_ARCH_ID it already had (AGENTS.md S4's "zero effect on existing builds").
    mix(static_cast<std::uint64_t>(USE_GATED_DELTANET ? GDN_K_HEADS    : 0));
    mix(static_cast<std::uint64_t>(USE_GATED_DELTANET ? GDN_V_HEADS    : 0));
    mix(static_cast<std::uint64_t>(USE_GATED_DELTANET ? GDN_K_HEAD_DIM : 0));
    mix(static_cast<std::uint64_t>(USE_GATED_DELTANET ? GDN_V_HEAD_DIM : 0));
    // N-gram embeddings: a SHAPE-changing axis (see its own section above), so PARAM_FLOATS already
    // discriminates it for the checkpoint-load gate and it does NOT join ARCH_FINGERPRINT -- but
    // MODEL_ARCH_ID covers every axis unconditionally, the same way DEPTH_ATTN_STRIDE's mix above
    // already changes this value for every build regardless of its setting.
    mix(static_cast<std::uint64_t>(NGRAM_MAX_N));
    mix(static_cast<std::uint64_t>(NGRAM_TABLES_PER_ORDER));
    mix(static_cast<std::uint64_t>(NGRAM_TABLE_SIZE));
    // Gated Residual: also a SHAPE-changing axis (docs/GATED_RESIDUAL.md S3c -- Delta is a pure,
    // unconditional addition of new tensors, strictly monotonic, so PARAM_FLOATS alone already
    // discriminates it and it does NOT join ARCH_FINGERPRINT), mixed in here unconditionally like
    // every other axis MODEL_ARCH_ID covers.
    mix(static_cast<std::uint64_t>(HC_COUNT));
    mix(static_cast<std::uint64_t>(HC_LOWRANK));
    // Mixture of Experts: NUM_EXPERTS is a SHAPE-changing axis (docs/MOE.md S3c -- moe_param_delta() is
    // strictly monotonic in the "on" direction), so PARAM_FLOATS alone already discriminates it and it
    // does NOT join ARCH_FINGERPRINT -- mixed in here unconditionally like every other axis MODEL_ARCH_ID
    // covers. EXPERTS_PER_TOK already joined ARCH_FINGERPRINT2 above (a shape-NEUTRAL, computation-
    // changing axis, mixed in via the earlier `mix(ARCH_FINGERPRINT2)` call, which now covers both GDN's
    // stride and MoE's experts_per_tok) and is not folded in a second time here.
    mix(static_cast<std::uint64_t>(NUM_EXPERTS));
    // QSA: the three INDEXER SHAPE axes are shape-changing (docs/QSA.md S3a -- qsa_param_delta() is
    // strictly positive and monotonic once on), so PARAM_FLOATS alone already discriminates them and they
    // do NOT join ARCH_FINGERPRINT2 -- mixed in here unconditionally like every other axis MODEL_ARCH_ID
    // covers. QSA_INDEXER_BUDGET/QSA_INDEXER_COMPRESS_RATIO already joined ARCH_FINGERPRINT2 above
    // (shape-NEUTRAL, computation-changing) and ride the earlier mix(ARCH_FINGERPRINT2) call rather than
    // being folded in a second time -- exactly EXPERTS_PER_TOK's own treatment.
    mix(static_cast<std::uint64_t>(QSA_INDEXER_N_HEADS));
    mix(static_cast<std::uint64_t>(QSA_INDEXER_KV_HEADS));
    mix(static_cast<std::uint64_t>(QSA_INDEXER_HEAD_DIM));
    return h;
}
inline constexpr std::uint64_t MODEL_ARCH_ID = make_model_arch_id();

// THE build's memplan::Dims. Every consumer that wants "this binary's shape" must call this rather than
// aggregate-initialise Dims itself.
//
// Why this exists: Dims is a 12-field aggregate and five sites were initialising it positionally by
// hand. That produced three separate real bugs in one feature cycle -- kFootprintDims silently
// under-predicted VRAM by 189-381 MiB after n_kv_heads was added; the configurator omitted exec_layers
// and under-predicted activation memory 1.43x (feeding gpu_batch_estimate, i.e. the batch we
// RECOMMEND); and sub0_memplan_stage passed six of twelve fields, costing a gated+tied+qk-norm build as
// an untied non-gated MHA one. Positional init also means a future field REORDER compiles silently with
// values shuffled. Adding a field now updates one place, and a site that needs a variant copies this and
// overrides the one field it varies.
inline constexpr memplan::Dims current_build_dims() {
    return memplan::Dims{
        .d_model     = D_MODEL,
        .n_layers    = N_LAYERS,
        .n_heads     = N_HEADS,
        .d_ff        = D_FF,
        .seq_len     = SEQ_LEN,
        .vocab       = VOCAB,
        .tied        = USE_TIED_EMBEDDINGS,
        .qk_norm     = USE_QK_NORM,
        .gated       = USE_GATED_FFN,
        .pos_emb     = HAS_POS_EMB,
        .n_kv_heads  = N_KV_HEADS,
        .exec_layers = LOOP_EXEC_COUNT,
        .depth_slots = DEPTH_CACHE_MAX,
    };
}

// Parameter tensor count: tok_emb, plus pos_emb only under absolute positions (see HAS_POS_EMB),
// then per transformer block either 10 (plain: ln1, ln2, Wq, Wk, Wv, Wo, W1, b1, W2, b2) or 9
// (gated: ln1, ln2, Wq, Wk, Wv, Wo, Wg, W1, W2 -- SwiGLU, no FFN biases, matching the GGUF/Llama
// convention this variant exists to import -- see gguf-import-feasibility-review), plus 2 more
// (q_norm, k_norm) when USE_QK_NORM, then the tail: ln_f alone when USE_TIED_EMBEDDINGS (the head
// reuses tok_emb, no separate lm_head/lm_bias slot -- the common tied-embedding convention also
// drops the head bias, matching GPT-2/GGUF-style tied models), else ln_f + lm_head + lm_bias (3).
// N-gram embeddings add NGRAM_NUM_EMBEDDERS table tensors plus one concat_proj tensor (0 when off).
//
// Per-LAYER aware (Stage 1): a layer is EITHER a softmax-attention layer (ln1,ln2 + Wq,Wk,Wv,Wo [+
// QNorm,KNorm] + FFN) OR a GDN layer (ln1,ln2 + the nine GDN tensors + FFN -- GDN never uses QNorm/
// KNorm, S1c), decided per layer by GDN_SCHEDULE.full_attn[l] (all true at GDN_FULL_ATTN_STRIDE == 0,
// reproducing the pre-Stage-1 count exactly -- see the identity check in tests/layout_tests.cpp). A
// plain runtime `if` on a compile-time-known array value inside a consteval function is the same
// pattern DEPTH_SCHEDULE.own[e] already uses at call sites -- this just does it during layout counting
// too, rather than only at dispatch time.
consteval int count_num_params() {
    int n = 1 + (HAS_POS_EMB ? 1 : 0);
    for (int l = 0; l < N_LAYERS; ++l) {
        n += 2;   // ln1, ln2
        if constexpr (USE_GATED_RESIDUAL) n += 4;   // GrHcNorm/MixDown/MixUp/BlockInject (attn-wrapping)
        if (MIXER_SCHEDULE[static_cast<std::size_t>(l)] == LayerMixer::Qsa) {
            // QSA layer (docs/QSA.md S3b): REPLACES Wq/Wk/Wv/Wo[+QNorm/KNorm] with its own ten tensors --
            // QsaQProj, QsaGateProj, QsaKProj, QsaVProj, QsaOProj, QsaQNorm, QsaKNorm (7) plus the
            // indexer's QsaIdxQkProj, QsaIdxQNorm, QsaIdxKNorm (3). QNorm/KNorm are ALWAYS present on a
            // QSA layer regardless of USE_QK_NORM -- the real Qwen4ExpTextAttention has them
            // unconditionally, so they are part of this mixer's identity, not an optional engine feature.
            n += 10;
        } else if (GDN_SCHEDULE.full_attn[static_cast<std::size_t>(l)]) {
            n += 4;   // Wq, Wk, Wv, Wo
            if constexpr (USE_QK_NORM) n += 2;   // QNorm, KNorm
        } else {
            n += 9;   // GdnInProjQkv/Z/B/A, GdnConv, GdnALog, GdnDtBias, GdnNorm, GdnOutProj
        }
        if constexpr (USE_GATED_RESIDUAL) n += 4;   // GrHcNorm/MixDown/MixUp/BlockInject (mlp-wrapping)
        // Mixture of Experts (Stage 1, docs/MOE.md S3b): replaces the FFN's own 3 (gated)/4 (plain)
        // tensors with MoeRouter (1) + NUM_EXPERTS*(MoeGate,MoeUp,MoeDown) + the shared expert's own
        // SwiGLU triple (3) + MoeSharedGateProj (1), for EVERY layer uniformly (no per-layer schedule,
        // unlike GDN -- MoE is on for the whole model or not at all, see USE_MOE's own comment).
        if constexpr (USE_MOE) n += 1 + 3 * NUM_EXPERTS + 3 + 1;
        else                   n += (USE_GATED_FFN ? 3 : 4);
    }
    n += (USE_TIED_EMBEDDINGS ? 1 : 3);
    n += (NGRAM_EMBED ? NGRAM_NUM_EMBEDDERS + 1 : 0);
    // Gated Residual's model-level exit collapse (docs/GATED_RESIDUAL.md S1c/S3b): ONE more instance,
    // WITHOUT GrBlockInject (use_combine=False in the real model -- see make_param_layout()'s own
    // placement), appended once regardless of N_LAYERS.
    if constexpr (USE_GATED_RESIDUAL) n += 3;   // GrHcNorm/MixDown/MixUp only
    return n;
}
inline constexpr int NUM_PARAMS = count_num_params();

// Role of each parameter tensor. Lets a backend special-case kinds (e.g. concat the
// Q/K/V projections into one GEMM, or keep the lm_head full precision) without
// re-parsing offsets. The enumerator order is NOT the layout order — PARAM_LAYOUT is.
// Wg (SwiGLU gate matrix) only appears when USE_GATED_FFN; W1/W2 are reused as the
// "up"/"down" projections in that mode (same role: a D_MODEL<->D_FF matmul weight).
// LmHead/LmBias are ABSENT from the layout entirely when USE_TIED_EMBEDDINGS -- the head
// reads tok_emb directly (see op_tied_head in backend_cpu.cpp), there is no separate slot.
// QNorm/KNorm only appear when USE_QK_NORM: a [1,D_HEAD] learned gamma applied per-head
// (shared across heads, same idea as ln1/ln2 but over one head's slice instead of the
// whole row) to Q/K right after their projection, before RoPE.
// NgramEmb: one of NGRAM_NUM_EMBEDDERS small hashed-n-gram embedding tables (role like TokEmb, but
// narrower and NOT the primary embedding). NgramProj: the single learned "concat_proj" linear that
// projects the concatenated table outputs down to D_MODEL (role like LmHead: a full-precision,
// AdamW-only GEMM weight -- see is_muon_kind below). Both ABSENT entirely when !NGRAM_EMBED.
// GDN kinds (Stage 1): a GDN layer replaces Wq/Wk/Wv/Wo (and QNorm/KNorm, which GDN never uses -- it
// does its own internal, ungained L2-norm, S1b) with these nine tensors -- see
// docs/GATED_DELTANET.md S1a/S3b for the real module this mirrors and S1c for GdnNorm's shape (shared
// across heads, [1,D_HEAD] like QNorm/KNorm, NOT D_MODEL-wide). GdnInProjQkv is the fused Q+K+V
// projection (S1a's real 4-way split, `in_proj_qkv`); GdnInProjZ is the output-gate projection
// (`in_proj_z`); GdnInProjB/GdnInProjA are the two per-V-head gate-input scalars (`in_proj_b`/
// `in_proj_a`); GdnConv is the depthwise causal conv weight (no in/out axis -- one row per channel);
// GdnALog/GdnDtBias are the two learned per-V-head recurrence-gate parameters; GdnOutProj restores
// D_MODEL width (role like LmHead: a full-precision, AdamW-only GEMM weight -- see is_muon_kind below).
// GrHcNorm/GrMixDown/GrMixUp/GrBlockInject (Stage 1, docs/GATED_RESIDUAL.md S3b/S4): one GatedResidual
// instance's four learned tensors. Reused for THREE independent instances per model when
// USE_GATED_RESIDUAL: two per layer (wrapping the attention/GDN sub-block and the FFN sub-block, each
// with its own GrBlockInject) plus one at the very end of the stack (the model-level exit collapse,
// WITHOUT GrBlockInject -- see make_param_layout()'s own placement). GrHcNorm is [1, HC_COUNT*D_MODEL]
// (one gain per (stream,channel) pair, NOT shared across streams like QNorm/KNorm -- a real divergence
// from this project's existing per-head norm convention, verified in the real source, S1a).
// MoE kinds (Stage 1, docs/MOE.md S3b/S4): a MoE-on layer replaces its FFN's Wg/W1/W2 (or W1/B1/W2/B2)
// with MoeRouter (the D_MODEL->NUM_EXPERTS gate, no bias) followed by NUM_EXPERTS repeats of
// (MoeGate, MoeUp, MoeDown) -- one routed expert's own SwiGLU triple -- then ONE shared-expert SwiGLU
// triple (MoeSharedGate/MoeSharedUp/MoeSharedDown, same D_FF width per MOE_DIMS) plus MoeSharedGateProj
// (the D_MODEL->1 sigmoid gate scaling the shared expert's own contribution).
// QSA kinds (Stage 1, docs/QSA.md S3b): a QSA layer replaces Wq/Wk/Wv/Wo (and QNorm/KNorm) with its own
// seven attention tensors plus the indexer's three. QsaQProj/QsaGateProj are the two halves of the real
// model's single DOUBLE-WIDTH q_proj, stored separately (docs/QSA.md S2b.4 -- exactly the same arithmetic,
// since chunking a bias-free Linear's output axis partitions its weight rows; a future weight transplant
// must respect the real PER-HEAD chunk order). QsaQNorm/QsaKNorm are [1,D_HEAD] per-head RMSNorm gains
// with the real model's (1 + w) zero-centered convention -- NOT this project's QNorm/KNorm convention,
// which is why they are separate kinds rather than a reuse. QsaIdxQkProj is the indexer's single fused
// [D_MODEL, (idx_n+idx_kv)*idx_hd] projection; QsaIdxQNorm/QsaIdxKNorm are its two [1, idx_hd] norms.
enum class PKind : unsigned char {
    TokEmb, PosEmb, Ln1, Ln2, Wq, Wk, Wv, Wo, W1, B1, W2, B2, LnF, LmHead, LmBias, Wg, QNorm, KNorm,
    NgramEmb, NgramProj,
    GdnInProjQkv, GdnInProjZ, GdnInProjB, GdnInProjA, GdnConv, GdnALog, GdnDtBias, GdnNorm, GdnOutProj,
    GrHcNorm, GrMixDown, GrMixUp, GrBlockInject,
    MoeRouter, MoeGate, MoeUp, MoeDown, MoeSharedGate, MoeSharedUp, MoeSharedDown, MoeSharedGateProj,
    QsaQProj, QsaGateProj, QsaKProj, QsaVProj, QsaOProj, QsaQNorm, QsaKNorm,
    QsaIdxQkProj, QsaIdxQNorm, QsaIdxKNorm
};

// One weight tensor: where it lives in the flat blob, its logical [rows x cols]
// shape, its role, whether AdamW weight-decay applies, and whether op_linear
// ternarizes it when USE_TERNARY (the lm_head stays full precision, so it is a
// matmul weight but ternary=false).
struct ParamDesc {
    std::size_t off;     // float offset into the flat parameter blob
    int         rows;    // logical rows
    int         cols;    // logical cols (n = rows * cols)
    PKind       kind;
    bool        decay;   // AdamW weight decay applies (the GEMM weight matrices)
    bool        ternary; // op_linear ternarizes this weight when USE_TERNARY
    constexpr std::size_t n() const { return static_cast<std::size_t>(rows) * cols; }
};

// Build the layout at compile time. The append order here defines the canonical
// parameter order; offsets are accumulated so PARAM_LAYOUT and PARAM_FLOATS stay
// consistent by construction.
consteval std::array<ParamDesc, NUM_PARAMS> make_param_layout() {
    std::array<ParamDesc, NUM_PARAMS> t{};
    std::size_t off = 0;
    int i = 0;
    auto add = [&](int r, int c, PKind k, bool decay, bool ternary) {
        t[i] = ParamDesc{off, r, c, k, decay, ternary};
        off += static_cast<std::size_t>(r) * c;
        ++i;
    };
    add(VOCAB,   D_MODEL, PKind::TokEmb, false, false);
    if constexpr (HAS_POS_EMB) add(SEQ_LEN, D_MODEL, PKind::PosEmb, false, false);
    for (int l = 0; l < N_LAYERS; ++l) {
        add(1,       D_MODEL, PKind::Ln1, false, false);
        add(1,       D_MODEL, PKind::Ln2, false, false);
        // Gated Residual (docs/GATED_RESIDUAL.md S3b/S4a): the attn_hyper_connection instance wrapping
        // this layer's attention/GDN sub-block. Placed here, before that sub-block's own weights, since
        // it is what turns `h` (the wide residual) into the D_MODEL-wide input ln1 actually reads.
        if constexpr (USE_GATED_RESIDUAL) {
            constexpr int WIDE = HC_COUNT * D_MODEL;
            add(1,          WIDE,       PKind::GrHcNorm,      false, false);
            add(WIDE,       HC_LOWRANK, PKind::GrMixDown,     true,  false);
            add(HC_LOWRANK, WIDE,       PKind::GrMixUp,       true,  false);
            add(WIDE,       HC_COUNT,   PKind::GrBlockInject, true,  false);
        }
        if (MIXER_SCHEDULE[static_cast<std::size_t>(l)] == LayerMixer::Qsa) {
            // QSA layer (docs/QSA.md S3b/S4): the gated-attention tensors then the indexer's own three.
            // Axis order is this project's [rows=in, cols=out] throughout (see qsa_math.hpp's header
            // comment for the explicit re-derivation from the real PyTorch [out,in] convention). The
            // projections are ternary-eligible like Wq/Wk/Wv/Wo; the two norms and the indexer's own
            // three tensors stay full precision (decay=false for the 1D gains; the indexer's projection
            // is a routing/SELECTION decision, the one place a ternarized weight is most damaging --
            // the same reasoning MoeRouter and LmHead stay full precision).
            add(D_MODEL, D_MODEL, PKind::QsaQProj,    true,  true);
            add(D_MODEL, D_MODEL, PKind::QsaGateProj, true,  true);
            add(D_MODEL, D_KV,    PKind::QsaKProj,    true,  true);
            add(D_MODEL, D_KV,    PKind::QsaVProj,    true,  true);
            add(D_MODEL, D_MODEL, PKind::QsaOProj,    true,  true);
            add(1, D_HEAD,        PKind::QsaQNorm,    false, false);
            add(1, D_HEAD,        PKind::QsaKNorm,    false, false);
            add(D_MODEL, QSA_IDX_QK_OUT,      PKind::QsaIdxQkProj, true,  false);
            add(1, QSA_INDEXER_HEAD_DIM,      PKind::QsaIdxQNorm,  false, false);
            add(1, QSA_INDEXER_HEAD_DIM,      PKind::QsaIdxKNorm,  false, false);
        } else if (GDN_SCHEDULE.full_attn[static_cast<std::size_t>(l)]) {
            add(D_MODEL, D_MODEL, PKind::Wq,  true,  true);
            // Wk/Wv are [in=D_MODEL, out=D_KV] -- narrower than Wq under GQA, identical when N_KV_HEADS
            // == N_HEADS. NOTE the axis order is this project's [rows=in, cols=out], the TRANSPOSE of the
            // PyTorch references this feature is modelled on (see AGENTS.md 5: re-derive conventions, do
            // not assume they match).
            add(D_MODEL, D_KV,    PKind::Wk,  true,  true);
            add(D_MODEL, D_KV,    PKind::Wv,  true,  true);
            add(D_MODEL, D_MODEL, PKind::Wo,  true,  true);
            if constexpr (USE_QK_NORM) {
                add(1, D_HEAD, PKind::QNorm, false, false);
                add(1, D_HEAD, PKind::KNorm, false, false);
            }
        } else {
            // Gated DeltaNet layer (docs/GATED_DELTANET.md S3a/S3b), using GDN's OWN four head axes
            // (WP4b blocker B -- see GDN_K_HEADS/GDN_V_HEADS/GDN_K_HEAD_DIM/GDN_V_HEAD_DIM above):
            // hidden_size=D_MODEL, key_dim=GDN_KEY_DIM, value_dim=GDN_VALUE_DIM. At the neutral setting
            // those resolve to D_KV and N_HEADS*D_HEAD (== D_MODEL), i.e. exactly the previous shapes.
            // Weight axis order is this project's own [in,out] convention throughout (see gdn_math.hpp's
            // header comment for the explicit re-derivation from the real model's PyTorch nn.Linear
            // [out,in] convention).
            add(D_MODEL, GDN_CONV_DIM,       PKind::GdnInProjQkv, true,  false);  // [q|k|v], 2*KD+VD
            add(D_MODEL, GDN_VALUE_DIM,      PKind::GdnInProjZ,  true,  false);
            add(D_MODEL, GDN_V_HEADS,        PKind::GdnInProjB,  true,  false);
            add(D_MODEL, GDN_V_HEADS,        PKind::GdnInProjA,  true,  false);
            // Depthwise conv weight: [GDN_CONV_DIM, GDN_CONV_KERNEL], no in/out axis (one row per
            // channel) -- decay=false (not a GEMM weight in the AdamW weight-decay sense), ternary=false.
            add(GDN_CONV_DIM, GDN_CONV_KERNEL, PKind::GdnConv,    false, false);
            add(1, GDN_V_HEADS, PKind::GdnALog,   false, false);
            add(1, GDN_V_HEADS, PKind::GdnDtBias, false, false);
            add(1, GDN_V_HEAD_DIM, PKind::GdnNorm, false, false);  // RMSNormGated gamma, shared across heads (S1c)
            add(GDN_VALUE_DIM, D_MODEL, PKind::GdnOutProj, true, false);   // stays full precision, like LmHead
        }
        // Gated Residual: the mlp_hyper_connection instance wrapping this layer's FFN sub-block --
        // same shape as the attn-wrapping instance just above, an independent set of tensors.
        if constexpr (USE_GATED_RESIDUAL) {
            constexpr int WIDE = HC_COUNT * D_MODEL;
            add(1,          WIDE,       PKind::GrHcNorm,      false, false);
            add(WIDE,       HC_LOWRANK, PKind::GrMixDown,     true,  false);
            add(HC_LOWRANK, WIDE,       PKind::GrMixUp,       true,  false);
            add(WIDE,       HC_COUNT,   PKind::GrBlockInject, true,  false);
        }
        if constexpr (USE_MOE) {
            // Mixture of Experts (docs/MOE.md S3b/S4): router (no bias) + NUM_EXPERTS routed-expert
            // SwiGLU triples + the shared expert's own SwiGLU triple + its own sigmoid-gate projection.
            // Router/shared-gate-proj stay full precision (ternary=false) -- routing decisions are the
            // one place a ternarized weight would be most damaging, same reasoning LmHead/GdnOutProj stay
            // full precision. Expert FFN matrices ARE ternary-eligible (ternary=true), matching Wg/W1/W2's
            // own treatment -- they are ordinary GEMM weights, just NUM_EXPERTS independent copies of one.
            add(D_MODEL, NUM_EXPERTS, PKind::MoeRouter, true, false);
            for (int e = 0; e < NUM_EXPERTS; ++e) {
                add(D_MODEL, D_FF, PKind::MoeGate, true, true);
                add(D_MODEL, D_FF, PKind::MoeUp,   true, true);
                add(D_FF, D_MODEL, PKind::MoeDown, true, true);
            }
            add(D_MODEL, D_FF, PKind::MoeSharedGate, true, true);
            add(D_MODEL, D_FF, PKind::MoeSharedUp,   true, true);
            add(D_FF, D_MODEL, PKind::MoeSharedDown, true, true);
            add(D_MODEL, 1,    PKind::MoeSharedGateProj, true, false);
        } else if constexpr (USE_GATED_FFN) {
            add(D_MODEL, D_FF,    PKind::Wg, true, true);   // gate
            add(D_MODEL, D_FF,    PKind::W1, true, true);   // up
            add(D_FF,    D_MODEL, PKind::W2, true, true);   // down (no biases -- GGUF/Llama convention)
        } else {
            add(D_MODEL, D_FF,    PKind::W1,  true,  true);
            add(1,       D_FF,    PKind::B1,  false, false);
            add(D_FF,    D_MODEL, PKind::W2,  true,  true);
            add(1,       D_MODEL, PKind::B2,  false, false);
        }
    }
    // Gated Residual's model-level exit collapse (docs/GATED_RESIDUAL.md S1c): ONE more instance,
    // appended once after every layer, WITHOUT GrBlockInject (the real model's hyper_connection_mixer
    // is built with use_combine=False -- block_inject_weight genuinely does not exist for this instance).
    if constexpr (USE_GATED_RESIDUAL) {
        constexpr int WIDE = HC_COUNT * D_MODEL;
        add(1,          WIDE,       PKind::GrHcNorm,  false, false);
        add(WIDE,       HC_LOWRANK, PKind::GrMixDown, true,  false);
        add(HC_LOWRANK, WIDE,       PKind::GrMixUp,   true,  false);
    }
    add(1,       D_MODEL, PKind::LnF,    false, false);
    if constexpr (!USE_TIED_EMBEDDINGS) {
        add(D_MODEL, VOCAB,   PKind::LmHead, true,  false);  // head stays full precision
        add(1,       VOCAB,   PKind::LmBias, false, false);
    }
    // N-gram embeddings: appended at the very end, additively (AGENTS.md 3 rule 2's "trailing
    // optional" pattern) -- NGRAM_NUM_EMBEDDERS small hashed tables (decay=false, like TokEmb: an
    // embedding row lookup, not a GEMM) followed by ONE concat_proj GEMM weight (decay=true, ternary=
    // false -- it stays full precision like LmHead, see is_muon_kind below).
    if constexpr (NGRAM_EMBED) {
        for (int e = 0; e < NGRAM_NUM_EMBEDDERS; ++e)
            add(NGRAM_VOCAB_DIMS[static_cast<std::size_t>(e)], NGRAM_EMB_DIM, PKind::NgramEmb, false, false);
        add(D_MODEL, D_MODEL, PKind::NgramProj, true, false);
    }
    return t;
}

inline constexpr std::array<ParamDesc, NUM_PARAMS> PARAM_LAYOUT = make_param_layout();

// Muon-eligible parameter kinds: the hidden 2D GEMM weight matrices that route through
// Newton-Schulz orthogonalized-momentum updates instead of AdamW's per-element update, when
// --optimizer muon is selected (Keller Jordan et al.; see include/sub0/muon.hpp). Embeddings,
// the output head, and 1D params (norm gains, biases) always stay on AdamW. Defined ONCE here
// (not duplicated per-backend) so the CPU hybrid dispatch (AdamW::step, backend_cpu.cpp) and the
// GPU hybrid dispatch (device_adam_step, backend_cuda.cu) can never drift out of sync on which
// kinds route through Muon -- backend_cpu.cpp used to inline this exact kind set locally.
inline constexpr bool is_muon_kind(PKind k) {
    return k == PKind::Wq || k == PKind::Wk || k == PKind::Wv || k == PKind::Wo ||
           k == PKind::W1 || k == PKind::W2 || k == PKind::Wg ||
           // MoE (Stage 1): the routed and shared experts' own SwiGLU matrices are ordinary hidden GEMM
           // weights, NUM_EXPERTS independent copies of the same role Wg/W1/W2 already play -- Muon-
           // eligible for the same reason. MoeRouter/MoeSharedGateProj stay on AdamW (see their own
           // ternary=false comment in make_param_layout() -- routing precision, same reasoning LmHead
           // stays off Muon/ternary both).
           k == PKind::MoeGate || k == PKind::MoeUp || k == PKind::MoeDown ||
           k == PKind::MoeSharedGate || k == PKind::MoeSharedUp || k == PKind::MoeSharedDown ||
           // QSA (Stage 1): the five gated-attention projections are the exact same role Wq/Wk/Wv/Wo
           // already play (ordinary hidden GEMM weights), so Muon-eligible for the same reason.
           // QsaIdxQkProj stays on AdamW -- a selection weight, same reasoning MoeRouter/LmHead do.
           k == PKind::QsaQProj || k == PKind::QsaGateProj || k == PKind::QsaKProj ||
           k == PKind::QsaVProj || k == PKind::QsaOProj;
}

// A compact [start,end) float-offset range where AdamW weight decay applies, merging adjacent
// decay=true PARAM_LAYOUT entries into one run (e.g. Wq,Wk,Wv,Wo,W1 are 5 consecutive decay=true
// tensors -- one range covers all of them). Lets a flat-index kernel (backend_cuda.cu's
// adam_step_kernel) test "is this parameter decayed" via a handful of compile-time-known range
// comparisons instead of maintaining a PARAM_FLOATS-long persistent mask buffer -- the CPU path never
// needed this (AdamW::step there iterates PARAM_LAYOUT per-TENSOR, not per-flat-index), only a
// flat-index-per-thread GPU kernel does.
struct DecayRange { std::size_t start, end; };

consteval int count_decay_ranges() {
    int n = 0;
    bool in_run = false;
    for (const ParamDesc& p : PARAM_LAYOUT) {
        if (p.decay) { if (!in_run) ++n; in_run = true; }
        else         { in_run = false; }
    }
    return n;
}
inline constexpr int NUM_DECAY_RANGES = count_decay_ranges();

consteval std::array<DecayRange, NUM_DECAY_RANGES> make_decay_ranges() {
    std::array<DecayRange, NUM_DECAY_RANGES> out{};
    int idx = -1;
    for (const ParamDesc& p : PARAM_LAYOUT) {
        if (!p.decay) continue;
        if (idx >= 0 && out[idx].end == p.off) out[idx].end += p.n();   // extend the current run
        else                                   out[++idx] = DecayRange{p.off, p.off + p.n()};
    }
    return out;
}
inline constexpr std::array<DecayRange, NUM_DECAY_RANGES> DECAY_RANGES = make_decay_ranges();

// Same [start,end) run-merging idea as DECAY_RANGES, but for is_muon_kind() instead of p.decay --
// lets a flat-index GPU kernel (backend_cuda.cu's adam_step_kernel) test "does this parameter
// route through Muon instead" via a handful of compile-time-known range comparisons, so the
// AdamW kernel can SKIP Muon-routed parameters (which the host-side per-matrix Muon loop already
// updated) instead of double-updating them. Deliberately a separate table from DECAY_RANGES, not
// reused/merged with it: a param can be muon-eligible AND decay=true simultaneously (every
// Muon-eligible kind currently is), so the two predicates are independent, not nested. Kept as a
// parallel, non-templated implementation of the same run-merging logic (not factored into a
// shared helper) so DECAY_RANGES -- exhaustively covered by layout_tests.cpp's offset-by-offset
// test -- is not touched by this change; the marginal DRY win of a generic template was judged
// not worth risking a regression in already-tested code.
consteval int count_muon_ranges() {
    int n = 0;
    bool in_run = false;
    for (const ParamDesc& p : PARAM_LAYOUT) {
        if (is_muon_kind(p.kind)) { if (!in_run) ++n; in_run = true; }
        else                      { in_run = false; }
    }
    return n;
}
inline constexpr int NUM_MUON_RANGES = count_muon_ranges();

consteval std::array<DecayRange, NUM_MUON_RANGES> make_muon_ranges() {
    std::array<DecayRange, NUM_MUON_RANGES> out{};
    int idx = -1;
    for (const ParamDesc& p : PARAM_LAYOUT) {
        if (!is_muon_kind(p.kind)) continue;
        if (idx >= 0 && out[idx].end == p.off) out[idx].end += p.n();   // extend the current run
        else                                   out[++idx] = DecayRange{p.off, p.off + p.n()};
    }
    return out;
}
inline constexpr std::array<DecayRange, NUM_MUON_RANGES> MUON_RANGES = make_muon_ranges();

// Upper bounds on a Muon-eligible matrix's element count (rows*cols, the "[m,n]" working size)
// and its square Gram-matrix element count (min(rows,cols)^2, the "[m,m]" working size), taken
// over every Muon-eligible PARAM_LAYOUT entry. The GPU Newton-Schulz scratch buffers (backend_cuda.cu)
// are sized ONCE to these ceilings and reused for every matrix/every step (AGENTS.md #1: no
// per-call heap/device allocation) -- every actual call is bounded by ITS OWN current rows*cols,
// never by the (possibly larger) scratch buffer's own capacity.
consteval std::size_t compute_muon_max_mn() {
    std::size_t mx = 0;
    for (const ParamDesc& p : PARAM_LAYOUT)
        if (is_muon_kind(p.kind)) mx = std::max(mx, p.n());
    return mx;
}
inline constexpr std::size_t MUON_MAX_MN = compute_muon_max_mn();

consteval std::size_t compute_muon_max_mm() {
    std::size_t mx = 0;
    for (const ParamDesc& p : PARAM_LAYOUT)
        if (is_muon_kind(p.kind)) {
            const std::size_t m = static_cast<std::size_t>(std::min(p.rows, p.cols));
            mx = std::max(mx, m * m);
        }
    return mx;
}
inline constexpr std::size_t MUON_MAX_MM = compute_muon_max_mm();

// Total trainable floats = sum of every tensor's element count. This replaces the
// hand-rolled size calculation; the on-disk checkpoint and every parameter arena
// are PARAM_FLOATS long.
consteval std::size_t total_param_floats() {
    std::size_t n = 0;
    for (const ParamDesc& p : PARAM_LAYOUT) n += p.n();
    return n;
}
inline constexpr std::size_t PARAM_FLOATS = total_param_floats();

}  // namespace sub0
