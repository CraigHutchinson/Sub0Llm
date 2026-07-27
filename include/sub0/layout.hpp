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
static_assert(LOOP_MIDDLE_LAYERS >= 0 && LOOP_MIDDLE_LAYERS <= 0xffff, "LOOP_MIDDLE_LAYERS must fit 16 bits");
static_assert(LOOP_REPEATS       >= 0 && LOOP_REPEATS       <= 0xffff, "LOOP_REPEATS must fit 16 bits");
inline constexpr std::uint64_t arch_fingerprint(int middle_layers, int repeats, float rope_theta) {
    return (static_cast<std::uint64_t>(static_cast<unsigned>(middle_layers) & 0xffffu) << 48)
         | (static_cast<std::uint64_t>(static_cast<unsigned>(repeats)       & 0xffffu) << 32)
         |  static_cast<std::uint64_t>(std::bit_cast<std::uint32_t>(rope_theta));
}
struct ArchAxes { int middle_layers; int repeats; float rope_theta; };
inline constexpr ArchAxes arch_axes_of(std::uint64_t fp) {
    return ArchAxes{ static_cast<int>((fp >> 48) & 0xffffu),
                     static_cast<int>((fp >> 32) & 0xffffu),
                     std::bit_cast<float>(static_cast<std::uint32_t>(fp & 0xffffffffu)) };
}
inline constexpr std::uint64_t ARCH_FINGERPRINT = arch_fingerprint(LOOP_MIDDLE_LAYERS, LOOP_REPEATS, ROPE_THETA);
// What a file carrying NO fingerprint must be assumed to have been built with. Such a file predates the
// record, so it is necessarily un-looped -- that part is a sound inference. ROPE_THETA is NOT inferable
// the same way (it was settable long before the record existed), so the legacy default is this build's
// own theta: it cannot catch a pre-record theta mismatch, and pretending otherwise would reject every
// legacy model on a non-default-theta build. Guarding theta starts from files written from here on.
inline constexpr std::uint64_t ARCH_FINGERPRINT_LEGACY = arch_fingerprint(0, 1, ROPE_THETA);

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
    };
}

// Parameter tensor count: tok_emb, plus pos_emb only under absolute positions (see HAS_POS_EMB),
// then per transformer block either 10 (plain: ln1, ln2, Wq, Wk, Wv, Wo, W1, b1, W2, b2) or 9
// (gated: ln1, ln2, Wq, Wk, Wv, Wo, Wg, W1, W2 -- SwiGLU, no FFN biases, matching the GGUF/Llama
// convention this variant exists to import -- see gguf-import-feasibility-review), plus 2 more
// (q_norm, k_norm) when USE_QK_NORM, then the tail: ln_f alone when USE_TIED_EMBEDDINGS (the head
// reuses tok_emb, no separate lm_head/lm_bias slot -- the common tied-embedding convention also
// drops the head bias, matching GPT-2/GGUF-style tied models), else ln_f + lm_head + lm_bias (3).
inline constexpr int NUM_PARAMS = 1 + (HAS_POS_EMB ? 1 : 0)
                                   + (USE_GATED_FFN ? 9 : 10) * N_LAYERS
                                   + (USE_QK_NORM ? 2 * N_LAYERS : 0)
                                   + (USE_TIED_EMBEDDINGS ? 1 : 3);

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
enum class PKind : unsigned char {
    TokEmb, PosEmb, Ln1, Ln2, Wq, Wk, Wv, Wo, W1, B1, W2, B2, LnF, LmHead, LmBias, Wg, QNorm, KNorm
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
        if constexpr (USE_GATED_FFN) {
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
    add(1,       D_MODEL, PKind::LnF,    false, false);
    if constexpr (!USE_TIED_EMBEDDINGS) {
        add(D_MODEL, VOCAB,   PKind::LmHead, true,  false);  // head stays full precision
        add(1,       VOCAB,   PKind::LmBias, false, false);
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
           k == PKind::W1 || k == PKind::W2 || k == PKind::Wg;
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
