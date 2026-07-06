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

#include <array>
#include <cstddef>

namespace sub0 {

// Parameter tensor count: tok_emb + pos_emb (2), then per transformer block either 10
// (plain: ln1, ln2, Wq, Wk, Wv, Wo, W1, b1, W2, b2) or 9 (gated: ln1, ln2, Wq, Wk, Wv, Wo,
// Wg, W1, W2 -- SwiGLU, no FFN biases, matching the GGUF/Llama convention this variant
// exists to import -- see gguf-import-feasibility-review), plus 2 more (q_norm, k_norm)
// when USE_QK_NORM, then the tail: ln_f alone when USE_TIED_EMBEDDINGS (the head reuses
// tok_emb, no separate lm_head/lm_bias slot -- the common tied-embedding convention also
// drops the head bias, matching GPT-2/GGUF-style tied models), else ln_f + lm_head +
// lm_bias (3).
inline constexpr int NUM_PARAMS = 2 + (USE_GATED_FFN ? 9 : 10) * N_LAYERS
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
    add(SEQ_LEN, D_MODEL, PKind::PosEmb, false, false);
    for (int l = 0; l < N_LAYERS; ++l) {
        add(1,       D_MODEL, PKind::Ln1, false, false);
        add(1,       D_MODEL, PKind::Ln2, false, false);
        add(D_MODEL, D_MODEL, PKind::Wq,  true,  true);
        add(D_MODEL, D_MODEL, PKind::Wk,  true,  true);
        add(D_MODEL, D_MODEL, PKind::Wv,  true,  true);
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
