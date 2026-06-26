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

#include "sub0_config.hpp"  // D_MODEL, N_LAYERS, N_HEADS, SEQ_LEN, D_FF, VOCAB, USE_TERNARY

#include <array>
#include <cstddef>

namespace sub0 {

// Parameter tensor count: tok_emb + pos_emb (2), then 10 per transformer block
// (ln1, ln2, Wq, Wk, Wv, Wo, W1, b1, W2, b2), then ln_f + lm_head + lm_bias (3).
inline constexpr int NUM_PARAMS = 2 + 10 * N_LAYERS + 3;

// Role of each parameter tensor. Lets a backend special-case kinds (e.g. concat the
// Q/K/V projections into one GEMM, or keep the lm_head full precision) without
// re-parsing offsets. The enumerator order is NOT the layout order — PARAM_LAYOUT is.
enum class PKind : unsigned char {
    TokEmb, PosEmb, Ln1, Ln2, Wq, Wk, Wv, Wo, W1, B1, W2, B2, LnF, LmHead, LmBias
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
        add(D_MODEL, D_FF,    PKind::W1,  true,  true);
        add(1,       D_FF,    PKind::B1,  false, false);
        add(D_FF,    D_MODEL, PKind::W2,  true,  true);
        add(1,       D_MODEL, PKind::B2,  false, false);
    }
    add(1,       D_MODEL, PKind::LnF,    false, false);
    add(D_MODEL, VOCAB,   PKind::LmHead, true,  false);  // head stays full precision
    add(1,       VOCAB,   PKind::LmBias, false, false);
    return t;
}

inline constexpr std::array<ParamDesc, NUM_PARAMS> PARAM_LAYOUT = make_param_layout();

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
