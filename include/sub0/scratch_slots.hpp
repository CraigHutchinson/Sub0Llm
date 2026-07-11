// sub0/scratch_slots.hpp -- the engine-facing foundation for scratch tokens (context-translation layer).
//
// A scratch slot is a reserved token id that is BOUND, per context, to a fragment sequence (an OOV
// construction). This header is the single, dependency-light home (casing.hpp + std only -- no tokenizer,
// so the BACKEND may include it) for: the reserved slot RANGE as a first-class constant, the per-context
// BINDING view the engine consumes, and the pluggable ENCODER that turns a slot's bound fragments into its
// embedding (content-derived slot embeddings -- so a scratch slot carries a signal of its content instead
// of a generic reserved-id vector; the fix the spike showed content-reasoning needs, see project memory
// scratch-tokens-context-translation-layer).
//
// The mechanism is range-AGNOSTIC (everything reads SCRATCH_SLOT_COUNT) and encoder-PLUGGABLE (dispatch on
// SlotEncoding): growing the pool or adding an encoder is a one-line/one-arm change, not a redesign.

#pragma once

#include "sub0/casing.hpp"   // TOK_RESERVED_* slot ids + TOK_MARKER_COUNT (engine-safe: no config/engine dep)

#include <cstddef>
#include <span>
#include <vector>

namespace sub0 {

// The reserved scratch-slot pool: slot i == SCRATCH_SLOT_BASE + i, for i in [0, SCRATCH_SLOT_COUNT). Today
// this is the reserved marker headroom (TOK_RESERVED_4..9 -> 6 slots). Growing it past the marker block is
// a casing.hpp marker-count bump (a new tokenizer format); the code below never hardcodes the count.
constexpr int  SCRATCH_SLOT_BASE  = casing::TOK_RESERVED_4;
constexpr int  SCRATCH_SLOT_COUNT = casing::TOK_MARKER_COUNT - casing::TOK_RESERVED_4;
constexpr int  scratch_slot_id(int i) { return SCRATCH_SLOT_BASE + i; }
constexpr bool is_scratch_slot(int token) {
    return token >= SCRATCH_SLOT_BASE && token < SCRATCH_SLOT_BASE + SCRATCH_SLOT_COUNT;
}

// How a slot's bound fragments become its embedding. MeanPool is implemented; CharEncoder (a learned
// char-CNN/MLP over the fragment rows, order-aware, adds params) and Hash (order-aware pooling without new
// params) are RESERVED extension points -- adding one is a new encode_slot/encode_slot_bwd arm below, with
// no change to any caller. (Until they exist, they fall through to MeanPool.)
enum class SlotEncoding { MeanPool, CharEncoder, Hash };

// The per-context binding view the engine consumes: slot i's bound fragment token ids (empty = unbound).
// Tokenizer-free -- sub0::ScratchTable (scratch.hpp) produces one; the engine only sees this.
struct ScratchBindings {
    std::span<const std::vector<int>> slots;                 // slots[i] = fragments of SCRATCH_SLOT_BASE+i
    SlotEncoding                      encoding = SlotEncoding::MeanPool;
    // CharEncoder only: the learned [C,C] projection weights + a grad accumulator. Carried here (not the
    // model's param arena) so the encoder can be spiked with no layout/checkpoint change -- the owner
    // trains enc_w against enc_w_grad. null for MeanPool. (Single-threaded training: enc_w_grad has no
    // per-thread reduction; a multi-threaded train_batch would race on it -- promote to a model param first.)
    const float*                      enc_w = nullptr;
    float*                            enc_w_grad = nullptr;

    bool bound(int token) const {
        const int i = token - SCRATCH_SLOT_BASE;
        return i >= 0 && i < static_cast<int>(slots.size()) && !slots[static_cast<std::size_t>(i)].empty();
    }
    std::span<const int> fragments(int token) const {
        const int i = token - SCRATCH_SLOT_BASE;
        if (i < 0 || i >= static_cast<int>(slots.size())) return {};
        return std::span<const int>(slots[static_cast<std::size_t>(i)]);
    }
};

// Forward: out[0..C) = f(fragments) under `enc`. `tok_emb` is the [vocab, C] table (row-major); `frags`
// are token ids; empty -> zero row.
//   MeanPool    = mean of the fragments' tok_emb rows (no params -> `enc_w` unused).
//   CharEncoder = sum over fragments of relu(enc_w . row): a learned per-fragment [C,C] projection then a
//                 relu, SUM-pooled. `enc_w` [C,C] (row-major) is REQUIRED. The per-fragment transform +
//                 relu give the model capacity to keep per-char features separable (the presence signal
//                 mean-pool dilutes), which the content-reasoning A/B tests need.
inline void encode_slot(const float* tok_emb, int C, std::span<const int> frags, SlotEncoding enc, float* out,
                        const float* enc_w = nullptr) {
    for (int j = 0; j < C; ++j) out[j] = 0.f;
    if (frags.empty()) return;
    switch (enc) {
        case SlotEncoding::CharEncoder: {
            for (int f : frags) {
                const float* e = tok_emb + static_cast<std::size_t>(f) * C;
                for (int c = 0; c < C; ++c) {
                    const float* w = enc_w + static_cast<std::size_t>(c) * C;
                    float z = 0.f;
                    for (int k = 0; k < C; ++k) z += w[k] * e[k];
                    out[c] += z > 0.f ? z : 0.f;                       // relu, summed over fragments
                }
            }
            break;
        }
        case SlotEncoding::MeanPool:
        default: {   // Hash reserved -> MeanPool until implemented
            const float inv = 1.f / static_cast<float>(frags.size());
            for (int f : frags) {
                const float* row = tok_emb + static_cast<std::size_t>(f) * C;
                for (int j = 0; j < C; ++j) out[j] += row[j];
            }
            for (int j = 0; j < C; ++j) out[j] *= inv;
            break;
        }
    }
}

// Backward (adjoint of encode_slot): scatter the slot-row grad `dout` into the fragment rows of
// `tok_emb_grad` (and, for CharEncoder, into `enc_w_grad`). MeanPool: dout/nfrags per fragment row.
// CharEncoder: recompute z per (fragment,channel), gate by relu, accumulate dW and the fragment-row grad.
inline void encode_slot_bwd(const float* dout, int C, std::span<const int> frags, SlotEncoding enc,
                            float* tok_emb_grad, const float* tok_emb = nullptr,
                            const float* enc_w = nullptr, float* enc_w_grad = nullptr) {
    if (frags.empty()) return;
    switch (enc) {
        case SlotEncoding::CharEncoder: {
            for (int f : frags) {
                const float* e  = tok_emb + static_cast<std::size_t>(f) * C;
                float*       eg = tok_emb_grad + static_cast<std::size_t>(f) * C;
                for (int c = 0; c < C; ++c) {
                    const float* w  = enc_w + static_cast<std::size_t>(c) * C;
                    float*       wg = enc_w_grad + static_cast<std::size_t>(c) * C;
                    float z = 0.f;
                    for (int k = 0; k < C; ++k) z += w[k] * e[k];
                    const float dz = (z > 0.f) ? dout[c] : 0.f;        // relu gate; da[c] = dout[c]
                    for (int k = 0; k < C; ++k) { wg[k] += dz * e[k]; eg[k] += dz * w[k]; }
                }
            }
            break;
        }
        case SlotEncoding::MeanPool:
        default: {
            const float inv = 1.f / static_cast<float>(frags.size());
            for (int f : frags) {
                float* g = tok_emb_grad + static_cast<std::size_t>(f) * C;
                for (int j = 0; j < C; ++j) g[j] += inv * dout[j];
            }
            break;
        }
    }
}

}  // namespace sub0
