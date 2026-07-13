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
#include <string>
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

// How a slot's bound fragments become its embedding. MeanPool + CharEncoder + Scalar are implemented; Hash
// (order-aware pooling without new params) is a RESERVED extension point -- adding one is a new
// encode_slot/encode_slot_bwd arm below, with no change to any caller. (Until it exists, it falls through
// to MeanPool.)
//   MeanPool    - mean of the fragments' tok_emb rows (no params).
//   CharEncoder - a learned per-fragment [C,C] projection + relu, sum-pooled (order-sensitive, adds params).
//   Scalar      - a fixed "scientific" encoding of the NUMERIC VALUE the fragments spell (sign, base-10
//                 magnitude, leading digits). No params, no grad -- the classical-compute result re-entering
//                 the model as ONE vector (the "brain-swap" scalar re-entry). Bounded regardless of
//                 magnitude, so powers/exponents (2^100, 1.23e45) map in by construction; see below.
enum class SlotEncoding { MeanPool, CharEncoder, Hash, Scalar };

// --- Scalar encoding of a numeric value -------------------------------------------------------------
// The classical computer resolves an op to an exact value (kept, exactly, in the slot's digit fragments);
// Scalar re-embeds that value as ONE fixed vector so an INTERMEDIATE result can re-enter the model at a
// single position instead of N digit tokens. It is deliberately floating-point-shaped -- sign, a base-10
// exponent (order of magnitude), and the leading significant digits -- so it is BOUNDED no matter how big
// the value is: this is what lets powers and exponents (2^100 ~ 1.267e30, 1.23e45) share one encoding with
// small integers. It is lossy in the low-order digits by design (those live in the exact fragment binding
// for the terminal/exact path); the vector carries magnitude + leading digits for the model to reason with.
constexpr int   SCALAR_MANT_DIGITS = 4;      // leading significant digits captured in the mantissa block
constexpr float SCALAR_EXP_SCALE   = 64.f;   // exponent normalizer: |E| up to ~64 decades stays O(1)
constexpr float SCALAR_AMP         = 1.f;    // output amplitude (match token-embed scale; see memory
                                             // conditioning-embedding-amplitude when training with it)

// The parsed scientific view of the value the fragments spell: value ~= sign * mant[0].mant[1..] * 10^exp.
// `ok` is false when the fragments are not a plain decimal/scientific literal (e.g. an un-evaluated symbolic
// form like "2^100") -- the encoder then emits a zero row rather than guessing.
struct ScalarParts {
    int  sign = 0;                          // -1, 0 (value is zero), or +1
    int  exp  = 0;                          // base-10 exponent of the leading significant digit
    int  mant[SCALAR_MANT_DIGITS] = {};     // leading significant digits, high-order first (0-padded)
    bool ok   = false;
};

// Parse a slot's fragment token ids (digit/sign/point/exponent chars -- byte tokens, id == ASCII) into a
// ScalarParts. Accepts [sign] digits [. digits] [ (e|E) [sign] digits ]; anything else -> ok=false.
inline ScalarParts parse_scalar(std::span<const int> frags) {
    ScalarParts p;
    // Reconstruct the literal (byte tokens carry their ASCII value directly; ignore non-byte ids).
    std::string s;
    s.reserve(frags.size());
    for (int f : frags) if (f >= 0 && f < 128) s.push_back(static_cast<char>(f));
    std::size_t i = 0;
    int sign = 1;
    if (i < s.size() && (s[i] == '+' || s[i] == '-')) { if (s[i] == '-') sign = -1; ++i; }

    std::vector<int> digits;                 // every digit, in order, across the decimal point
    int  int_digits = 0;                     // count of digits before the '.'
    bool saw_point = false, saw_digit = false;
    for (; i < s.size(); ++i) {
        const char c = s[i];
        if (c >= '0' && c <= '9') { digits.push_back(c - '0'); if (!saw_point) ++int_digits; saw_digit = true; }
        else if (c == '.' && !saw_point) saw_point = true;
        else break;                          // hand off to the exponent scan (or reject)
    }
    if (!saw_digit) return p;                // no mantissa digits -> not a number

    int exp_explicit = 0;
    if (i < s.size() && (s[i] == 'e' || s[i] == 'E')) {
        ++i;
        int esign = 1;
        if (i < s.size() && (s[i] == '+' || s[i] == '-')) { if (s[i] == '-') esign = -1; ++i; }
        bool esaw = false;
        for (; i < s.size() && s[i] >= '0' && s[i] <= '9'; ++i) { exp_explicit = exp_explicit * 10 + (s[i] - '0'); esaw = true; }
        if (!esaw) return p;                 // dangling exponent marker -> reject
        exp_explicit *= esign;
    }
    if (i != s.size()) return p;             // trailing junk (e.g. a '^' symbolic form) -> not a plain literal

    // Locate the first non-zero significant digit; its base-10 exponent anchors the mantissa.
    int k = 0;
    while (k < static_cast<int>(digits.size()) && digits[static_cast<std::size_t>(k)] == 0) ++k;
    if (k == static_cast<int>(digits.size())) { p.sign = 0; p.ok = true; return p; }   // value is exactly zero

    // E = position of the leading significant digit relative to the ones place, plus any explicit exponent.
    p.exp  = (int_digits - 1 - k) + exp_explicit;
    p.sign = sign;
    for (int d = 0; d < SCALAR_MANT_DIGITS && k + d < static_cast<int>(digits.size()); ++d)
        p.mant[d] = digits[static_cast<std::size_t>(k + d)];
    p.ok = true;
    return p;
}

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
        case SlotEncoding::Scalar: {
            const ScalarParts p = parse_scalar(frags);
            if (!p.ok) break;                    // not a plain literal -> zero row (already zeroed)
            // Structured, bounded layout: [ sign, exp/scale, mant0/9, mant1/9, ... ]; the rest stays 0.
            int idx = 0;
            auto put = [&](float v) { if (idx < C) out[idx++] = v * SCALAR_AMP; };
            put(static_cast<float>(p.sign));
            put(static_cast<float>(p.exp) / SCALAR_EXP_SCALE);
            for (int d = 0; d < SCALAR_MANT_DIGITS; ++d) put(static_cast<float>(p.mant[d]) / 9.f);
            break;
        }
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
        case SlotEncoding::Scalar:
            break;   // fixed encoding of discrete digit tokens -> stop-gradient (no params, no tok_emb read)
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
