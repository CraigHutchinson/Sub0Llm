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

#include <cmath>
#include <cstddef>
#include <random>
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

// How a slot's bound fragments become its embedding. All four arms are implemented.
// MeanPool and CharEncoder are BOTH permutation-invariant -- `sum/mean of a per-fragment function` is the
// Deep Sets canonical form (Zaheer et al. 2017) for any function invariant to reordering its inputs, so
// neither can carry fragment ORDER no matter how much training (confirmed: b7896ab's own CharEncoder
// finding was on the order-AGNOSTIC "contains X" probe, not "starts with X" -- see project memory
// scratch-content-embedding-meanpool-vs-charencoder). Hash below closes that gap: a RoPE-style positional
// rotation applied to each fragment (by its position within the slot) BEFORE summing, so the composed
// vector is no longer symmetric under reordering -- see project memory meanpool-alternatives-prior-art-and-math
// for the full prior-art survey (Deep Sets; Holographic Reduced Representations / Plate 1995; Fourier
// HRR / VSA phase-binding, of which this rotate-then-bundle IS the real-valued form) this candidate came
// from, and TODO(order-sensitive-slot-encoding) below for the still-open verification (A/B against
// `content_select_task`, the #4 "starts with X" probe currently excluded from production because nothing
// else beats chance at it).
//   MeanPool    - mean of the fragments' tok_emb rows (no params). Permutation-invariant (order-blind).
//   CharEncoder - a learned per-fragment [C,C] projection + relu, sum-pooled, adds params. Also
//                 permutation-invariant (same canonical form as MeanPool) -- helps PRESENCE/contains
//                 discrimination, not position.
//   Hash        - RoPE-style positional binding: rotate fragment p's row by a p-dependent angle (position
//                 0 = identity, so it passes through unrotated), then SUM across fragments. No learned
//                 params. NOT permutation-invariant (a rotate-then-sum is order-sensitive by construction)
//                 -- spike-verified real-but-modest/noisy edge, see TODO(order-sensitive-slot-encoding).
//   ConvPool    - Kim et al. 2016 ("Character-Aware Neural Language Models") style: a learned width-2
//                 convolution (two [C,C] projections, one per window offset -- packed into `enc_w` as
//                 [2,C,C], reusing CharEncoder's own params/grad pointers rather than adding new fields)
//                 over consecutive fragment pairs, relu, then MAX-POOLED across window positions (a
//                 single-fragment slot falls back to its raw row -- no width-2 window exists). NOT
//                 permutation-invariant: the two offset projections are DIFFERENT matrices, so swapping a
//                 pair's order changes the window's activation, and maxpool's hard per-channel selection
//                 depends on WHICH adjacent pair fired hardest. Candidate 3 from
//                 meanpool-alternatives-prior-art-and-math -- spike-verified real, consistent edge, then
//                 SUPERSEDED by HRR below (see the FINDING comments further down).
//   HRR         - Holographic Reduced Representations (Plate 1995): bind(role_p, filler_p) = CIRCULAR
//                 CONVOLUTION of a fixed pseudo-random unit-RMS "role" vector (keyed by fragment position
//                 p, from a lazily-built, program-lifetime-cached table -- see hrr_role_table below) with
//                 the fragment's tok_emb row, then SUM (bundle) across fragments. No learned params. NOT
//                 permutation-invariant: circular convolution with a position-DEPENDENT role vector means
//                 swapping two fragments' order binds them to different roles, changing the result --
//                 mathematically distinct from Hash's rotation (a different real-valued VSA binding
//                 operator, not just a re-derivation of it), Candidate 2 from
//                 meanpool-alternatives-prior-art-and-math -- spike-verified DECISIVE winner over both
//                 Hash and ConvPool (see the FINDING comments further down); the current leading
//                 candidate, and needs no checkpoint-persistence work to reach production (no params).
//   Scalar      - a fixed "scientific" encoding of the NUMERIC VALUE the fragments spell (sign, base-10
//                 magnitude, leading digits). No params, no grad -- the classical-compute result re-entering
//                 the model as ONE vector (the "brain-swap" scalar re-entry). Bounded regardless of
//                 magnitude, so powers/exponents (2^100, 1.23e45) map in by construction; see below.
enum class SlotEncoding { MeanPool, CharEncoder, Hash, ConvPool, HRR, Scalar };

// The PARAMETER-FREE SlotEncoding arms a content-embed run may select, as a STABLE persisted index --
// CharEncoder/ConvPool are deliberately excluded (they need enc_w, which needs
// TODO(charencoder-production)/[[slot-encoder-checkpoint-persistence-design]]'s param-layout work before
// they can be selected safely). A named enum, not a bare int, specifically so callers (train_stage.cpp,
// gen_stage.cpp, tests) never spell out 0/1/2 directly -- the underlying values ARE part of the on-disk
// config.json contract (registry.hpp's RunConfig::content_embed_kind persists this exact int), so they
// must stay stable, but nothing outside this enum's own definition should need to know that; use
// `sub0::ContentEmbedKind::HRR` etc., cast to `int` only at the RunConfig boundary. Registry-agnostic on
// purpose (this header stays dependency-light -- no registry.hpp include); registry.hpp's own
// `enum_name("content_embed_kind", ...)` list ({"meanpool","hash","hrr"}) must stay in the SAME order as
// this enum -- cross-referenced in both places since the two headers deliberately don't share code.
enum class ContentEmbedKind : int { MeanPool = 0, Hash = 1, HRR = 2 };

constexpr SlotEncoding content_embed_encoding_of(int idx) {
    switch (static_cast<ContentEmbedKind>(idx)) {
        case ContentEmbedKind::Hash: return SlotEncoding::Hash;
        case ContentEmbedKind::HRR:  return SlotEncoding::HRR;
        case ContentEmbedKind::MeanPool: default: return SlotEncoding::MeanPool;
    }
}

// The one canonical index -> display-name mapping (log lines, error messages) -- every call site
// (train_stage.cpp, gen_stage.cpp) uses THIS instead of spelling out its own ternary chain, so the
// names can never drift out of sync between them. Derives from content_embed_encoding_of (not a second
// switch over the raw int) so there is exactly ONE place that interprets the persisted index.
constexpr const char* content_embed_kind_name(int idx) {
    switch (content_embed_encoding_of(idx)) {
        case SlotEncoding::Hash: return "hash";
        case SlotEncoding::HRR:  return "hrr";
        default: return "meanpool";
    }
}

// HRR's fixed "role" vectors: one per fragment POSITION (0-indexed, capped at HRR_MAX_POS -- generous
// headroom over SCRATCH_SLOT_COUNT=6 for any future longer-fragment use, e.g. multi-token pattern spans),
// pseudo-random with a FIXED seed (deterministic across runs/processes -- no state to persist, unlike a
// learned param) and unit-RMS-ish scale (matches the token-embed amplitude convention, see project memory
// conditioning-embedding-amplitude, so a single binding's magnitude lands near an ordinary embedding row's).
// Built ONCE, lazily, on first use -- C is effectively a compile-time constant (D_MODEL) for the life of a
// process, so this is the same "cache once, reuse" shape as cpu_affinity.hpp's detect_core_order().
// THREAD-SAFETY (the lambda-init form is load-bearing, not style): train_batch embeds windows on multiple
// OMP worker threads, each of which reaches encode_slot(HRR) -> here on the FIRST content-embed training
// step -- with HRR as the production content-embed default, several threads genuinely arrive concurrently.
// A magic static's INITIALIZATION is guaranteed exactly-once/blocking by the language; a lazy
// `if (empty()) resize+fill` after it is NOT (two threads both resizing the same vector is UB -- a real
// latent race found in review 2026-07-16, never bitten only by first-arrival timing luck). So the whole
// build runs inside the initializer.
constexpr int HRR_MAX_POS = 16;
inline const std::vector<float>& hrr_role_table(int C) {
    static const std::vector<float> table = [C] {
        std::vector<float> t(static_cast<std::size_t>(HRR_MAX_POS) * static_cast<std::size_t>(C));
        std::mt19937 rng(0xC0FFEEu);
        std::normal_distribution<float> nd(0.f, 1.f / std::sqrt(static_cast<float>(C)));
        for (float& x : t) x = nd(rng);
        return t;
    }();
    return table;
}

// FINDING (2026-07-16, spike scale d196, tests/scratchspike_engine_tests.cpp "CONTENT (starts-with)
// reasoning -- Hash/RoPE positional-binding A/B"): on content_select_task (#4, "which slot's OOV starts
// with letter X" -- excluded from build_dataset_scratch because nothing else beats chance ~1/K): a first
// single run looked strong (MeanPool held-out 0.375, Hash 0.450); a 3-training-seed follow-up (weight init
// is deterministic here, only training ORDER varies) tempered that -- mean held-out MeanPool 0.356, Hash
// 0.403 (chance 0.333), with one seed showing BOTH arms collapse to an IDENTICAL degenerate output (no gap
// that seed -- the same failure mode b7896ab's own CONTAINS A/B first noted) and the other two showing a
// clear Hash win. Directionally consistent (Hash never underperforms MeanPool across seeds) but NOT
// statistically decisive at n=3 (paired t~2.0, df=2). Read this as "promising direction confirmed across
// seeds, not yet a strong/reliable effect" -- weaker confidence than the first run alone suggested. Not yet
// wired into any production curriculum or CLI flag -- SlotEncoding::Hash exists and works but nothing
// selects it outside this spike test.
// NOTE (motivation, not a correctness caveat): positional queries are NOT otherwise unanswerable -- the
// existing combine/uncombine mechanism already gives a resolve-then-attend route (reveal a slot's real
// fragment bytes into the stream, then plain attention handles "starts with"/"ends with"/Nth-char over
// literal tokens), exactly what #4b's CoT variant already validates. What a working one-hop encoder would
// buy is answering in ONE attention hop over the slot token instead of a multi-step resolve trace -- an
// efficiency win, not the only path to correctness. See project memory
// meanpool-alternatives-prior-art-and-math for the full reframing.
//
// FINDING (2026-07-16, same day, ConvPool): multi-seeded from the start this time, baseline = fully
// plain (no content-embed), matching CharEncoder's own methodology. Mean held-out: plain 0.344 (at chance
// 0.333, as expected), ConvPool 0.458 (gap +0.114 -- ~2.4x Hash's own gap). ConvPool won EVERY seed, no
// ties or degenerate collapse (unlike Hash's seed 2). SUPERSEDED same day by HRR below.
//
// FINDING (2026-07-16, same day, later still, HRR): mean held-out MeanPool 0.356, HRR 0.744 -- more than
// DOUBLE chance (0.333), strong and consistent across all 3 seeds (no ties, no collapse), held-out
// tracking drilled closely every seed (genuine generalization). Gap +0.388 is ~3.4x ConvPool's and ~8x
// Hash's -- not incremental, qualitatively different. HRR is now the CLEAR LEADING candidate, and unlike
// ConvPool/CharEncoder it needs NO learned params, so it does not depend on
// TODO(charencoder-production)-style checkpoint persistence to reach production -- see project memory
// meanpool-alternatives-prior-art-and-math for the full per-seed numbers and the plausible reason
// (quasi-orthogonal random role vectors interfere far less than Hash's low-rank pairwise rotation or
// ConvPool's locally-windowed view -- the STRUCTURAL binding operator matters more than learned capacity).
constexpr float HASH_ROPE_THETA = 10000.f;   // matches the engine's own ROPE_THETA (backend_cpu.cpp) --
                                             // this header can't see that constant (no core.hpp dep), so
                                             // it's a local copy; the value is conventional, not coupled.

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
    // TODO(charencoder-production): enc_w has no checkpoint persistence -- promoting it to a real
    // PARAM_LAYOUT entry (surviving save_model/load_model) is the prerequisite before --content-embed can
    // select CharEncoder instead of hardcoded MeanPool (see the smooth-noodling-kurzweil plan's
    // "explicitly out of scope" list and project memory scratch-content-embedding-meanpool-vs-charencoder).
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

// --- Persistent (unbounded) slot range -- SPIKE ------------------------------------------------------
// A SECOND, structurally different id range from the ephemeral pool above: ids >= `base` (the caller's
// VOCAB), open-ended (not capped at a fixed pool size), for a persistent compound-word cache (e.g. a
// recurring multi-token OOV gets ONE stable id forever, composed live from its fragments -- see
// docs/DETERMINISTIC_MECHANISMS.md's "persistent compound-word cache" extension note). Deliberately a
// SEPARATE type from ScratchBindings, not a runtime-configurable base on it: the two ranges have
// different LIFETIMES (this one is global/read-once/immutable for the process, never swapped per-window
// the way the ephemeral pool's thread_local binding is) and different SIZE characteristics (a handful of
// ephemeral slots reused per context vs. a potentially large, monotonically-growing persistent table) --
// forcing them into one type would blur both. `base` is NOT baked in as a constant (unlike
// SCRATCH_SLOT_BASE) because it's VOCAB, a fact this dependency-light header deliberately doesn't know
// (no core.hpp/config include) -- callers that DO have VOCAB in scope pass it explicitly.
//
// Engine safety invariant this exists to close: VOCAB is compile-time-fixed, so an id >= VOCAB can never
// be sampled by the model (sample_token is hard-bounded to [0,VOCAB)) -- such an id can only ever be
// INJECTED by a harness. But nothing previously stopped one from reaching the plain `tab[id, ...]`
// embedding-table lookup and reading out of bounds if it ever did arrive unbound. `is_persistent_slot`
// is meant to be checked UNCONDITIONALLY for any id >= base, before any raw table index -- see
// persistent_fragments() below, which is null/unbound-safe by construction (never a source of the OOB
// this is closing).
struct PersistentBindings {
    std::span<const std::vector<int>> slots;                 // slots[i] = fragments of base+i
    int                                base = 0;              // typically VOCAB; NOT compile-time (see above)
    SlotEncoding                       encoding = SlotEncoding::MeanPool;
    // Learned-encoder weights (ConvPool [2,C,C] / CharEncoder [C,C]) + grad accumulator, mirroring
    // ScratchBindings' own enc_w/enc_w_grad exactly -- null for the parameter-free encodings
    // (MeanPool/Hash/HRR), REQUIRED non-null before selecting a learned one (encode_slot dereferences
    // enc_w unconditionally for those arms; before these fields existed, the backend hardcoded nullptr
    // at every persistent call site, so merely SETTING encoding=ConvPool here was undefined behaviour --
    // a real hazard closed by this plumbing, found during the 2026-07-16 encoder-shootout review). Same
    // caveat as the ephemeral pool's: enc_w_grad has no per-thread reduction, so only a single-threaded
    // training loop may drive a learned encoder (the multi-threaded train_batch path doesn't accept
    // PersistentBindings anyway -- see persistent_scratchspike_engine_tests.cpp).
    const float*                       enc_w = nullptr;
    float*                             enc_w_grad = nullptr;

    bool bound(int token) const {
        const int i = token - base;
        return i >= 0 && i < static_cast<int>(slots.size()) && !slots[static_cast<std::size_t>(i)].empty();
    }
    std::span<const int> fragments(int token) const {
        const int i = token - base;
        if (i < 0 || i >= static_cast<int>(slots.size())) return {};
        return std::span<const int>(slots[static_cast<std::size_t>(i)]);
    }
};

constexpr bool is_persistent_slot(int token, int base) { return token >= base; }

// Null-safe fragment lookup: no table set, or `token` not bound in it, -> an EMPTY span (encode_slot's
// documented empty-fragments contract is a zero row) rather than requiring every call site to branch on
// `pb` being null. This is what lets engine call sites route EVERY id >= base through the same compose
// path unconditionally, closing the OOB risk above regardless of whether a real table is bound yet.
inline std::span<const int> persistent_fragments(const PersistentBindings* pb, int token) {
    return (pb && pb->bound(token)) ? pb->fragments(token) : std::span<const int>{};
}

// Forward: out[0..C) = f(fragments) under `enc`. `tok_emb` is the [vocab, C] table (row-major); `frags`
// are token ids; empty -> zero row.
//   MeanPool    = mean of the fragments' tok_emb rows (no params -> `enc_w` unused).
//   CharEncoder = sum over fragments of relu(enc_w . row): a learned per-fragment [C,C] projection then a
//                 relu, SUM-pooled. `enc_w` [C,C] (row-major) is REQUIRED. The per-fragment transform +
//                 relu give the model capacity to keep per-char features separable (the presence signal
//                 mean-pool dilutes), which the content-reasoning A/B tests need.
//   Hash        = sum over fragments of rotate(row, ang(position)): each fragment's row is rotated by an
//                 angle depending on its POSITION within `frags` (0-indexed; position 0 = identity), then
//                 summed. No params (`enc_w` unused). Interleaved-pair rotation, same convention as
//                 op_rope (backend_cpu.cpp) but full-width (no head split) and over fragment position
//                 instead of token position -- see HASH_ROPE_THETA above.
//   ConvPool    = max over adjacent-pair window positions p of relu(enc_w[0] . row_p + enc_w[1] . row_{p+1}):
//                 a learned width-2 convolution (TWO [C,C] projections packed into `enc_w` as [2,C,C] --
//                 enc_w[0..C*C) and enc_w[C*C..2*C*C)) then max-pooled over window position. A single
//                 fragment (no width-2 window exists) falls through as its own raw row.
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
        case SlotEncoding::Hash: {
            const int half = C / 2;
            for (std::size_t p = 0; p < frags.size(); ++p) {
                const float* row = tok_emb + static_cast<std::size_t>(frags[p]) * C;
                for (int m = 0; m < half; ++m) {
                    const float ang = static_cast<float>(p) * std::pow(HASH_ROPE_THETA, -2.f * m / C);
                    const float cs = std::cos(ang), sn = std::sin(ang);
                    const float x0 = row[2 * m], x1 = row[2 * m + 1];
                    out[2 * m]     += x0 * cs - x1 * sn;
                    out[2 * m + 1] += x0 * sn + x1 * cs;
                }
                if (2 * half < C) out[C - 1] += row[C - 1];   // odd tail: identity, still summed
            }
            break;
        }
        case SlotEncoding::ConvPool: {
            const int n = static_cast<int>(frags.size());
            if (n == 1) {   // no width-2 window -- pass the lone fragment through as-is
                const float* row = tok_emb + static_cast<std::size_t>(frags[0]) * C;
                for (int c = 0; c < C; ++c) out[c] = row[c];
                break;
            }
            const float* w0 = enc_w;              // [C,C]
            const float* w1 = enc_w + static_cast<std::size_t>(C) * C;   // [C,C]
            for (int c = 0; c < C; ++c) {
                float best = 0.f;                 // relu floor: a channel with no positive window stays 0
                const float* wr0 = w0 + static_cast<std::size_t>(c) * C;
                const float* wr1 = w1 + static_cast<std::size_t>(c) * C;
                for (int p = 0; p + 1 < n; ++p) {
                    const float* e0 = tok_emb + static_cast<std::size_t>(frags[static_cast<std::size_t>(p)]) * C;
                    const float* e1 = tok_emb + static_cast<std::size_t>(frags[static_cast<std::size_t>(p + 1)]) * C;
                    float z = 0.f;
                    for (int k = 0; k < C; ++k) z += wr0[k] * e0[k] + wr1[k] * e1[k];
                    if (z > best) best = z;        // relu + max-pool over window positions, fused
                }
                out[c] = best;
            }
            break;
        }
        case SlotEncoding::HRR: {
            const std::vector<float>& roles = hrr_role_table(C);
            for (std::size_t p = 0; p < frags.size(); ++p) {
                const float* filler = tok_emb + static_cast<std::size_t>(frags[p]) * C;
                const std::size_t pi = p < static_cast<std::size_t>(HRR_MAX_POS) ? p : static_cast<std::size_t>(HRR_MAX_POS - 1);
                const float* role = roles.data() + pi * static_cast<std::size_t>(C);
                // Circular convolution: out[n] += sum_k role[k] * filler[(n-k) mod C] -- bind(role_p, filler_p).
                for (int n = 0; n < C; ++n) {
                    float s = 0.f;
                    for (int k = 0; k < C; ++k) {
                        int idx = n - k; if (idx < 0) idx += C;
                        s += role[k] * filler[idx];
                    }
                    out[n] += s;
                }
            }
            break;
        }
        case SlotEncoding::MeanPool:
        default: {
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
// Hash: rotation is orthogonal, so its adjoint is the INVERSE rotation (transpose = rotate by -position)
// applied to dout -- no params, no tok_emb read (the angle depends only on position, not the row's value).
// ConvPool: standard maxpool adjoint -- only the ARGMAX window position (recomputed, not cached) gets
// gradient per channel, gated by the same relu-vs-zero-floor condition the forward used.
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
        case SlotEncoding::Hash: {
            const int half = C / 2;
            for (std::size_t p = 0; p < frags.size(); ++p) {
                float* g = tok_emb_grad + static_cast<std::size_t>(frags[p]) * C;
                for (int m = 0; m < half; ++m) {
                    const float ang = static_cast<float>(p) * std::pow(HASH_ROPE_THETA, -2.f * m / C);
                    const float cs = std::cos(ang), sn = std::sin(ang);
                    const float d0 = dout[2 * m], d1 = dout[2 * m + 1];
                    g[2 * m]     += d0 * cs + d1 * sn;
                    g[2 * m + 1] += -d0 * sn + d1 * cs;
                }
                if (2 * half < C) g[C - 1] += dout[C - 1];
            }
            break;
        }
        case SlotEncoding::ConvPool: {
            const int n = static_cast<int>(frags.size());
            if (n == 1) {
                float* g = tok_emb_grad + static_cast<std::size_t>(frags[0]) * C;
                for (int c = 0; c < C; ++c) g[c] += dout[c];
                break;
            }
            const float* w0  = enc_w;
            const float* w1  = enc_w + static_cast<std::size_t>(C) * C;
            float*       wg0 = enc_w_grad;
            float*       wg1 = enc_w_grad + static_cast<std::size_t>(C) * C;
            for (int c = 0; c < C; ++c) {
                const float* wr0 = w0 + static_cast<std::size_t>(c) * C;
                const float* wr1 = w1 + static_cast<std::size_t>(c) * C;
                float best = 0.f; int best_p = -1;   // -1: no window beat the relu floor -> no gradient
                for (int p = 0; p + 1 < n; ++p) {
                    const float* e0 = tok_emb + static_cast<std::size_t>(frags[static_cast<std::size_t>(p)]) * C;
                    const float* e1 = tok_emb + static_cast<std::size_t>(frags[static_cast<std::size_t>(p + 1)]) * C;
                    float z = 0.f;
                    for (int k = 0; k < C; ++k) z += wr0[k] * e0[k] + wr1[k] * e1[k];
                    if (z > best) { best = z; best_p = p; }
                }
                if (best_p < 0) continue;
                const float dz = dout[c];
                const float* e0 = tok_emb + static_cast<std::size_t>(frags[static_cast<std::size_t>(best_p)]) * C;
                const float* e1 = tok_emb + static_cast<std::size_t>(frags[static_cast<std::size_t>(best_p + 1)]) * C;
                float* g0 = tok_emb_grad + static_cast<std::size_t>(frags[static_cast<std::size_t>(best_p)]) * C;
                float* g1 = tok_emb_grad + static_cast<std::size_t>(frags[static_cast<std::size_t>(best_p + 1)]) * C;
                float* wgr0 = wg0 + static_cast<std::size_t>(c) * C;
                float* wgr1 = wg1 + static_cast<std::size_t>(c) * C;
                for (int k = 0; k < C; ++k) {
                    wgr0[k] += dz * e0[k]; g0[k] += dz * wr0[k];
                    wgr1[k] += dz * e1[k]; g1[k] += dz * wr1[k];
                }
            }
            break;
        }
        case SlotEncoding::HRR: {
            // Adjoint of circular convolution by a FIXED role vector is circular CORRELATION by that same
            // role vector: d(filler)[j] = sum_n dout[n] * role[(n-j) mod C]. No params, no tok_emb read
            // (role depends only on position, like Hash's rotation).
            const std::vector<float>& roles = hrr_role_table(C);
            for (std::size_t p = 0; p < frags.size(); ++p) {
                float* g = tok_emb_grad + static_cast<std::size_t>(frags[p]) * C;
                const std::size_t pi = p < static_cast<std::size_t>(HRR_MAX_POS) ? p : static_cast<std::size_t>(HRR_MAX_POS - 1);
                const float* role = roles.data() + pi * static_cast<std::size_t>(C);
                for (int j = 0; j < C; ++j) {
                    float s = 0.f;
                    for (int n = 0; n < C; ++n) {
                        int idx = n - j; if (idx < 0) idx += C;
                        s += dout[n] * role[idx];
                    }
                    g[j] += s;
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
