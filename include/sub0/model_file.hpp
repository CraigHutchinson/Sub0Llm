// sub0/model_file.hpp -- the on-disk model file's own layout: the `S0L5` header struct and the three
// trailing records that follow the parameter blob.
//
// WHY THIS IS A HEADER NOW. It was an anonymous-namespace struct inside engine_core.cpp, which was
// exactly right while engine_core.cpp was the only thing that could write one. WP4c adds a second
// writer -- tools/sub0llm-transplant.cpp, which is compiled against a DIFFERENT config header (the
// real Qwen4 axes, 4 layers) and so cannot link the engine at all. Copying the struct into the tool
// would create a second definition of a fixed-size binary format, and AGENTS.md S3 names that as this
// project's highest-blast-radius category of change: a field reordered on one side and not the other
// makes every existing checkpoint fail to load, silently, on a multi-day training run.
//
// So the definition moves here and BOTH writers include it. Nothing about the bytes changes -- the
// struct is transcribed field for field, the static_assert below pins its size, and every consumer's
// arithmetic is unchanged.
//
// FORMAT, in order (see engine_core.cpp's save_model/load_model for the reading side and for why each
// trailer is appended rather than folded into the header):
//   Header                    sizeof(Header) bytes, config-pinning
//   param_t[PARAM_FLOATS]     the blob, in PARAM_LAYOUT order -- element type per Header::param_dtype
//   uint64 tokenizer_fp       which vocab these weights were trained against (0 = unknown)
//   uint64 arch_fingerprint   the shape-NEUTRAL, computation-changing axes (LoopSplit, ROPE_THETA)
//   uint64 arch_fingerprint2  the second such word (GDN stride, experts_per_tok, QSA, rotary_dim)

#pragma once

#include "sub0_config.hpp"
#include "sub0/layout.hpp"

#include <cstddef>
#include <cstdint>

namespace sub0 {

// --- B24 phase 1: the parameter blob's element type ------------------------------------------------
//
// The blob was implicitly always f32. docs/BACKBONE_PRECISION.md S1 halves the resident backbone by
// storing it at bf16 instead, which means a reader must be told how many bytes an element is BEFORE it
// reads the blob -- so this cannot be a trailing record the way the tokenizer/architecture fingerprints
// are, and it has to be in the Header.
//
// Defined HERE, not as the generated config's `Dtype`, on purpose: sub0llm-transplant is a second
// writer of this format compiled against tests/qwen4_real_axes/sub0_config.hpp, which has no `Dtype`
// enum at all (it is a model-axes header, not a full generated config). The FILE's own vocabulary of
// element types must not depend on which of the two config headers a writer happened to be built
// against. The engine maps its own `PARAM_DTYPE` onto this in engine_core.cpp.
// B33 (docs/BACKBONE_PRECISION.md S2): FP8 adds a third element width. It is a flat E4M3 float
// (sub0/fp8.hpp), NOT any GGUF-style per-block-scaled quant format -- see that header's own comment for
// why the two must not be confused.
enum class ParamDtype : std::int32_t { F32 = 0, BF16 = 1, FP8 = 2 };

// Bytes per stored element. Returns 0 for an unrecognised tag -- callers use that as the "refuse"
// signal rather than guessing 4 (see load_model, which would otherwise read the wrong number of bytes
// and produce a silently wrong model, the exact outcome AGENTS.md S3 exists to prevent).
[[nodiscard]] constexpr int param_dtype_bytes(std::int32_t tag) {
    switch (tag) {
        case static_cast<std::int32_t>(ParamDtype::F32):  return 4;
        case static_cast<std::int32_t>(ParamDtype::BF16): return 2;
        case static_cast<std::int32_t>(ParamDtype::FP8):  return 1;
        default: return 0;
    }
}
[[nodiscard]] constexpr const char* param_dtype_name(std::int32_t tag) {
    switch (tag) {
        case static_cast<std::int32_t>(ParamDtype::F32):  return "f32";
        case static_cast<std::int32_t>(ParamDtype::BF16): return "bf16";
        case static_cast<std::int32_t>(ParamDtype::FP8):  return "fp8";
        default: return "<unrecognised>";
    }
}

// Total on-disk size of a well-formed model file at a given element width: header + blob + the three
// 8-byte trailers. Used BOTH as a write-side expectation and, on the read side, as an INDEPENDENT
// cross-check of the declared tag -- see load_model's own comment for why an independent check matters
// here specifically (the tag lives in bytes that were structure padding in every file written before
// B24, and padding was never guaranteed to be zero).
[[nodiscard]] constexpr std::uint64_t model_file_bytes(std::uint64_t header_bytes,
                                                       std::uint64_t param_floats, int elem_bytes) {
    return header_bytes + param_floats * static_cast<std::uint64_t>(elem_bytes) + 3u * 8u;
}

// NOTE: adding a field here changes sizeof(Header) and breaks resuming EVERY existing checkpoint
// (old bytes no longer line up) -- so USE_GATED_FFN deliberately does NOT get its own field. It does
// not need one: param_floats already differs between a gated and a non-gated build at identical
// d_model/n_layers/n_heads/seq_len/vocab (gated: 3*C*F per layer; plain: 2*C*F+F+C; these are never
// equal for this project's fixed F=4*C convention), so the existing param_floats check in load_model
// already catches a gated/non-gated mismatch -- one less checkpoint-format break for an orthogonal
// feature.
struct ModelHeader {
    char magic[4] = {'S', '0', 'L', '5'};
    int d_model = D_MODEL, n_layers = N_LAYERS, n_heads = N_HEADS;
    int d_ff = D_FF, seq_len = SEQ_LEN, vocab = VOCAB, ternary = USE_TERNARY;
    int pos_encoding = static_cast<int>(POS_ENCODING);   // 0 = absolute learned, 1 = RoPE
    // B24: the blob's element type (ParamDtype above). This field occupies the FOUR BYTES THAT WERE
    // ALREADY STRUCTURE PADDING at offset 36 -- 4 magic + 8 ints = 36, and the uint64 below forced a
    // pad to 40 -- so sizeof(ModelHeader) is UNCHANGED at 48 and every field that existed before B24
    // still sits at exactly the byte offset it always did. That is AGENTS.md S3's rule 2 (additive,
    // never reshuffle) taken as far as it can go for a field that cannot be a trailing record.
    //
    // The one thing it does NOT give for free is a defined value in OLDER files: `Header h;` is
    // default-initialisation, which leaves padding indeterminate, so a pre-B24 file's bytes here are
    // whatever was on that writer's stack. load_model therefore treats the file's TOTAL SIZE as the
    // authoritative discriminator and this tag as corroboration -- see its comment. New writers always
    // set it, because it is now a real member with a default.
    std::int32_t param_dtype = static_cast<std::int32_t>(ParamDtype::F32);
    std::uint64_t param_floats = PARAM_FLOATS;
};

// The size this format has always had on this toolchain: 4 magic + 8 ints = 36, padded to 40 for the
// uint64's alignment, + 8 = 48. Pinned so a compiler or field change that moves it is a build error
// rather than a file every existing model silently fails to load. B24's param_dtype was placed INTO
// that pad precisely so this assertion keeps holding unchanged.
static_assert(sizeof(ModelHeader) == 48, "the S0L5 header's on-disk size must not change");
static_assert(alignof(ModelHeader) == 8);
static_assert(offsetof(ModelHeader, param_dtype) == 36,
              "param_dtype must occupy the pre-existing pad at byte 36, not move any older field");
static_assert(offsetof(ModelHeader, param_floats) == 40,
              "param_floats must stay at byte 40, exactly where every pre-B24 file has it");

}  // namespace sub0
