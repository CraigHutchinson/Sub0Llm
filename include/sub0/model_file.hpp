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
//   float[PARAM_FLOATS]       the blob, in PARAM_LAYOUT order
//   uint64 tokenizer_fp       which vocab these weights were trained against (0 = unknown)
//   uint64 arch_fingerprint   the shape-NEUTRAL, computation-changing axes (LoopSplit, ROPE_THETA)
//   uint64 arch_fingerprint2  the second such word (GDN stride, experts_per_tok, QSA, rotary_dim)

#pragma once

#include "sub0_config.hpp"
#include "sub0/layout.hpp"

#include <cstdint>

namespace sub0 {

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
    std::uint64_t param_floats = PARAM_FLOATS;
};

// The size this format has always had on this toolchain: 4 magic + 8 ints = 36, padded to 40 for the
// uint64's alignment, + 8 = 48. Pinned so a compiler or field change that moves it is a build error
// rather than a file every existing model silently fails to load.
static_assert(sizeof(ModelHeader) == 48, "the S0L5 header's on-disk size must not change");
static_assert(alignof(ModelHeader) == 8);

}  // namespace sub0
