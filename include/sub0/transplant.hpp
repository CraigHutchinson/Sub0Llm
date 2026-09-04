// sub0/transplant.hpp -- the GGUF-to-PARAM_LAYOUT weight mapping: names, axis order, and the three
// tensor-granularity mismatches. Engine-free (no sub0_config.hpp, no layout.hpp), like gguf.hpp and
// memplan.hpp, so the mapping can be unit-tested and fixture-replayed WITHOUT compiling a model at the
// real axes -- which matters a great deal here, since the real axes make PARAM_LAYOUT a ~6,200-tensor
// consteval walk and a 45 GB destination blob.
//
// WHY A HEADER AND NOT JUST THE TOOL (docs/WP4_SCOPE.md S4c): levels 3 and 4 of that section's gate
// replay this mapping against the EXISTING real-weight fixtures. Those fixtures are sliced-down shapes
// that this project's own Model cannot be built at (gdn_qwen4_fixture_tests.cpp's header comment
// explains why), so the gate can only run if the mapping is callable independently of a compiled
// layout. Putting the mapping in the tool would have made its strongest correctness check impossible
// to write -- the same reasoning that put gdn::Dims/qsa::Dims in engine-free math headers.
//
// --- THE ONE CONVENTION THAT APPLIES EVERYWHERE, and its exceptions -------------------------------
//
// GGUF's `ne` array is fastest-varying-first, so a PyTorch nn.Linear weight stored as [out, in]
// row-major is DECLARED as ne = [in, out]. That is a trap worth naming explicitly: the declared dims
// read as [in, out], which LOOKS like this project's own [rows=in, cols=out] convention already -- and
// the bytes are still the transpose of it. Element (o, i) sits at flat index o*in + i in the file and
// must land at i*out + o in the destination. Shape assertions cannot catch a missed transpose here,
// because the declared shape was never wrong.
//
// Two 2-D tensors do NOT transpose, and both were checked against the real file rather than assumed:
//   * `token_embd.weight` is an nn.Embedding table, [num_embeddings, embedding_dim] -- which IS this
//     project's TokEmb [VOCAB, D_MODEL]. Straight copy.
//   * `blk.N.ssm_conv1d.weight` is a depthwise Conv1d [channels, 1, kernel], ne = [kernel, channels],
//     i.e. row-major [channels][kernel] -- which IS GdnConv [GDN_CONV_DIM, GDN_CONV_KERNEL]. Straight
//     copy. (gdn_qwen4_fixture_tests.cpp's own conv1d gradient comparison already relies on this.)
// `token_embd.weight` and `output.weight` have IDENTICAL declared dims ([2560, 248320]) and opposite
// treatment -- copy vs transpose. Getting that backwards produces a file of exactly the right size.
//
// --- THE THREE GRANULARITY MISMATCHES -------------------------------------------------------------
//
// docs/WP4_SCOPE.md S3a-bis named two; reading the real file for THIS pass found the picture is
// different in one place and larger in another:
//
//  1. QSA's q_proj is a SPLIT, not a concat, and it is PER HEAD. GGUF has one `blk.N.attn_q.weight`
//     with 2*n_heads*head_dim output rows; this engine has QsaQProj and QsaGateProj separately. The
//     reference does `q_proj(h).view(..., -1, head_dim*2)` then `chunk(2, dim=-1)`, so head h's rows
//     are [query_h | gate_h] ADJACENT, and splitting the row block down the middle is wrong for every
//     head but head 0 (docs/QSA.md S2b.4). The fixture at tests/fixtures/qwen4_preview/ has 2 heads
//     precisely so a down-the-middle split is distinguishable -- see transplant_fixture_tests.cpp.
//  2. QSA's indexer IS a concat: `indexer.q_proj.weight` then `indexer.k_proj.weight`, in that order,
//     along the OUTPUT axis, matching the reference's torch.split([n*hd, kv*hd], dim=-1).
//  3. MoE needs NO pair reconstruction at all. S3a-bis flagged `ffn_gate_exps`/`ffn_up_exps` as a
//     split of a fused `gate_up_proj` needing concatenation -- but that fusion is the HF/safetensors
//     granularity. THIS engine already stores MoeGate and MoeUp as separate PARAM_LAYOUT tensors, so
//     GGUF's granularity and the destination's coincide exactly and each is a plain per-expert slice.
//     What IS needed is the 3-D EXPERT SLICE: one GGUF tensor supplies NUM_EXPERTS destinations.

#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string>

namespace sub0::transplant {

// Every destination this transplant fills, one enumerator per (PKind, instance) pair. Deliberately a
// SEPARATE enum from layout.hpp's PKind rather than a reuse: PKind lives behind sub0_config.hpp, and
// the whole point of this header is to be reachable without one. The tool carries the single explicit
// PKind -> Dest switch, which is greppable and exhaustive; nothing else may translate between them.
//
// Gated Residual gets nine enumerators, not four, because its four tensors appear in THREE independent
// instances with three different GGUF prefixes (`hc_attn_*` / `hc_ffn_*` / `output_hc_*`) -- and the
// exit instance genuinely has no inject tensor (docs/GATED_RESIDUAL.md S1c: use_combine=False), which
// is why GrExit* is three and not four.
enum class Dest {
    TokEmb, LnF, LmHead, LmBias,
    GdnInProjQkv, GdnInProjZ, GdnInProjB, GdnInProjA, GdnConv, GdnALog, GdnDtBias, GdnNorm, GdnOutProj,
    GrAttnNorm, GrAttnDown, GrAttnUp, GrAttnInject,
    GrFfnNorm,  GrFfnDown,  GrFfnUp,  GrFfnInject,
    GrExitNorm, GrExitDown, GrExitUp,
    MoeRouter, MoeGate, MoeUp, MoeDown,
    MoeSharedGate, MoeSharedUp, MoeSharedDown, MoeSharedGateProj,
    QsaQProj, QsaGateProj, QsaKProj, QsaVProj, QsaOProj, QsaQNorm, QsaKNorm,
    QsaIdxQkProj, QsaIdxQNorm, QsaIdxKNorm,
    Count
};

// How a destination is built from its source bytes.
enum class Op {
    Copy,          // 1-D vector, or a 2-D tensor whose GGUF byte order already IS [rows, cols]
    Transpose,     // 2-D [out, in] row-major -> [in, out]  (the default for every nn.Linear)
    PerHeadHalf,   // QSA q_proj: take half `half` of each head's 2*head_dim row group, then transpose
    ConcatOut,     // two sources joined along the OUTPUT axis (src then src2), then transposed
    ExpertSlice,   // 3-D [n_experts][out][in]: take expert e's slice, then transpose
    Synthetic,     // NO GGUF source exists -- filled with `fill` (see recipe_for's own comment)
};

struct Recipe {
    Op          op = Op::Copy;
    const char* src = nullptr;    // GGUF name; "%d" (if present) is the layer index
    const char* src2 = nullptr;   // ConcatOut's second source
    int         half = 0;         // PerHeadHalf: 0 = query, 1 = gate
    float       fill = 0.f;       // Synthetic: the value to write
};

// The name/axis table. Every entry was checked against the real
// D:\ModelWeights\Qwen3.8-Flash-Next-GGUF\UD-IQ1_S file's own tensor table -- name, declared dims and
// type id read directly, not taken from docs/WP4_SCOPE.md S3a-bis on trust (that table got MoE's
// granularity and the Gated Residual exit instance wrong, and did not know the real file has no final
// norm at all; see this header's own notes and the two Synthetic entries below).
//
// The two Synthetic entries are a REAL finding, recorded rather than papered over:
//   * `output_norm.weight` DOES NOT EXIST in the real file. Not under that name, not under any other:
//     the only non-`blk.` tensors in the whole 1,224-tensor set are output.weight, output_hc_{down,
//     norm,up}.weight, per_layer_token_embd.weight and token_embd.weight. That is architecturally
//     coherent -- the same reason WP4b blocker D removed Ln1/Ln2, namely that the Gated Residual
//     instance's own hc_norm IS the norm at that point -- but docs/WP4_SCOPE.md S2 blocker D assumed a
//     real LnF counterpart existed, and it does not. LnF is filled with 1.0, the RMSNorm identity
//     gain, so the tensor this engine's layout insists on is at least a no-op rather than a zero.
//   * `LmBias` likewise has no source (the real head is bias-free, `attention_bias: false` and no
//     `output.bias` in the file). Filled with 0.0, the additive identity.
// Both are reported by name in the tool's own level-1 reconciliation, so "2 destinations synthesized"
// is a number the operator sees, not a silent default.
constexpr Recipe recipe_for(Dest d) {
    switch (d) {
        case Dest::TokEmb:      return {Op::Copy,      "token_embd.weight"};
        case Dest::LnF:         return {Op::Synthetic, nullptr, nullptr, 0, 1.0f};
        case Dest::LmHead:      return {Op::Transpose, "output.weight"};
        case Dest::LmBias:      return {Op::Synthetic, nullptr, nullptr, 0, 0.0f};

        // Gated DeltaNet. GGUF calls the fused Q|K|V projection `attn_qkv` and the output gate
        // `attn_gate` -- names that look like ordinary attention but belong to the linear-attention
        // mixer; the `ssm_*` names carry the rest. ssm_alpha <- in_proj_a and ssm_beta <- in_proj_b is
        // NOT read off the names: it is llama.cpp's own gguf-py/gguf/tensor_mapping.py, which maps
        // MODEL_TENSOR.SSM_ALPHA to "model.layers.{bid}.linear_attn.in_proj_a" and SSM_BETA to
        // "...in_proj_b". Confirming that from a real converter mattered: a/b are same-shaped
        // [hidden, num_v_heads] vectors feeding different gates, which is exactly the identity-swap
        // shape of the dt_bias/A_log bug WP-GDN Stage 3 found (docs/GATED_DELTANET.md S6).
        case Dest::GdnInProjQkv: return {Op::Transpose, "blk.%d.attn_qkv.weight"};
        case Dest::GdnInProjZ:   return {Op::Transpose, "blk.%d.attn_gate.weight"};
        case Dest::GdnInProjB:   return {Op::Transpose, "blk.%d.ssm_beta.weight"};
        case Dest::GdnInProjA:   return {Op::Transpose, "blk.%d.ssm_alpha.weight"};
        case Dest::GdnConv:      return {Op::Copy,      "blk.%d.ssm_conv1d.weight"};   // depthwise: no transpose
        case Dest::GdnALog:      return {Op::Copy,      "blk.%d.ssm_a"};               // note: no ".weight"
        case Dest::GdnDtBias:    return {Op::Copy,      "blk.%d.ssm_dt.bias"};
        case Dest::GdnNorm:      return {Op::Copy,      "blk.%d.ssm_norm.weight"};
        case Dest::GdnOutProj:   return {Op::Transpose, "blk.%d.ssm_out.weight"};

        case Dest::GrAttnNorm:   return {Op::Copy,      "blk.%d.hc_attn_norm.weight"};
        case Dest::GrAttnDown:   return {Op::Transpose, "blk.%d.hc_attn_down.weight"};
        case Dest::GrAttnUp:     return {Op::Transpose, "blk.%d.hc_attn_up.weight"};
        case Dest::GrAttnInject: return {Op::Transpose, "blk.%d.hc_attn_inject.weight"};
        case Dest::GrFfnNorm:    return {Op::Copy,      "blk.%d.hc_ffn_norm.weight"};
        case Dest::GrFfnDown:    return {Op::Transpose, "blk.%d.hc_ffn_down.weight"};
        case Dest::GrFfnUp:      return {Op::Transpose, "blk.%d.hc_ffn_up.weight"};
        case Dest::GrFfnInject:  return {Op::Transpose, "blk.%d.hc_ffn_inject.weight"};
        case Dest::GrExitNorm:   return {Op::Copy,      "output_hc_norm.weight"};
        case Dest::GrExitDown:   return {Op::Transpose, "output_hc_down.weight"};
        case Dest::GrExitUp:     return {Op::Transpose, "output_hc_up.weight"};

        case Dest::MoeRouter:        return {Op::Transpose,   "blk.%d.ffn_gate_inp.weight"};
        case Dest::MoeGate:          return {Op::ExpertSlice, "blk.%d.ffn_gate_exps.weight"};
        case Dest::MoeUp:            return {Op::ExpertSlice, "blk.%d.ffn_up_exps.weight"};
        case Dest::MoeDown:          return {Op::ExpertSlice, "blk.%d.ffn_down_exps.weight"};
        case Dest::MoeSharedGate:    return {Op::Transpose,   "blk.%d.ffn_gate_shexp.weight"};
        case Dest::MoeSharedUp:      return {Op::Transpose,   "blk.%d.ffn_up_shexp.weight"};
        case Dest::MoeSharedDown:    return {Op::Transpose,   "blk.%d.ffn_down_shexp.weight"};
        // Declared [2560] in the file and [D_MODEL, 1] in the layout -- the same 2560 floats in the
        // same order, so a copy. (A "transpose" of an Nx1 is the identity anyway; Copy says why.)
        case Dest::MoeSharedGateProj: return {Op::Copy,       "blk.%d.ffn_gate_inp_shexp.weight"};

        case Dest::QsaQProj:     return {Op::PerHeadHalf, "blk.%d.attn_q.weight", nullptr, 0};
        case Dest::QsaGateProj:  return {Op::PerHeadHalf, "blk.%d.attn_q.weight", nullptr, 1};
        case Dest::QsaKProj:     return {Op::Transpose,   "blk.%d.attn_k.weight"};
        case Dest::QsaVProj:     return {Op::Transpose,   "blk.%d.attn_v.weight"};
        case Dest::QsaOProj:     return {Op::Transpose,   "blk.%d.attn_output.weight"};
        case Dest::QsaQNorm:     return {Op::Copy,        "blk.%d.attn_q_norm.weight"};
        case Dest::QsaKNorm:     return {Op::Copy,        "blk.%d.attn_k_norm.weight"};
        case Dest::QsaIdxQkProj: return {Op::ConcatOut,   "blk.%d.indexer.q_proj.weight",
                                                          "blk.%d.indexer.k_proj.weight"};
        case Dest::QsaIdxQNorm:  return {Op::Copy,        "blk.%d.indexer.q_norm.weight"};
        case Dest::QsaIdxKNorm:  return {Op::Copy,        "blk.%d.indexer.k_norm.weight"};
        case Dest::Count:        break;
    }
    return {};
}

// Substitutes the layer index into a pattern containing "%d". Hand-rolled rather than std::format so
// this header keeps its "std only, no <format> in a C++20-reachable header" posture, and so a pattern
// with NO %d (the three model-level Gated Residual tensors, token_embd, output) passes through
// untouched instead of needing a separate call.
inline std::string gguf_name(const char* pattern, int layer) {
    std::string out;
    if (pattern == nullptr) return out;
    for (const char* p = pattern; *p; ++p) {
        if (p[0] == '%' && p[1] == 'd') { out += std::to_string(layer); ++p; }
        else                             { out += *p; }
    }
    return out;
}

// --- the array operations -------------------------------------------------------------------------
// Every one writes EXACTLY rows*cols floats into `dst` and reads only within the declared source
// extent. They take raw pointers plus explicit extents rather than spans of computed size because
// each is called once per destination tensor from a loop that already knows both -- and because the
// fixture replay calls them directly with the fixture's own dims.

// [out_f, in_f] row-major -> [in_f, out_f] row-major.
inline void transpose_out_in(const float* src, int out_f, int in_f, float* dst) {
    for (int o = 0; o < out_f; ++o)
        for (int i = 0; i < in_f; ++i)
            dst[static_cast<std::size_t>(i) * out_f + o] = src[static_cast<std::size_t>(o) * in_f + i];
}

// QSA q_proj. `src` is [n_heads * 2 * head_dim, in_f] row-major. Head h's output rows are
// [h*2*hd, h*2*hd + 2*hd), of which the FIRST hd are the query and the SECOND hd are the gate --
// the reference's `.view(..., -1, head_dim*2)` then `chunk(2, dim=-1)`. Writes
// [in_f, n_heads*head_dim], with destination column h*hd + d.
//
// The failure this exists to prevent: taking rows [0, n_heads*hd) as the query. That is correct for
// head 0 and wrong for every other head, and produces output of exactly the right shape and a
// plausible magnitude (docs/QSA.md S2b.4).
inline void per_head_half_transpose(const float* src, int n_heads, int head_dim, int in_f, int half,
                                     float* dst) {
    const int out_f = n_heads * head_dim;
    for (int h = 0; h < n_heads; ++h)
        for (int d = 0; d < head_dim; ++d) {
            const std::size_t src_row =
                static_cast<std::size_t>(h) * 2 * head_dim + static_cast<std::size_t>(half) * head_dim + d;
            const int out_col = h * head_dim + d;
            for (int i = 0; i < in_f; ++i)
                dst[static_cast<std::size_t>(i) * out_f + out_col] = src[src_row * in_f + i];
        }
}

// Two [out, in] sources joined along the OUTPUT axis in the order (a, then b), transposed into
// [in, out_a + out_b]. `a` supplies destination columns [0, out_a), `b` supplies [out_a, out_a+out_b).
inline void concat_out_transpose(const float* a, int out_a, const float* b, int out_b, int in_f,
                                  float* dst) {
    const int out_f = out_a + out_b;
    for (int o = 0; o < out_a; ++o)
        for (int i = 0; i < in_f; ++i)
            dst[static_cast<std::size_t>(i) * out_f + o] = a[static_cast<std::size_t>(o) * in_f + i];
    for (int o = 0; o < out_b; ++o)
        for (int i = 0; i < in_f; ++i)
            dst[static_cast<std::size_t>(i) * out_f + out_a + o] = b[static_cast<std::size_t>(o) * in_f + i];
}

// --- per-tensor statistics (docs/WP4_SCOPE.md S4c level 2) ----------------------------------------
// Computed on BOTH sides of a reshaping op. Every op here is a PERMUTATION of its input's elements,
// so n / min / max must match EXACTLY and mean / std to floating-point summation noise. That is a
// genuine detector of a transpose applied twice or not at all on a non-square tensor (the offsets
// differ, so the multiset differs), of a wrong slice, and of a per-head split taken down the middle.
// It is NOT a detector of a same-shaped identity swap -- which is what levels 3 and 4 are for.
struct Stats {
    std::size_t n = 0;
    double mean = 0.0, stddev = 0.0, min = 0.0, max = 0.0;
    std::size_t nonfinite = 0;
};

inline Stats stats_of(std::span<const float> v) {
    Stats s;
    s.n = v.size();
    if (v.empty()) return s;
    double sum = 0.0, sum2 = 0.0;
    double mn = 0.0, mx = 0.0;
    bool first = true;
    for (float x : v) {
        const double d = static_cast<double>(x);
        if (!(d == d) || d > 1e308 || d < -1e308) { ++s.nonfinite; continue; }
        sum += d;
        sum2 += d * d;
        if (first) { mn = mx = d; first = false; }
        else { if (d < mn) mn = d; if (d > mx) mx = d; }
    }
    const double n = static_cast<double>(s.n);
    s.mean = sum / n;
    const double var = sum2 / n - s.mean * s.mean;
    s.stddev = var > 0.0 ? std::sqrt(var) : 0.0;
    s.min = mn;
    s.max = mx;
    return s;
}

// True when `a` and `b` are consistent with being permutations of one another: identical element
// count, identical extrema, and mean/std agreeing to `rel` relative. The extrema comparison is EXACT
// -- a permutation cannot change them, and allowing slack there would let a wrong slice through.
inline bool stats_consistent(const Stats& a, const Stats& b, double rel = 1e-6) {
    if (a.n != b.n || a.nonfinite != b.nonfinite) return false;
    if (a.min != b.min || a.max != b.max) return false;
    const double scale = (a.max > -a.min ? a.max : -a.min);
    const double tol = rel * (scale > 0.0 ? scale : 1.0);
    const double dm = a.mean - b.mean, ds = a.stddev - b.stddev;
    return (dm < tol && dm > -tol) && (ds < tol && ds > -tol);
}

}  // namespace sub0::transplant
