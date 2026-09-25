// sub0/backbone_quant.hpp -- O5 phase 2b-1: the `S0B1` native-quant BACKBONE sidecar -- its on-disk
// format and its read-only mmap loader. Mirrors include/sub0/moe_quant.hpp's `S0Q1` shape (Header, Desc,
// Store, mmap, on-disk static_asserts) applied to the backbone instead of the routed experts, per this
// project's own reuse-the-precedent discipline (AGENTS.md S3/S10).
//
// WHAT THIS FILE IS, AND IS NOT (docs/BACKBONE_NATIVE_QUANT.md S9/S10). This is the STORAGE half of O5
// phase 2b: it produces (via the writer in tools/sub0llm-transplant.cpp) and loads a second sidecar file,
// `<model>.bin.bbq`, holding the unsloth GGUF's own Q8_0/Q4_K/Q5_K/Q6_K backbone bytes verbatim. It does
// NOT change engine math, decode, or `op_linear` -- nothing in src/ includes this header yet, the same
// "phase 2a is isolated" posture backbone_quant_dot.hpp itself documents. include/sub0/backbone_quant_dot.hpp
// (a separate, parallel work package) is the KERNEL that will eventually consume what this format stores;
// this header only produces the bytes and the {span, type_raw, n_rows, row_elems} tuple that kernel wants.
//
// --- THE TWO-FILE RELATIONSHIP, same shape as moe_quant.hpp's own ---------------------------------
//
//     <model>.bin        the existing S0L5 header + flat PARAM_LAYOUT-ordered blob (f32/bf16/fp8) --
//                        UNCHANGED. Every backbone tensor this sidecar also carries STAYS in the blob,
//                        at its existing dtype; this is an ADDITIVE alternate source, not a replacement
//                        (S9's "additive, gracefully-degrading format" precedent, AGENTS.md S3 rule 2).
//     <model>.bin.bbq    THIS format: a subset of the backbone's own 2-D weight tensors, in their native
//                        GGUF encoding, byte-for-byte as the source shard held them.
//
// `model_param_floats` below pins which S0L5 blob this sidecar belongs beside, exactly as moeq::Header's
// own field does -- a mismatched pair refuses to load rather than silently combining two different
// models' worth of weights (AGENTS.md S3).
//
// --- WHICH TENSORS ARE IN HERE, AND WHY OTHERS ARE NOT (the real work of this phase) ----------------
//
// The role table below is not "every backbone tensor" -- it is deliberately narrower, for three
// independent reasons, each checked against the real census and against transplant.hpp's own recipe
// table (the project's single source for what each destination is built from, AGENTS.md S5), not assumed:
//
//   1. WRONG FORMAT. F32 tensors (norms, the dense router, biases, small per-head-scalar GDN gates) have
//      no smaller native form to keep -- they are excluded by construction, the same reasoning
//      docs/BACKBONE_NATIVE_QUANT.md S2a gives for leaving them out of the byte totals. The QSA indexer
//      (`indexer.q_proj`/`k_proj`) is BF16, already the resident target format -- also excluded.
//   2. ALREADY HAS ITS OWN SIDECAR. The 512 routed experts per layer are `moe_quant.hpp`'s own `S0Q1`
//      format's job; this header's role table does not duplicate them (docs/BACKBONE_NATIVE_QUANT.md S2
//      itself scopes the census the same way: "routed-expert tensors... explicitly OUT of this census").
//   3. VALUE-ORDER TRANSFORM (the one this phase had to actually resolve, not just note). GDN's own
//      value-head grouped-to-tiled reorder (`transplant::vperm_for`/`ungroup_v_heads`) touches THREE
//      tensors that ARE otherwise-native-quant candidates: `GdnInProjQkv`/`GdnInProjZ` (Q5_K) and
//      `GdnOutProj` (Q6_K). A permutation commutes with raw block-quantized bytes only when it moves
//      WHOLE, self-contained quantization blocks -- and the two axes this reorder actually uses are NOT
//      symmetric that way:
//        * `GdnInProjQkv`/`GdnInProjZ` reorder the destination's COLUMN axis, which (Op::Transpose,
//          transplant.hpp's own convention) is the RAW GGUF tensor's own ROW axis -- i.e. it reorders
//          WHOLE ROWS. Each row is independently, self-containedly block-quantized along `in_f`, so this
//          particular reorder COULD in principle be carried as a permuted row COPY at sidecar-build time,
//          with no re-quantization.
//        * `GdnOutProj` reorders the destination's ROW axis, which is the raw GGUF tensor's own COLUMN
//          axis -- i.e. it reorders ELEMENTS WITHIN each row, across the very axis K-quant super-blocks
//          are computed over. A within-row element permutation does NOT commute with the raw quantized
//          bytes: reassembling a valid, differently-ordered block stream would need real re-blocking
//          (recomputing which elements fall in which sub-block), not a byte copy -- exactly the
//          "must be handled explicitly, or stay bf16" fork this phase's brief names.
//      Phase 2b-1 made ONE decision for all three, deliberately conservative: EXCLUDE every VPerm-affected
//      role from the sidecar. They stayed in the existing bf16 blob. That gave up part of the Q5_K byte
//      win (the GDN in-proj share of it) and almost all of Q6_K's INTENDED role coverage (`GdnOutProj` is
//      Q6_K's only NAMED consumer in this table) -- a real, quantified cost, reported by the writer/tool
//      rather than glossed over (see docs/BACKBONE_NATIVE_QUANT.md's S13 resolution section).
//
//      **O5 phase 2b-2b (docs/BACKBONE_NATIVE_QUANT.md S16) RECOVERS all three, exactly along the two
//      lines named above, once a real kernel consumer existed to validate against**: `GdnInProjQkv`/
//      `GdnInProjZ` are carried as a PERMUTED ROW COPY (the writer copies each raw GGUF row to its
//      HF-ordered destination row, byte for byte, no re-quantization -- `gdn_row_permutation` below); the
//      sidecar's own row order for these two roles is therefore HF/grouped, matching the .bin blob's own
//      column order, and a consumer needs no runtime permutation to read them. `GdnOutProj` is carried
//      RAW, GGUF/tiled order, unpermuted (a within-row permutation still cannot be carried as a byte
//      copy) -- its consumer must gather the ACTIVATION into GGUF/tiled order before the dot
//      (`gdn_out_gather_index` below), not the weight.
//
//      ONE MORE THING FOUND BY ACTUALLY RUNNING THIS AGAINST THE REAL FILE (AGENTS.md S9), not assumed
//      from the aggregate census table: the backbone is NOT uniformly one format per role the way the
//      design doc's S2a totals table might suggest -- layer 2 specifically is a real per-layer outlier,
//      the same kind of per-layer mixed quantization `docs/WP4_SCOPE.md` S3a-bis already documented for
//      the routed experts. `blk.2.ffn_gate_shexp.weight`/`blk.2.ffn_up_shexp.weight` (role
//      `MoeSharedGate`/`MoeSharedUp`, no VPerm, no Fold) are Q6_K where every other layer's are Q5_K.
//      Because this format's writer discovers each tensor's real `type_raw` by lookup (never assumes a
//      role's format from its name), this is handled correctly with no special case: those two planes are
//      included, tagged `Q6_K`, exactly like any other role/layer. So Q6_K is NOT entirely absent from a
//      real sidecar the way "GdnOutProj is its only consumer" would suggest on paper -- it is merely rare
//      (one layer's shared-expert gate/up, not the far larger GDN out-proj population the census's
//      per-format totals table counts).
//
// `fold_for()` (RMSNorm `1+w`, `ssm_a`'s `-exp(A_log)`) never applies to any role in the table below --
// checked, not assumed: every Dest `fold_for` returns non-`None` for is a norm, a bias, or an
// F32/per-head-scalar GDN gate, all already excluded by point 1. So no role here needs its raw quantized
// bytes value-transformed at all; every included tensor really is storable verbatim.
//
// --- BYTE ORDER: GGUF'S OWN, VERBATIM, UNTRANSPOSED (the opposite of the .bin blob) ------------------
//
// A GGUF tensor is `[out][in]`, row-major, contiguous over `in` (`ne0`) -- exactly the DOT-friendly shape
// `backbone_quant_dot.hpp`'s kernels want (that header's own "LAYOUT" note). The existing S0L5 blob
// instead stores every 2-D weight TRANSPOSED to this project's own `[in, out]` AXPY convention
// (`transplant::transpose_out_in`, applied by every `Op::Transpose`/`Op::Copy` destination the ordinary
// transplant path fills). This sidecar does the OPPOSITE: `dequantize_expert`'s `transpose_out_in` step is
// NOT applied here. `Desc::in_f`/`out_f` below name the GGUF-declared extents directly (`ne[0]`/`ne[1]`),
// the same naming `moeq::Desc` already uses for the identical reason ("dims[0] is the input width is the
// most inversion-prone line" -- transplant.hpp's own words) -- but unlike `moeq::Desc`, `out_f` here IS
// the raw file's own row count (`n_rows` a GEMV kernel would iterate), not a post-transpose column count.
// A reader of this format must not assume "the same shape as the .bin blob's tensors" -- it is the
// GGUF shard's own shape, because that is what the fused kernel this format exists to feed actually wants.
//
// --- TOKEN EMBEDDING: A GATHER, STORED THE SAME WAY AS EVERY OTHER ROLE -------------------------------
//
// `TokEmb` (`token_embd.weight`, Q4_K) is a per-token row GATHER in the engine, not a GEMV
// (docs/BACKBONE_NATIVE_QUANT.md S2b names this as out of scope for phase 2a's row-range GEMV kernel).
// It is included here anyway: this file only stores addressable per-row native bytes, and a gather is
// exactly "read one row's own bytes and dequantize just that row" -- no different in STORAGE terms from
// any GEMV-consumed role. Whether/how a future consumer wires a single-row dequant path is that
// consumer's decision, not a reason to leave the largest Q4_K tensor's bytes out of the sidecar.
//
// Engine-free (gguf.hpp + transplant.hpp + file_map.hpp + backbone_quant_dot.hpp's pure format
// predicates only -- no sub0_config.hpp, no layout.hpp), like moe_quant.hpp, so the format and the loader
// are unit-testable without compiling a model at the real axes.

#pragma once

#include "sub0/backbone_quant_dot.hpp"   // bbqd::fusable/plane_bytes -- the one format-validity table
#include "sub0/file_map.hpp"
#include "sub0/gguf.hpp"
#include "sub0/transplant.hpp"

#include <array>
#include <climits>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <span>
#include <string>
#include <unordered_map>
#include <vector>

namespace sub0::bbq {

// An S0B1 sidecar is usable only with the exact blob written alongside it. This separate record
// preserves the existing S0B1 byte layout while binding both complete files, including their headers.
// FNV-1a is an accidental-mismatch guard, not an authentication mechanism.
struct PairIdentity {
    char          magic[4] = {'S', '0', 'B', 'I'};
    std::uint32_t version = 1;
    std::uint64_t model_bytes = 0, model_hash = 0;
    std::uint64_t sidecar_bytes = 0, sidecar_hash = 0;
};
static_assert(sizeof(PairIdentity) == 40);

[[nodiscard]] inline bool file_identity(const std::string& path, std::uint64_t& bytes,
                                        std::uint64_t& hash, std::string& err) {
    std::ifstream file(path, std::ios::binary);
    if (!file) { err = "cannot open " + path + " for identity check"; return false; }
    std::vector<char> block(1 << 20);
    bytes = 0;
    hash = 14695981039346656037ull;
    while (file) {
        file.read(block.data(), block.size());
        const auto n = file.gcount();
        for (std::streamsize i = 0; i < n; ++i) {
            hash ^= static_cast<std::uint8_t>(block[static_cast<std::size_t>(i)]);
            hash *= 1099511628211ull;
        }
        bytes += static_cast<std::uint64_t>(n);
    }
    if (!file.eof()) { err = "cannot read " + path + " for identity check"; return false; }
    return true;
}

[[nodiscard]] inline bool write_pair_identity(const std::string& model_path,
                                              const std::string& sidecar_path, std::string& err) {
    PairIdentity id;
    if (!file_identity(model_path, id.model_bytes, id.model_hash, err) ||
        !file_identity(sidecar_path, id.sidecar_bytes, id.sidecar_hash, err)) return false;
    std::ofstream out(sidecar_path + ".pair", std::ios::binary | std::ios::trunc);
    if (!out) { err = "cannot create " + sidecar_path + ".pair"; return false; }
    out.write(reinterpret_cast<const char*>(&id), sizeof id);
    out.close();
    if (!out) { err = "cannot write " + sidecar_path + ".pair"; return false; }
    return true;
}

[[nodiscard]] inline bool verify_pair_identity(const std::string& model_path,
                                               const std::string& sidecar_path, std::string& err) {
    const std::string pair_path = sidecar_path + ".pair";
    std::ifstream in(pair_path, std::ios::binary | std::ios::ate);
    if (!in || in.tellg() != static_cast<std::streamoff>(sizeof(PairIdentity))) {
        err = pair_path + ": missing or invalid pair identity";
        return false;
    }
    in.seekg(0);
    PairIdentity id;
    in.read(reinterpret_cast<char*>(&id), sizeof id);
    if (!in || std::memcmp(id.magic, "S0BI", 4) != 0 || id.version != 1) {
        err = pair_path + ": invalid pair identity header";
        return false;
    }
    std::uint64_t bytes = 0, hash = 0;
    if (!file_identity(model_path, bytes, hash, err)) return false;
    if (bytes != id.model_bytes || hash != id.model_hash) {
        err = pair_path + ": model content does not match the sidecar pair";
        return false;
    }
    if (!file_identity(sidecar_path, bytes, hash, err)) return false;
    if (bytes != id.sidecar_bytes || hash != id.sidecar_hash) {
        err = pair_path + ": sidecar content does not match the model pair";
        return false;
    }
    return true;
}

// --- roles: the stable, engine-facing tensor identity ----------------------------------------------
//
// A GGUF tensor NAME is not a stable key for the engine to ask for -- the engine has its own per-layer
// weight roles (transplant::Dest), and the whole point of a role enum is that a caller who does not know
// or care what the GGUF file called something can still ask "give me this layer's GDN out-proj". Mirrors
// moeq::Which's "(layer, expert, which)" precedent with a flat per-role enum instead (no "expert" axis
// here), consistent with docs/BACKBONE_NATIVE_QUANT.md's own field naming for this design.
//
// EVERY role here maps 1:1 to exactly one `transplant::Dest`, EXCEPT `QsaQGateProj`: QSA's `attn_q.weight`
// supplies BOTH `Dest::QsaQProj` and `Dest::QsaGateProj` (transplant.hpp's `PerHeadHalf` op, a
// non-contiguous per-head row selection -- see `role_pattern`'s own comment for why this sidecar stores
// the tensor whole rather than pre-splitting it).
// O5 phase 2b-2b: the three GDN roles phase 2b-1 excluded on VPerm grounds (this file's own header
// comment, point 3) are APPENDED here, after every phase-2b-1 role and before Count -- never inserted
// earlier -- because Desc::role is the role's plain enum-underlying int, written to disk (AGENTS.md S3):
// reordering the existing members would silently reinterpret every already-written sidecar's role ids.
// See docs/BACKBONE_NATIVE_QUANT.md S16 for how each of the three is now resolved (a permuted row COPY
// for the two Cols-axis roles, a raw verbatim store + consumer-side activation gather for the one
// Rows-axis role) rather than excluded.
enum class Role : std::int32_t {
    TokEmb = 0, LmHead,
    GrAttnDown, GrAttnUp, GrFfnDown, GrFfnUp, GrExitDown, GrExitUp,
    MoeSharedGate, MoeSharedUp, MoeSharedDown,
    QsaQGateProj, QsaKProj, QsaVProj, QsaOProj,
    GdnInProjQkv, GdnInProjZ, GdnOutProj,
    Count
};
inline constexpr int kRoleCount = static_cast<int>(Role::Count);

// True for a role that exists once per layer (GrAttnDown, QsaKProj, ...) rather than once for the whole
// model (TokEmb, LmHead, the Gated-Residual EXIT instance). A per-layer role's own tensor may still be
// ABSENT at a given layer -- QSA roles only exist on layers whose mixer is QSA, never GDN -- which the
// writer discovers by lookup rather than this flag predicting it (MIXER_SCHEDULE is a layout.hpp/
// sub0_config.hpp fact this engine-free header does not have access to).
[[nodiscard]] constexpr bool role_per_layer(Role r) {
    switch (r) {
        case Role::TokEmb:
        case Role::LmHead:
        case Role::GrExitDown:
        case Role::GrExitUp:      return false;
        default:                  return true;
    }
}

// The GGUF source name pattern for a role ("%d" substituted with the layer index by
// `transplant::gguf_name`, exactly as every other consumer of `transplant::Recipe::src` already does).
// Read from `transplant::recipe_for` -- the project's single name/axis table (AGENTS.md S5) -- rather
// than a second, hand-copied string literal per role, so the two tables cannot silently drift apart.
//
// `QsaQGateProj` reads `Dest::QsaQProj`'s own source (`blk.%d.attn_q.weight`) -- `Dest::QsaGateProj`
// shares the identical `src` string (both are `PerHeadHalf` slices of the SAME GGUF tensor), so either
// works as the lookup key. This sidecar stores that source tensor WHOLE, in GGUF's own row order,
// covering both destinations at once: `PerHeadHalf`'s own per-head row selection
// (`h*2*head_dim + half*head_dim`, transplant.hpp's `per_head_half_transpose`) picks specific WHOLE rows
// out of a contiguous, self-contained block-quantized tensor, so a future consumer can apply that exact
// row-selection arithmetic directly against this role's own raw bytes -- no pre-split, no permutation,
// no re-blocking, unlike the GDN roles excluded above.
[[nodiscard]] inline const char* role_pattern(Role r) {
    using transplant::Dest;
    using transplant::recipe_for;
    switch (r) {
        case Role::TokEmb:        return recipe_for(Dest::TokEmb).src;
        case Role::LmHead:        return recipe_for(Dest::LmHead).src;
        case Role::GrAttnDown:    return recipe_for(Dest::GrAttnDown).src;
        case Role::GrAttnUp:      return recipe_for(Dest::GrAttnUp).src;
        case Role::GrFfnDown:     return recipe_for(Dest::GrFfnDown).src;
        case Role::GrFfnUp:       return recipe_for(Dest::GrFfnUp).src;
        case Role::GrExitDown:    return recipe_for(Dest::GrExitDown).src;
        case Role::GrExitUp:      return recipe_for(Dest::GrExitUp).src;
        case Role::MoeSharedGate: return recipe_for(Dest::MoeSharedGate).src;
        case Role::MoeSharedUp:   return recipe_for(Dest::MoeSharedUp).src;
        case Role::MoeSharedDown: return recipe_for(Dest::MoeSharedDown).src;
        case Role::QsaQGateProj:  return recipe_for(Dest::QsaQProj).src;
        case Role::QsaKProj:      return recipe_for(Dest::QsaKProj).src;
        case Role::QsaVProj:      return recipe_for(Dest::QsaVProj).src;
        case Role::QsaOProj:      return recipe_for(Dest::QsaOProj).src;
        case Role::GdnInProjQkv:  return recipe_for(Dest::GdnInProjQkv).src;
        case Role::GdnInProjZ:    return recipe_for(Dest::GdnInProjZ).src;
        case Role::GdnOutProj:    return recipe_for(Dest::GdnOutProj).src;
        case Role::Count:         break;
    }
    return nullptr;
}

[[nodiscard]] inline const char* role_name(Role r) {
    switch (r) {
        case Role::TokEmb:        return "TokEmb";
        case Role::LmHead:        return "LmHead";
        case Role::GrAttnDown:    return "GrAttnDown";
        case Role::GrAttnUp:      return "GrAttnUp";
        case Role::GrFfnDown:     return "GrFfnDown";
        case Role::GrFfnUp:       return "GrFfnUp";
        case Role::GrExitDown:    return "GrExitDown";
        case Role::GrExitUp:      return "GrExitUp";
        case Role::MoeSharedGate: return "MoeSharedGate";
        case Role::MoeSharedUp:   return "MoeSharedUp";
        case Role::MoeSharedDown: return "MoeSharedDown";
        case Role::QsaQGateProj:  return "QsaQGateProj";
        case Role::QsaKProj:      return "QsaKProj";
        case Role::QsaVProj:      return "QsaVProj";
        case Role::QsaOProj:      return "QsaOProj";
        case Role::GdnInProjQkv:  return "GdnInProjQkv";
        case Role::GdnInProjZ:    return "GdnInProjZ";
        case Role::GdnOutProj:    return "GdnOutProj";
        case Role::Count:         break;
    }
    return "?";
}

// --- O5 phase 2b-2b: the GDN value-head permutation, shared by the sidecar WRITER (a permuted ROW copy
// for GdnInProjQkv/GdnInProjZ) and the DECODE consumer (an activation GATHER for GdnOutProj) -- one
// derivation, not two hand-copied ones (AGENTS.md S5/S10). Both mirror transplant::ungroup_v_heads's own
// src/dst head mapping exactly (dst_head = k*rep+r, the HF/repeat_interleave order; src_head =
// r*num_k_heads+k, the GGUF/tiled ggml-broadcast order), generalized from PER-FLOAT to PER-HEAD-GROUP so
// a whole row (or a whole gather-index run) can be permuted/derived at once instead of per element.
//
// WHY A ROW COPY IS LOSSLESS FOR GdnInProjQkv/Z BUT A GATHER (NOT A COPY) IS NEEDED FOR GdnOutProj: see
// this file's own header comment, point 3 -- the two "Cols" roles permute the raw GGUF tensor's ROW axis
// (whole, independently block-quantized rows), the one "Rows" role permutes its COLUMN axis (elements
// WITHIN a row, across superblock boundaries a K-quant block cannot survive being re-cut at). The sidecar
// therefore stores GdnOutProj's raw bytes completely unpermuted, GGUF order, and it is the ACTIVATION fed
// into that role's dot product that must be reordered instead, once per (token, layer), before the dot.
//
// PROOF OF THE GATHER'S DIRECTION (re-derived here, not assumed -- AGENTS.md S5): let `blob_col[o][j]`
// be the .bin blob's own (already-ungrouped, HF-ordered) weight value at output o, HF position j, and
// `raw_row[o][i]` the sidecar's raw row o's value at GGUF/tiled position i. `ungroup_v_heads` establishes
// `blob_col[o][gdn_out_gather_index()[i]] == raw_row[o][i]` for every i (its own dst[d_idx] = src[s_idx],
// with d_idx = gather_index[i] and s_idx = i in this function's own naming). So for any activation x_hf
// (HF order) and its gather x_ggml[i] = x_hf[gather_index[i]]:
//   dot(raw_row[o], x_ggml) = sum_i raw_row[o][i] * x_hf[gather_index[i]]
//                           = sum_i blob_col[o][gather_index[i]] * x_hf[gather_index[i]]
// and because i -> gather_index[i] is a bijection over the V block (a permutation of [0, value_dim)),
// substituting j = gather_index[i] re-sums this to sum_j blob_col[o][j] * x_hf[j] = dot(blob_col[o], x_hf)
// exactly -- QED, checked once in prose here and re-checked numerically by
// tests/backbone_quant_tests.cpp's own "GdnOutProj gather direction" case, on both a synthetic multi-head
// fixture and the real sidecar/blob pair (AGENTS.md S9).
[[nodiscard]] constexpr int gdn_v_src_head(int dst_head, int num_k_heads, int rep) {
    const int k = dst_head / rep, r = dst_head % rep;
    return r * num_k_heads + k;
}

// Fills `row_source[d] = the RAW GGUF row index that must be copied into sidecar row d`, for d in
// [0, out_f) -- identity outside the V block [base, base + num_v_heads*head_v_dim). `row_source.size()`
// must be >= out_f; extra tail entries (there are none at a real call site) are left untouched.
inline void gdn_row_permutation(std::uint32_t out_f, std::uint32_t base, int num_k_heads, int num_v_heads,
                                std::uint32_t head_v_dim, std::span<std::uint32_t> row_source) {
    for (std::uint32_t i = 0; i < out_f && i < row_source.size(); ++i) row_source[i] = i;
    if (num_k_heads <= 0 || num_v_heads <= 0 || num_v_heads % num_k_heads != 0) return;
    const int rep = num_v_heads / num_k_heads;
    for (int dst_head = 0; dst_head < num_v_heads; ++dst_head) {
        const int src_head = gdn_v_src_head(dst_head, num_k_heads, rep);
        for (std::uint32_t g = 0; g < head_v_dim; ++g) {
            const std::uint32_t d = base + static_cast<std::uint32_t>(dst_head) * head_v_dim + g;
            const std::uint32_t s = base + static_cast<std::uint32_t>(src_head) * head_v_dim + g;
            if (d < out_f && s < out_f && d < row_source.size()) row_source[d] = s;
        }
    }
}

// Fills `idx[i] = the HF/grouped activation position GGUF/tiled column i needs`, for i in
// [0, num_v_heads*head_v_dim) -- GdnOutProj's whole input axis is the V-head axis (base == 0, see this
// file's header comment), so no base parameter is needed here the way gdn_row_permutation has one.
// `idx.size()` must be >= num_v_heads*head_v_dim.
inline void gdn_out_gather_index(int num_k_heads, int num_v_heads, std::uint32_t head_v_dim,
                                 std::span<std::uint32_t> idx) {
    const std::uint32_t value_dim = static_cast<std::uint32_t>(num_v_heads) * head_v_dim;
    for (std::uint32_t i = 0; i < value_dim && i < idx.size(); ++i) idx[i] = i;
    if (num_k_heads <= 0 || num_v_heads <= 0 || num_v_heads % num_k_heads != 0) return;
    const int rep = num_v_heads / num_k_heads;
    for (int src_head = 0; src_head < num_v_heads; ++src_head) {
        const int k = src_head % num_k_heads, r = src_head / num_k_heads;
        const int dst_head = k * rep + r;
        for (std::uint32_t g = 0; g < head_v_dim; ++g) {
            const std::uint32_t i = static_cast<std::uint32_t>(src_head) * head_v_dim + g;
            const std::uint32_t j = static_cast<std::uint32_t>(dst_head) * head_v_dim + g;
            if (i < value_dim && i < idx.size()) idx[i] = j;
        }
    }
}

// --- on-disk format --------------------------------------------------------------------------------
//
// Fixed-size, written directly to disk. Pinned by static_assert below for the same reason
// ModelHeader's/moeq::Header's own sizes are (AGENTS.md S3): a field change that moves it must be a
// build error, not a file that reads as garbage.
struct Header {
    char           magic[4] = {'S', '0', 'B', '1'};
    std::uint32_t  version  = 1;
    // The one axis this sidecar's own indexing needs: how many layers the descriptor table covers.
    // Deliberately NOT d_model/d_ff/num_experts (moeq::Header carries those, but every role here already
    // states its own in_f/out_f per Desc, and this format has no expert axis to size) -- AGENTS.md S8,
    // no field nothing reads.
    std::int32_t   n_layers = 0;
    std::int32_t   reserved = 0;
    std::uint64_t  n_tensors = 0;    // number of Desc entries actually present (a SPARSE count -- see below)
    std::uint64_t  data_off  = 0;    // byte offset from file start to the payload
    std::uint64_t  data_bytes = 0;
    // The PARAM_FLOATS of the S0L5 blob this sidecar belongs beside -- the same "these two files are one
    // model" check moeq::Header::model_param_floats makes.
    std::uint64_t  model_param_floats = 0;
};
static_assert(sizeof(Header) == 48, "the S0B1 header's on-disk size must not change");
static_assert(alignof(Header) == 8);

// One backbone tensor. Unlike moeq::Desc (a dense `(layer, expert, which)` grid, every cell always
// present), this table is SPARSE: a per-layer role like QsaKProj has a real source tensor only on QSA
// layers, and the writer discovers that by lookup rather than a caller predicting layer counts per role
// (this header has no access to MIXER_SCHEDULE, an sub0_config.hpp fact). So each Desc carries its own
// `(role, layer)` identity rather than being addressed by a formula -- `Store::find` below resolves it
// through a small hash map built once at open(), not a per-lookup scan.
struct Desc {
    std::int32_t   role  = 0;        // Role, stored as its underlying int (AGENTS.md: on-disk format
                                      // fields are plain ints, never the enum type itself)
    std::int32_t   layer = -1;       // -1 for a role with role_per_layer(role) == false
    std::uint32_t  type_raw = 0;     // raw GGML type id -- Q8_0/Q4_K/Q5_K/Q6_K only, checked at write time
    std::uint32_t  in_f  = 0;        // GGUF ne[0]: elements per row (bbqd's own `row_elems`)
    std::uint32_t  out_f = 0;        // GGUF ne[1]: row count (bbqd's own `n_rows`)
    std::uint32_t  reserved = 0;
    std::uint64_t  off = 0;          // byte offset from Header::data_off
    std::uint64_t  bytes = 0;        // encoded length
};
static_assert(sizeof(Desc) == 40, "the S0B1 descriptor's on-disk size must not change");
static_assert(alignof(Desc) == 8);

// Packs (role, layer) into one lookup key. `layer + 1` so the model-level roles' `layer == -1` maps to a
// non-negative key component, matching the convention transplant.hpp's own Slot::layer uses for the same
// "-1 means model-level" case.
[[nodiscard]] constexpr std::uint64_t role_key(Role role, int layer) {
    return (static_cast<std::uint64_t>(static_cast<std::uint32_t>(role)) << 32)
           | static_cast<std::uint32_t>(layer + 1);
}

// --- the one decode, for tooling/verification use only -- NOT a decode hot path ----------------------
//
// bbqd::gemv_plane consumes a Desc's raw bytes DIRECTLY (no float intermediate at all -- that is the
// entire point of native-quant residency); nothing in the decode path this sidecar feeds ever calls a
// function like this one. It exists for round-trip verification (this header's own tests, and the
// transplant tool's own post-write check) and is written with `std::vector` accordingly -- an allocating
// helper is correct here precisely because it is NOT a per-token/per-call path (AGENTS.md S1 constrains
// hot paths, not a one-shot tool/test decode).
[[nodiscard]] inline bool dequantize_role_to_f32(const Desc& d, std::span<const std::uint8_t> raw,
                                                 std::vector<float>& out) {
    gguf::TensorInfo t;
    t.type_raw = d.type_raw;
    t.dims = {static_cast<std::uint64_t>(d.in_f) * d.out_f};
    return gguf::to_f32(t, raw, out);
}

// Decode one GGUF-order row into caller-owned storage. Token embedding is a gather, so decoding the
// whole tensor (or allocating a temporary vector) on each token would defeat native residency.
[[nodiscard]] inline bool dequantize_row(const Desc& d, std::span<const std::uint8_t> raw,
                                         std::uint32_t row, std::span<float> out) {
    if (row >= d.out_f || out.size() != d.in_f || d.in_f > static_cast<std::uint32_t>(INT_MAX) ||
        !bbqd::fusable(d.type_raw, static_cast<int>(d.in_f)) ||
        bbqd::plane_bytes(d.type_raw, d.in_f) == 0 ||
        bbqd::plane_bytes(d.type_raw, static_cast<std::uint64_t>(d.in_f) * d.out_f) > raw.size())
        return false;
    for (std::uint32_t col = 0; col < d.in_f; col += bbqd::GROUP) {
        const std::uint64_t p = static_cast<std::uint64_t>(row) * d.in_f + col;
        bbqd::WeightGroup group;
        switch (static_cast<gguf::TensorType>(d.type_raw)) {
            case gguf::TensorType::Q8_0: group = bbqd::Q8_0Plane{raw.data()}.group(p); break;
            case gguf::TensorType::Q4_K: group = bbqd::Q4KPlane{raw.data()}.group(p); break;
            case gguf::TensorType::Q5_K: group = bbqd::Q5KPlane{raw.data()}.group(p); break;
            case gguf::TensorType::Q6_K: group = bbqd::Q6KPlane{raw.data()}.group(p); break;
            default: return false;
        }
        for (int j = 0; j < bbqd::GROUP / 2; ++j)
            out[col + static_cast<std::uint32_t>(j)] = group.scale_lo * group.q[static_cast<std::size_t>(j)] + group.bias_lo;
        for (int j = bbqd::GROUP / 2; j < bbqd::GROUP; ++j)
            out[col + static_cast<std::uint32_t>(j)] = group.scale_hi * group.q[static_cast<std::size_t>(j)] + group.bias_hi;
    }
    return true;
}

// --- reader ------------------------------------------------------------------------------------------
//
// Holds the sidecar's descriptor table and a read-only MAPPING of its encoded payload, in its native
// form -- same "mapped, not eagerly read" reasoning as moeq::Store (docs/WP4_SCOPE.md WP5b): the backbone
// sidecar is smaller than the MoE one (S9's measured total), but there is no reason to pay an eager read
// twice over when one mapping mechanism already exists and is proven.
class Store {
public:
    // Returns false and fills `err` on any problem; never throws, never partially initializes.
    //
    // `expect_model_param_floats`: when non-zero, the caller's OWN `.bin` blob's `PARAM_FLOATS` -- open()
    // refuses a sidecar paired with a different model rather than silently combining two models' worth of
    // weights (AGENTS.md S3, the same check `src/backends/cpu/backend.cpp`'s `load_moe_quant_sidecar`
    // makes against `moeq::Header::model_param_floats`). Unlike that engine-side check, this header has no
    // compile-time `PARAM_FLOATS` of its own to compare against (it is deliberately engine-free), so the
    // expected value is caller-supplied. 0 means "skip the check" -- a real model always has billions of
    // parameters, so 0 is never a legitimate value to guard against by coincidence.
    [[nodiscard]] bool open(const std::string& path, std::string& err,
                            std::uint64_t expect_model_param_floats = 0) {
        map_.close();
        descs_.clear();
        index_.clear();
        h_ = Header{};
        data_ = nullptr;

        FileMap m;
        if (!m.open(path, err)) return false;
        const std::uint8_t* p = m.data();
        const std::uint64_t file_bytes = m.size();

        if (file_bytes < sizeof(Header)) { err = path + ": truncated header"; return false; }
        std::memcpy(&h_, p, sizeof h_);
        if (std::memcmp(h_.magic, "S0B1", 4) != 0) { err = path + ": not an S0B1 sidecar"; return false; }
        if (h_.version != 1) { err = path + ": unsupported S0B1 version"; return false; }
        if (expect_model_param_floats != 0 && h_.model_param_floats != expect_model_param_floats) {
            err = path + ": model_param_floats does not match the paired .bin blob (sidecar belongs to "
                         "a different model build)";
            return false;
        }

        if (h_.n_layers <= 0 || h_.n_tensors > (file_bytes - sizeof(Header)) / sizeof(Desc)) {
            err = path + ": truncated descriptor table";
            return false;
        }
        const std::uint64_t table_bytes = h_.n_tensors * sizeof(Desc);
        if (h_.data_off < sizeof(Header) + table_bytes) {
            err = path + ": payload overlaps the descriptor table";
            return false;
        }
        descs_.resize(static_cast<std::size_t>(h_.n_tensors));
        std::memcpy(descs_.data(), p + sizeof(Header), static_cast<std::size_t>(table_bytes));

        // The payload must lie wholly inside the file, and every descriptor wholly inside the payload --
        // same two checks moeq::Store::open makes, for the same reason (a mapping's addressing failure is
        // an access violation, not a short read, so this has to be explicit).
        if (h_.data_off > file_bytes || h_.data_bytes > file_bytes - h_.data_off) {
            err = path + ": the payload runs past the end of the file";
            return false;
        }
        index_.reserve(descs_.size());
        for (std::size_t i = 0; i < descs_.size(); ++i) {
            const Desc& d = descs_[i];
            if (d.off > h_.data_bytes || d.bytes > h_.data_bytes - d.off) {
                err = path + ": a descriptor's byte range runs past the payload";
                return false;
            }
            if (d.role < 0 || d.role >= kRoleCount) {
                err = path + ": a descriptor names an unrecognised role";
                return false;
            }
            const Role role = static_cast<Role>(d.role);
            if ((role_per_layer(role) && (d.layer < 0 || d.layer >= h_.n_layers)) ||
                (!role_per_layer(role) && d.layer != -1)) {
                err = path + ": a descriptor names an invalid layer for its role";
                return false;
            }
            if (d.in_f > static_cast<std::uint32_t>(INT_MAX) ||
                d.out_f > static_cast<std::uint32_t>(INT_MAX) || d.out_f == 0 ||
                !bbqd::fusable(d.type_raw, static_cast<int>(d.in_f)) ||
                bbqd::plane_bytes(d.type_raw, d.in_f) == 0 ||
                bbqd::plane_bytes(d.type_raw, static_cast<std::uint64_t>(d.in_f) * d.out_f) != d.bytes) {
                err = path + ": a descriptor has invalid plane geometry";
                return false;
            }
            const std::uint64_t key = role_key(role, d.layer);
            // A duplicate (role, layer) pair would make find() silently prefer one over the other --
            // refused rather than the writer's own bug becoming a loader-side ambiguity.
            if (!index_.emplace(key, i).second) {
                err = path + ": duplicate (role, layer) descriptor";
                return false;
            }
        }
        map_ = std::move(m);
        data_ = map_.data() + h_.data_off;
        return true;
    }

    [[nodiscard]] const Header& header() const { return h_; }
    [[nodiscard]] bool loaded() const { return data_ != nullptr; }

    // nullptr when this (role, layer) has no native-quant plane in this sidecar -- an excluded role
    // (F32/BF16/VPerm-affected, see this file's header comment), or a per-layer role whose mixer branch
    // is not present at that layer (a QSA role on a GDN layer). A caller must treat that as "fall back to
    // the .bin blob's own bf16/f32 value for this tensor", never as an error.
    [[nodiscard]] const Desc* find(Role role, int layer = -1) const {
        const auto it = index_.find(role_key(role, layer));
        return it == index_.end() ? nullptr : &descs_[it->second];
    }

    [[nodiscard]] std::span<const std::uint8_t> raw(const Desc& d) const {
        return {data_ + d.off, static_cast<std::size_t>(d.bytes)};
    }

    [[nodiscard]] std::uint64_t resident_bytes() const { return h_.data_bytes; }

    /** Read one byte of every 4 KiB page of the mapped payload, so decode never takes a first-touch
     * soft page fault on it.
     *
     * The bf16 blob is read into memory at load, and the batched forward() touches it again; this
     * sidecar is only mapped, and decode is its only reader. Without this, the first decode tokens pay
     * about one fault per page of native weights inside the timed path: ~470k extra faults over six
     * tokens with the GDN roles present, which hid their whole gain (docs/BACKBONE_NATIVE_QUANT.md S17).
     * @return a byte sum, so the reads cannot be optimized away.
     */
    [[nodiscard]] std::uint64_t prefault() const noexcept {
        constexpr std::uint64_t kPage = 4096;
        std::uint64_t sum = 0;
        for (std::uint64_t off = 0; off < h_.data_bytes; off += kPage) sum += data_[off];
        return sum;
    }

private:
    Header                                    h_{};
    std::vector<Desc>                         descs_;
    std::unordered_map<std::uint64_t, std::size_t> index_;   // role_key() -> descs_ index
    FileMap                                   map_;
    const std::uint8_t*                       data_ = nullptr;   // map_.data() + h_.data_off, or null
};

}  // namespace sub0::bbq
