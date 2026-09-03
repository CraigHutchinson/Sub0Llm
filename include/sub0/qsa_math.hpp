// sub0/qsa_math.hpp -- Qwen Sparse Attention (QSA) + lightning indexer, Stage 1: the shared,
// engine-free forward math core, mirroring gdn_math.hpp's / gated_residual_math.hpp's / moe_math.hpp's
// role for their own mechanisms.
//
// Every equation here is re-derived from transformers==5.16.1's REAL, installed
// `transformers.models.qwen4_exp.modeling_qwen4_exp` source (AGENTS.md S5 -- fetched via
// `python -c "import inspect, transformers.models.qwen4_exp.modeling_qwen4_exp as m; ..."`, not
// recalled), specifically `Qwen4ExpTextQSAIndexer.forward`, `Qwen4ExpTextAttention.forward`,
// `apply_rotary_pos_emb`, `rotate_half`, `eager_attention_forward`, `repeat_kv` and
// `Qwen4ExpTextRMSNorm`. See docs/QSA.md S1 for the full quoted source.
//
// Weight-layout convention: every 2D weight here uses THIS PROJECT'S OWN [rows=in, cols=out] layout
// (row p contiguous over output, `w[p*out+o]`) -- the transpose of the PyTorch nn.Linear
// `[out_features, in_features]` convention the fixture's raw weight files are extracted in. Re-derived
// explicitly rather than assumed, per AGENTS.md S5 -- the same axis-order flip gdn_math.hpp's /
// gated_residual_math.hpp's / moe_math.hpp's own header comments already document.
//
// Norm convention: Qwen4ExpTextRMSNorm's gain is `(1.0 + weight)`, NOT `weight` (a zero-initialized
// weight is the IDENTITY norm). This project's own op_rmsnorm/op_qknorm use `weight` directly -- a real,
// verified divergence, documented in docs/GATED_RESIDUAL.md S1a and honoured here rather than reusing
// the engine's convention (which would make a QSA layer numerically NOT the real model's layer, defeating
// the fixture gate's whole purpose -- docs/QSA.md S2a).
//
// RoPE convention: apply_rotary_pos_emb's HALF-SPLIT rotation over a `rotary_dim` PREFIX of each head
// vector, with the remaining (head_dim - rotary_dim) channels passed through UNROTATED. This project's
// own op_rope is INTERLEAVED-pair and full-width -- another real divergence, so cos/sin are taken as
// CALLER-SUPPLIED tables here rather than derived from ROPE_THETA internally (which also lets the
// fixture feed the real model's own partial-rotary cos/sin unchanged).
//
// No heap allocation (AGENTS.md S1): every buffer this needs is caller-supplied, sized by the
// *_scratch_floats() helpers below.

#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>

namespace sub0::qsa {

// Every dimension the mechanism needs, explicit rather than closed over a build's own constants -- same
// reasoning as gdn_math.hpp's / gated_residual_math.hpp's / moe_math.hpp's own Dims (lets a standalone
// test exercise the real fixture's own shape regardless of what this project's compiled build happens to
// be configured for). NOTE head_dim is INDEPENDENT of hidden_size/n_heads here: the real model has
// hidden_size=2560, n_heads=24, head_dim=256, so n_heads*head_dim != hidden_size (docs/QSA.md S2b.1).
// The engine instantiates this at head_dim = D_HEAD = D_MODEL/N_HEADS; the fixture does not have to.
struct Dims {
    int hidden_size;       // hidden_size (real: 2560)
    int n_heads;           // num_attention_heads (real: 24)
    int head_dim;          // head_dim (real: 256)
    int n_kv_heads;        // num_key_value_heads (real: 2)
    int idx_n_heads;       // indexer_n_heads (real: 4)
    int idx_kv_heads;      // indexer_kv_heads (real: 1 -- the reference's .squeeze(2) requires exactly 1)
    int idx_head_dim;      // indexer_head_dim (real: 128)
    int budget;            // indexer_budget (real: 2048)
    int compress_ratio;    // indexer_compress_ratio (real: 4)
    int rotary_dim;        // cos/sin width = partial_rotary_factor * head_dim (real: 64). Applies to BOTH
                            // the attention heads AND the indexer heads -- the reference reuses the SAME
                            // cos/sin for its indexer_head_dim=128 vectors (docs/QSA.md S1b).

    constexpr int q_width()     const { return n_heads * head_dim; }
    constexpr int kv_width()    const { return n_kv_heads * head_dim; }
    constexpr int idx_q_width() const { return idx_n_heads * idx_head_dim; }
    constexpr int idx_qk_out()  const { return (idx_n_heads + idx_kv_heads) * idx_head_dim; }
    // block_topk = token_budget // compress_ratio (Qwen4ExpTextQSAIndexer.__init__, verbatim).
    constexpr int block_topk()  const { return compress_ratio > 0 ? budget / compress_ratio : 0; }
    constexpr int gqa_group()   const { return n_kv_heads > 0 ? n_heads / n_kv_heads : 1; }
};

// The real model's own rms_norm_eps (config.json: 1e-06), hardcoded as a verified real value rather than
// exposed as a knob -- the same precedent gdn_math.hpp's RMS_EPS/L2_EPS and gated_residual_math.hpp's own
// 1e-6f set. Callers may pass their own eps where the signature takes one.
inline constexpr float RMS_EPS = 1e-6f;

namespace detail {
inline float sigmoid(float x) { return 1.f / (1.f + std::exp(-x)); }
}  // namespace detail

// --- scratch sizing -----------------------------------------------------------------------------
// indexer_select_row: the per-block score array (kv_len/compress_ratio, +1 slack) plus one pooled-key
// buffer (idx_head_dim). Destroyed by the call.
inline constexpr std::size_t select_scratch_floats(const Dims& d, int max_kv) {
    return static_cast<std::size_t>(max_kv / (d.compress_ratio > 0 ? d.compress_ratio : 1) + 1)
         + static_cast<std::size_t>(d.idx_head_dim);
}
// attn_row: the per-query score array over the kv window (max_kv) plus the pre-o_proj attention output
// (q_width).
inline constexpr std::size_t attn_scratch_floats(const Dims& d, int max_kv) {
    return static_cast<std::size_t>(max_kv) + static_cast<std::size_t>(d.q_width());
}
// forward(): the K/V caches and the indexer's raw-key cache for the whole T-row prefill, plus one row's
// worth of every per-row temporary, plus the two row helpers' own scratch. Scaled by T (unlike
// moe_math.hpp's row-independent scratch) because attention is genuinely NOT row-independent -- every
// query reads every earlier row's K/V.
inline constexpr std::size_t scratch_floats(const Dims& d, int T) {
    return 2u * static_cast<std::size_t>(T) * static_cast<std::size_t>(d.kv_width())   // k_cache, v_cache
         + static_cast<std::size_t>(T) * static_cast<std::size_t>(d.idx_head_dim)      // raw indexer keys
         + static_cast<std::size_t>(d.idx_q_width())                                    // indexer q (1 row)
         + 2u * static_cast<std::size_t>(d.q_width())                                   // q, gate (1 row)
         + static_cast<std::size_t>(T)                                                  // visibility mask
         + select_scratch_floats(d, T) + attn_scratch_floats(d, T);
}

// --- primitives ---------------------------------------------------------------------------------

// Qwen4ExpTextRMSNorm (non-grouped): out = x * rsqrt(mean(x^2) + eps) * (1 + w). NOTE the (1 + w) gain
// (this file's header comment). `out` may alias `x`.
inline void rms_norm_row(const float* x, const float* w, int n, float eps, float* out) {
    float ms = 0.f;
    for (int i = 0; i < n; ++i) ms += x[i] * x[i];
    const float inv = 1.f / std::sqrt(ms / static_cast<float>(n) + eps);
    for (int i = 0; i < n; ++i) out[i] = x[i] * inv * (1.f + w[i]);
}

// apply_rotary_pos_emb for ONE head vector of width `head_dim`, rotating only its first `rotary_dim`
// channels (half-split within that prefix); channels [rotary_dim, head_dim) pass through untouched.
// cos/sin are this position's own `rotary_dim`-wide rows. In place.
inline void rope_apply_row(float* x, const float* cos, const float* sin, int rotary_dim) {
    const int half = rotary_dim / 2;
    for (int i = 0; i < half; ++i) {
        const float a = x[i], b = x[i + half];
        // q_rope = q_rope*cos + rotate_half(q_rope)*sin, rotate_half = (-x2, x1)
        x[i]        = a * cos[i]        + (-b) * sin[i];
        x[i + half] = b * cos[i + half] +   a  * sin[i + half];
    }
}

// out[o] = sum_i x[i] * w[i*out_n + o]  -- this project's [rows=in, cols=out] convention, no bias
// (attention_bias=false everywhere in the real config).
inline void linear_row(const float* x, const float* w, int in_n, int out_n, float* out) {
    for (int o = 0; o < out_n; ++o) out[o] = 0.f;
    for (int i = 0; i < in_n; ++i) {
        const float xi = x[i];
        if (xi == 0.f) continue;
        const float* wr = w + static_cast<std::size_t>(i) * out_n;
        for (int o = 0; o < out_n; ++o) out[o] += xi * wr[o];
    }
}

// --- the indexer --------------------------------------------------------------------------------

// One row's indexer projection (Qwen4ExpTextQSAIndexer.forward's first half, docs/QSA.md S1a):
// index_qk_proj -> torch.split into [idx_n_heads*hd | idx_kv_heads*hd] -> the QUERY half is
// q_layernorm'd per head and rotated at this row's position; the KEY half is left COMPLETELY RAW
// (k_layernorm and RoPE happen later, on the POOLED block key, at the block's own start position).
// qk_proj_w: [hidden_size, idx_qk_out()]. q_ln_w: [idx_head_dim].
inline void indexer_project_row(const Dims& d, const float* x, const float* qk_proj_w,
                                 const float* q_ln_w, const float* cos_pos, const float* sin_pos,
                                 float eps, float* out_q, float* out_raw_k) {
    // Project straight into out_q for the query part; the key part needs its own tail slot, so the
    // projection is done in two passes over the SAME weight columns rather than needing a joint buffer.
    for (int o = 0; o < d.idx_q_width(); ++o) out_q[o] = 0.f;
    for (int o = 0; o < d.idx_kv_heads * d.idx_head_dim; ++o) out_raw_k[o] = 0.f;
    const int out_n = d.idx_qk_out();
    for (int i = 0; i < d.hidden_size; ++i) {
        const float xi = x[i];
        if (xi == 0.f) continue;
        const float* wr = qk_proj_w + static_cast<std::size_t>(i) * out_n;
        for (int o = 0; o < d.idx_q_width(); ++o) out_q[o] += xi * wr[o];
        for (int o = 0; o < d.idx_kv_heads * d.idx_head_dim; ++o)
            out_raw_k[o] += xi * wr[d.idx_q_width() + o];
    }
    for (int h = 0; h < d.idx_n_heads; ++h) {
        float* qh = out_q + static_cast<std::size_t>(h) * d.idx_head_dim;
        rms_norm_row(qh, q_ln_w, d.idx_head_dim, eps, qh);
        rope_apply_row(qh, cos_pos, sin_pos, d.rotary_dim);
    }
}

// Block selection for ONE query (Qwen4ExpTextQSAIndexer.forward's per-query loop body, docs/QSA.md S1a),
// restricted to this engine's only mask -- a contiguous causal prefix [0, kv_len) (docs/QSA.md S2b.3;
// at a contiguous prefix `local_visible_indices == arange(kv_len)`, so the reference's general
// torch.nonzero path and this one are identical by inspection).
//
//   num_complete_blocks = kv_len / compress_ratio
//   pooled_b            = mean over the block's compress_ratio raw keys      (float mean)
//   block_key_b         = rope(k_layernorm(pooled_b), at the block's FIRST token's position)
//   score_b             = sum_h relu(q_h . block_key_b) / sqrt(idx_head_dim)   (ReLU BEFORE the head-sum)
//   selected            = topk(score, min(block_topk, num_complete_blocks))'s tokens
//                         UNION the incomplete tail [nb*ratio, kv_len)         (ALWAYS visible)
//
// `raw_keys`: [kv_len, idx_head_dim], the UNNORMED, UNROTATED per-token indexer keys.
// `cos`/`sin`: [>= kv_len, rotary_dim] full-position tables.
// `out_mask`: [kv_len] floats, written 1.0f for visible / 0.0f for masked-out.
// `scratch`: >= select_scratch_floats(d, kv_len).
// Returns the number of visible positions.
inline int indexer_select_row(const Dims& d, const float* q, const float* raw_keys, int kv_len,
                               const float* k_ln_w, const float* cos, const float* sin, float eps,
                               float* out_mask, float* scratch) {
    for (int j = 0; j < kv_len; ++j) out_mask[j] = 0.f;
    const int ratio = d.compress_ratio;
    const int nb    = kv_len / ratio;
    float* scores = scratch;                        // [nb]
    float* pooled = scratch + nb + 1;               // [idx_head_dim]
    const float inv_sqrt_hd = 1.f / std::sqrt(static_cast<float>(d.idx_head_dim));
    for (int b = 0; b < nb; ++b) {
        const int start = b * ratio;
        for (int j = 0; j < d.idx_head_dim; ++j) pooled[j] = 0.f;
        for (int t = start; t < start + ratio; ++t) {
            const float* kt = raw_keys + static_cast<std::size_t>(t) * d.idx_head_dim;
            for (int j = 0; j < d.idx_head_dim; ++j) pooled[j] += kt[j];
        }
        const float inv_ratio = 1.f / static_cast<float>(ratio);
        for (int j = 0; j < d.idx_head_dim; ++j) pooled[j] *= inv_ratio;
        rms_norm_row(pooled, k_ln_w, d.idx_head_dim, eps, pooled);
        // Rotated at the BLOCK's own first token position (`group_starts`), not at the query's.
        rope_apply_row(pooled, cos + static_cast<std::size_t>(start) * d.rotary_dim,
                        sin + static_cast<std::size_t>(start) * d.rotary_dim, d.rotary_dim);
        float s = 0.f;
        for (int h = 0; h < d.idx_n_heads; ++h) {
            const float* qh = q + static_cast<std::size_t>(h) * d.idx_head_dim;
            float dot = 0.f;
            for (int j = 0; j < d.idx_head_dim; ++j) dot += qh[j] * pooled[j];
            s += dot > 0.f ? dot : 0.f;             // relu BEFORE the head-sum (docs/QSA.md S1a)
        }
        scores[b] = s * inv_sqrt_hd;
    }
    // Top-k over blocks: the same bounded repeated-max scan moe_math.hpp's router_topk_row uses, marking
    // each pick consumed. Ties resolve to the lowest block index. Host-side scalar code inside the math
    // core, deliberately NOT a separate differentiable Node -- docs/QSA.md S4b/S5.
    const int k = std::min(d.block_topk(), nb);
    int visible = 0;
    for (int pick = 0; pick < k; ++pick) {
        int best = -1; float bestv = -std::numeric_limits<float>::infinity();
        for (int b = 0; b < nb; ++b) if (scores[b] > bestv) { bestv = scores[b]; best = b; }
        if (best < 0) break;
        scores[best] = -std::numeric_limits<float>::infinity();
        for (int t = best * ratio; t < best * ratio + ratio; ++t) { out_mask[t] = 1.f; ++visible; }
    }
    // The incomplete TAIL is ALWAYS visible, regardless of any score (docs/QSA.md S1a).
    for (int t = nb * ratio; t < kv_len; ++t) { out_mask[t] = 1.f; ++visible; }
    return visible;
}

// --- the attention sublayer ---------------------------------------------------------------------

// One row's attention projections (Qwen4ExpTextAttention.forward, docs/QSA.md S1b). The real model's
// q_proj is ONE [hidden, n_heads*head_dim*2] Linear chunked PER HEAD into (query | gate); this port
// stores the two halves as two separate [hidden, n_heads*head_dim] tensors, which is exactly the same
// arithmetic (a chunk of a bias-free Linear's output axis is a partition of its weight rows) -- see
// docs/QSA.md S2b.4 for why, and for the per-head chunk order a future weight transplant must respect.
// q_norm_w/k_norm_w: [head_dim], applied PER HEAD, BEFORE RoPE. v is neither normed nor rotated.
inline void attn_project_row(const Dims& d, const float* x, const float* q_w, const float* gate_w,
                              const float* k_w, const float* v_w, const float* q_norm_w,
                              const float* k_norm_w, const float* cos_pos, const float* sin_pos,
                              float eps, float* out_q, float* out_gate, float* out_k, float* out_v) {
    linear_row(x, q_w,    d.hidden_size, d.q_width(),  out_q);
    linear_row(x, gate_w, d.hidden_size, d.q_width(),  out_gate);
    linear_row(x, k_w,    d.hidden_size, d.kv_width(), out_k);
    linear_row(x, v_w,    d.hidden_size, d.kv_width(), out_v);
    for (int h = 0; h < d.n_heads; ++h) {
        float* qh = out_q + static_cast<std::size_t>(h) * d.head_dim;
        rms_norm_row(qh, q_norm_w, d.head_dim, eps, qh);
        rope_apply_row(qh, cos_pos, sin_pos, d.rotary_dim);
    }
    for (int h = 0; h < d.n_kv_heads; ++h) {
        float* kh = out_k + static_cast<std::size_t>(h) * d.head_dim;
        rms_norm_row(kh, k_norm_w, d.head_dim, eps, kh);
        rope_apply_row(kh, cos_pos, sin_pos, d.rotary_dim);
    }
}

// Masked softmax attention for ONE query row over the kv window [0, kv_len), followed by the sigmoid
// output gate and o_proj (eager_attention_forward + Qwen4ExpTextAttention.forward's tail, docs/QSA.md
// S1b). `mask[j] != 0` means position j is visible; a fully-masked head would divide by zero, which
// cannot happen here because indexer_select_row always keeps at least the query's own tail position.
// k_cache/v_cache: [kv_len, kv_width()]. o_proj_w: [q_width(), hidden_size].
// `scratch`: >= attn_scratch_floats(d, kv_len).
inline void attn_row(const Dims& d, const float* q, const float* gate, const float* k_cache,
                      const float* v_cache, int kv_len, const float* mask, const float* o_proj_w,
                      float* out, float* scratch) {
    float* sc = scratch;                     // [kv_len]
    float* ao = scratch + kv_len;            // [q_width()]
    const float scale = 1.f / std::sqrt(static_cast<float>(d.head_dim));
    const int group = d.gqa_group();
    for (int h = 0; h < d.n_heads; ++h) {
        const int off    = h * d.head_dim;
        const int off_kv = (h / group) * d.head_dim;   // repeat_kv: the shared KV head for this group
        float mx = -std::numeric_limits<float>::infinity();
        for (int j = 0; j < kv_len; ++j) {
            if (mask[j] == 0.f) { sc[j] = -std::numeric_limits<float>::infinity(); continue; }
            const float* kj = k_cache + static_cast<std::size_t>(j) * d.kv_width() + off_kv;
            float s = 0.f;
            for (int a = 0; a < d.head_dim; ++a) s += q[off + a] * kj[a];
            s *= scale;
            sc[j] = s;
            mx = std::max(mx, s);
        }
        float Z = 0.f;
        for (int j = 0; j < kv_len; ++j) {
            sc[j] = (sc[j] == -std::numeric_limits<float>::infinity()) ? 0.f : std::exp(sc[j] - mx);
            Z += sc[j];
        }
        for (int a = 0; a < d.head_dim; ++a) ao[off + a] = 0.f;
        for (int j = 0; j < kv_len; ++j) {
            if (sc[j] == 0.f) continue;
            const float p = sc[j] / Z;
            const float* vj = v_cache + static_cast<std::size_t>(j) * d.kv_width() + off_kv;
            for (int a = 0; a < d.head_dim; ++a) ao[off + a] += p * vj[a];
        }
    }
    // attn_output = attn_output * sigmoid(gate), elementwise over the flat [n_heads*head_dim] row.
    for (int o = 0; o < d.q_width(); ++o) ao[o] *= detail::sigmoid(gate[o]);
    linear_row(ao, o_proj_w, d.q_width(), d.hidden_size, out);
}

// The whole QSA mixer sublayer for a T-row prefill (positions 0..T-1, causal). A thin loop over the row
// helpers above, so the engine's batched Model::forward and its single-token Model::forward_one provably
// run the SAME arithmetic (the forward-vs-forward_one parity test is what gates that -- docs/QSA.md S9).
// `cos`/`sin`: [>= T, rotary_dim]. `out`: [T, hidden_size]. `scratch`: >= scratch_floats(d, T).
inline void forward(const Dims& d, int T, const float* hidden,
                     const float* idx_qk_proj_w, const float* idx_q_ln_w, const float* idx_k_ln_w,
                     const float* q_w, const float* gate_w, const float* k_w, const float* v_w,
                     const float* q_norm_w, const float* k_norm_w, const float* o_proj_w,
                     const float* cos, const float* sin, float eps, float* out, float* scratch) {
    const std::size_t kvw = static_cast<std::size_t>(d.kv_width());
    float* k_cache  = scratch;
    float* v_cache  = k_cache + static_cast<std::size_t>(T) * kvw;
    float* raw_keys = v_cache + static_cast<std::size_t>(T) * kvw;
    float* idx_q    = raw_keys + static_cast<std::size_t>(T) * d.idx_head_dim;
    float* q_row    = idx_q + d.idx_q_width();
    float* gate_row = q_row + d.q_width();
    float* mask     = gate_row + d.q_width();
    float* sel_scr  = mask + T;
    float* att_scr  = sel_scr + select_scratch_floats(d, T);

    for (int t = 0; t < T; ++t) {
        const float* x = hidden + static_cast<std::size_t>(t) * d.hidden_size;
        const float* cos_t = cos + static_cast<std::size_t>(t) * d.rotary_dim;
        const float* sin_t = sin + static_cast<std::size_t>(t) * d.rotary_dim;
        // The indexer's raw key for THIS token must exist before it can be selected, and the reference
        // caches raw keys for the whole visible prefix -- so project every row's key as we go.
        indexer_project_row(d, x, idx_qk_proj_w, idx_q_ln_w, cos_t, sin_t, eps, idx_q,
                             raw_keys + static_cast<std::size_t>(t) * d.idx_head_dim);
        attn_project_row(d, x, q_w, gate_w, k_w, v_w, q_norm_w, k_norm_w, cos_t, sin_t, eps,
                          q_row, gate_row, k_cache + static_cast<std::size_t>(t) * kvw,
                          v_cache + static_cast<std::size_t>(t) * kvw);
        indexer_select_row(d, idx_q, raw_keys, t + 1, idx_k_ln_w, cos, sin, eps, mask, sel_scr);
        attn_row(d, q_row, gate_row, k_cache, v_cache, t + 1, mask, o_proj_w,
                  out + static_cast<std::size_t>(t) * d.hidden_size, att_scr);
    }
}

}  // namespace sub0::qsa
