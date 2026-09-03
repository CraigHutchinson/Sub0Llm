// sub0/moe_math.hpp -- Mixture-of-Experts (MoE) Stage 1: the shared, engine-free forward math core,
// mirroring gdn_math.hpp's/gated_residual_math.hpp's role for their own mechanisms.
//
// Every equation here is re-derived from transformers==5.16.1's REAL, installed
// `transformers.models.qwen4_exp.modeling_qwen4_exp` source (AGENTS.md S5 -- fetched via
// `python3 -c "import inspect, transformers.models.qwen4_exp.modeling_qwen4_exp as m; ..."`, not
// recalled), specifically `Qwen4ExpTextSparseMoeBlock.forward`, `Qwen4ExpTextTopKRouter.forward`,
// `Qwen4ExpTextExperts.forward` and `Qwen4ExpTextMLP.forward` (the shared expert's own module). See
// docs/MOE.md S1 for the full quoted source.
//
// Weight-layout convention: every 2D weight here uses THIS PROJECT'S OWN [rows=in, cols=out] layout
// (row p contiguous over output, `w[p*out+o]`) -- the transpose of the PyTorch nn.Linear
// `[out_features, in_features]` convention (and the real `gate_up_proj`/`down_proj`'s own
// `[num_experts, out, in]`/`[num_experts, out, in]` 3D layout) the fixture's raw weight files are
// extracted in. Re-derived explicitly rather than assumed, per AGENTS.md S5 -- the same axis-order flip
// gdn_math.hpp's/gated_residual_math.hpp's own header comments already document.
//
// No heap allocation (AGENTS.md S1): every buffer this needs is caller-supplied. Row-independence (S2 of
// docs/MOE.md): unlike Gated DeltaNet's recurrent state, MoE routing has no cross-token memory at all --
// every row's router/expert computation depends only on that row's own input, so a single reusable
// scratch buffer serves an entire [T, hidden_size] batch (scratch_floats() below is NOT scaled by T,
// unlike gdn_math.hpp's own scratch_floats(d, T)).

#pragma once

#include <cmath>
#include <cstddef>

namespace sub0::moe {

// Every dimension the module needs, explicit rather than closed over a build's own constants -- same
// reasoning as gdn_math.hpp's/gated_residual_math.hpp's own Dims (lets a standalone test exercise the
// real fixture's own shape regardless of what this project's compiled build happens to be configured
// for).
struct Dims {
    int hidden_size;      // D_MODEL
    int d_ff;              // per-expert AND shared-expert SwiGLU intermediate width (this project reuses
                            // one width for both -- the real model's moe_intermediate_size==
                            // shared_expert_intermediate_size==640 happen to already be equal, docs/MOE.md
                            // S3a, so this is a real-model-faithful simplification, not an approximation)
    int num_experts;       // total routed experts (real model: 512)
    int experts_per_tok;   // top-k selected per token (real model: 10)
};

// Compile-time-in-spirit cap on experts_per_tok for this math core's own internal fixed-size stack
// buffers (topk_w/topk_idx in forward_row below) -- both this stage's own test scale (2) and the real
// model's own value (10) fit comfortably under it. `experts_per_tok` is a runtime `int` here (this file
// is dims-parameterized so a standalone test can exercise arbitrary hypothetical shapes, per this
// project's own gdn_math.hpp/gated_residual_math.hpp precedent), so a literal, generously-sized fixed
// bound -- not a heap allocation -- is what keeps this a no-heap function while still supporting that.
inline constexpr int TOPK_MAX = 16;

// scratch_floats(): num_experts (router logits/probs, reused as scratch, S4b) + d_ff (one expert's SwiGLU
// pre-activation) + hidden_size (one expert's FFN output before its top-k/shared weighting is applied).
// Deliberately NOT scaled by T (see this file's own header comment) -- one call's worth, reused across
// forward()'s own T-row loop below.
inline constexpr std::size_t scratch_floats(const Dims& d) {
    return static_cast<std::size_t>(d.num_experts)
         + static_cast<std::size_t>(d.d_ff)
         + static_cast<std::size_t>(d.hidden_size);
}

namespace detail {
inline float silu(float x) { return x / (1.f + std::exp(-x)); }
inline float sigmoid(float x) { return 1.f / (1.f + std::exp(-x)); }
}  // namespace detail

// One expert's SwiGLU FFN for a single row: out = down(silu(gate(x)) * up(x)) -- Qwen4ExpTextMLP.forward
// (also what Qwen4ExpTextExperts.forward computes per selected expert, chunked gate_up_proj split in
// half). gate_w/up_w: [hidden_size, d_ff]; down_w: [d_ff, hidden_size], this project's own [in,out]
// convention. `pre_scratch`: >= d_ff floats. Streams gate/up per output unit (no separate [d_ff] buffer
// needed for each) so scratch stays a single d_ff-wide buffer.
inline void expert_ffn_row(const Dims& d, const float* x, const float* gate_w, const float* up_w,
                            const float* down_w, float* out, float* pre_scratch) {
    for (int o = 0; o < d.d_ff; ++o) {
        float g = 0.f, u = 0.f;
        for (int i = 0; i < d.hidden_size; ++i) {
            const float xi = x[i];
            g += xi * gate_w[static_cast<std::size_t>(i) * d.d_ff + o];
            u += xi * up_w[static_cast<std::size_t>(i) * d.d_ff + o];
        }
        pre_scratch[o] = detail::silu(g) * u;
    }
    for (int j = 0; j < d.hidden_size; ++j) {
        float acc = 0.f;
        for (int o = 0; o < d.d_ff; ++o)
            acc += pre_scratch[o] * down_w[static_cast<std::size_t>(o) * d.hidden_size + j];
        out[j] = acc;
    }
}

// The router (Qwen4ExpTextTopKRouter.forward, docs/MOE.md S1a): logits = linear(x, router_w) (no bias,
// full-width softmax over ALL num_experts, THEN top-k -- not top-k-then-softmax); optionally renormalized
// (norm_topk_prob, real config default True) so the selected weights sum to 1. router_w: [hidden_size,
// num_experts]. `probs_scratch`: >= num_experts floats, destroyed (used as working storage for the
// selection scan below, S4b -- this row's own softmax values are not needed again after this call).
// `out_weight`/`out_idx`: caller buffers of length >= experts_per_tok (TOPK_MAX-capped internally).
inline void router_topk_row(const Dims& d, const float* x, const float* router_w, float* probs_scratch,
                             float* out_weight, int* out_idx, bool norm_topk_prob) {
    for (int e = 0; e < d.num_experts; ++e) {
        float s = 0.f;
        for (int i = 0; i < d.hidden_size; ++i)
            s += x[i] * router_w[static_cast<std::size_t>(i) * d.num_experts + e];
        probs_scratch[e] = s;
    }
    float mx = probs_scratch[0];
    for (int e = 1; e < d.num_experts; ++e) mx = std::max(mx, probs_scratch[e]);
    float sum = 0.f;
    for (int e = 0; e < d.num_experts; ++e) { probs_scratch[e] = std::exp(probs_scratch[e] - mx); sum += probs_scratch[e]; }
    for (int e = 0; e < d.num_experts; ++e) probs_scratch[e] /= sum;
    // Top-k selection: a plain O(num_experts * experts_per_tok) repeated-max scan, marking each pick
    // consumed by setting its own probability to -1 in probs_scratch (bounded, tiny compute -- this
    // stage's own explicit design decision, docs/MOE.md S4b, in place of a genuinely new differentiable
    // Node-graph selection op; AGENTS.md's own "a top-10-of-8-or-512 selection is tiny, bounded compute"
    // framing). Ties resolve to the lowest index, matching torch.topk's own stable-first-max convention
    // closely enough that a real (non-adversarial) fixture is not expected to exercise the difference.
    float wsum = 0.f;
    for (int k = 0; k < d.experts_per_tok; ++k) {
        int best = -1; float bestv = -1.f;
        for (int e = 0; e < d.num_experts; ++e) {
            if (probs_scratch[e] > bestv) { bestv = probs_scratch[e]; best = e; }
        }
        out_idx[k] = best;
        out_weight[k] = bestv;
        wsum += bestv;
        probs_scratch[best] = -1.f;
    }
    if (norm_topk_prob) {
        const float inv = 1.f / wsum;
        for (int k = 0; k < d.experts_per_tok; ++k) out_weight[k] *= inv;
    }
}

// The full block for one row (Qwen4ExpTextSparseMoeBlock.forward, docs/MOE.md S1): route, run only the
// experts_per_tok SELECTED experts' SwiGLU FFNs, weighted-sum them, then add the ALWAYS-ON shared
// expert's own SwiGLU FFN output scaled by sigmoid(shared_gate_proj(x)) -- NOT re-weighted by the
// router's own topk output, a real, verified divergence from a naive "shared expert is just expert #0"
// reading (docs/MOE.md S1a). `expert_gate_w`/`expert_up_w`/`expert_down_w`: arrays of `num_experts`
// pointers, one per routed expert's own weight tensors. `scratch`: >= scratch_floats(d) floats.
inline void forward_row(const Dims& d, const float* x, const float* router_w,
                         const float* const* expert_gate_w, const float* const* expert_up_w,
                         const float* const* expert_down_w,
                         const float* shared_gate_w, const float* shared_up_w, const float* shared_down_w,
                         const float* shared_gate_proj_w,
                         float* out, float* scratch, bool norm_topk_prob = true) {
    float* probs      = scratch;                    // [num_experts]
    float* ffn_scratch = probs + d.num_experts;      // [d_ff]
    float* expert_out = ffn_scratch + d.d_ff;        // [hidden_size]
    float topk_w[TOPK_MAX];
    int   topk_idx[TOPK_MAX];
    router_topk_row(d, x, router_w, probs, topk_w, topk_idx, norm_topk_prob);

    for (int j = 0; j < d.hidden_size; ++j) out[j] = 0.f;
    for (int k = 0; k < d.experts_per_tok; ++k) {
        const int e = topk_idx[k];
        expert_ffn_row(d, x, expert_gate_w[e], expert_up_w[e], expert_down_w[e], expert_out, ffn_scratch);
        for (int j = 0; j < d.hidden_size; ++j) out[j] += topk_w[k] * expert_out[j];
    }

    // Shared expert: an ordinary (non-routed) SwiGLU FFN with its OWN weights, gated by
    // sigmoid(Linear(hidden_size, 1, bias=False)(x)) -- Qwen4ExpTextSparseMoeBlock.forward's own
    // `F.sigmoid(self.shared_expert_gate(hidden_states_reshaped)) * shared_expert_output`.
    expert_ffn_row(d, x, shared_gate_w, shared_up_w, shared_down_w, expert_out, ffn_scratch);
    float gate_logit = 0.f;
    for (int i = 0; i < d.hidden_size; ++i) gate_logit += x[i] * shared_gate_proj_w[i];  // [hidden_size,1]
    const float sg = detail::sigmoid(gate_logit);
    for (int j = 0; j < d.hidden_size; ++j) out[j] += sg * expert_out[j];
}

// Batched T-row wrapper. Rows are independent (this file's own header comment) so `scratch` is reused
// across the loop, not sized per-row -- one scratch_floats(d)-sized buffer serves any T.
inline void forward(const Dims& d, int T, const float* x, const float* router_w,
                     const float* const* expert_gate_w, const float* const* expert_up_w,
                     const float* const* expert_down_w,
                     const float* shared_gate_w, const float* shared_up_w, const float* shared_down_w,
                     const float* shared_gate_proj_w,
                     float* out, float* scratch, bool norm_topk_prob = true) {
    for (int t = 0; t < T; ++t) {
        forward_row(d, x + static_cast<std::size_t>(t) * d.hidden_size, router_w,
                    expert_gate_w, expert_up_w, expert_down_w,
                    shared_gate_w, shared_up_w, shared_down_w, shared_gate_proj_w,
                    out + static_cast<std::size_t>(t) * d.hidden_size, scratch, norm_topk_prob);
    }
}

}  // namespace sub0::moe
