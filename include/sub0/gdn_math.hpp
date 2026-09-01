// sub0/gdn_math.hpp -- Gated DeltaNet Stage 1: the shared, engine-free forward math core.
//
// Every equation here is re-derived from transformers==5.16.1's REAL, installed
// `transformers.models.qwen4_exp.modeling_qwen4_exp` source (AGENTS.md S5 -- fetched via
// `python3 -c "import inspect, transformers.models.qwen4_exp.modeling_qwen4_exp as m; ..."`, not
// recalled), specifically:
//   Qwen4ExpTextGatedDeltaNet.forward, torch_recurrent_gated_delta_rule, torch_chunk_gated_delta_rule,
//   causal_conv1d_fn / causal_conv1d_update, Qwen4ExpTextRMSNormGated, l2norm.
// See docs/GATED_DELTANET.md S5 for why the SEQUENTIAL form (torch_recurrent_gated_delta_rule's exact
// unrolling) is Stage 1's target rather than the chunked form, and this file's own test
// (tests/gdn_qwen4_fixture_tests.cpp) for the numeric proof that it reproduces the chunked reference
// path's real output bit-for-bit (to float32 rounding).
//
// ONE REAL CORRECTION vs docs/GATED_DELTANET.md S1b, found only by re-fetching the actual source
// rather than trusting the doc's already-quoted snippet: both `torch_recurrent_gated_delta_rule` and
// `torch_chunk_gated_delta_rule` scale QUERY by `1/sqrt(head_k_dim)` (`scale = 1 / (query.shape[-1] **
// 0.5); query = query * scale`) immediately after L2-normalizing it and BEFORE the quoted recurrence
// loop even starts. The design doc's S1b snippet begins reading right after that line, so its own
// claim ("No explicit 1/sqrt(d_k) scale appears anywhere in this recurrence") is true of the lines it
// quotes but misses that the scale was already applied one line earlier in the same function. Omitting
// this scale reproduces a plausible-looking but numerically WRONG output (confirmed by first getting a
// large mismatch against the fixture without it, then an exact match after adding it) -- exactly the
// AGENTS.md S5 hazard this file's own header comment exists to flag for the next reader.
//
// Weight-layout convention: every 2D weight here uses THIS PROJECT'S OWN [rows=in, cols=out] layout
// (row p contiguous over output, `w[p*out+o]`) -- the transpose of the PyTorch nn.Linear
// `[out_features, in_features]` convention the fixture's raw weight files were extracted in (see
// tests/fixtures/qwen4_preview/gdn_layer0_small_manifest.json). Re-derived explicitly rather than
// assumed, per AGENTS.md S5 -- this is the SAME axis-order flip layout.hpp's own Wq/Wk/Wv comment
// already documents for softmax attention. Callers sourcing weights from a PyTorch-convention file
// (the fixture test) must transpose before calling; callers using this project's own PARAM_LAYOUT
// (op_gdn in backend_cpu.cpp) already store weights in this convention and pass them straight through.
// The depthwise conv weight and the two head-indexed vectors (dt_bias, A_log, RMSNormGated's gamma)
// have no in/out axis at all, so no such ambiguity applies to them.
//
// No heap allocation (AGENTS.md S1): every buffer this needs is caller-supplied, sized once by
// scratch_floats() below and sliced internally. `state` (the recurrent state) and `conv_hist` (the
// causal conv's short ring buffer) are READ AND WRITTEN IN PLACE so a caller can thread them across
// calls (forward_one's decode persistence) or pass a call-local, zeroed scratch buffer (forward()'s
// training-scratch convention, docs/GATED_DELTANET.md S2) -- this file has no opinion on which.

#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>

namespace sub0::gdn {

// Every dimension the recurrence needs, explicit rather than closed over a build's own constants --
// same reasoning as layout.hpp's depth_schedule_for/gdn_schedule_for: lets a standalone test (this
// file's own fixture test, whose shape does NOT satisfy this project's own D_MODEL == N_HEADS*D_HEAD
// invariant -- see docs/GATED_DELTANET.md S3b's flagged mismatch) exercise the exact real-model shape,
// not just whatever shape the engine happens to be compiled for.
struct Dims {
    int hidden_size;   // input width (and output width -- out_proj restores it)
    int num_k_heads;   // Q/K head count (this project's N_KV_HEADS in the engine-conforming case)
    int num_v_heads;   // V/Z/gate head count (this project's N_HEADS); must be a multiple of num_k_heads
    int head_k_dim;    // Q/K per-head width (this project's D_HEAD)
    int head_v_dim;    // V/Z per-head width (this project's D_HEAD; == head_k_dim in the real model)
    int conv_kernel;   // depthwise causal conv kernel size (real model: 4)

    constexpr int key_dim()   const { return head_k_dim * num_k_heads; }
    constexpr int value_dim() const { return head_v_dim * num_v_heads; }
    constexpr int conv_dim()  const { return 2 * key_dim() + value_dim(); }
    constexpr int rep()       const { return num_v_heads / num_k_heads; }   // repeat_interleave factor
};

// Floats of caller-provided scratch forward() needs for T positions (the T-scaling intermediates:
// pre-conv/post-conv QKV, the output gate projection, and the two per-(t,head) gate scalars). Does
// NOT include `state` or `conv_hist`, whose sizes (state_floats/conv_hist_floats below) are constant
// in T and so are the caller's own arena/persistent-slot allocation, not part of this per-call scratch.
inline constexpr std::size_t scratch_floats(const Dims& d, int T) {
    const std::size_t T_ = static_cast<std::size_t>(T);
    return T_ * static_cast<std::size_t>(d.conv_dim()) * 2      // qkv_pre, qkv_post
         + T_ * static_cast<std::size_t>(d.value_dim())         // z (output-gate projection)
         + T_ * static_cast<std::size_t>(d.num_v_heads) * 2;    // beta, g
}
inline constexpr std::size_t state_floats(const Dims& d) {
    return static_cast<std::size_t>(d.num_v_heads) * static_cast<std::size_t>(d.head_k_dim)
         * static_cast<std::size_t>(d.head_v_dim);
}
inline constexpr std::size_t conv_hist_floats(const Dims& d) {
    return static_cast<std::size_t>(d.conv_dim()) * static_cast<std::size_t>(d.conv_kernel - 1);
}

namespace detail {
inline float silu(float x) { return x / (1.f + std::exp(-x)); }
inline float sigmoid(float x) { return 1.f / (1.f + std::exp(-x)); }
inline float softplus(float x) { return std::log1p(std::exp(-std::fabs(x))) + std::max(x, 0.f); }
}  // namespace detail

// One (t, v-head) step of the sequential delta-rule recurrence -- S1b's exact formula, quoted/verified
// against the real `torch_recurrent_gated_delta_rule` (this file's header comment): `S = g_t*S; kv_mem
// = S^T k_t; delta = beta_t*(v_t - kv_mem); S += k_t (x) delta; o_t = S^T q_t`. Factored out of
// forward() as its own tiny, independently-testable primitive (AGENTS.md S6: new forward math needs a
// numerical property check beyond "it compiles") -- see tests/gdn_qwen4_fixture_tests.cpp's
// degenerate-to-cumulative-sum and zero-beta-is-a-no-op property checks, which call this directly with
// hand-picked g_t/beta_t/k_t/q_t/v_t rather than routing through the full weight pipeline.
//
// `S` ([dk,dv], row-major) is READ AND WRITTEN IN PLACE. `k_t`/`q_t` ([dk]) and `v_t` ([dv]) are
// read-only. `out_o` ([dv]) is written (not accumulated). No heap allocation; `dk`/`dv` bounded by the
// same generous 256 this file uses elsewhere (real model: 128). `out_o` MAY safely alias `v_t`: every
// read of `v_t` (into the local `delta` buffer) happens before `out_o`'s first write.
inline void recurrence_step(int dk, int dv, float g_t, float beta_t,
                             const float* k_t, const float* q_t, const float* v_t,
                             float* S, float* out_o) {
    for (int i = 0; i < dk * dv; ++i) S[i] *= g_t;
    float kv_mem[256];
    for (int j = 0; j < dv; ++j) kv_mem[j] = 0.f;
    for (int i = 0; i < dk; ++i) {
        const float ki = k_t[i];
        const float* Sr = S + static_cast<std::size_t>(i) * dv;
        for (int j = 0; j < dv; ++j) kv_mem[j] += ki * Sr[j];
    }
    float delta[256];
    for (int j = 0; j < dv; ++j) delta[j] = (v_t[j] - kv_mem[j]) * beta_t;
    for (int i = 0; i < dk; ++i) {
        const float ki = k_t[i];
        float* Sr = S + static_cast<std::size_t>(i) * dv;
        for (int j = 0; j < dv; ++j) Sr[j] += ki * delta[j];
    }
    for (int j = 0; j < dv; ++j) out_o[j] = 0.f;
    for (int i = 0; i < dk; ++i) {
        const float qi = q_t[i];
        const float* Sr = S + static_cast<std::size_t>(i) * dv;
        for (int j = 0; j < dv; ++j) out_o[j] += qi * Sr[j];
    }
}

// The real forward, per S1b/S1c/the class's own forward() (quoted/re-derived in this file's header
// comment): in_proj_qkv -> causal depthwise conv1d + SiLU -> split Q/K/V -> beta=sigmoid(b),
// g=-exp(A_log)*softplus(a+dt_bias) -> repeat_interleave Q/K to num_v_heads -> L2-norm Q/K ->
// Q *= 1/sqrt(head_k_dim) -> sequential delta-rule recurrence per (t, v-head) -> RMSNormGated(gamma,
// sigmoid(z)) -> out_proj. `x`: [T,hidden_size] in, row-major. `out`: [T,hidden_size], row-major.
// `state`/`conv_hist` in/out, see the file header. `scratch`: >= scratch_floats(d,T) floats, fully
// consumed here, not retained.
//
// The output GATE activation is hardcoded SIGMOID here, not a generic `hidden_act` fallback: the real
// Qwen4-preview config sets `output_gate_type="sigmoid"` explicitly (manifest's config_small, and
// `activation=config.output_gate_type or config.hidden_act` in the real __init__ -- a truthy
// `output_gate_type` always wins), so this is the one real, verified value rather than a speculative
// knob (AGENTS.md S8: don't add surface area nothing has asked for yet). The causal conv's own
// activation is separately, correctly SiLU (`config.hidden_act`) -- the two are genuinely different
// activations in the real model, not the same choice reused twice.
inline void forward(const Dims& d, int T,
                     const float* x,
                     const float* w_qkv, const float* w_z, const float* w_b, const float* w_a,
                     const float* conv_w, const float* dt_bias, const float* a_log, const float* norm_w,
                     const float* w_out,
                     float* state, float* conv_hist,
                     float* out,
                     float* scratch) {
    const int hs = d.hidden_size;
    const int key_dim = d.key_dim(), value_dim = d.value_dim(), conv_dim = d.conv_dim();
    const int K = d.conv_kernel;
    const int Hk = d.num_k_heads, Hv = d.num_v_heads, rep = d.rep();
    const int dk = d.head_k_dim, dv = d.head_v_dim;
    constexpr float RMS_EPS = 1e-6f;   // Qwen4ExpTextRMSNormGated's real eps (rms_norm_eps in the manifest)
    constexpr float L2_EPS  = 1e-6f;   // l2norm's real eps

    float* qkv_pre  = scratch;                                    scratch += static_cast<std::size_t>(T) * conv_dim;
    float* qkv_post = scratch;                                    scratch += static_cast<std::size_t>(T) * conv_dim;
    float* zb       = scratch;                                    scratch += static_cast<std::size_t>(T) * value_dim;
    float* beta     = scratch;                                    scratch += static_cast<std::size_t>(T) * Hv;
    float* gg       = scratch;                                    /* scratch += T*Hv, unused after */

    // in_proj_qkv, in_proj_z, in_proj_b, in_proj_a -- this project's [in,out] weight convention.
    for (int t = 0; t < T; ++t) {
        const float* xt = x + static_cast<std::size_t>(t) * hs;
        float* qkvr = qkv_pre + static_cast<std::size_t>(t) * conv_dim;
        std::fill(qkvr, qkvr + conv_dim, 0.f);
        float* zr = zb + static_cast<std::size_t>(t) * value_dim;
        std::fill(zr, zr + value_dim, 0.f);
        for (int i = 0; i < hs; ++i) {
            const float xi = xt[i];
            const float* Wq = w_qkv + static_cast<std::size_t>(i) * conv_dim;
            for (int o = 0; o < conv_dim; ++o) qkvr[o] += xi * Wq[o];
            const float* Wz = w_z + static_cast<std::size_t>(i) * value_dim;
            for (int o = 0; o < value_dim; ++o) zr[o] += xi * Wz[o];
        }
        for (int hh = 0; hh < Hv; ++hh) {
            float bs = 0.f, as_ = 0.f;
            for (int i = 0; i < hs; ++i) {
                bs  += xt[i] * w_b[static_cast<std::size_t>(i) * Hv + hh];
                as_ += xt[i] * w_a[static_cast<std::size_t>(i) * Hv + hh];
            }
            beta[static_cast<std::size_t>(t) * Hv + hh] = detail::sigmoid(bs);
            gg[static_cast<std::size_t>(t) * Hv + hh] =
                -std::exp(a_log[hh]) * detail::softplus(as_ + dt_bias[hh]);
        }
    }

    // Causal depthwise conv1d + SiLU. Virtual sequence = concat(conv_hist[0..K-2], qkv_pre[0..T-1]);
    // out[t] = sum_k w[k] * virtual[t+k] -- exactly causal_conv1d_fn's F.conv1d(padding=K-1)[:, :, :T]
    // when conv_hist is all-zero (a fresh training-scratch call), and exactly causal_conv1d_update's
    // cat([conv_state, hidden_states]) when conv_hist carries real decode history.
    for (int c = 0; c < conv_dim; ++c) {
        for (int t = 0; t < T; ++t) {
            float s = 0.f;
            for (int k = 0; k < K; ++k) {
                const int v = t + k;
                const float val = (v < K - 1) ? conv_hist[static_cast<std::size_t>(v) * conv_dim + c]
                                               : qkv_pre[static_cast<std::size_t>(v - (K - 1)) * conv_dim + c];
                s += conv_w[static_cast<std::size_t>(c) * K + k] * val;
            }
            qkv_post[static_cast<std::size_t>(t) * conv_dim + c] = detail::silu(s);
        }
    }
    // Roll the history forward: new_hist[j] = virtual[T+j] for j in [0,K-2]. Buffer first (tmp) since
    // this overwrites conv_hist in place and some new_hist entries alias OLD conv_hist entries.
    {
        constexpr int MAX_K = 32;   // generous bound on conv_kernel; GDN_CONV_KERNEL is 4 in this project
        float tmp[MAX_K - 1];
        for (int c = 0; c < conv_dim; ++c) {
            for (int j = 0; j < K - 1; ++j) {
                const int v = T + j;
                tmp[j] = (v < K - 1) ? conv_hist[static_cast<std::size_t>(v) * conv_dim + c]
                                     : qkv_pre[static_cast<std::size_t>(v - (K - 1)) * conv_dim + c];
            }
            for (int j = 0; j < K - 1; ++j) conv_hist[static_cast<std::size_t>(j) * conv_dim + c] = tmp[j];
        }
    }

    // Split into Q/K/V column ranges of qkv_post: Q [0,key_dim), K [key_dim,2*key_dim), V [2*key_dim,conv_dim).
    const int q_off = 0, k_off = key_dim, v_off = 2 * key_dim;

    // Sequential delta-rule recurrence, per S1b's exact form (verified against the real installed
    // reference, see this file's header comment for the one real correction found doing so).
    for (int t = 0; t < T; ++t) {
        const float* row = qkv_post + static_cast<std::size_t>(t) * conv_dim;
        float* orow = out + static_cast<std::size_t>(t) * hs;   // written at the very end via out_proj instead;
        (void)orow;
        for (int hk = 0; hk < Hk; ++hk) {
            // L2-normalize this physical (t, k-head)'s Q and K once; every one of its `rep` virtual
            // v-heads reads the SAME normalized vector (repeat_interleave duplicates the value, so
            // normalizing before or after duplication is identical -- see this project's own note).
            float qn[256], kn[256];   // generous bound on head_k_dim (real model: 128)
            const float* qraw = row + q_off + hk * dk;
            const float* kraw = row + k_off + hk * dk;
            float qss = 0.f, kss = 0.f;
            for (int a = 0; a < dk; ++a) { qss += qraw[a] * qraw[a]; kss += kraw[a] * kraw[a]; }
            const float qinv = 1.f / std::sqrt(qss + L2_EPS);
            const float kinv = 1.f / std::sqrt(kss + L2_EPS);
            const float qscale = qinv / std::sqrt(static_cast<float>(dk));   // L2-norm THEN 1/sqrt(dk) -- see header note
            for (int a = 0; a < dk; ++a) { qn[a] = qraw[a] * qscale; kn[a] = kraw[a] * kinv; }

            for (int r = 0; r < rep; ++r) {
                const int hv = hk * rep + r;
                const float* vraw = row + v_off + hv * dv;
                const float g_t = std::exp(gg[static_cast<std::size_t>(t) * Hv + hv]);
                const float beta_t = beta[static_cast<std::size_t>(t) * Hv + hv];
                float* S = state + static_cast<std::size_t>(hv) * dk * dv;
                // o_t[hv] = S^T q, written straight back into qkv_post's own V slot (no longer needed
                // raw -- vraw and out_o safely alias, see recurrence_step's own aliasing note) --
                // reused as the pre-gated-norm "core_attn_out" scratch for this head.
                float* core = qkv_post + static_cast<std::size_t>(t) * conv_dim + v_off + hv * dv;
                recurrence_step(dk, dv, g_t, beta_t, kn, qn, vraw, S, core);
            }
        }
    }

    // RMSNormGated, per head_v_dim group (norm_w is [head_v_dim], shared across heads -- S1c), gated
    // by sigmoid(z) (this real config's output_gate_type), then out_proj back to hidden_size.
    for (int t = 0; t < T; ++t) {
        float gated[/*value_dim, generous*/ 4096];
        const float* core_row = qkv_post + static_cast<std::size_t>(t) * conv_dim + v_off;
        const float* z_row = zb + static_cast<std::size_t>(t) * value_dim;
        for (int hh = 0; hh < Hv; ++hh) {
            const float* cv = core_row + hh * dv;
            const float* zv = z_row + hh * dv;
            float ms = 0.f;
            for (int j = 0; j < dv; ++j) ms += cv[j] * cv[j];
            ms /= dv;
            const float rinv = 1.f / std::sqrt(ms + RMS_EPS);
            for (int j = 0; j < dv; ++j)
                gated[hh * dv + j] = norm_w[j] * (cv[j] * rinv) * detail::sigmoid(zv[j]);
        }
        float* ot = out + static_cast<std::size_t>(t) * hs;
        std::fill(ot, ot + hs, 0.f);
        for (int i = 0; i < value_dim; ++i) {
            const float gi = gated[i];
            const float* Wo = w_out + static_cast<std::size_t>(i) * hs;
            for (int o = 0; o < hs; ++o) ot[o] += gi * Wo[o];
        }
    }
}

}  // namespace sub0::gdn
