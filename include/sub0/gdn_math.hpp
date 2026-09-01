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
#include <utility>

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

// =============================================================================================
// Stage 2 -- CPU backward. RECOMPUTE-based (docs/GATED_DELTANET.md S4's closing paragraph, and its
// own precedent -- docs/DEPTH_ATTENTION.md S5b's "backward_device already recomputes qkv per
// execution"): forward() above never retains the state trajectory S_0..S_{T-1} (each S_t is
// overwritten in place by the next step, and V's raw pre-recurrence value is overwritten by o_t) --
// backward() below reruns the SAME forward math from the retained inputs (x + every weight tensor,
// both already retained by the engine's own Node/arena machinery) into `scratch`, this time keeping
// everything the backward sweep needs, rather than retaining an equivalent amount of state from
// forward itself. This costs the same ORDER of memory forward's in-place trick avoided (S4), but only
// as a single, reused, backward-time buffer -- see backend_cpu.cpp's GdnBwdScratch for why that buffer
// is a lazily-sized thread_local std::vector, not a raw thread_local array (Worker's own comment on
// why multi-MB thread_local statics are unsafe on Windows applies here too).
//
// Every equation below was hand-derived from forward()'s exact recurrence (S3b algebra: S_t =
// g_t*(I - beta_t k_t k_t^T) S_{t-1} + beta_t k_t (x) v_t, a per-step affine map in S with transition
// A_t and additive term B_t = k_t (x) delta_t), then verified -- BEFORE this port -- against two
// independent oracles at TWO shapes: (1) the real transformers==5.16.1 Qwen4ExpTextGatedDeltaNet's own
// chunked-path autograd gradients on the real extracted layer-0 weights
// (tests/fixtures/qwen4_preview/gdn_layer0_small_grad_*.bin, loss = dot(out, a fixed random vector) so
// every output element carries a distinct nonzero upstream gradient) -- max relative diff ~3e-7 for
// every one of the 9 parameter tensors plus the input, double-precision NumPy vs the real oracle's
// float32 autograd; and (2) a from-scratch synthetic Hk=2/Hv=4 (multi-group repeat_interleave, unlike
// the real fixture's Hk=1) shape against torch.autograd on an independent from-scratch PyTorch
// reimplementation of this exact math, matching to machine precision (~1e-15 relative). See
// tests/gdn_qwen4_fixture_tests.cpp for the C++-level port of oracle (1).

// Floats of caller-provided scratch backward() needs for T positions -- the recompute-and-retain
// buffer described above. Dominated by the state trajectory (S_t for every t, not just the current
// one, unlike forward()'s O(1)-in-T `state` buffer) -- see this file's header comment on why that is
// the deliberate, bounded cost of the recompute approach, not an oversight.
inline constexpr std::size_t bwd_scratch_floats(const Dims& d, int T) {
    const std::size_t T_ = static_cast<std::size_t>(T);
    const std::size_t Hv_ = static_cast<std::size_t>(d.num_v_heads);
    const std::size_t dk_ = static_cast<std::size_t>(d.head_k_dim), dv_ = static_cast<std::size_t>(d.head_v_dim);
    return T_ * static_cast<std::size_t>(d.conv_dim()) * 3          // qkv_pre, conv_pre_silu, qkv_post
         + T_ * static_cast<std::size_t>(d.num_k_heads) * 2         // qinv, kinv
         + T_ * Hv_ * dk_ * 2                                       // qn, kn (post-repeat_interleave)
         + T_ * Hv_ * 3                                             // beta, g_exp, a_logit
         + T_ * static_cast<std::size_t>(d.value_dim()) * 2         // z_proj, gated_flat
         + T_ * Hv_ * dk_ * dv_                                     // S_traj (S_t, t=0..T-1)
         + T_ * Hv_ * dv_ * 2                                       // delta_traj, core
         + T_ * Hv_                                                 // rinv
         + Hv_ * dk_ * dv_ * 2                                      // G_a, G_b (backward-recurrence ping-pong)
         + dk_ * dv_;                                                // zero_state (shared, all-zero S_{-1})
}

namespace detail {
inline float dsilu(float x) {   // silu(x) = x*sigmoid(x); dsilu/dx = s*(1+x*(1-s)), s=sigmoid(x)
    const float s = sigmoid(x);
    return s * (1.f + x * (1.f - s));
}
}  // namespace detail

// Backward of ONE (t, v-head) step of the sequential delta-rule recurrence -- the adjoint of
// recurrence_step's forward math, factored out the same way (AGENTS.md S6: independently testable).
// `S_prev` = S_{t-1} ([dk,dv], the state BEFORE this step, zero at t=0), `S_cur` = S_t (AFTER this
// step, as produced by recurrence_step at the time it ran), `delta_t` = this step's forward delta
// ([dv], retained from the recompute sweep) -- all READ-ONLY, all recomputed/retained by the caller's
// forward-recompute pass, never by this function. `do_t` ([dv]) is the EXTERNAL gradient on o_t (from
// downstream, e.g. RMSNormGated's backward). `G_next` ([dk,dv]) is dS_{t+1} carried back from the
// FUTURE step (the caller passes an all-zero buffer for the last position, T-1). `G_t` ([dk,dv]) is
// SCRATCH the caller owns: this function fills it with dS_t (needed internally) and then OVERWRITES it
// IN PLACE with g_t * d(S~_t) -- exactly the value the caller should pass as `G_next` for step t-1 (the
// backward-recurrence's own "dS_{t-1} contribution from this step"), so a caller can walk t = T-1..0
// reusing two ping-ponged [dk,dv] buffers with no further bookkeeping.
//
// `dq_t`/`dk_t` ([dk]) and `dv_t` ([dv]) are ACCUMULATED (+=) into by the caller across every v-head
// this (t, physical k-head) group participates in (the repeat_interleave fold, done by the caller after
// this loop, per S1b's Q/K broadcast). `d_beta_t`/`d_g_t` (scalars, +=) are this head's contribution to
// d(loss)/d(beta_t) and d(loss)/d(g_t) -- g_t is the GATE VALUE exp(alpha_t), not alpha_t itself; the
// caller chains d(alpha_t) = d_g_t * g_t afterward (g_t = exp(alpha_t)).
inline void recurrence_step_backward(int dk, int dv, float g_t, float beta_t,
                                      const float* k_t, const float* q_t, const float* v_t,
                                      const float* S_prev, const float* S_cur, const float* delta_t,
                                      const float* do_t, const float* G_next,
                                      float* dq_t, float* dk_t, float* dv_t,
                                      float* d_beta_t, float* d_g_t, float* G_t) {
    // G_t = dS_t = G_next + q_t (x) do_t  (o_t = S_t^T q_t contributes q_t (x) do_t to dS_t directly).
    for (int i = 0; i < dk; ++i) {
        const float qi = q_t[i];
        const float* Gn = G_next + static_cast<std::size_t>(i) * dv;
        float* Gr = G_t + static_cast<std::size_t>(i) * dv;
        for (int j = 0; j < dv; ++j) Gr[j] = Gn[j] + qi * do_t[j];
    }
    // dq_t += S_t @ do_t  (o_t[j] = sum_i q_t[i]*S_t[i,j]  ->  dq_t[i] = sum_j S_t[i,j]*do_t[j]).
    for (int i = 0; i < dk; ++i) {
        const float* Sr = S_cur + static_cast<std::size_t>(i) * dv;
        float s = 0.f;
        for (int j = 0; j < dv; ++j) s += Sr[j] * do_t[j];
        dq_t[i] += s;
    }
    // u = k_t^T G_t == d(delta_t), from S_t = S~_t + k_t (x) delta_t's b-slot (outer-product adjoint).
    float u[256];   // dv bound, matches this file's other generous head-dim bounds (real model: 128)
    for (int j = 0; j < dv; ++j) u[j] = 0.f;
    for (int i = 0; i < dk; ++i) {
        const float ki = k_t[i];
        const float* Gr = G_t + static_cast<std::size_t>(i) * dv;
        for (int j = 0; j < dv; ++j) u[j] += ki * Gr[j];
    }
    // dk_t += G_t @ delta_t, from the SAME outer product's a-slot.
    for (int i = 0; i < dk; ++i) {
        const float* Gr = G_t + static_cast<std::size_t>(i) * dv;
        float s = 0.f;
        for (int j = 0; j < dv; ++j) s += Gr[j] * delta_t[j];
        dk_t[i] += s;
    }
    // kv_mem_t = k_t^T S~_t, S~_t = g_t*S_prev (recomputed here, cheap -- one extra O(dk*dv) pass).
    float kv_mem[256];
    for (int j = 0; j < dv; ++j) kv_mem[j] = 0.f;
    for (int i = 0; i < dk; ++i) {
        const float kgd = k_t[i] * g_t;
        const float* Sr = S_prev + static_cast<std::size_t>(i) * dv;
        for (int j = 0; j < dv; ++j) kv_mem[j] += kgd * Sr[j];
    }
    // delta_t = beta_t*(v_t - kv_mem_t)  ->  d(beta_t) = dot(u, v_t - kv_mem_t);
    // dv_t += beta_t*u ; d(kv_mem_t) = -beta_t*u.
    float db = 0.f;
    for (int j = 0; j < dv; ++j) db += u[j] * (v_t[j] - kv_mem[j]);
    *d_beta_t += db;
    for (int j = 0; j < dv; ++j) dv_t[j] += beta_t * u[j];
    // kv_mem_t = k_t^T S~_t  ->  dk_t += S~_t @ d(kv_mem_t) = -beta_t * (S~_t @ u);
    // d(S~_t) (direct, from S_t=S~_t+B_t) + (via kv_mem_t) = G_t[i,j] + k_t[i]*(-beta_t*u[j]).
    for (int i = 0; i < dk; ++i) {
        const float sd_row_scale = g_t;   // S~_t[i,j] = g_t*S_prev[i,j]
        const float* Sr = S_prev + static_cast<std::size_t>(i) * dv;
        float s = 0.f;
        for (int j = 0; j < dv; ++j) s += (sd_row_scale * Sr[j]) * (-beta_t * u[j]);
        dk_t[i] += s;
    }
    // d(g_t) = sum_ij d(S~_t)[i,j] * S_prev[i,j]  (S~_t = g_t*S_prev); also overwrite G_t IN PLACE with
    // g_t * d(S~_t) -- the value the caller threads to the PREVIOUS step as its `G_next`.
    float dgt = 0.f;
    for (int i = 0; i < dk; ++i) {
        const float ki = k_t[i];
        float* Gr = G_t + static_cast<std::size_t>(i) * dv;
        const float* Sr = S_prev + static_cast<std::size_t>(i) * dv;
        for (int j = 0; j < dv; ++j) {
            const float dSd_ij = Gr[j] + ki * (-beta_t * u[j]);
            dgt += dSd_ij * Sr[j];
            Gr[j] = g_t * dSd_ij;
        }
    }
    *d_g_t += dgt;
}

// The full Stage 2 backward: recompute forward()'s math from x + every weight (retaining everything
// this needs, per this section's header comment), then walk the chain in reverse to produce
// d(loss)/d(x) and d(loss)/d(every one of the 9 weight tensors). Every output pointer is ACCUMULATED
// (+=), matching this engine's own backward_node convention (param grads sum across the whole backward
// walk) -- callers must zero them first (the engine's arena already does, via graph_reset()).
//
// `dOut` ([T,hidden_size]) is the upstream gradient on forward()'s `out`. `scratch` must be
// >= bwd_scratch_floats(d,T) floats. Weight layout: this project's own [rows=in,cols=out] convention
// throughout (same as forward(), see this file's header comment) -- NOT the raw fixture files'
// PyTorch [out,in] convention (transpose first, as gdn_qwen4_fixture_tests.cpp's own transpose()
// helper already does for forward()).
inline void backward(const Dims& d, int T,
                      const float* x,
                      const float* w_qkv, const float* w_z, const float* w_b, const float* w_a,
                      const float* conv_w, const float* dt_bias, const float* a_log, const float* norm_w,
                      const float* w_out,
                      const float* dOut,
                      float* dx,
                      float* dw_qkv, float* dw_z, float* dw_b, float* dw_a,
                      float* dconv_w, float* ddt_bias, float* da_log, float* dnorm_w, float* dw_out,
                      float* scratch) {
    const int hs = d.hidden_size;
    const int key_dim = d.key_dim(), value_dim = d.value_dim(), conv_dim = d.conv_dim();
    const int K = d.conv_kernel;
    const int Hk = d.num_k_heads, Hv = d.num_v_heads, rep = d.rep();
    const int dk = d.head_k_dim, dv = d.head_v_dim;
    constexpr float RMS_EPS = 1e-6f;
    constexpr float L2_EPS  = 1e-6f;
    const std::size_t T_ = static_cast<std::size_t>(T);

    // --- Partition `scratch` (mirrors forward()'s own partitioning style) --------------------------
    float* qkv_pre       = scratch; scratch += T_ * conv_dim;
    float* conv_pre_silu = scratch; scratch += T_ * conv_dim;
    float* qkv_post      = scratch; scratch += T_ * conv_dim;
    float* qinv_buf      = scratch; scratch += T_ * Hk;
    float* kinv_buf      = scratch; scratch += T_ * Hk;
    float* qn            = scratch; scratch += T_ * Hv * dk;
    float* kn            = scratch; scratch += T_ * Hv * dk;
    float* beta          = scratch; scratch += T_ * Hv;
    float* g_exp         = scratch; scratch += T_ * Hv;
    float* a_logit       = scratch; scratch += T_ * Hv;
    float* z_proj        = scratch; scratch += T_ * value_dim;
    float* gated_flat    = scratch; scratch += T_ * value_dim;
    float* S_traj        = scratch; scratch += T_ * Hv * dk * dv;
    float* delta_traj    = scratch; scratch += T_ * Hv * dv;
    float* core          = scratch; scratch += T_ * Hv * dv;          // later overwritten IN PLACE: d(core)
    float* rinv_buf      = scratch; scratch += T_ * Hv;
    float* G_a           = scratch; scratch += static_cast<std::size_t>(Hv) * dk * dv;
    float* G_b           = scratch; scratch += static_cast<std::size_t>(Hv) * dk * dv;
    float* zero_state    = scratch; /* scratch += dk*dv, unused after */

    // --- Recompute forward, retaining everything (S4's recompute-during-backward direction) --------
    for (int t = 0; t < T; ++t) {
        const float* xt = x + static_cast<std::size_t>(t) * hs;
        float* qkvr = qkv_pre + static_cast<std::size_t>(t) * conv_dim;
        std::fill(qkvr, qkvr + conv_dim, 0.f);
        float* zr = z_proj + static_cast<std::size_t>(t) * value_dim;
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
            a_logit[static_cast<std::size_t>(t) * Hv + hh] = as_;
            g_exp[static_cast<std::size_t>(t) * Hv + hh] =
                std::exp(-std::exp(a_log[hh]) * detail::softplus(as_ + dt_bias[hh]));
        }
    }
    for (int c = 0; c < conv_dim; ++c) {
        for (int t = 0; t < T; ++t) {
            float s = 0.f;
            for (int k = 0; k < K; ++k) {
                const int v = t + k;
                const float val = (v < K - 1) ? 0.f   // training path: always-zero initial conv history
                                               : qkv_pre[static_cast<std::size_t>(v - (K - 1)) * conv_dim + c];
                s += conv_w[static_cast<std::size_t>(c) * K + k] * val;
            }
            conv_pre_silu[static_cast<std::size_t>(t) * conv_dim + c] = s;
            qkv_post[static_cast<std::size_t>(t) * conv_dim + c] = detail::silu(s);
        }
    }
    const int q_off = 0, k_off = key_dim, v_off = 2 * key_dim;
    for (int t = 0; t < T; ++t) {
        for (int hk = 0; hk < Hk; ++hk) {
            const float* qraw = qkv_post + static_cast<std::size_t>(t) * conv_dim + q_off + hk * dk;
            const float* kraw = qkv_post + static_cast<std::size_t>(t) * conv_dim + k_off + hk * dk;
            float qss = 0.f, kss = 0.f;
            for (int a = 0; a < dk; ++a) { qss += qraw[a] * qraw[a]; kss += kraw[a] * kraw[a]; }
            const float qinv = 1.f / std::sqrt(qss + L2_EPS);
            const float kinv = 1.f / std::sqrt(kss + L2_EPS);
            qinv_buf[static_cast<std::size_t>(t) * Hk + hk] = qinv;
            kinv_buf[static_cast<std::size_t>(t) * Hk + hk] = kinv;
            const float qscale = qinv / std::sqrt(static_cast<float>(dk));
            for (int r = 0; r < rep; ++r) {
                const int hv = hk * rep + r;
                float* qnr = qn + (static_cast<std::size_t>(t) * Hv + hv) * dk;
                float* knr = kn + (static_cast<std::size_t>(t) * Hv + hv) * dk;
                for (int a = 0; a < dk; ++a) { qnr[a] = qraw[a] * qscale; knr[a] = kraw[a] * kinv; }
            }
        }
    }
    // Recurrence forward, retaining the FULL trajectory S_traj[t] = S_t for every t (unlike forward()'s
    // O(1)-in-T `state` buffer, which overwrites in place) -- S_{t-1} is read directly as S_traj[t-1]
    // (an all-zero virtual S_traj[-1] at t==0, per the training path's fresh-window invariant), so no
    // separate running accumulator is needed and nothing here heap-allocates (AGENTS.md S1).
    for (int hv = 0; hv < Hv; ++hv) {
        for (int t = 0; t < T; ++t) {
            const float g_t = g_exp[static_cast<std::size_t>(t) * Hv + hv];
            const float beta_t = beta[static_cast<std::size_t>(t) * Hv + hv];
            const float* kt = kn + (static_cast<std::size_t>(t) * Hv + hv) * dk;
            const float* qt = qn + (static_cast<std::size_t>(t) * Hv + hv) * dk;
            const float* vt = qkv_post + static_cast<std::size_t>(t) * conv_dim + v_off + hv * dv;
            const float* Sprev = (t == 0) ? nullptr
                                           : S_traj + (static_cast<std::size_t>(t - 1) * Hv + hv) * dk * dv;
            float* Sdst = S_traj + (static_cast<std::size_t>(t) * Hv + hv) * dk * dv;
            float* deltad = delta_traj + (static_cast<std::size_t>(t) * Hv + hv) * dv;
            float* cored = core + (static_cast<std::size_t>(t) * Hv + hv) * dv;
            // S~_t = g_t*S_{t-1}; kv_mem = S~_t^T k_t; delta = beta_t*(v_t-kv_mem); S_t = S~_t + k(x)delta.
            for (int i = 0; i < dk; ++i) {
                float* Sr = Sdst + static_cast<std::size_t>(i) * dv;
                if (Sprev) { const float* Pr = Sprev + static_cast<std::size_t>(i) * dv;
                             for (int j = 0; j < dv; ++j) Sr[j] = g_t * Pr[j]; }
                else       { for (int j = 0; j < dv; ++j) Sr[j] = 0.f; }
            }
            float kv_mem[256];
            for (int j = 0; j < dv; ++j) kv_mem[j] = 0.f;
            for (int i = 0; i < dk; ++i) {
                const float ki = kt[i];
                const float* Sr = Sdst + static_cast<std::size_t>(i) * dv;   // == S~_t so far
                for (int j = 0; j < dv; ++j) kv_mem[j] += ki * Sr[j];
            }
            for (int j = 0; j < dv; ++j) deltad[j] = beta_t * (vt[j] - kv_mem[j]);
            for (int i = 0; i < dk; ++i) {
                const float ki = kt[i];
                float* Sr = Sdst + static_cast<std::size_t>(i) * dv;
                for (int j = 0; j < dv; ++j) Sr[j] += ki * deltad[j];   // S~_t -> S_t, in place
            }
            for (int j = 0; j < dv; ++j) cored[j] = 0.f;
            for (int i = 0; i < dk; ++i) {
                const float qi = qt[i];
                const float* Sr = Sdst + static_cast<std::size_t>(i) * dv;
                for (int j = 0; j < dv; ++j) cored[j] += qi * Sr[j];
            }
        }
    }
    for (int t = 0; t < T; ++t) {
        const float* core_row = core + static_cast<std::size_t>(t) * Hv * dv;
        const float* z_row = z_proj + static_cast<std::size_t>(t) * value_dim;
        float* gated_row = gated_flat + static_cast<std::size_t>(t) * value_dim;
        for (int hh = 0; hh < Hv; ++hh) {
            const float* cv = core_row + hh * dv;
            const float* zv = z_row + hh * dv;
            float ms = 0.f;
            for (int j = 0; j < dv; ++j) ms += cv[j] * cv[j];
            ms /= dv;
            const float rinv = 1.f / std::sqrt(ms + RMS_EPS);
            rinv_buf[static_cast<std::size_t>(t) * Hv + hh] = rinv;
            for (int j = 0; j < dv; ++j)
                gated_row[hh * dv + j] = norm_w[j] * (cv[j] * rinv) * detail::sigmoid(zv[j]);
        }
    }

    // --- Backward: out_proj, then RMSNormGated, per position --------------------------------------
    // out = gated_flat @ w_out ([in=value_dim,out=hs], same convention as forward()'s own out_proj loop
    // and backend_cpu.cpp's Op::Linear backward): d(gated_flat) = dOut @ w_out^T, d(w_out) accumulates
    // gated_flat^T @ dOut. RMSNormGated's backward then overwrites `core` and `z_proj` IN PLACE with
    // d(core) / d(z_logit) respectively -- safe because every read of a given (t,head)'s core/z values
    // happens-before that same (t,head)'s first overwrite (see the two-pass structure below).
    float d_gated_row[/*value_dim, generous*/ 4096];
    float gate_cache[256], d_normed_cache[256];   // dv bound, matches this file's other head-dim bounds
    for (int t = 0; t < T; ++t) {
        const float* dYr = dOut + static_cast<std::size_t>(t) * hs;
        for (int i = 0; i < value_dim; ++i) {
            const float* Wr = w_out + static_cast<std::size_t>(i) * hs;
            float s = 0.f;
            for (int o = 0; o < hs; ++o) s += dYr[o] * Wr[o];
            d_gated_row[i] = s;
        }
        const float* Xr = gated_flat + static_cast<std::size_t>(t) * value_dim;
        for (int i = 0; i < value_dim; ++i) {
            float* Wg = dw_out + static_cast<std::size_t>(i) * hs;
            const float xip = Xr[i];
            for (int o = 0; o < hs; ++o) Wg[o] += xip * dYr[o];
        }

        float* core_row = core + static_cast<std::size_t>(t) * Hv * dv;      // -> becomes d(core)
        float* z_row = z_proj + static_cast<std::size_t>(t) * value_dim;     // -> becomes d(z_logit)
        for (int hh = 0; hh < Hv; ++hh) {
            float* cv = core_row + hh * dv;
            float* zv = z_row + hh * dv;
            const float rinv = rinv_buf[static_cast<std::size_t>(t) * Hv + hh];
            const float r3_over_dv = (rinv * rinv * rinv) / static_cast<float>(dv);
            float Ssum = 0.f;
            for (int j = 0; j < dv; ++j) {
                const float gate = detail::sigmoid(zv[j]);
                gate_cache[j] = gate;
                const float normed = cv[j] * rinv;
                const float dg = d_gated_row[hh * dv + j];
                const float d_normed = dg * norm_w[j] * gate;
                d_normed_cache[j] = d_normed;
                dnorm_w[j] += dg * normed * gate;
                const float d_gate = dg * norm_w[j] * normed;
                zv[j] = d_gate * gate * (1.f - gate);          // overwrite z_proj IN PLACE: d(z_logit)
                Ssum += d_normed * cv[j];
            }
            for (int j = 0; j < dv; ++j)
                cv[j] = d_normed_cache[j] * rinv - cv[j] * r3_over_dv * Ssum;   // overwrite core IN PLACE
        }
    }

    // --- Backward: the sequential recurrence, walking t = T-1 .. 0 --------------------------------
    // G_cur/G_next: [Hv,dk,dv] ping-pong of dS_t / dS_{t+1} (no future beyond T-1, so G_next starts
    // all-zero). `core` (now holding d(core), i.e. do_t) and `qn`/`kn` (still the FORWARD Q/K, read-only
    // here) feed recurrence_step_backward; `qkv_post`'s Q/K/V columns are overwritten IN PLACE, per
    // head/position, with d(q_raw)/d(k_raw)/d(v_raw) as each is computed -- by the time this whole
    // reverse walk finishes, `qkv_post` holds the FULL d(qkv_post) (SiLU's own backward input) rather
    // than the original post-conv activations, and `beta`/`a_logit` hold d(b_logit)/d(a_logit).
    std::fill(zero_state, zero_state + static_cast<std::size_t>(dk) * dv, 0.f);
    std::fill(G_b, G_b + static_cast<std::size_t>(Hv) * dk * dv, 0.f);   // G_next for t=T-1: no future
    float* G_cur = G_a;
    float* G_next = G_b;
    float d_qn_hk_t[/*Hk*dk, generous*/ 4096], d_kn_hk_t[4096];
    const float inv_sqrt_dk = 1.f / std::sqrt(static_cast<float>(dk));

    for (int t = T - 1; t >= 0; --t) {
        std::fill(d_qn_hk_t, d_qn_hk_t + Hk * dk, 0.f);
        std::fill(d_kn_hk_t, d_kn_hk_t + Hk * dk, 0.f);
        const float* do_row = core + static_cast<std::size_t>(t) * Hv * dv;   // d(core) at this t
        float* qkv_row_w = qkv_post + static_cast<std::size_t>(t) * conv_dim;

        for (int hv = 0; hv < Hv; ++hv) {
            const int hk = hv / rep;
            const float g_t = g_exp[static_cast<std::size_t>(t) * Hv + hv];
            const float beta_t = beta[static_cast<std::size_t>(t) * Hv + hv];
            const float* kt = kn + (static_cast<std::size_t>(t) * Hv + hv) * dk;
            const float* qt = qn + (static_cast<std::size_t>(t) * Hv + hv) * dk;
            const float* vt = qkv_row_w + v_off + hv * dv;
            const float* Sprev = (t == 0) ? zero_state
                                           : S_traj + (static_cast<std::size_t>(t - 1) * Hv + hv) * dk * dv;
            const float* Scur = S_traj + (static_cast<std::size_t>(t) * Hv + hv) * dk * dv;
            const float* deltat = delta_traj + (static_cast<std::size_t>(t) * Hv + hv) * dv;
            const float* do_t = do_row + hv * dv;
            float* Gn = G_next + static_cast<std::size_t>(hv) * dk * dv;
            float* Gc = G_cur + static_cast<std::size_t>(hv) * dk * dv;

            float dq_head[256] = {}, dk_head[256] = {}, dv_head[256] = {};
            float d_beta_t = 0.f, d_g_t = 0.f;
            recurrence_step_backward(dk, dv, g_t, beta_t, kt, qt, vt, Sprev, Scur, deltat, do_t, Gn,
                                      dq_head, dk_head, dv_head, &d_beta_t, &d_g_t, Gc);

            for (int a = 0; a < dk; ++a) { d_qn_hk_t[hk * dk + a] += dq_head[a]; d_kn_hk_t[hk * dk + a] += dk_head[a]; }
            float* vw = qkv_row_w + v_off + hv * dv;
            for (int j = 0; j < dv; ++j) vw[j] = dv_head[j];   // overwrite qkv_post's V slot: d(v_raw)

            const float bt = beta_t;
            beta[static_cast<std::size_t>(t) * Hv + hv] = d_beta_t * bt * (1.f - bt);   // -> d(b_logit)

            const float d_alpha = d_g_t * g_t;                 // g_t = exp(alpha_t)
            const float expA = std::exp(a_log[hv]);
            const float sp_arg = a_logit[static_cast<std::size_t>(t) * Hv + hv] + dt_bias[hv];
            const float d_sp_arg = (d_alpha * -expA) * detail::sigmoid(sp_arg);   // softplus' = sigmoid
            a_logit[static_cast<std::size_t>(t) * Hv + hv] = d_sp_arg;             // -> d(a_logit)
            ddt_bias[hv] += d_sp_arg;
            da_log[hv] += d_alpha * (-expA * detail::softplus(sp_arg));           // d(exp(A_log))/d(A_log)=exp(A_log)
        }
        std::swap(G_cur, G_next);

        // L2-norm(+scale) backward for Q/K at this t, folding the repeat_interleave sum computed above.
        // Overwrites qkv_post's Q/K slots IN PLACE with d(q_raw)/d(k_raw) -- V was already overwritten
        // above and does not alias these column ranges.
        for (int hk = 0; hk < Hk; ++hk) {
            const float* qraw = qkv_row_w + q_off + hk * dk;
            const float* kraw = qkv_row_w + k_off + hk * dk;
            const float qinv = qinv_buf[static_cast<std::size_t>(t) * Hk + hk];
            const float kinv = kinv_buf[static_cast<std::size_t>(t) * Hk + hk];
            const float* dqhk = d_qn_hk_t + hk * dk;
            const float* dkhk = d_kn_hk_t + hk * dk;
            float dotq = 0.f, dotk = 0.f;
            for (int a = 0; a < dk; ++a) { dotq += qraw[a] * dqhk[a]; dotk += kraw[a] * dkhk[a]; }
            float* qw = qkv_row_w + q_off + hk * dk;
            float* kw = qkv_row_w + k_off + hk * dk;
            const float qinv3 = qinv * qinv * qinv, kinv3 = kinv * kinv * kinv;
            for (int a = 0; a < dk; ++a) {
                qw[a] = inv_sqrt_dk * (qinv * dqhk[a] - qraw[a] * qinv3 * dotq);
                kw[a] = kinv * dkhk[a] - kraw[a] * kinv3 * dotk;
            }
        }
    }

    // --- Backward: SiLU, then the causal depthwise conv (a GATHER over output positions, so the
    // original qkv_pre value at each input position is read exactly once and can be overwritten in
    // place with d(qkv_pre) immediately after) ------------------------------------------------------
    for (std::size_t i = 0; i < T_ * static_cast<std::size_t>(conv_dim); ++i)
        qkv_post[i] *= detail::dsilu(conv_pre_silu[i]);   // qkv_post now holds d(conv_pre_silu)

    for (int c = 0; c < conv_dim; ++c) {
        for (int src_t = 0; src_t < T; ++src_t) {
            float acc = 0.f;
            for (int k = 0; k < K; ++k) {
                const int t = src_t + (K - 1) - k;   // v_idx = t+k = src_t+(K-1)  ->  t = src_t+(K-1)-k
                if (t < 0 || t >= T) continue;
                const float dg = qkv_post[static_cast<std::size_t>(t) * conv_dim + c];
                acc += conv_w[static_cast<std::size_t>(c) * K + k] * dg;
                dconv_w[static_cast<std::size_t>(c) * K + k] += qkv_pre[static_cast<std::size_t>(src_t) * conv_dim + c] * dg;
            }
            qkv_pre[static_cast<std::size_t>(src_t) * conv_dim + c] = acc;   // overwrite IN PLACE: d(qkv_pre)
        }
    }

    // --- Backward: the four input projections (in_proj_qkv/z/b/a), Op::Linear's own dX/dW shapes ----
    // dx[t,p] += sum_o dY[t,o]*W[p,o] ; dW[p,o] += sum_t x[t,p]*dY[t,o]. qkv_pre/z_proj/beta/a_logit now
    // hold d(qkv_pre)/d(z_logit)/d(b_logit)/d(a_logit) respectively, per the in-place overwrites above.
    for (int t = 0; t < T; ++t) {
        const float* dqkvr = qkv_pre + static_cast<std::size_t>(t) * conv_dim;
        const float* dzr   = z_proj  + static_cast<std::size_t>(t) * value_dim;
        const float* dbr   = beta    + static_cast<std::size_t>(t) * Hv;
        const float* dar   = a_logit + static_cast<std::size_t>(t) * Hv;
        float* dxr = dx + static_cast<std::size_t>(t) * hs;
        for (int p = 0; p < hs; ++p) {
            float s = 0.f;
            const float* Wq = w_qkv + static_cast<std::size_t>(p) * conv_dim;
            for (int o = 0; o < conv_dim; ++o) s += dqkvr[o] * Wq[o];
            const float* Wz = w_z + static_cast<std::size_t>(p) * value_dim;
            for (int o = 0; o < value_dim; ++o) s += dzr[o] * Wz[o];
            const float* Wb = w_b + static_cast<std::size_t>(p) * Hv;
            for (int o = 0; o < Hv; ++o) s += dbr[o] * Wb[o];
            const float* Wa = w_a + static_cast<std::size_t>(p) * Hv;
            for (int o = 0; o < Hv; ++o) s += dar[o] * Wa[o];
            dxr[p] += s;
        }
    }
    for (int p = 0; p < hs; ++p) {
        float* Wg_qkv = dw_qkv + static_cast<std::size_t>(p) * conv_dim;
        float* Wg_z   = dw_z   + static_cast<std::size_t>(p) * value_dim;
        float* Wg_b   = dw_b   + static_cast<std::size_t>(p) * Hv;
        float* Wg_a   = dw_a   + static_cast<std::size_t>(p) * Hv;
        for (int t = 0; t < T; ++t) {
            const float xtp = x[static_cast<std::size_t>(t) * hs + p];
            const float* dqkvr = qkv_pre + static_cast<std::size_t>(t) * conv_dim;
            for (int o = 0; o < conv_dim; ++o) Wg_qkv[o] += xtp * dqkvr[o];
            const float* dzr = z_proj + static_cast<std::size_t>(t) * value_dim;
            for (int o = 0; o < value_dim; ++o) Wg_z[o] += xtp * dzr[o];
            const float* dbr = beta + static_cast<std::size_t>(t) * Hv;
            for (int o = 0; o < Hv; ++o) Wg_b[o] += xtp * dbr[o];
            const float* dar = a_logit + static_cast<std::size_t>(t) * Hv;
            for (int o = 0; o < Hv; ++o) Wg_a[o] += xtp * dar[o];
        }
    }
}

}  // namespace sub0::gdn
