// sub0/gated_residual_math.hpp -- Gated Residual (hyper-connections) Stage 1: the shared, engine-free
// forward math core, mirroring gdn_math.hpp's role for Gated DeltaNet.
//
// Every equation here is re-derived from transformers==5.16.1's REAL, installed
// `transformers.models.qwen4_exp.modeling_qwen4_exp` source (AGENTS.md S5 -- fetched via
// `python3 -c "import inspect, transformers.models.qwen4_exp.modeling_qwen4_exp as m; ..."`, not
// recalled), specifically `Qwen4ExpTextGatedResidual.__init__`/`.forward` and `Qwen4ExpTextRMSNorm`
// (the grouped-norm primitive GatedResidual's own `hc_norm` uses). See docs/GATED_RESIDUAL.md S1 for
// the full quoted source and S4 for why this file splits the real module's single `forward()` into four
// independently-callable pieces (hc_norm/mix/gate/combine) plus a fifth (tile) for the model-level entry
// step that lives in `Qwen4ExpTextModel.forward` rather than the module itself.
//
// Weight-layout convention: every 2D weight here uses THIS PROJECT'S OWN [rows=in, cols=out] layout
// (row p contiguous over output, `w[p*out+o]`) -- the transpose of the PyTorch nn.Linear
// `[out_features, in_features]` convention the fixture's raw weight files were extracted in (see
// tests/fixtures/qwen4_preview/gated_residual_layer0_small_manifest.json). Re-derived explicitly rather
// than assumed, per AGENTS.md S5 -- the SAME axis-order flip gdn_math.hpp's own header comment already
// documents. Callers sourcing weights from a PyTorch-convention file (the fixture test) must transpose
// before calling; callers using this project's own PARAM_LAYOUT (op_gr_* in backend_cpu.cpp) already
// store weights in this convention and pass them straight through.
//
// No heap allocation (AGENTS.md S1): every buffer this needs is caller-supplied, sized once by the
// *_scratch_floats() helpers below and sliced internally.

#pragma once

#include <cmath>
#include <cstddef>

namespace sub0::gr {

// Every dimension the module needs, explicit rather than closed over a build's own constants -- same
// reasoning as gdn_math.hpp's own Dims (lets a standalone test exercise the real fixture's own shape
// regardless of what this project's compiled build happens to be configured for).
struct Dims {
    int hidden_size;   // D_MODEL -- one stream's width
    int hc_count;      // number of parallel residual streams (real model: 4)
    int hc_lowrank;    // low-rank bottleneck width of the input-mixer gate (real model: 320)

    constexpr int wide() const { return hc_count * hidden_size; }   // total width of the wide stream
};

inline constexpr std::size_t normed_scratch_floats(const Dims& d, int T) {
    return static_cast<std::size_t>(T) * static_cast<std::size_t>(d.wide());
}
// mix()'s own scratch need, on top of the ALREADY-NORMALIZED `normed` buffer it takes as an input (see
// mix()'s own comment) -- just the down-projection's pre-activation, [T, hc_lowrank]. A caller that has
// not yet normalized its wide input separately needs normed_scratch_floats(d,T) MORE, for that step.
inline constexpr std::size_t mix_scratch_floats(const Dims& d, int T) {
    return static_cast<std::size_t>(T) * static_cast<std::size_t>(d.hc_lowrank);
}

namespace detail {
inline float silu(float x) { return x / (1.f + std::exp(-x)); }
inline float sigmoid(float x) { return 1.f / (1.f + std::exp(-x)); }
}  // namespace detail

// Grouped RMSNorm with a ZERO-CENTERED weight convention (docs/GATED_RESIDUAL.md S1a): each of the
// hc_count streams is normalized independently over its OWN hidden_size-wide slice (group_size ==
// hidden_size, the real model's own value), but the gain is `1 + norm_w[c]` -- NOT `norm_w[c]` directly,
// the way this project's existing op_rmsnorm/Ln1/Ln2/QNorm/KNorm all work -- so a freshly-initialized
// (all-zero) norm_w is the identity RMS-norm here, unlike every other norm this engine has. `norm_w` is
// [wide] (one gain per (stream, channel) pair, not shared across streams -- another real divergence from
// this project's existing QNorm/KNorm convention, verified in the real source, not assumed by analogy).
// eps is the real model's own verified value (rms_norm_eps=1e-6 in the fixture manifest), hardcoded
// rather than exposed as a knob -- same precedent as gdn_math.hpp's own RMS_EPS/L2_EPS constants.
//
// `wide_in`: [T, wide], row-major. `norm_w`: [wide]. `out_normed`: [T, wide], written (not accumulated).
inline void hc_norm(const Dims& d, int T, const float* wide_in, const float* norm_w, float* out_normed) {
    constexpr float EPS = 1e-6f;
    const int hs = d.hidden_size, hc = d.hc_count;
    for (int t = 0; t < T; ++t) {
        const float* xr = wide_in + static_cast<std::size_t>(t) * d.wide();
        float* yr = out_normed + static_cast<std::size_t>(t) * d.wide();
        for (int s = 0; s < hc; ++s) {
            const float* xg = xr + static_cast<std::size_t>(s) * hs;
            float* yg = yr + static_cast<std::size_t>(s) * hs;
            float ms = 0.f;
            for (int j = 0; j < hs; ++j) ms += xg[j] * xg[j];
            ms /= hs;
            const float rinv = 1.f / std::sqrt(ms + EPS);
            const float* wg = norm_w + static_cast<std::size_t>(s) * hs;
            for (int j = 0; j < hs; ++j) yg[j] = xg[j] * rinv * (1.f + wg[j]);
        }
    }
}

// The READ step's "mixed_input" half (docs/GATED_RESIDUAL.md S1a/S4b): mixed = mean_over_streams(
// sigmoid(up(silu(down(normed)/hc_count))) * normed ). `normed` is the hc_norm() output for THIS call's
// own weights (docs/GATED_RESIDUAL.md S4c: op_gr_mix and op_gr_gate each call hc_norm() independently on
// their own scratch rather than sharing one precomputed buffer -- a deliberate, documented Stage 1
// simplification, not an oversight; see that section for why the duplicated cost is bounded and cheap).
// down_w: [wide, hc_lowrank], up_w: [hc_lowrank, wide], this project's own [in,out] convention.
// out_mixed: [T, hidden_size]. scratch: >= T*hc_lowrank floats (the down-projection's pre-activation).
inline void mix(const Dims& d, int T, const float* normed, const float* down_w, const float* up_w,
                 float* out_mixed, float* scratch) {
    const int hs = d.hidden_size, hc = d.hc_count, wide = d.wide(), lr = d.hc_lowrank;
    float* down_pre = scratch;   // [T, hc_lowrank]
    for (int t = 0; t < T; ++t) {
        const float* xr = normed + static_cast<std::size_t>(t) * wide;
        float* dr = down_pre + static_cast<std::size_t>(t) * lr;
        for (int o = 0; o < lr; ++o) dr[o] = 0.f;
        for (int i = 0; i < wide; ++i) {
            const float xi = xr[i];
            const float* Wr = down_w + static_cast<std::size_t>(i) * lr;
            for (int o = 0; o < lr; ++o) dr[o] += xi * Wr[o];
        }
        for (int o = 0; o < lr; ++o) dr[o] = detail::silu(dr[o] / static_cast<float>(hc));
    }
    for (int t = 0; t < T; ++t) {
        const float* dr = down_pre + static_cast<std::size_t>(t) * lr;
        const float* xr = normed + static_cast<std::size_t>(t) * wide;
        float* out = out_mixed + static_cast<std::size_t>(t) * hs;
        for (int j = 0; j < hs; ++j) out[j] = 0.f;
        for (int s = 0; s < hc; ++s) {
            for (int j = 0; j < hs; ++j) {
                const int col = s * hs + j;
                float up_val = 0.f;
                for (int i = 0; i < lr; ++i) up_val += dr[i] * up_w[static_cast<std::size_t>(i) * wide + col];
                const float gate_v = detail::sigmoid(up_val);
                out[j] += gate_v * xr[static_cast<std::size_t>(s) * hs + j];
            }
        }
        const float inv_hc = 1.f / static_cast<float>(hc);
        for (int j = 0; j < hs; ++j) out[j] *= inv_hc;
    }
}

// The READ step's "injection_weights" half (docs/GATED_RESIDUAL.md S1a/S1b): inj = 2*sigmoid(
// block_inject(normed)/hc_count), [T, hc_count]. block_inject_w: [wide, hc_count], this project's own
// [in,out] convention. No scratch needed.
inline void gate(const Dims& d, int T, const float* normed, const float* block_inject_w, float* out_inj) {
    const int wide = d.wide(), hc = d.hc_count;
    const float inv_hc = 1.f / static_cast<float>(hc);
    for (int t = 0; t < T; ++t) {
        const float* xr = normed + static_cast<std::size_t>(t) * wide;
        float* orow = out_inj + static_cast<std::size_t>(t) * hc;
        for (int o = 0; o < hc; ++o) orow[o] = 0.f;
        for (int i = 0; i < wide; ++i) {
            const float xi = xr[i];
            const float* Wr = block_inject_w + static_cast<std::size_t>(i) * hc;
            for (int o = 0; o < hc; ++o) orow[o] += xi * Wr[o];
        }
        for (int o = 0; o < hc; ++o) orow[o] = 2.f * detail::sigmoid(orow[o] * inv_hc);
    }
}

// The WRITE step (docs/GATED_RESIDUAL.md S1b, done by the decoder layer itself in the real model):
// out_wide[t, s*hidden+j] = wide_in[t,s*hidden+j] + inj[t,s] * mixer_out[t,j]. `wide_in`: [T,wide] (the
// ORIGINAL wide stream from before this sub-block's read step -- untouched by hc_norm). `mixer_out`:
// [T,hidden_size] (the sub-block's own output -- attention/GDN/FFN, shared across every stream).
// `inj`: [T,hc_count]. `out_wide`: [T,wide], written (not accumulated; may alias wide_in).
inline void combine(const Dims& d, int T, const float* wide_in, const float* mixer_out, const float* inj,
                     float* out_wide) {
    const int hs = d.hidden_size, hc = d.hc_count, wide = d.wide();
    for (int t = 0; t < T; ++t) {
        const float* wr = wide_in + static_cast<std::size_t>(t) * wide;
        const float* mr = mixer_out + static_cast<std::size_t>(t) * hs;
        const float* ir = inj + static_cast<std::size_t>(t) * hc;
        float* orow = out_wide + static_cast<std::size_t>(t) * wide;
        for (int s = 0; s < hc; ++s) {
            const float w = ir[s];
            for (int j = 0; j < hs; ++j) orow[s * hs + j] = wr[s * hs + j] + w * mr[j];
        }
    }
}

// The model-level ENTRY step (docs/GATED_RESIDUAL.md S1c): tile the ordinary D_MODEL-wide embedding
// hc_count times to seed every stream identically -- torch.repeat, literal duplication, not zero-pad or
// independent init. `h`: [T,hidden_size]. `out_wide`: [T,wide], written.
inline void tile(const Dims& d, int T, const float* h, float* out_wide) {
    const int hs = d.hidden_size, hc = d.hc_count;
    for (int t = 0; t < T; ++t) {
        const float* hr = h + static_cast<std::size_t>(t) * hs;
        float* orow = out_wide + static_cast<std::size_t>(t) * d.wide();
        for (int s = 0; s < hc; ++s)
            for (int j = 0; j < hs; ++j) orow[s * hs + j] = hr[j];
    }
}

}  // namespace sub0::gr
