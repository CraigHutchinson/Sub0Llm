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

#include <algorithm>   // std::max, used by the router's own softmax -- this header compiled only
                        // because every existing consumer happened to include it first (WP5b)
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
// pre-activation, "up" half then overwritten with silu(gate)*up) + d_ff again (the "gate" accumulator
// expert_ffn_row needs alongside it, kept separate for a cache-friendly summation order -- see that
// function's own comment) + hidden_size (the SHARED expert's FFN output) + experts_per_tok * hidden_size
// (one output buffer per SELECTED routed expert). Deliberately NOT scaled by T (see this file's own
// header comment) -- one call's worth, reused across forward()'s own T-row loop below.
//
// THE LAST TERM IS WHY THIS GREW (B20 part 2). The routed experts used to share ONE hidden_size buffer,
// because each was consumed -- weighted and added into `out` -- before the next was computed. Giving
// each selected expert its own buffer is what makes the k-loop's ORDER irrelevant to the answer, so it
// can be run on several threads without the weighted sum's float accumulation order changing. See
// forward_row_via_run below; the sum itself is still done afterwards, sequentially, in k order.
inline constexpr std::size_t scratch_floats(const Dims& d) {
    return static_cast<std::size_t>(d.num_experts)
         + 2u * static_cast<std::size_t>(d.d_ff)
         + static_cast<std::size_t>(d.hidden_size)
         + static_cast<std::size_t>(d.experts_per_tok) * static_cast<std::size_t>(d.hidden_size);
}

namespace detail {
inline float silu(float x) { return x / (1.f + std::exp(-x)); }
inline float sigmoid(float x) { return 1.f / (1.f + std::exp(-x)); }
}  // namespace detail

// One expert's SwiGLU FFN for a single row: out = down(silu(gate(x)) * up(x)) -- Qwen4ExpTextMLP.forward
// (also what Qwen4ExpTextExperts.forward computes per selected expert, chunked gate_up_proj split in
// half). gate_w/up_w: [hidden_size, d_ff]; down_w: [d_ff, hidden_size], this project's own [in,out]
// convention. `pre_scratch`/`g_scratch`: >= d_ff floats each.
//
// Both GEMVs are INPUT-major (outer loop over the projection's input dimension, contiguous inner loop
// over its output dimension) -- matching gate_w/up_w/down_w's own row-major [in,out] layout and this
// project's `linear_row` convention everywhere else. An earlier form of this function was OUTPUT-major
// (outer loop over d_ff/hidden_size, inner loop striding the weight by d_ff or hidden_size floats per
// step) -- correct, but a real, measured cache-locality defect found in a post-merge performance review
// (every inner-loop weight read a cache miss at production d_ff=640/hidden_size=2560). This form needs
// TWO d_ff-wide accumulators (`g_scratch` for the gate projection, `pre_scratch` for the up projection,
// combined into `pre_scratch` only once both are complete) where the old form needed one, since it can no
// longer interleave the two projections' accumulation per output unit.
// B24: the three weight-plane pointers are a template parameter (`WP`). This function has TWO kinds of
// caller and they do not agree on the weight type: a ROUTED expert's planes come out of the sidecar's
// dequantize pool and are always genuine `float*` (moe_quant.hpp produces f32), while the SHARED
// expert's come straight out of `g_param_data`, whose element type is PARAM_DTYPE. Templating is what
// lets both keep calling the one function -- which is the property docs/MOE.md already relies on for
// WP4e's bit-identical gate. `const float*` deduces exactly as before in every F32 build.
template <class WP>
inline void expert_ffn_row(const Dims& d, const float* x, WP gate_w, WP up_w,
                            WP down_w, float* out, float* pre_scratch, float* g_scratch) {
    for (int o = 0; o < d.d_ff; ++o) { pre_scratch[o] = 0.f; g_scratch[o] = 0.f; }
    for (int i = 0; i < d.hidden_size; ++i) {
        const float xi = x[i];
        if (xi == 0.f) continue;
        const auto gr = gate_w + static_cast<std::size_t>(i) * d.d_ff;
        const auto ur = up_w   + static_cast<std::size_t>(i) * d.d_ff;
        for (int o = 0; o < d.d_ff; ++o) { g_scratch[o] += xi * gr[o]; pre_scratch[o] += xi * ur[o]; }
    }
    for (int o = 0; o < d.d_ff; ++o) pre_scratch[o] = detail::silu(g_scratch[o]) * pre_scratch[o];

    for (int j = 0; j < d.hidden_size; ++j) out[j] = 0.f;
    for (int o = 0; o < d.d_ff; ++o) {
        const float po = pre_scratch[o];
        if (po == 0.f) continue;
        const auto dr = down_w + static_cast<std::size_t>(o) * d.hidden_size;
        for (int j = 0; j < d.hidden_size; ++j) out[j] += po * dr[j];
    }
}

// --- fused, no-transpose variant of expert_ffn_row (B31, docs/INDEPENDENT_REVIEW_BACKLOG.md) -------
//
// The plain expert_ffn_row above needs its three weight planes already in THIS PROJECT'S [in,out]
// convention -- for the routed-expert resolve path, that meant dequantize_expert (moe_quant.hpp) had to
// run gguf::to_f32 (write) then transplant::transpose_out_in (read+write) before a single FFN read
// could happen: 3 full touches of each ~19.66 MB plane-set instead of 1. This function instead consumes
// the planes in GGUF's own SOURCE order (moeq::ExpertCacheSource pairs with it, moe_quant.hpp) --
// gate_src/up_src are [d_ff, hidden_size] row-major (row o = output unit o's own hidden_size-wide
// weight vector, contiguous), down_src is [hidden_size, d_ff] row-major (row j = output unit j's own
// d_ff-wide weight vector, contiguous) -- eliminating the transpose stage's round trip entirely: a
// resolve now touches DRAM once (the dequantize write) instead of three times.
//
// WHY THIS IS BIT-EXACT, NOT AN APPROXIMATION. expert_ffn_row's gate/up accumulation is an INPUT-major
// scatter: outer loop i ascending 0..hidden_size-1, inner loop o, `g_scratch[o] += x[i]*gate_w[i*d_ff+o]`
// -- so for a FIXED o, the additions happen in i = 0, 1, 2, ... ascending order, one term per i (terms
// where x[i]==0 are skipped, not added as zero). This function computes the SAME g_scratch[o] as a single
// dot product, `sum_i x[i]*gate_src[o*hidden_size+i]`, with the loop over i ALSO ascending and the same
// x[i]==0 skip -- i.e. the exact same sequence of partial sums, only computed as a per-output REDUCTION
// instead of a scatter. Same addends, same order -> bit-for-bit identical, not merely close (checked by
// running both functions on the same dequantized-vs-source-order planes in tests/moe_quant_tests.cpp's
// B31 case, not just argued here). The down projection is the identical argument with i/o swapped:
// expert_ffn_row's own INPUT-major loop over o (d_ff) scatter-accumulates out[j] in o = 0..d_ff-1
// ascending order for a fixed j (skipping pre_scratch[o]==0); this function's per-j dot product sums the
// same terms in the same o-ascending order with the same skip.
//
// TILING. Each row read here is already fully contiguous, and the vector it is dotted against (x for
// gate/up, pre_scratch for down) is small enough to stay L1-resident for the WHOLE call (hidden_size*4 =
// 10 KiB, d_ff*4 = 2.5 KiB at the real Qwen4 axes -- both far under this host's 48 KiB P-core L1d, docs/
// host-cpu-arrow-lake-hx.md) -- so unlike transpose_block's 2D blocking (which exists to fix a STRIDED
// destination write), there is no strided axis to block away here: this is already the textbook
// cache-optimal GEMV access pattern (stream the matrix once, reuse the small vector out of L1/L2). A
// ROW_TILE grouping is still applied below, sized `consteval` against this host's smaller P-core L2 (3
// MiB, per B31's own brief -- a single decode thread, B29, can land on either core type, so the SMALLER
// P-core figure is the safe target) purely to bound how much of the source plane one chunk of the loop
// touches before moving on to the next. Measured against the plain unblocked per-row loop on the real
// 48-layer artifact: no reliable difference (same shape of honest null result as this project's own B28
// software-prefetch finding, docs/INDEPENDENT_REVIEW_BACKLOG.md) -- kept anyway because it costs nothing
// and documents the real cache budget this build was sized against, per this project's own
// compile-time-over-runtime preference (AGENTS.md S1/S2).
inline constexpr int kP_CORE_L2_BYTES = 3 * 1024 * 1024;
// A conservative fraction of L2, leaving headroom for x/pre_scratch/g_scratch and everything else live
// on this core (router probs, the shared-expert accumulators, OS/runtime state) -- not the whole 3 MiB.
inline constexpr int kL2_TILE_BUDGET_BYTES = kP_CORE_L2_BYTES / 2;
// How many D_FF=640-wide source rows (hidden_size=2560 floats = 10 KiB each at the real axes) fit the
// budget -- computed from the real dims rather than hand-picked, per AGENTS.md S2's "derive bounds from
// scale" precedent. Clamped to at least 1 (never fewer rows than exist to still make progress) and to the
// call's own d_ff/hidden_size so a tiny test-scale Dims never asks for a tile bigger than the matrix.
inline constexpr int row_tile(int row_bytes, int n_rows) {
    const int budget_rows = row_bytes > 0 ? (kL2_TILE_BUDGET_BYTES / row_bytes) : n_rows;
    int t = budget_rows < 1 ? 1 : budget_rows;
    if (t > n_rows) t = n_rows > 0 ? n_rows : 1;
    return t;
}

template <class WP>
inline void expert_ffn_row_source(const Dims& d, const float* x, WP gate_src, WP up_src, WP down_src,
                                   float* out, float* pre_scratch, float* g_scratch) {
    const int gu_tile = row_tile(d.hidden_size * static_cast<int>(sizeof(float)), d.d_ff);
    for (int ot = 0; ot < d.d_ff; ot += gu_tile) {
        const int o_end = std::min(ot + gu_tile, d.d_ff);
        for (int o = ot; o < o_end; ++o) {
            const auto gr = gate_src + static_cast<std::size_t>(o) * d.hidden_size;
            const auto ur = up_src   + static_cast<std::size_t>(o) * d.hidden_size;
            float gs = 0.f, ps = 0.f;
            for (int i = 0; i < d.hidden_size; ++i) {
                const float xi = x[i];
                if (xi == 0.f) continue;
                gs += xi * gr[i];
                ps += xi * ur[i];
            }
            g_scratch[o] = gs;
            pre_scratch[o] = ps;
        }
    }
    for (int o = 0; o < d.d_ff; ++o) pre_scratch[o] = detail::silu(g_scratch[o]) * pre_scratch[o];

    const int down_tile = row_tile(d.d_ff * static_cast<int>(sizeof(float)), d.hidden_size);
    for (int jt = 0; jt < d.hidden_size; jt += down_tile) {
        const int j_end = std::min(jt + down_tile, d.hidden_size);
        for (int j = jt; j < j_end; ++j) {
            const auto dr = down_src + static_cast<std::size_t>(j) * d.d_ff;
            float s = 0.f;
            for (int o = 0; o < d.d_ff; ++o) {
                const float po = pre_scratch[o];
                if (po == 0.f) continue;
                s += po * dr[o];
            }
            out[j] = s;
        }
    }
}

// The router (Qwen4ExpTextTopKRouter.forward, docs/MOE.md S1a): logits = linear(x, router_w) (no bias,
// full-width softmax over ALL num_experts, THEN top-k -- not top-k-then-softmax); optionally renormalized
// (norm_topk_prob, real config default True) so the selected weights sum to 1. router_w: [hidden_size,
// num_experts]. `probs_scratch`: >= num_experts floats, destroyed (used as working storage for the
// selection scan below, S4b -- this row's own softmax values are not needed again after this call).
// `out_weight`/`out_idx`: caller buffers of length >= experts_per_tok (TOPK_MAX-capped internally).
template <class WP>
inline void router_topk_row(const Dims& d, const float* x, WP router_w, float* probs_scratch,
                             float* out_weight, int* out_idx, bool norm_topk_prob) {
    // INPUT-major/contiguous (this project's own linear_row convention, matching router_w's own
    // [hidden_size, num_experts] row-major layout): outer over the contraction dim (hidden_size), inner
    // CONTIGUOUS over num_experts -- found by VTune (router_topk_row was 17.1% of CPU time in a
    // realistic GR+MoE+QSA hotspot profile, second only to expert_ffn_row) after the original
    // output-major form (outer e, inner i striding router_w by num_experts floats per step) survived
    // the earlier code-reading-only perf review that fixed gr::mix()/expert_ffn_row() but never looked
    // at this function. Pure summation reorder, same tolerance argument as that fix.
    for (int e = 0; e < d.num_experts; ++e) probs_scratch[e] = 0.f;
    for (int i = 0; i < d.hidden_size; ++i) {
        const float xi = x[i];
        if (xi == 0.f) continue;
        const auto wr = router_w + static_cast<std::size_t>(i) * d.num_experts;
        for (int e = 0; e < d.num_experts; ++e) probs_scratch[e] += xi * wr[e];
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

// One routed expert's three weight planes. `gate`/`up` are [hidden_size, d_ff] and `down` is
// [d_ff, hidden_size], this project's own [in,out] convention -- exactly the three pointers
// expert_ffn_row already takes.
// B24: parameterised on the plane pointer type so an f32-resident caller (whose planes come from
// `g_param_data`, i.e. PARAM_DTYPE) and the sidecar resolver (whose planes are always dequantized f32)
// can each name the one that is true for them. `ExpertWeights` keeps its original name and meaning --
// the sidecar's f32 planes -- because that is what every existing call site means by it.
template <class WP>
struct ExpertWeightsOf { WP gate; WP up; WP down; };
using ExpertWeights = ExpertWeightsOf<const float*>;

// The full block for one row (Qwen4ExpTextSparseMoeBlock.forward, docs/MOE.md S1): route, run only the
// experts_per_tok SELECTED experts' SwiGLU FFNs, weighted-sum them, then add the ALWAYS-ON shared
// expert's own SwiGLU FFN output scaled by sigmoid(shared_gate_proj(x)) -- NOT re-weighted by the
// router's own topk output, a real, verified divergence from a naive "shared expert is just expert #0"
// reading (docs/MOE.md S1a). `scratch`: >= scratch_floats(d) floats.
//
// WHERE THE EXPERT WEIGHTS COME FROM IS A PARAMETER (WP4e, docs/WP4_SCOPE.md). `resolve(e)` returns
// expert e's three planes and is called ONLY for the experts_per_tok experts this row actually selects
// -- which is what lets a caller hold the routed experts in their native quantized form and dequantize
// on demand (backend_cpu.cpp under MOE_QUANT_EXPERTS) instead of keeping all num_experts resident as
// f32. The f32-resident caller passes a resolver that indexes its own pointer arrays (forward_row just
// below), so BOTH residency strategies run this one function, with the same arithmetic in the same
// order. That is not a tidiness point: it is why WP4e's gate -- bitwise-identical output between the two
// -- is a structural property of having one code path rather than an empirical hope about two.
//
// `resolve` must not allocate and must leave the returned planes valid until this row's whole selected
// set has been computed (a caller running the k-loop serially through a one-slot pool is fine: the
// planes are consumed by expert_ffn_row before resolve is called again; a caller running it in parallel
// needs one pool PER THREAD, since several resolves are then live at once). AGENTS.md S1 -- this is a
// per-token, per-selected-expert path.
//
// --- RUNNING THE SELECTED EXPERTS ON MORE THAN ONE THREAD (B20 part 2) ---------------------------
//
// The `experts_per_tok` selected experts are independent by construction: each reads the same `x` and
// its own three weight planes, and writes its own output. The only thing that was NOT order-independent
// was the WEIGHTED SUM that consumed each expert's output the moment it was produced -- float addition
// is not associative, so a thread-completion order would have changed the last bits of `out` even
// though nothing was wrong.
//
// So the loop is in two phases, and it is in two phases for the SERIAL path too, not just the parallel
// one: phase 1 computes every selected expert into ITS OWN buffer (order-independent -- no two writes
// touch the same float), phase 2 does the weighted sum afterwards, single-threaded, in the original k
// order. That is exactly the same sequence of float additions per output element as the old interleaved
// form (out[j] starts at 0 and takes w0*e0[j], then w1*e1[j], ...), so this restructuring is itself
// bit-for-bit -- which is what lets one code path serve both, rather than a parallel variant whose
// agreement with the serial one would be an empirical hope.
//
// `run_experts(n, ffn_scratch, g_scratch, body)` invokes `body(k, ffn, g)` for every k in [0, n). The
// default runner below is the plain serial loop, reusing the one scratch pair; a parallel runner hands
// each thread its own pair. Nothing about WHICH threads, or how many, reaches this file: this header is
// engine-free (docs/MOE.md), and OpenMP lives at the call site.
struct SerialExperts {
    template <class Body>
    void operator()(int n, float* ffn, float* g, Body&& body) const {
        for (int k = 0; k < n; ++k) body(k, ffn, g);
    }
};

// B31: the generalized form, letting the caller decide HOW a selected expert's row is produced instead
// of this function hardcoding "resolve weight pointers, then call expert_ffn_row" -- which is what lets
// a caller fuse the resolve and the FFN together (moeq::ExpertCacheSource + expert_ffn_row_source above)
// without a second, parallel copy of this whole two-phase/order-independence machinery. `compute_expert`
// is called once per selected expert, exactly as `resolve` used to be, but its OWN job is now "produce
// this row's `d.hidden_size` output floats", not merely "hand back three pointers". Same order-
// independence argument as before: each call writes only its own `out_ptr`, nothing is read back until
// phase 2.
//
// B24: `WP` is the PARAMETER-ARENA pointer type (router + shared expert).
template <class WP, class ComputeExpert, class RunExperts>
inline void forward_row_via_run_ex(const Dims& d, const float* x, WP router_w,
                                    ComputeExpert&& compute_expert, RunExperts&& run_experts,
                                    WP shared_gate_w, WP shared_up_w,
                                    WP shared_down_w, WP shared_gate_proj_w,
                                    float* out, float* scratch, bool norm_topk_prob = true) {
    float* probs      = scratch;                    // [num_experts]
    float* ffn_scratch = probs + d.num_experts;      // [d_ff]
    float* g_scratch  = ffn_scratch + d.d_ff;        // [d_ff]
    float* expert_out = g_scratch + d.d_ff;          // [hidden_size]   -- the SHARED expert's
    float* routed_out = expert_out + d.hidden_size;  // [experts_per_tok][hidden_size]
    float topk_w[TOPK_MAX];
    int   topk_idx[TOPK_MAX];
    router_topk_row(d, x, router_w, probs, topk_w, topk_idx, norm_topk_prob);

    // Phase 1: every selected expert into its own buffer. Order-independent by construction.
    run_experts(d.experts_per_tok, ffn_scratch, g_scratch, [&](int k, float* ffn, float* g) {
        compute_expert(k, topk_idx[k], routed_out + static_cast<std::size_t>(k) * d.hidden_size, ffn, g);
    });
    // Phase 2: the weighted sum, in the original selection order, on one thread.
    for (int j = 0; j < d.hidden_size; ++j) out[j] = 0.f;
    for (int k = 0; k < d.experts_per_tok; ++k) {
        const float* ek = routed_out + static_cast<std::size_t>(k) * d.hidden_size;
        for (int j = 0; j < d.hidden_size; ++j) out[j] += topk_w[k] * ek[j];
    }

    // Shared expert: an ordinary (non-routed) SwiGLU FFN with its OWN weights, gated by
    // sigmoid(Linear(hidden_size, 1, bias=False)(x)) -- Qwen4ExpTextSparseMoeBlock.forward's own
    // `F.sigmoid(self.shared_expert_gate(hidden_states_reshaped)) * shared_expert_output`. Always
    // f32-resident: it runs for EVERY token, so there is nothing for a quantized-resident form to save
    // (see include/sub0/moe_quant.hpp's own header comment).
    expert_ffn_row(d, x, shared_gate_w, shared_up_w, shared_down_w, expert_out, ffn_scratch, g_scratch);
    float gate_logit = 0.f;
    for (int i = 0; i < d.hidden_size; ++i) gate_logit += x[i] * shared_gate_proj_w[i];  // [hidden_size,1]
    const float sg = detail::sigmoid(gate_logit);
    for (int j = 0; j < d.hidden_size; ++j) out[j] += sg * expert_out[j];
}

// The original, unchanged-signature entry point every existing caller (op_moe, forward_row_via,
// tests) uses: `resolve(idx) -> ExpertWeights`, then this function's own expert_ffn_row call, exactly as
// before B31 -- now a thin wrapper over forward_row_via_run_ex so there is exactly one copy of the two-
// phase/order-independence machinery, not two that could drift apart.
template <class WP, class Resolve, class RunExperts>
inline void forward_row_via_run(const Dims& d, const float* x, WP router_w, Resolve&& resolve,
                                 RunExperts&& run_experts,
                                 WP shared_gate_w, WP shared_up_w,
                                 WP shared_down_w, WP shared_gate_proj_w,
                                 float* out, float* scratch, bool norm_topk_prob = true) {
    forward_row_via_run_ex(
        d, x, router_w,
        [&](int /*k*/, int idx, float* out_ptr, float* ffn, float* g) {
            const ExpertWeights w = resolve(idx);
            expert_ffn_row(d, x, w.gate, w.up, w.down, out_ptr, ffn, g);
        },
        static_cast<RunExperts&&>(run_experts), shared_gate_w, shared_up_w, shared_down_w,
        shared_gate_proj_w, out, scratch, norm_topk_prob);
}

// The single-threaded form, and the one every caller that has no reason to fan out uses: the selected
// experts run one after another on this thread, reusing the one scratch pair. Kept as its own name (and
// its own unchanged signature) because that is what op_moe's batched T-row path, the f32-resident
// wrappers below, and every test call -- none of which are the decode hot path B20 measured -- want.
template <class WP, class Resolve>
inline void forward_row_via(const Dims& d, const float* x, WP router_w, Resolve&& resolve,
                             WP shared_gate_w, WP shared_up_w,
                             WP shared_down_w, WP shared_gate_proj_w,
                             float* out, float* scratch, bool norm_topk_prob = true) {
    forward_row_via_run(d, x, router_w, resolve, SerialExperts{}, shared_gate_w, shared_up_w,
                        shared_down_w, shared_gate_proj_w, out, scratch, norm_topk_prob);
}

// The f32-resident form: `expert_gate_w`/`expert_up_w`/`expert_down_w` are arrays of `num_experts`
// pointers, one per routed expert. A thin wrapper, deliberately -- see forward_row_via's comment.
template <class WP>
inline void forward_row(const Dims& d, const float* x, WP router_w,
                         const WP* expert_gate_w, const WP* expert_up_w,
                         const WP* expert_down_w,
                         WP shared_gate_w, WP shared_up_w, WP shared_down_w,
                         WP shared_gate_proj_w,
                         float* out, float* scratch, bool norm_topk_prob = true) {
    forward_row_via(d, x, router_w,
                    [&](int e) {
                        return ExpertWeightsOf<WP>{expert_gate_w[e], expert_up_w[e], expert_down_w[e]};
                    },
                    shared_gate_w, shared_up_w, shared_down_w, shared_gate_proj_w, out, scratch,
                    norm_topk_prob);
}

// Batched T-row wrapper. Rows are independent (this file's own header comment) so `scratch` is reused
// across the loop, not sized per-row -- one scratch_floats(d)-sized buffer serves any T.
template <class WP>
inline void forward(const Dims& d, int T, const float* x, WP router_w,
                     const WP* expert_gate_w, const WP* expert_up_w,
                     const WP* expert_down_w,
                     WP shared_gate_w, WP shared_up_w, WP shared_down_w,
                     WP shared_gate_proj_w,
                     float* out, float* scratch, bool norm_topk_prob = true) {
    for (int t = 0; t < T; ++t) {
        forward_row(d, x + static_cast<std::size_t>(t) * d.hidden_size, router_w,
                    expert_gate_w, expert_up_w, expert_down_w,
                    shared_gate_w, shared_up_w, shared_down_w, shared_gate_proj_w,
                    out + static_cast<std::size_t>(t) * d.hidden_size, scratch, norm_topk_prob);
    }
}

// The same batched wrapper for a resolver-driven caller (WP4e). Separate from forward() rather than a
// default argument because the two have genuinely different call sites, and because `resolve` must be
// reused across the T-row loop -- resolving per row is what makes the small pool pay for itself when
// several rows of one batch pick the same expert.
template <class WP, class Resolve>
inline void forward_via(const Dims& d, int T, const float* x, WP router_w, Resolve&& resolve,
                         WP shared_gate_w, WP shared_up_w, WP shared_down_w,
                         WP shared_gate_proj_w,
                         float* out, float* scratch, bool norm_topk_prob = true) {
    for (int t = 0; t < T; ++t) {
        forward_row_via(d, x + static_cast<std::size_t>(t) * d.hidden_size, router_w, resolve,
                        shared_gate_w, shared_up_w, shared_down_w, shared_gate_proj_w,
                        out + static_cast<std::size_t>(t) * d.hidden_size, scratch, norm_topk_prob);
    }
}

}  // namespace sub0::moe
