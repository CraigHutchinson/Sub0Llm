// sub0/memplan.hpp — pure, backend-agnostic device-memory footprint model.
//
// The CUDA backend lays out a fixed set of resident allocations for a training step: the
// parameter mirror, the optimizer arenas, the fused-QKV weights, and the forward + backward
// activation scratch. Their sizes are 100% determined by the model dimensions and the
// minibatch, so we express the TOTAL device footprint ONCE here as a pure constexpr function
// of EXPLICIT dimensions -- deliberately NO dependency on the generated sub0_config.hpp (or
// layout.hpp, which includes it). That lets the two consumers that must reason about memory
// before a device exists both call it:
//   * the runtime GPU tuner -- skip (don't even time) a batch that would not fit in VRAM;
//   * the configurator -- turn a baked DEFAULT_GPU_BATCH that cannot fit into a HARD build
//     failure (a misconfiguration caught at configure time, not as a silent WDDM spill).
//
// This is a PREDICTION, and predictions rot. It is kept honest by sub0_cuda_train_footprint(),
// which measures the ACTUAL device delta (cudaMemGetInfo around the real allocations) and
// compares the two: a mismatch means a buffer was added/removed/resized in backend_cuda.cu
// without updating the mirror below. The CUDA footprint test asserts they agree, so a drift
// fails CI rather than silently mis-predicting in production.
//
// Keep every term in lock-step with backend_cuda.cu: the persistent g_dev_* arenas, wqkv_alloc,
// fwd_alloc and train_alloc. Each term below names the buffer(s) it mirrors.

#pragma once

#include <cstddef>

namespace sub0::memplan {

// The handful of model dimensions the footprint depends on (C/L/H/F/T/V in the kernel comments).
struct Dims {
    int d_model;   // C: embedding / residual width
    int n_layers;  // L: transformer block count
    int n_heads;   // H: attention heads
    int d_ff;      // F: feed-forward width (= 4*C in this engine)
    int seq_len;   // T: context window
    int vocab;     // V: token count
    // Tied embeddings (USE_TIED_EMBEDDINGS): the LM head reuses tok_emb instead of its own
    // matrix+bias, dropping param_floats()'s "head" term from ln_f+lm_head+lm_bias down to ln_f
    // alone (see layout.hpp's make_param_layout tail). Defaults false so every pre-existing
    // aggregate-init call site (which only lists the first 6 fields) keeps modeling the untied
    // shape exactly as before -- only a caller that actually builds a tied model needs to pass it.
    bool tied = false;
    // QK-norm (USE_QK_NORM): adds a [1,D_HEAD] q_norm + k_norm gamma pair per layer (see layout.hpp's
    // make_param_layout, inserted between Wo and the FFN block) and one extra TRAINING-only scratch
    // buffer (the pre-norm Q/K stash backward needs -- see train_scratch_bytes' qk_pre term below).
    // Defaults false for the same reason `tied` does: every pre-existing aggregate-init call site
    // keeps modeling the non-qk-norm shape exactly as before.
    bool qk_norm = false;
    // Gated (SwiGLU) FFN (USE_GATED_FFN): replaces the plain FFN's W1[C,F]+b1[F]+W2[F,C]+b2[C] (2*C*F +
    // F + C floats) with Wg[C,F]+W1[C,F]+W2[F,C], no biases (3*C*F floats) -- see layout.hpp's
    // make_param_layout. A GPU build also keeps a THIRD bf16 GEMM-weight mirror (g_wg16, alongside
    // g_w1_16/g_w2_16) instead of two -- see persistent_bytes' BF16 term below. Training scratch is
    // UNCHANGED (backward_device's checkpoint-recompute reuses the plain path's existing ff1/gact/
    // dff1/dgact [M,F] buffers under role-remapping rather than adding new ones -- see that function's
    // own comment for the exact buffer-role sequence). Defaults false for the same reason `tied`/
    // `qk_norm` do: every pre-existing aggregate-init call site keeps modeling the plain-FFN shape.
    bool gated = false;
    // Absolute learned positions (POS_ENCODING == Absolute, i.e. layout.hpp's HAS_POS_EMB): adds a
    // pos_emb [T,C] table. RoPE omits it entirely -- and that omission is what removes the ONLY
    // SEQ_LEN dependence from the parameter set, letting a checkpoint move between window sizes.
    // Defaults TRUE (unlike the flags above) because every pre-existing aggregate-init call site was
    // written when the table was unconditional, so true is what preserves their meaning.
    bool pos_emb = true;
    // Grouped-query attention (N_KV_HEADS): K/V project to n_kv_heads*D_HEAD instead of d_model, so
    // Wk/Wv shrink and the KV cache shrinks with them. 0 means "same as n_heads" -- plain MHA, what
    // every pre-existing aggregate-init call site means.
    int n_kv_heads = 0;
    // LoopSplit: total layer EXECUTIONS per forward (sub0::LOOP_EXEC_COUNT). Per-execution activation
    // checkpoints scale on this, not on n_layers; 0 means "unset" and falls back to n_layers.
    int exec_layers = 0;
    // Depth attention: cache SLOTS (sub0::DEPTH_CACHE_MAX), one per participating execution. 0 = off,
    // which is what every pre-existing aggregate-init call site means. This is the term that decides
    // which stride fits: unlike the checkpoints above, each slot is RETAINED for the whole step AND
    // carries a f32 gradient accumulator, so it is ~1.5x the cost a forward-only reading suggests --
    // see train_scratch_bytes' `depth` term and docs/DEPTH_ATTENTION.md 5b.
    int depth_slots = 0;
    // Forced logits row-chunk COUNT, overriding logits_n_chunks' derivation (0 = derive, what every
    // pre-existing aggregate-init call site means). This field exists so the chunk lever has exactly ONE
    // source of truth: the CUDA allocator used to branch on its own runtime override while the predictor
    // always re-derived, so setting the override made the prediction disagree with the allocation by up
    // to 1.35 GiB at the LoopSplit arms' shape -- over 4x FOOTPRINT_OVERPREDICT_TOLERANCE_MB. Nothing
    // caught it because the only caller that set the override happened to be declared AFTER the footprint
    // test. Routing the override through Dims means train_scratch_bytes models whatever the allocator
    // will actually do, so the lever can be raised in production without silently under-sizing the batch.
    int logits_chunks = 0;
};

using u64 = unsigned long long;

inline constexpr u64 FLOAT = sizeof(float);
inline constexpr u64 INT   = sizeof(int);
inline constexpr u64 DBL   = sizeof(double);

// Hard ceiling on the minibatch any device entry point will size scratch for (backend_cuda.cu
// validates/clamps against this as MAX_FWD_BATCH). Lives here -- not in the backend -- because it is
// also (a) the size of the constant per-window `lengths` device buffer counted in
// train_scratch_bytes below, and (b) the bound the training loop's token-budget batch scheduler
// clamps its per-step effective batch to (train_stage.cpp), so all three stay one number.
inline constexpr int MAX_DEVICE_BATCH = 4096;

// Trainable float count -- the exact sum PARAM_LAYOUT produces in layout.hpp, re-derived here
// from dims so the configurator (which cannot include layout.hpp) gets the same number. The CUDA
// footprint test asserts param_floats(dims) == sub0::PARAM_FLOATS, catching any layout drift.
// K/V projection width (sub0::D_KV) for these runtime Dims. n_kv_heads == 0 means "unset" and falls
// back to n_heads, i.e. plain MHA, where this is exactly d_model.
constexpr u64 d_kv(const Dims& d) {
    const u64 C = u64(d.d_model), H = u64(d.n_heads);
    const u64 DH = H > 0 ? C / H : 0;
    return u64(d.n_kv_heads > 0 ? d.n_kv_heads : d.n_heads) * DH;
}
// Runtime mirrors of sub0::QKV_STRIDE / sub0::QK_PRE_STRIDE (see layout.hpp, which explains the fused
// buffer's column layout and why these live in two places). memplan is included BY layout.hpp, so it
// cannot use the constexpr forms and must re-derive them from Dims -- keep the two in lock-step.
constexpr u64 qkv_stride(const Dims& d)    { return u64(d.d_model) + 2 * d_kv(d); }  // == 3*C under MHA
constexpr u64 qk_pre_stride(const Dims& d) { return u64(d.d_model) + d_kv(d); }      // == 2*C under MHA

constexpr u64 param_floats(const Dims& d) {
    const u64 C = u64(d.d_model), L = u64(d.n_layers), F = u64(d.d_ff), T = u64(d.seq_len), V = u64(d.vocab);
    const u64 CKV = d_kv(d);                    // D_KV: K/V width (== C for plain MHA)
    const u64 H  = u64(d.n_heads), DH = H > 0 ? C / H : 0;   // D_HEAD = C/H
    const u64 emb   = V * C + (d.pos_emb ? T * C : 0);   // tok_emb [V,C] + pos_emb [T,C] if present
    const u64 block = 2 * C                     // ln1 + ln2          [C] each
                    + 2 * C * C                 // Wq, Wo             [C,C]
                    + 2 * C * CKV               // Wk, Wv             [C,D_KV] (== [C,C] under MHA)
                    + (d.qk_norm ? 2 * DH : 0)  // q_norm + k_norm [1,D_HEAD] each (USE_QK_NORM only)
                    // gated: Wg [C,F], W1 [C,F], W2 [F,C], no biases (3*C*F). plain: W1 [C,F], b1 [F],
                    // W2 [F,C], b2 [C] (2*C*F + F + C).
                    + (d.gated ? 3 * C * F : (2 * C * F + F + C));
    // head: ln_f [C] alone when tied (the head reuses tok_emb, already counted in `emb` above --
    // no separate lm_head/lm_bias slot, see layout.hpp's make_param_layout); ln_f + lm_head [C,V] +
    // lm_bias [V] otherwise.
    const u64 head  = d.tied ? C : (C + C * V + V);
    return emb + L * block + head;
}

// Persistent, batch-independent device memory: the param mirror (g_dev_params) + the three
// optimizer arenas (g_dev_grad/m/vel) + the norm accumulator (g_dev_normsq) + the on-device
// grad-clip scale (g_dev_gs). Weight decay costs ZERO persistent bytes (2026-07): it used to be a
// PARAM_FLOATS-long uint8 mask, now it's a handful of compile-time-known ranges in __constant__
// memory (g_decay_ranges, backend_cuda.cu) -- see layout.hpp's DECAY_RANGES.
// A = activation dtype width (FLOAT for an F32 build, 2 for BF16). A BF16 build keeps a second,
// narrower copy of every GEMM weight (g_w1_16/g_w2_16/g_wo16/g_wqkv16, built DIRECTLY from the F32
// master by build_qkv_weights()) instead of the F32 fused-QKV staging buffer (g_fwd.wqkv) -- that
// buffer is F32-build-only now (2026-07): a BF16 training run never allocates it at all (dead weight
// during training, ~108 MiB at production dims), only the CUDA inference paths (sub0_cuda_forward,
// sub0_cuda_forward_one) lazily build+keep it on demand, via ensure_wqkv_f32 in backend_cuda.cu -- so
// it is correctly excluded from a BF16 TRAINING footprint. An F32 build has no separate mirrors
// (they alias the master weights directly, build_qkv_weights()'s F32 branch), so the second term is
// 0 there. sub0_cuda_train_footprint() calls build_qkv_weights() before measuring so the PREDICTED
// and MEASURED deltas stay comparable.
constexpr u64 persistent_bytes(const Dims& d, u64 A = FLOAT) {
    const u64 C = u64(d.d_model), L = u64(d.n_layers), F = u64(d.d_ff);
    const u64 QKVW = qkv_stride(d);                      // C + 2*D_KV -- fused row width, == 3*C under MHA
    const u64 ffn_mirrors = (d.gated ? 3 : 2) * C * F;   // gated: g_wg16+g_w1_16+g_w2_16; plain: g_w1_16+g_w2_16
    return 4 * param_floats(d) * FLOAT          // g_dev_params + grad + m + vel
         + DBL                                  // g_dev_normsq [1] (double)
         + FLOAT                                // g_dev_gs [1] (on-device grad-clip scale, 2026-07)
         + (A >= FLOAT ? L * (C * QKVW) * FLOAT : 0)                     // g_fwd.wqkv[l] = [C,QKVW] F32 (F32 only)
         + (A < FLOAT ? L * (ffn_mirrors + C * C + C * QKVW) * A : 0);   // BF16: g_w[g]1_16+g_w2_16+g_wo16+g_wqkv16
}

// Row-chunk count for the TRAINING [chunk_rows,V] logits/dlogits scratch buffer (train_alloc /
// g_tr.logits in backend_cuda.cu -- NOT the inference-only fwd_alloc/g_fwd.logits below, which never
// runs cross-entropy and stays a plain [M,V] buffer). At this project's typical V >> C/F scale, a
// full [M,V] logits buffer is the single largest per-window TRAINING scratch buffer (~55-60% of it
// at production dims) -- see train_scratch_bytes' `logits` term. Chunking the lm_head/tied-head
// forward -> cross-entropy -> lm_head/tied-head backward chain over row-chunks of the M dimension
// (instead of one [M,V] shot) lets that one buffer shrink to [chunk_rows,V], reused across chunks.
//
// ceil(vocab/d_ff) is the ratio, not a fixed constant: it targets the CHUNKED buffer to land at
// roughly the same byte-scale as this model's OTHER large per-row activations, which all scale with
// d_ff (ff1/gact/dff1/dgact, the next-widest per-row buffers after logits) rather than vocab -- i.e.
// "how many d_ff-sized pieces would a vocab-sized buffer need to be split into to match that scale."
// Clamped to [1,8]: 1 when vocab is already small relative to d_ff (no chunking needed -- chunk_rows
// == total_rows, degenerating to exactly today's single-shot behavior, zero overhead); capped at 8
// because the marginal memory win beyond that is small relative to the extra GEMM/kernel launches per
// training step -- a real production-dims calculation (D_MODEL=448, D_FF=1792, VOCAB=16517) shows
// K=8 already captures ~87% of the theoretical maximum reduction this lever can offer (48.9% of
// 55.9%), while K=16 only adds another ~3.5 percentage points for double the launches.
// The [1,8] clamp's upper bound is CALIBRATED FOR A PARTICULAR d_ff and truncates badly away from it.
// The 87%-of-maximum figure above was computed at production dims (D_MODEL 448, D_FF 1792, VOCAB 16517),
// where ceil(vocab/d_ff) is 10 and the cap costs almost nothing. At the LoopSplit arms' shape (D_FF 512,
// VOCAB 16508) the desired ratio is 33, so a cap of 8 truncates it FOUR-FOLD and the logits buffer stops
// tracking the d_ff scale it is supposed to match: measured 1806 MiB at batch 448, which is 22% of the
// whole training footprint and the largest single scratch term after the per-execution checkpoints.
// Raising the cap to 32 there takes it to 452 MiB.
//
// Overridable per build rather than raised globally: more chunks means more kernel launches, so the
// value belongs to whoever is trading launches for VRAM. Changing it CANNOT change results -- the chunk
// loop is mathematically identical to the unchunked path (see head_ce_chunked) -- and that invariant is
// pinned by a test that runs the same backward at several chunk counts and compares gradients, rather
// than being asserted in a comment as it was before.
#if !defined(SUB0_LOGITS_MAX_CHUNKS)
  #define SUB0_LOGITS_MAX_CHUNKS 8
#endif
inline constexpr int LOGITS_MAX_CHUNKS = SUB0_LOGITS_MAX_CHUNKS;
static_assert(LOGITS_MAX_CHUNKS >= 1, "LOGITS_MAX_CHUNKS must be at least 1 (1 = no chunking)");

// `forced` > 0 bypasses the derivation entirely and uses that chunk count verbatim (Dims::logits_chunks
// / the CUDA backend's sub0_cuda_set_logits_chunks). It is NOT clamped to LOGITS_MAX_CHUNKS: the clamp
// exists to stop the *automatic* derivation from trading unbounded kernel launches for VRAM, whereas a
// forced count is an explicit caller decision that must be modelled exactly as given -- clamping it here
// would reintroduce the predictor/allocator disagreement this parameter was added to remove.
constexpr int logits_n_chunks(int vocab, int d_ff, int forced = 0) {
    if (forced > 0) return forced;
    int k = (vocab + d_ff - 1) / d_ff;   // ceil(vocab/d_ff)
    if (k < 1) k = 1;
    if (k > LOGITS_MAX_CHUNKS) k = LOGITS_MAX_CHUNKS;
    return k;
}
// Row-chunk CAPACITY: the [chunk_rows,V] training logits/dlogits buffer is allocated ONCE at this
// size (train_alloc) and reused across every chunk iteration and every training step within the same
// row budget; `total_rows` is that reserved row budget (train_alloc's Mm, i.e. batch*seq_len).
constexpr u64 logits_chunk_rows(int vocab, int d_ff, u64 total_rows, int forced = 0) {
    const u64 n = u64(logits_n_chunks(vocab, d_ff, forced));
    return (total_rows + n - 1) / n;
}

// Resident forward scratch (fwd_alloc), grown to `batch`. One term per cudaMalloc in fwd_alloc.
constexpr u64 fwd_scratch_bytes(const Dims& d, int batch) {
    const u64 C = u64(d.d_model), F = u64(d.d_ff), V = u64(d.vocab);
    const u64 Mm = u64(batch) * u64(d.seq_len);
    return Mm * INT                             // dids   [M]
         + 6 * Mm * C * FLOAT                   // h, a, att, proj, fbuf, ff2  [M,C]
         + Mm * qkv_stride(d) * FLOAT           // qkv    [M,C+2*D_KV] (== [M,3C] under MHA)
         + 2 * Mm * F * FLOAT                   // ff1, gact  [M,F]
         + Mm * V * FLOAT;                      // logits [M,V]
}

// Itemized training scratch: one field per GROUP of cudaMallocs in train_alloc, so a predicted-vs-measured
// gap can be ATTRIBUTED to a term instead of only observed on the total. This exists because the total was
// the only thing memplan exposed, which meant the allocation census in docs/MEMORY_AUDIT.md had to be
// hand-recomputed from the formula -- and a hand recomputation that silently ignored logits_n_chunks'
// [1,LOGITS_MAX_CHUNKS] clamp put the largest single term out by 4x (438 MiB claimed vs 1806 MiB real)
// while the total still looked plausible. Terms are the unit of review; the total is a derived quantity.
struct ScratchTerms {
    u64 per_exec  = 0;   // EL x (h_in, h_mid, rinv1, rinv2) -- scales with EXECUTIONS (LoopSplit)
    u64 final_blk = 0;   // h_final, a_final, rinv_f + the single checkpoint scratches (a, fbuf, ff1, gact, qkv, att)
    u64 logits    = 0;   // [chunk_rows,V] logits/dlogits -- the chunk lever's target, and the largest single term
    u64 grad      = 0;   // the backward's dh/da/datt/dfbuf/dqkv/dff1/dgact + dwqkv, dtargets, lengths, active, loss
    u64 qk_pre    = 0;   // pre-norm Q/K stash (USE_QK_NORM only)
    u64 depth     = 0;   // depth-attention cache slots + their f32 grad accumulators (USE_DEPTH_ATTN only)
    constexpr u64 total() const { return per_exec + final_blk + logits + grad + qk_pre + depth; }
};

// Resident training scratch (train_alloc), grown to `batch`, itemized. One term per cudaMalloc group.
// USE_GATED_FFN needs NO extra terms here: backward_device's checkpoint-recompute reuses the plain
// path's existing ff1/gact/dff1/dgact [M,F] buffers under role-remapping (ff1<-gate_pre, gact<-up_pre,
// dff1<-hswi then dgate, dgact<-d_hswi then dup) instead of allocating a dedicated SwiGLU-output
// buffer -- see backward_device's own comment for the exact per-step sequence this relies on.
constexpr ScratchTerms train_scratch_terms(const Dims& d, int batch, u64 A = FLOAT) {
    const u64 C = u64(d.d_model), F = u64(d.d_ff);
    const u64 T = u64(d.seq_len), V = u64(d.vocab);
    const u64 Mm = u64(batch) * T;
    const u64 EL = u64(d.exec_layers > 0 ? d.exec_layers : d.n_layers);   // executions, not layers
    const u64 per_layer = 2 * Mm * C * A        // h_in, h_mid  [M,C] bf16 residual stream
                        + 2 * Mm * FLOAT;       // rinv1, rinv2  [M]  (a/qkv/att/ff1/gact checkpointed -> single, below)
    const u64 final_blk = Mm * C * A            // h_final [M,C] bf16 residual
                        + Mm * C * FLOAT        // a_final [M,C] f32 (lm_head input stays f32)
                        + Mm * FLOAT            // rinv_f  [M]
                        + Mm * C * A            // a [M,C] single checkpoint scratch (bf16)
                        + Mm * C * A            // fbuf [M,C] bf16 FFN checkpoint scratch
                        + 2 * Mm * F * A        // ff1, gact  [M,F] bf16 FFN scratch
                        + Mm * qkv_stride(d) * A // qkv [M,C+2*D_KV] single checkpoint scratch (bf16, recomputed in bwd)
                        + Mm * C * A;           // att [M,C] single checkpoint scratch (bf16, recomputed in bwd)
    // logits [chunk_rows,V]: honours a forced chunk count (Dims::logits_chunks) so the prediction tracks
    // what train_alloc will really allocate -- see that field's comment for the divergence this closes.
    const u64 logits    = logits_chunk_rows(int(V), int(F), Mm, d.logits_chunks) * V * FLOAT;
    const u64 grad      = 2 * Mm * C * FLOAT    // dh, da  [M,C] f32
                        + Mm * C * A            // datt [M,C] bf16
                        + Mm * C * (FLOAT + A)  // dfbuf [M,C] f32 + dh16 [M,C] bf16
                        + Mm * qkv_stride(d) * A // dqkv  [M,C+2*D_KV] bf16
                        + 2 * Mm * F * A        // dff1, dgact  [M,F] bf16
                        + C * qkv_stride(d) * FLOAT  // dwqkv [C,C+2*D_KV] (batch-independent temp)
                        + Mm * INT              // dtargets  [M]
                        + u64(MAX_DEVICE_BATCH) * INT   // lengths [MAX_DEVICE_BATCH] (constant: covers any
                                                        // effective batch the row budget admits, see train_alloc)
                        + u64(MAX_DEVICE_BATCH) * INT   // active [MAX_DEVICE_BATCH] (CE loss-mask normalizer,
                                                        // mirrors lengths -- see ce_backward_kernel)
                        + DBL;                  // loss [1] (double)
    // qk_pre [M,C+D_KV] act_t: the pre-norm Q/K stash QK-norm's backward needs (USE_QK_NORM only). The
    // backward recompute overwrites g_tr.qkv in place with RoPE's output immediately after the QKV
    // GEMM, before qknorm's own backward (which sits BEFORE RoPE in the forward graph) ever runs --
    // by the time it would need the pre-norm values, they no longer survive anywhere else. Zero bytes
    // when qk_norm is off (the common case).
    const u64 qk_pre    = d.qk_norm ? Mm * qk_pre_stride(d) * A : 0;
    // Depth attention: per cache SLOT, a retained [M,D_KV] K and mixed V (act_t) plus their two f32
    // gradient accumulators -- the accumulators are what make this ~1.5x a forward-only estimate, and
    // they cannot be folded into the per-execution checkpoints because a slot is written by executions
    // AFTER its owner and so must stay live across the whole backward. Plus ONE shared pre-mix V (not
    // one per execution: the backward handles one execution at a time). Zero bytes at stride 0.
    const u64 depth = d.depth_slots > 0
        ? u64(d.depth_slots) * Mm * d_kv(d) * (2 * A + 2 * FLOAT) + Mm * d_kv(d) * A
        : 0;
    return ScratchTerms{ .per_exec = EL * per_layer, .final_blk = final_blk, .logits = logits,
                         .grad = grad, .qk_pre = qk_pre, .depth = depth };
}

// Total resident training scratch (train_alloc), grown to `batch` -- the sum of train_scratch_terms above.
constexpr u64 train_scratch_bytes(const Dims& d, int batch, u64 A = FLOAT) {
    return train_scratch_terms(d, batch, A).total();
}

// Forward scratch a training step keeps resident: only the token-id buffer. Training reads from the
// activation-saving g_tr arenas, so fwd_alloc(batch, full=false) skips h..logits -- just dids[M].
constexpr u64 fwd_dids_bytes(const Dims& d, int batch) {
    return u64(batch) * u64(d.seq_len) * INT;   // dids [M]
}

// Total resident device bytes for one training step at `batch` (persistent + dids + train scratch).
// fwd_scratch_bytes is the inference-only path; training carries just dids, not the full forward set.
constexpr u64 train_resident_bytes(const Dims& d, int batch, u64 A = FLOAT) {
    return persistent_bytes(d, A) + fwd_dids_bytes(d, batch) + train_scratch_bytes(d, batch, A);
}

// Footprint in whole MiB, rounded UP -- the same unit GPU_VRAM_MB (from nvidia-smi) is expressed in,
// so a `predicted_mb > GPU_VRAM_MB` comparison is apples-to-apples.
constexpr int train_resident_mb(const Dims& d, int batch, u64 A = FLOAT) {
    const u64 b = train_resident_bytes(d, batch, A);
    return int((b + (u64(1) << 20) - 1) >> 20);
}

// Allowed gap (MiB) between this PREDICTION and the measured device delta when validating the model
// against reality (sub0_cuda_train_footprint). The prediction is an EXACT byte sum; the measured
// delta sits a little off it (per-cudaMalloc rounding over ~dozens of allocations + WDDM free-memory
// noise) -- both bounded and near-batch-independent, so a FIXED absolute slack, not a percentage, is
// the right tolerance. The two directions are NOT symmetric in CONSEQUENCE, so the band is asymmetric:
//   * UNDER-prediction (predicted < measured) is the DANGEROUS drift -- a run trusting the prediction
//     could under-reserve and OOM, or it means a buffer was added to backend_cuda.cu without updating
//     this file -- so it must stay within the tight tolerance below.
//   * OVER-prediction (predicted > measured) is SAFE (reserve more than used -> never OOMs). At large
//     scale (d768-class) the predictor over-estimates by a steady ~140 MiB -- a known, uninvestigated,
//     safe gap (see project memory memplan-vram-prediction-gap-at-scale) -- so the over side gets a
//     wider band that still catches a GROSS "mirror drifted" regression (hundreds of MiB) without
//     failing on that documented offset.
// Either way a genuine buffer add/remove/resize is hundreds of MiB at any non-trivial batch, outside
// both bands, so the check stays sensitive to real breakage.
constexpr double FOOTPRINT_TOLERANCE_MB             = 128.0;   // max safe UNDER-prediction (OOM-risk side)
constexpr double FOOTPRINT_OVERPREDICT_TOLERANCE_MB = 320.0;   // max tolerated OVER-prediction (safe side)

// Largest minibatch whose resident training footprint fits `vram_mb`, bounded by `hard_cap`. Inverts
// train_resident_mb (monotonically increasing in batch) by binary search. Returns hard_cap when VRAM
// is unknown (vram_mb <= 0), or 0 if even batch 1 cannot fit. Lets the GPU tuner size its batch
// ladder to the ACTUAL device at runtime instead of a baked dimension list. act_bytes = 2 bf16 / 4 f32.
constexpr int max_batch_for_vram(const Dims& d, int vram_mb, int hard_cap, u64 act_bytes = FLOAT) {
    if (vram_mb <= 0) return hard_cap;
    int lo = 1, hi = hard_cap, best = 0;
    while (lo <= hi) {
        const int mid = lo + (hi - lo) / 2;
        if (train_resident_mb(d, mid, act_bytes) <= vram_mb) { best = mid; lo = mid + 1; }
        else hi = mid - 1;
    }
    return best;
}

}  // namespace sub0::memplan
