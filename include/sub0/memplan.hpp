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
};

using u64 = unsigned long long;

inline constexpr u64 FLOAT = sizeof(float);
inline constexpr u64 INT   = sizeof(int);
inline constexpr u64 DBL   = sizeof(double);

// Trainable float count -- the exact sum PARAM_LAYOUT produces in layout.hpp, re-derived here
// from dims so the configurator (which cannot include layout.hpp) gets the same number. The CUDA
// footprint test asserts param_floats(dims) == sub0::PARAM_FLOATS, catching any layout drift.
constexpr u64 param_floats(const Dims& d) {
    const u64 C = u64(d.d_model), L = u64(d.n_layers), F = u64(d.d_ff), T = u64(d.seq_len), V = u64(d.vocab);
    const u64 emb   = V * C + T * C;            // tok_emb [V,C] + pos_emb [T,C]
    const u64 block = 2 * C                     // ln1 + ln2          [C] each
                    + 4 * C * C                 // Wq Wk Wv Wo        [C,C]
                    + 2 * C * F + F + C;         // W1 [C,F], b1 [F], W2 [F,C], b2 [C]
    const u64 head  = C + C * V + V;            // ln_f [C] + lm_head [C,V] + lm_bias [V]
    return emb + L * block + head;
}

// Persistent, batch-independent device memory: the param mirror (g_dev_params) + the four
// optimizer arenas (g_dev_grad/m/vel/decay) + the norm accumulator (g_dev_normsq), plus the
// per-layer fused-QKV weights (g_fwd.wqkv, [C,3C] each, built once at upload).
constexpr u64 persistent_bytes(const Dims& d) {
    const u64 C = u64(d.d_model), L = u64(d.n_layers);
    return 5 * param_floats(d) * FLOAT          // g_dev_params + grad + m + vel + decay
         + DBL                                  // g_dev_normsq [1] (double)
         + L * (3 * C * C) * FLOAT;             // g_fwd.wqkv[l] = [C, 3C]
}

// Resident forward scratch (fwd_alloc), grown to `batch`. One term per cudaMalloc in fwd_alloc.
constexpr u64 fwd_scratch_bytes(const Dims& d, int batch) {
    const u64 C = u64(d.d_model), F = u64(d.d_ff), V = u64(d.vocab);
    const u64 Mm = u64(batch) * u64(d.seq_len);
    return Mm * INT                             // dids   [M]
         + 6 * Mm * C * FLOAT                   // h, a, att, proj, fbuf, ff2  [M,C]
         + Mm * 3 * C * FLOAT                   // qkv    [M,3C]
         + 2 * Mm * F * FLOAT                   // ff1, gact  [M,F]
         + Mm * V * FLOAT;                      // logits [M,V]
}

// Resident training scratch (train_alloc), grown to `batch`. One term per cudaMalloc in train_alloc.
constexpr u64 train_scratch_bytes(const Dims& d, int batch) {
    const u64 C = u64(d.d_model), L = u64(d.n_layers), F = u64(d.d_ff);
    const u64 T = u64(d.seq_len), V = u64(d.vocab);
    const u64 Mm = u64(batch) * T;
    const u64 per_layer = 5 * Mm * C * FLOAT    // h_in, a, att, h_mid, fbuf  [M,C]
                        + 2 * Mm * FLOAT        // rinv1, rinv2  [M]
                        + Mm * 3 * C * FLOAT;   // qkv   [M,3C]  (P-free flash: no [batch,H,T,T]; ff1/gact checkpointed)
    const u64 final_blk = 2 * Mm * C * FLOAT    // h_final, a_final  [M,C]
                        + Mm * FLOAT            // rinv_f  [M]
                        + Mm * V * FLOAT        // logits  [M,V]
                        + 2 * Mm * F * FLOAT;   // ff1, gact  [M,F] single checkpoint scratch (recomputed in bwd)
    const u64 grad      = 4 * Mm * C * FLOAT    // dh, da, datt, dfbuf  [M,C]
                        + Mm * 3 * C * FLOAT    // dqkv  [M,3C]
                        + 2 * Mm * F * FLOAT    // dff1, dgact  [M,F]
                        + 3 * C * C * FLOAT     // dwqkv [C,3C] (batch-independent temp)
                        + Mm * INT              // dtargets  [M]
                        + u64(batch) * INT      // lengths   [batch] (per-window padding mask)
                        + DBL;                  // loss [1] (double)
    return L * per_layer + final_blk + grad;
}

// Forward scratch a training step keeps resident: only the token-id buffer. Training reads from the
// activation-saving g_tr arenas, so fwd_alloc(batch, full=false) skips h..logits -- just dids[M].
constexpr u64 fwd_dids_bytes(const Dims& d, int batch) {
    return u64(batch) * u64(d.seq_len) * INT;   // dids [M]
}

// Total resident device bytes for one training step at `batch` (persistent + dids + train scratch).
// fwd_scratch_bytes is the inference-only path; training carries just dids, not the full forward set.
constexpr u64 train_resident_bytes(const Dims& d, int batch) {
    return persistent_bytes(d) + fwd_dids_bytes(d, batch) + train_scratch_bytes(d, batch);
}

// Footprint in whole MiB, rounded UP -- the same unit GPU_VRAM_MB (from nvidia-smi) is expressed in,
// so a `predicted_mb > GPU_VRAM_MB` comparison is apples-to-apples.
constexpr int train_resident_mb(const Dims& d, int batch) {
    const u64 b = train_resident_bytes(d, batch);
    return int((b + (u64(1) << 20) - 1) >> 20);
}

// Allowed gap (MiB) between this PREDICTION and the measured device delta when validating the model
// against reality (sub0_cuda_train_footprint). The prediction is an EXACT byte sum; the measured
// delta sits a little above it for two reasons that are both bounded and (near enough) batch-
// independent, so a FIXED absolute slack -- not a percentage -- is the right tolerance:
//   * per-cudaMalloc rounding: the backend makes ~dozens of allocations (the per-layer scratch
//     scales the count with n_layers), each rounded up to the driver's allocation granularity, for
//     a roughly constant aggregate overhead regardless of batch;
//   * WDDM free-memory noise: cudaMemGetInfo reports whole-device free VRAM, which the desktop
//     compositor and other GPU clients perturb between the two samples.
// Observed gap on an 8 GB laptop is tens of MiB and non-monotonic in batch -- pure offset + noise.
// A genuine "the mirror drifted" regression (a buffer added/removed/resized) is hundreds of MiB at
// any non-trivial batch, far outside this band, so the check stays sensitive to real breakage.
constexpr double FOOTPRINT_TOLERANCE_MB = 128.0;

}  // namespace sub0::memplan
