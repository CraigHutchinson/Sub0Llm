// sub0/device_backend.hpp -- the ONE canonical, backend-NEUTRAL device seam (docs/BACKENDS.md).
//
// Everything a production consumer may ask of a device accelerator, named sub0_dev_* (not sub0_cuda_*):
// the seam must outlive any single vendor. extern "C" + POD types only -- it crosses a DLL boundary
// (sub0_backend_<kind>.dll), same reasoning registry.hpp documents for its own boundary. Selection is
// LINK-TIME (one device backend per build, cmake/Backends.cmake) per the project's compile-time-decision
// philosophy -- no runtime registry, no vtables; "is any device present" stays a compile-time gate
// (SUB0_BUILD_DEVICE), everything finer-grained is a sub0_dev_caps() query made at SETUP time, never in
// a hot loop.
//
// TODAY this is a BRIDGE: under SUB0_BUILD_CUDA the neutral functions are zero-cost inline forwards to
// the existing sub0_cuda_* exports (declared here canonically -- consumers should stop declaring their
// own externs as they migrate); with no device backend they are fail-fast stubs, so migrated consumers
// need no #if scaffolding of their own. Migration order + the symbol flip that eventually renames the
// exports natively: docs/BACKENDS.md "Migration path". Per-backend DIAGNOSTIC symbols (the
// sub0_cuda_*_check parity functions) are deliberately NOT part of this seam -- each backend's parity
// suite is backend-specific by nature (see the doc's "abstraction theater" note).

#pragma once

#include <cstdint>

// Compile-time "is any device backend linked" gate, backend-neutral. Aliases today's CUDA-specific
// macro so existing build plumbing keeps working; a future SUB0_DEVICE=ROCM branch defines
// SUB0_BUILD_DEVICE via its own path.
#if (defined(SUB0_BUILD_CUDA) || defined(SUB0_BUILD_MOCK_DEVICE)) && !defined(SUB0_BUILD_DEVICE)
  #define SUB0_BUILD_DEVICE 1
#endif

// What THIS build's device backend can actually do -- consumers branch on these bits, never on the
// backend's name and never on vendor-specific #ifs (see docs/BACKENDS.md for each field's consumer).
// `name` is a static string ("cuda" | "rocm" | "openvino" | "none"), for logs/telemetry only.
struct Sub0DeviceCaps {
    const char* name;
    int supports_train;             // train_reserve/train_step/backward + device-resident step
    int supports_eval;              // batched forward-only cross-entropy (forward_loss) -- STRICTLY
                                    // weaker than supports_train, and its own bit because a backend
                                    // may serve evaluation without ever implementing a backward
    int supports_decode;            // kv_reset/forward_one
    int supports_interception;      // decode-time expand/combine/compute injection mid-generation
    int supports_binding_compose;   // encode_slot-class composed embeddings (scratch/persistent/sentinel)
                                    // -- the hybrid router keeps binding windows on CPU while this is 0
    int supports_opt_state;         // upload_opt/download_opt round-trip (optimizer moments on-device)
    int supports_tf32;              // the set_tf32 precision hint is meaningful on this hardware
};

// --- Binding-compose override-table wire layout (sub0_dev_set_window_bindings) -----------------
// The host precomputes, per step (or per decode session), WHICH flat positions embed as composed
// rows -- the device never parses binding semantics (docs/BACKENDS.md "Design: binding-compose on
// CUDA"). POD ints only across the seam:
//   override_idx[n_positions] : position m composes entries[override_idx[m]]; -1 = plain lookup.
//                               Training: m = b*T + t (flat batch row). Decode: m = the token's pos.
//   entries                   : n_entries consecutive {frag_offset, frag_len, encoding} int triples
//                               (SUB0_DEV_BIND_ENTRY_INTS per entry).
//   frags[n_frags]            : concatenated fragment token ids, each in [0, VOCAB).
// `encoding` carries sub0::SlotEncoding's underlying value; only the param-free arms below have
// device kernels -- anything else is REJECTED at install (nonzero return, nothing changes), never
// silently mis-composed. Cross-referenced with the kBindEntryInts/kBindEnc* block in
// src/backend_cuda.cu (a static_assert there pins the values; the two headers deliberately don't
// share code -- same convention as scratch_slots.hpp vs registry.hpp's enum-name list).
constexpr int SUB0_DEV_BIND_ENTRY_INTS   = 3;
constexpr int SUB0_DEV_BIND_ENC_MEANPOOL = 0;   // == (int)sub0::SlotEncoding::MeanPool
constexpr int SUB0_DEV_BIND_ENC_HASH     = 2;   // == (int)sub0::SlotEncoding::Hash
constexpr int SUB0_DEV_BIND_ENC_HRR      = 4;   // == (int)sub0::SlotEncoding::HRR

#if defined(SUB0_BUILD_MOCK_DEVICE)

// --- TEST-ONLY backend: an in-process device that is really the CPU engine -------------------------
// Defined ONLY by the sub0_eval_seam_tests target (tests/CMakeLists.txt) and deliberately checked
// BEFORE SUB0_BUILD_CUDA, so a CUDA-enabled tree still builds this target against the mock. It exists
// because the interesting failures in a device consumer are not in the kernels: they are in the
// PLUMBING around the seam -- how windows are batched, how ids and targets are paired (an off-by-one
// there shifts the whole prediction task by one token and still returns a plausible number), how a
// ragged final group is weighted back together. None of that needs a GPU to get wrong, and none of it
// was reachable by a test while the only implementation of the seam required one.
//
// The mock answers sub0_dev_forward_loss by running the real CPU engine over the ids/targets it was
// handed, so "device" and "CPU" must agree EXACTLY when the plumbing is right, and disagree the moment
// it is not. See tests/mock_device_backend.cpp for the implementation and the contract it pins.
extern "C" [[nodiscard]] int  sub0_mock_init();
extern "C" void               sub0_mock_shutdown();
extern "C" [[nodiscard]] int  sub0_mock_upload_params(const float* host);
extern "C" [[nodiscard]] int  sub0_mock_forward_loss(const int* ids, const int* targets, int batch, int T,
                                                     double* out_loss, const int* lengths);
// How many forward_loss calls the mock has served, and the batch sizes it saw -- the test asserts on
// the GROUPING, which is invisible in the returned mean.
extern "C" int                sub0_mock_call_count();
extern "C" int                sub0_mock_batch_at(int i);
extern "C" void               sub0_mock_reset_log();

inline Sub0DeviceCaps sub0_dev_caps() {
    // Honest: this backend evaluates and nothing else. A consumer that branches correctly on caps
    // therefore exercises exactly the eval path here and takes its own fallback for everything else.
    return { "mock", /*train=*/0, /*eval=*/1, /*decode=*/0, /*interception=*/0,
             /*binding_compose=*/0, /*opt_state=*/0, /*tf32=*/0 };
}
[[nodiscard]] inline int  sub0_dev_init()                        { return sub0_mock_init(); }
inline void               sub0_dev_shutdown()                    { sub0_mock_shutdown(); }
[[nodiscard]] inline int  sub0_dev_upload_params(const float* h) { return sub0_mock_upload_params(h); }
[[nodiscard]] inline int  sub0_dev_forward_loss(const int* ids, const int* targets, int batch, int T,
                                                double* out_loss, const int* lengths) {
    return sub0_mock_forward_loss(ids, targets, batch, T, out_loss, lengths);
}
// Everything this backend does not claim in caps: same fail-fast contract as the no-device stubs.
[[nodiscard]] inline int  sub0_dev_download_params(float*)       { return -1; }
[[nodiscard]] inline int  sub0_dev_upload_opt(const float*, const float*) { return -1; }
[[nodiscard]] inline int  sub0_dev_download_opt(float*, float*)  { return -1; }
inline void               sub0_dev_set_tf32(int)                 {}
[[nodiscard]] inline int  sub0_dev_train_reserve(int)            { return -1; }
[[nodiscard]] inline int  sub0_dev_train_step(const int*, const int*, int, int, float, long, double*,
                                              const int*, float)                 { return -1; }
[[nodiscard]] inline int  sub0_dev_backward(const int*, const int*, int, int, float*, double*,
                                            const int*)                          { return -1; }
[[nodiscard]] inline int  sub0_dev_time_train_step(int, int, double, double*)    { return -1; }
[[nodiscard]] inline int  sub0_dev_train_footprint(int, double*, double*)        { return -1; }
inline int                sub0_dev_free_mem_mb()                 { return 0; }
[[nodiscard]] inline int  sub0_dev_kv_reset()                    { return -1; }
[[nodiscard]] inline int  sub0_dev_forward_one(int, int, float*) { return -1; }
[[nodiscard]] inline int  sub0_dev_set_window_bindings(const int*, int, const int*, int,
                                                       const int*, int)          { return -1; }

#elif defined(SUB0_BUILD_CUDA)

// --- The current CUDA backend's exports (canonical declarations; consumers: prefer sub0_dev_*) -------
extern "C" [[nodiscard]] int  sub0_cuda_init();
extern "C" void               sub0_cuda_shutdown();
extern "C" [[nodiscard]] int  sub0_cuda_upload_params(const float* host);
extern "C" [[nodiscard]] int  sub0_cuda_download_params(float* host);
extern "C" [[nodiscard]] int  sub0_cuda_upload_opt(const float* host_m, const float* host_v);
extern "C" [[nodiscard]] int  sub0_cuda_download_opt(float* host_m, float* host_v);
extern "C" void               sub0_cuda_set_tf32(int on);
extern "C" [[nodiscard]] int  sub0_cuda_train_reserve(int batch);
extern "C" [[nodiscard]] int  sub0_cuda_train_step(const int* ids, const int* targets, int batch, int T,
                                                   float lr, long t, double* out_loss, const int* lengths,
                                                   float muon_lr);
extern "C" [[nodiscard]] int  sub0_cuda_backward(const int* ids, const int* targets, int batch, int T,
                                                 float* out_grad, double* out_loss, const int* lengths);
extern "C" [[nodiscard]] int  sub0_cuda_forward_loss(const int* ids, const int* targets, int batch, int T,
                                                     double* out_loss, const int* lengths);
extern "C" [[nodiscard]] int  sub0_cuda_time_train_step(int batch, int T, double budget_ms, double* out_ms);
extern "C" [[nodiscard]] int  sub0_cuda_train_footprint(int batch, double* predicted_mb, double* actual_mb);
extern "C" int                sub0_cuda_free_vram_mb();
extern "C" [[nodiscard]] int  sub0_cuda_kv_reset();
extern "C" [[nodiscard]] int  sub0_cuda_forward_one(int id, int pos, float* out_logits);
extern "C" [[nodiscard]] int  sub0_cuda_set_window_bindings(const int* override_idx, int n_positions,
                                                            const int* entries, int n_entries,
                                                            const int* frags, int n_frags);

// --- The neutral seam: zero-cost forwards over the CUDA exports (the bridge, docs/BACKENDS.md) -------
inline Sub0DeviceCaps sub0_dev_caps() {
    // The TRUTHFUL current capability set: CUDA trains + decodes with device-resident opt state and
    // TF32 control, and (2026-07-17, the first caps flip -- docs/BACKENDS.md "Design: binding-compose
    // on CUDA") composes binding-override embed rows on device for the param-free encoder arms
    // (MeanPool/Hash/HRR; sub0_dev_set_window_bindings REJECTS anything else at install). Flipped
    // only after the four-gate parity suite (per-arm forward + backward differentials vs
    // encode_slot/encode_slot_bwd, inertness, end-to-end binding-heavy train-step differential) ran
    // green on real hardware -- see cuda_tests.cpp's "binding-compose" cases. No interception path
    // yet. The hybrid router's consuming flip (needs_cpu &= !supports_binding_compose) is the
    // documented follow-up; nothing routes differently until train_stage.cpp adopts it.
    //
    // supports_eval (2026-07-29): sub0_cuda_forward_loss runs the batched forward-only cross-entropy
    // on device. Set only after the CPU/GPU nelbo parity gate in cuda_tests.cpp ("forward_loss ==
    // CPU evaluate") ran green -- an eval that silently disagreed with the CPU number would corrupt
    // every A/B comparison scored with it, and a mean nelbo is exactly where a small systematic
    // difference hides.
    // supports_decode is CONDITIONAL (2026-07-30): the single-token decode path does not implement
    // depth attention (backend_cuda.cu's forward_one_device -- training and batched inference do), so a
    // depth-attention build must report 0 and let every decode consumer degrade to the CPU. Reporting a
    // capability the backend does not have is the one thing this struct exists to prevent; it is also
    // what `report`'s sample battery would otherwise trip over mid-run, not just an explicit `gen`.
    return { "cuda", /*train=*/1, /*eval=*/1, /*decode=*/!sub0::USE_DEPTH_ATTN, /*interception=*/0,
             /*binding_compose=*/1, /*opt_state=*/1, /*tf32=*/1 };
}
[[nodiscard]] inline int  sub0_dev_init()                       { return sub0_cuda_init(); }
inline void               sub0_dev_shutdown()                   { sub0_cuda_shutdown(); }
[[nodiscard]] inline int  sub0_dev_upload_params(const float* h){ return sub0_cuda_upload_params(h); }
[[nodiscard]] inline int  sub0_dev_download_params(float* h)    { return sub0_cuda_download_params(h); }
[[nodiscard]] inline int  sub0_dev_upload_opt(const float* m, const float* v) { return sub0_cuda_upload_opt(m, v); }
[[nodiscard]] inline int  sub0_dev_download_opt(float* m, float* v)           { return sub0_cuda_download_opt(m, v); }
inline void               sub0_dev_set_tf32(int on)             { sub0_cuda_set_tf32(on); }
[[nodiscard]] inline int  sub0_dev_train_reserve(int batch)     { return sub0_cuda_train_reserve(batch); }
[[nodiscard]] inline int  sub0_dev_train_step(const int* ids, const int* targets, int batch, int T,
                                              float lr, long t, double* out_loss, const int* lengths,
                                              float muon_lr) {
    return sub0_cuda_train_step(ids, targets, batch, T, lr, t, out_loss, lengths, muon_lr);
}
[[nodiscard]] inline int  sub0_dev_backward(const int* ids, const int* targets, int batch, int T,
                                            float* out_grad, double* out_loss, const int* lengths) {
    return sub0_cuda_backward(ids, targets, batch, T, out_grad, out_loss, lengths);
}
// Mean cross-entropy over `batch` windows of `T` tokens, forward only -- the evaluation counterpart
// of sub0_dev_backward, computing the SAME loss without the backward half or its gradient buffers.
// `lengths` (optional, [batch]) marks each window's real length so padded tails are not graded;
// targets < 0 are excluded from both the loss and its per-window normalizer. Gated by
// sub0_dev_caps().supports_eval.
[[nodiscard]] inline int  sub0_dev_forward_loss(const int* ids, const int* targets, int batch, int T,
                                                double* out_loss, const int* lengths) {
    return sub0_cuda_forward_loss(ids, targets, batch, T, out_loss, lengths);
}
[[nodiscard]] inline int  sub0_dev_time_train_step(int batch, int T, double budget_ms, double* out_ms) {
    return sub0_cuda_time_train_step(batch, T, budget_ms, out_ms);
}
[[nodiscard]] inline int  sub0_dev_train_footprint(int batch, double* predicted_mb, double* actual_mb) {
    return sub0_cuda_train_footprint(batch, predicted_mb, actual_mb);
}
inline int                sub0_dev_free_mem_mb()                { return sub0_cuda_free_vram_mb(); }
[[nodiscard]] inline int  sub0_dev_kv_reset()                   { return sub0_cuda_kv_reset(); }
[[nodiscard]] inline int  sub0_dev_forward_one(int id, int pos, float* out_logits) {
    return sub0_cuda_forward_one(id, pos, out_logits);
}
// Install (or clear: null/0) the binding-compose override table -- see the wire-layout block above.
// Rejects (nonzero) any entry whose encoding has no device kernel; clear always succeeds. The table
// stays installed until cleared/replaced (the trainer clears after each step).
[[nodiscard]] inline int  sub0_dev_set_window_bindings(const int* override_idx, int n_positions,
                                                       const int* entries, int n_entries,
                                                       const int* frags, int n_frags) {
    return sub0_cuda_set_window_bindings(override_idx, n_positions, entries, n_entries, frags, n_frags);
}

#else  // no device backend in this build ------------------------------------------------------------

// Fail-fast stubs: a migrated consumer needs no #if of its own -- init reports failure, caps report
// "none"/zeros, and everything else is unreachable-by-contract (callers gate on init/caps first).
inline Sub0DeviceCaps sub0_dev_caps()                            { return { "none", 0, 0, 0, 0, 0, 0, 0 }; }
[[nodiscard]] inline int  sub0_dev_init()                        { return -1; }
inline void               sub0_dev_shutdown()                    {}
[[nodiscard]] inline int  sub0_dev_upload_params(const float*)   { return -1; }
[[nodiscard]] inline int  sub0_dev_download_params(float*)       { return -1; }
[[nodiscard]] inline int  sub0_dev_upload_opt(const float*, const float*) { return -1; }
[[nodiscard]] inline int  sub0_dev_download_opt(float*, float*)  { return -1; }
inline void               sub0_dev_set_tf32(int)                 {}
[[nodiscard]] inline int  sub0_dev_train_reserve(int)            { return -1; }
[[nodiscard]] inline int  sub0_dev_train_step(const int*, const int*, int, int, float, long, double*,
                                              const int*, float)                 { return -1; }
[[nodiscard]] inline int  sub0_dev_backward(const int*, const int*, int, int, float*, double*,
                                            const int*)                          { return -1; }
[[nodiscard]] inline int  sub0_dev_forward_loss(const int*, const int*, int, int, double*,
                                                const int*)                      { return -1; }
[[nodiscard]] inline int  sub0_dev_time_train_step(int, int, double, double*)    { return -1; }
[[nodiscard]] inline int  sub0_dev_train_footprint(int, double*, double*)        { return -1; }
inline int                sub0_dev_free_mem_mb()                 { return 0; }
[[nodiscard]] inline int  sub0_dev_kv_reset()                    { return -1; }
[[nodiscard]] inline int  sub0_dev_forward_one(int, int, float*) { return -1; }
[[nodiscard]] inline int  sub0_dev_set_window_bindings(const int*, int, const int*, int,
                                                       const int*, int)          { return -1; }

#endif
