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
#if defined(SUB0_BUILD_CUDA) && !defined(SUB0_BUILD_DEVICE)
  #define SUB0_BUILD_DEVICE 1
#endif

// What THIS build's device backend can actually do -- consumers branch on these bits, never on the
// backend's name and never on vendor-specific #ifs (see docs/BACKENDS.md for each field's consumer).
// `name` is a static string ("cuda" | "rocm" | "openvino" | "none"), for logs/telemetry only.
struct Sub0DeviceCaps {
    const char* name;
    int supports_train;             // train_reserve/train_step/backward + device-resident step
    int supports_decode;            // kv_reset/forward_one
    int supports_interception;      // decode-time expand/combine/compute injection mid-generation
    int supports_binding_compose;   // encode_slot-class composed embeddings (scratch/persistent/sentinel)
                                    // -- the hybrid router keeps binding windows on CPU while this is 0
    int supports_opt_state;         // upload_opt/download_opt round-trip (optimizer moments on-device)
    int supports_tf32;              // the set_tf32 precision hint is meaningful on this hardware
};

#if defined(SUB0_BUILD_CUDA)

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
extern "C" [[nodiscard]] int  sub0_cuda_time_train_step(int batch, int T, double budget_ms, double* out_ms);
extern "C" [[nodiscard]] int  sub0_cuda_train_footprint(int batch, double* predicted_mb, double* actual_mb);
extern "C" int                sub0_cuda_free_vram_mb();
extern "C" [[nodiscard]] int  sub0_cuda_kv_reset();
extern "C" [[nodiscard]] int  sub0_cuda_forward_one(int id, int pos, float* out_logits);

// --- The neutral seam: zero-cost forwards over the CUDA exports (the bridge, docs/BACKENDS.md) -------
inline Sub0DeviceCaps sub0_dev_caps() {
    // The TRUTHFUL current capability set: CUDA trains + decodes with device-resident opt state and
    // TF32 control, but has no interception path and no binding-compose kernels (those windows route to
    // CPU under the hybrid split until kernels exist and this flips -- the designed landing zone for
    // GPU-side scratch/sentinel work).
    return { "cuda", 1, 1, 0, 0, 1, 1 };
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

#else  // no device backend in this build ------------------------------------------------------------

// Fail-fast stubs: a migrated consumer needs no #if of its own -- init reports failure, caps report
// "none"/zeros, and everything else is unreachable-by-contract (callers gate on init/caps first).
inline Sub0DeviceCaps sub0_dev_caps()                            { return { "none", 0, 0, 0, 0, 0, 0 }; }
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
[[nodiscard]] inline int  sub0_dev_time_train_step(int, int, double, double*)    { return -1; }
[[nodiscard]] inline int  sub0_dev_train_footprint(int, double*, double*)        { return -1; }
inline int                sub0_dev_free_mem_mb()                 { return 0; }
[[nodiscard]] inline int  sub0_dev_kv_reset()                    { return -1; }
[[nodiscard]] inline int  sub0_dev_forward_one(int, int, float*) { return -1; }

#endif
