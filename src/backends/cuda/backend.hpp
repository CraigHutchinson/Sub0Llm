#pragma once
// CUDA backend interface — included by ops.cpp regardless of build config
// so the dispatch code compiles cleanly; the function bodies are no-ops /
// throw when CUDA is not compiled in.
#include "sub0llm/core/tensor.hpp"
#include "sub0llm/backends/cpu/quant.hpp"   // BlockQ8_0

#include <memory>
#include <vector>

namespace sub0llm::backend::cuda {

// Allocate device memory and return a Storage with a cuda-free deleter.
// Throws if CUDA is not compiled in.
[[nodiscard]] std::shared_ptr<Storage> alloc(std::size_t byte_size, int device_index);

// Copy between host and device (or device to device). All throw on CUDA error.
void memcpy_h2d(void* dst, const void* src, std::size_t bytes, int device_index);
void memcpy_d2h(void* dst, const void* src, std::size_t bytes, int device_index);
void memcpy_d2d(void* dst, const void* src, std::size_t bytes, int device_index);
void memset_zero(void* dst, std::size_t bytes, int device_index);

// ── Ops (each throws if not compiled in) ─────────────────────────────────────
[[nodiscard]] Tensor add(const Tensor& a, const Tensor& b);
[[nodiscard]] Tensor mul(const Tensor& a, const Tensor& b);
[[nodiscard]] Tensor relu(const Tensor& a);
[[nodiscard]] Tensor matmul(const Tensor& a, const Tensor& b);
// C = Aᵀ·B for 2D A(M,K), B(M,N) → C(K,N). The weight-gradient op; caller (ops::matmul_tb)
// validates shapes. Device-resident in/out.
[[nodiscard]] Tensor matmul_tb(const Tensor& a, const Tensor& b);
// Row-wise softmax (dim=-1) on a device-resident 1D/2D f32 tensor. The caller (ops::softmax)
// validates shape/dtype; this allocates the device output and launches the kernel.
[[nodiscard]] Tensor softmax(const Tensor& t, int dim);

// rms_norm training fwd+bwd on DEVICE pointers (T rows × D), signatures mirroring
// backend::cpu::rms_norm_*_f32 so autograd::rms_norm can dispatch by device. Throw if not built.
void rms_norm_fwd(const float* x, const float* w, float* x_norm, float* inv_rms,
                  float* out, int T, int D, float eps);
void rms_norm_bwd_x(const float* g, const float* x_norm, const float* inv_rms,
                    const float* w, float* gx, int T, int D);
void rms_norm_bwd_w(const float* g, const float* x_norm, float* gw, int T, int D);

// rope (half-split) fwd+bwd on DEVICE pointers — x/out/g/gx are (T,Dh), cos/sin are (T,Dh/2).
// Match autograd::rope. Throw if not built.
void rope_fwd(const float* x, const float* cos, const float* sin, float* out, int T, int Dh);
void rope_bwd(const float* g, const float* cos, const float* sin, float* gx, int T, int Dh);

// Benchmark the row-wise softmax KERNEL in isolation (no per-iter device alloc): H2D the
// (rows×cols) host input once, run `reps` timed launches (GPU events, after warm-up), D2H the
// last result into `host_out`. Returns total GPU kernel time (seconds) for `reps` launches, so
// the caller can compute throughput and compare host_out to the CPU reference. Mirrors
// matmul_q8_0_bench. Throws if CUDA is not compiled in.
[[nodiscard]] double softmax_rows_bench(const float* host_in, float* host_out,
                                        int rows, int cols, int reps);

// Benchmark the rms_norm FORWARD kernel in isolation (T×D), same idiom as softmax_rows_bench:
// H2D x/w once, preallocate the x_norm/inv_rms scratch, run `reps` timed launches, D2H `out`.
// Returns total GPU kernel seconds. Throws if CUDA is not compiled in.
[[nodiscard]] double rms_norm_fwd_bench(const float* host_x, const float* host_w,
                                        float* host_out, int T, int D, float eps, int reps);

// Benchmark the matmul_tb kernel (C = Aᵀ·B) in isolation: H2D A(M,K)/B(M,N) once, run `reps`
// timed launches, D2H C(K,N). Returns total GPU kernel seconds. Throws if CUDA not compiled in.
[[nodiscard]] double matmul_tb_bench(const float* host_a, const float* host_b, float* host_c,
                                     int M, int N, int K, int reps);

// Benchmark the rope FORWARD kernel (T×Dh, cos/sin T×Dh/2) in isolation. Same idiom as the
// other *_bench. Returns total GPU kernel seconds. Throws if CUDA is not compiled in.
[[nodiscard]] double rope_fwd_bench(const float* host_x, const float* host_cos,
                                    const float* host_sin, float* host_out,
                                    int T, int Dh, int reps);

// Benchmark + validate the device Q8 matmul end-to-end: H2D-copy host Wq[M,K/32] and
// Xq[T,K/32], run `reps` timed kernel launches (GPU events, after a warm-up), D2H-copy the
// last result into Y[M,T]. Returns total GPU kernel time (seconds) for `reps` launches.
// `variant`: 0 = dp4a (CUDA cores), 1 = IMMA (int8 tensor cores). Throws if CUDA is not
// compiled in. Lets the CPU-side caller (qbench) compare Y to the CPU reference and compute
// GFLOP/s without touching the CUDA API itself.
[[nodiscard]] double matmul_q8_0_bench(const cpu::BlockQ8_0* Wq, const cpu::BlockQ8_0* Xq,
                                       float* Y, int M, int K, int T, int reps, int variant);

// ── Layer sub-kernel validation (host in/out: H2D → launch → D2H) ────────────────────────
// Thin wrappers over the kernels::launch_* layer ops. Each runs the GPU kernel on host data
// and copies the result back, so qbench can compare it to the CPU reference (relRMS) before
// the kernels are wired into the device-resident single-layer forward. Throw if CUDA absent.
void rmsnorm_dev(const float* x, const float* w, float* y, int n, float eps);   // w=nullptr → no weight
void rope_neox_dev(const float* x_in, float* x_out, int dh, int pos, float base, const float* ff); // ff=nullptr
void geglu_dev(const float* gate, const float* up, float* out, int n);
void flash_attn_decode_dev(const float* q, const float* K, const float* V, float* o,
                           int dh, int kvlen);
void quantize_q8_dev(const float* x, cpu::BlockQ8_0* y, int n);   // n = row width (n/32 blocks)

// Validate batched causal PREFILL attention (H2D→launch→D2H). Q[T·nH·dh], f32 KV cache
// kcache/vcache[nKV·max_pos·dh] (filled for the T positions), out[T·nH·dh]. group=nH/nKV.
void flash_attn_prefill_dev(const float* Q, const float* kcache, const float* vcache, float* out,
                            int T, int nH, int nKV, int dh, int window, int max_pos);

// ── On-device single Gemma layer (Stage 2) ──────────────────────────────────────────────
// One transformer layer's weights + shape as HOST pointers (the GPU forward uploads them).
// Q8 weights are row-major (out,in) exactly as GemmaLayer stores them; norms are f32. This is
// the validation form (in-call upload); the Stage-3 perf form keeps everything device-resident.
struct GpuLayerDesc {
    int   D = 0, d_ff = 0, dh = 0, n_head = 0, n_kv_head = 0, window = 0;
    float eps = 1e-6f, rope_base = 1e4f, out_scale = 1.0f;
    bool  has_wv = true;
    const cpu::BlockQ8_0 *wq = nullptr, *wk = nullptr, *wv = nullptr, *wo = nullptr;
    const cpu::BlockQ8_0 *gate = nullptr, *up = nullptr, *down = nullptr;
    const float *attn_norm = nullptr, *post_attn_norm = nullptr, *ffn_norm = nullptr;
    const float *post_ffw_norm = nullptr, *q_norm = nullptr, *k_norm = nullptr;
    const float *rope_freqs = nullptr;   // global freq_factors (dh/2) or nullptr
};

// Run ONE decode step of a single Gemma layer on the GPU, matching GemmaModel::forward_one's
// per-layer body exactly. `x` is the f32 layer input (D). `kcache`/`vcache` are the host-side
// KV caches laid out [kv_head][max_pos][dh]; positions [0,pos) must already hold the (RoPE'd K /
// normed V) for this layer, and this call writes position `pos` into both before attending over
// the window. Writes the layer output (D) to `out`. Uploads weights+caches, runs the device
// kernel pipeline, downloads the result. Throws if CUDA is not compiled in.
void gemma_layer_decode_dev(const GpuLayerDesc& L, const float* x, int pos,
                            float* kcache, float* vcache, int max_pos, float* out);

// Persistent device-resident run of a contiguous block of Gemma layers (Stage 3 perf form):
// uploads every layer's weights ONCE at construction, holds a device KV cache + reused scratch,
// and on each decode keeps activations device-resident across all layers (no per-token weight
// re-upload). Numerically identical to chaining gemma_layer_decode_dev layer-by-layer — same
// kernels, same order. Construction throws if CUDA is not compiled in.
class GemmaGpuLayers {
public:
    // kv_q8: q8 KV cache (~3.8× less KV memory/bandwidth — long context) vs f32 (greedy-exact vs
    // CPU). q8 is LOSSY (~1% per value) and shifts Gemma's scale=1.0 attention enough to change
    // greedy tokens, so it is off by default (the parity-exact path).
    GemmaGpuLayers(const std::vector<GpuLayerDesc>& layers, int max_pos, bool kv_q8 = false);
    ~GemmaGpuLayers();
    GemmaGpuLayers(const GemmaGpuLayers&) = delete;
    GemmaGpuLayers& operator=(const GemmaGpuLayers&) = delete;
    // Decode one token at `pos`: x (D, host f32) → out (D, host f32), updating the device KV
    // cache. `out` is the activation after the LAST uploaded layer (feeds the CPU layers / head).
    void decode(const float* x, int pos, float* out);

    // BATCHED prefill of T tokens through all uploaded layers (the compute-bound IMMA path): x
    // (T·D, host f32) → out (T·D, host f32, the activation after the last GPU layer per token),
    // filling the device KV cache at [start_pos, start_pos+T) so decode() can continue from it.
    // Requires the default f32 KV (throws if kv_q8). Numerically ~equal to T sequential decode()s.
    void prefill(const float* x, int T, int start_pos, float* out);
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace sub0llm::backend::cuda
