// backend_cuda.cu — CUDA compute backend skeleton (part of the GPU build).
//
// Compiled by nvcc (CUDA 13, sm_120) on the multi-compiler path; the C++ engine stays
// on clang and meets this translation unit only at the extern "C" boundary. This is the
// file that grows into the full GPU backend (forward / backward / train_batch / AdamW as
// batched device kernels + CUDA-graph-captured step), eventually replacing backend_cpu.cpp
// when SUB0_COMPUTE=GPU.
//
// Phase 2a (this file's current scope): prove the nvcc build + device execution + the
// host<->device parameter mirror that the full backend relies on. The differentiable
// kernels land in Phase 2b/2c. See include/sub0/core.hpp for the API to implement and
// the memory plan in the repo notes (VRAM = hard fast budget, +shared = soft ceiling).

#include "sub0/layout.hpp"   // PARAM_FLOATS + model dims + COMPUTE_MODE / CUDA_ARCH / HAS_CUDA
#include "sub0/knob.hpp"     // Knob<T,Baked>: compile-time-baked / runtime-tunable knobs
#include "sub0/bench.hpp"    // adaptive_time: budget-sized measurement shared with the CPU tuner
#include "sub0/memplan.hpp"  // train_resident_bytes: the predicted footprint this file is validated against
#include "sub0/muon.hpp"     // sub0::muon::scale_factor -- the GPU Newton-Schulz math itself is
                              // reimplemented below via cuBLAS (see muon_newton_schulz_device);
                              // only the tiny host-side post-orthogonalization scale is shared
#include "sub0/scratch_slots.hpp"  // binding-compose (docs/BACKENDS.md "first caps flip"): SlotEncoding's
                              // wire values, HASH_ROPE_THETA/HRR_MAX_POS, the HOST-built hrr_role_table
                              // the device copy is uploaded from, and encode_slot/encode_slot_bwd as the
                              // in-process CPU reference for sub0_cuda_binding_compose_check. The header
                              // is deliberately dependency-light so the backend may include it (its own
                              // top comment documents exactly this consumer).

#include <algorithm>
#include <cstdio>
#include <vector>
#include <cmath>
#include <type_traits>

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cublas_v2.h>

// Export macro for the CUDA backend's C-ABI entry points (the clang-built host links
// against the import lib, so the symbols must leave the DLL). The full backend will
// reuse SUB0_API from core.hpp for the engine API; this covers the extra self-test hook.
#if defined(_WIN32)
  #define SUB0_CUDA_API extern "C" __declspec(dllexport)
#else
  #define SUB0_CUDA_API extern "C" __attribute__((visibility("default")))
#endif

// The GPU backend is dense-FP only for now; ternary/BitNet stays on the CPU backend. A
// GPU build with ternary weights would silently ignore the quantization -> hard stop
// here (mirrors the CMake FATAL_ERROR guard in cmake/Backends.cmake).
// TODO(ternary-gpu): implement absmean re-quant + straight-through estimator on device
// (mirror ternarize_into() in src/backend_cpu.cpp), then lift this guard.
static_assert(!USE_TERNARY,
    "the CUDA backend is dense-FP only; ternary/BitNet is CPU-only for now (TODO(ternary-gpu)).");

// Depth attention (DEPTH_ATTN_STRIDE > 0) is CPU-only as of Stage 1 -- see docs/DEPTH_ATTENTION.md 6.
// This is a HARD STOP rather than a silent no-op on purpose: the op adds no parameters and rewrites only
// V, so a GPU build that skipped it would train and score a DIFFERENT architecture while every shape
// check, every checkpoint field and PARAM_FLOATS all still agreed. That is precisely the class of
// silent-divergence bug ARCH_FINGERPRINT was extended to catch at load time; refusing to compile catches
// it one step earlier. Lift this when Stage 2 lands the cross-execution dK/dV accumulation.
// TODO(depth-attn-gpu): implement op_depth_attn's forward + backward on device, then remove this guard.
static_assert(!USE_DEPTH_ATTN,
    "depth attention is CPU-only until docs/DEPTH_ATTENTION.md Stage 2 lands the CUDA cross-execution "
    "dK/dV accumulation; configure with -DSUB0_COMPUTE=CPU or --depth-attn-stride 0.");

// SwiGLU-gated FFN (USE_GATED_FFN, for importing GGUF/Llama-family weights): GPU support landed.
// swiglu_kernel/swiglu_act_kernel/swiglu_backward_act_kernel below implement op_swiglu's forward/
// backward exactly (see backend_cpu.cpp), `if constexpr (USE_GATED_FFN)`-gated at each forward/
// forward_one/backward call site, mirroring backend_cpu.cpp's Model struct. The per-layer PARAM_LAYOUT
// offsets this file indexes are resolved through the named kWg/kW1/kW2/etc. constants (see "per-layer
// parameter slot offsets" below), which now bake the gated 9/11-slot layer shape (Wg,W1,W2, no
// b1/b2) alongside the plain 10/12-slot one. The backward's checkpoint-recompute (fbuf/ff1/gact are
// NOT saved per-layer, see forward_train's own comment) means the gated backward recomputes THREE
// intermediates (gate_pre, up_pre, and their SwiGLU combination) by reusing the plain path's existing
// ff1/gact/dff1/dgact `[M,F]` scratch buffers under role-remapping rather than adding new ones -- see
// backward_device's gated FFN branch for the exact sequence and buffer roles at each step. The
// configurator's --gated-ffn/--compute refusal has been lifted to match. (No static_assert left for
// this axis -- ternary above remains CPU-only and keeps its guard.)

// Tied embeddings (USE_TIED_EMBEDDINGS): the LM head reuses tok_emb (transposed) instead of its own
// matrix+bias -- see op_tied_head in backend_cpu.cpp for the reference semantics. Every forward/
// forward_one/backward path below now branches on USE_TIED_EMBEDDINGS with `if constexpr`, mirroring
// backend_cpu.cpp's Model struct: the tied-head forward is one new GEMM shape (launch_tied_head,
// tok_emb read transposed via CUBLAS_OP_T -- cuBLAS already supports this operand natively, same
// mechanism as launch_linear_bwd's existing dX GEMM), and the backward has TWO accumulating
// contributions into tok_emb's grad (the tied head's own dtable = dY^T @ a_final, reusing
// launch_linear_bwd's dW GEMM shape with accumulate=true, plus the pre-existing embedding-lookup
// scatter-add) -- see launch_tied_head_bwd below. The configurator's --tie-embeddings/--compute
// refusal has been lifted to match. (No static_assert left for this axis -- ternary above remains
// CPU-only and keeps its guard; this one is now a real, tested GPU capability.)

// QK-norm (USE_QK_NORM): per-head RMSNorm on Q/K right after their projection, before RoPE (see
// op_qknorm in backend_cpu.cpp for the reference semantics). GPU support landed: qknorm_act_kernel /
// qknorm_save_act_kernel / qknorm_backward_act_kernel below, `if constexpr (USE_QK_NORM)`-gated at
// each forward/backward call site, mirroring backend_cpu.cpp's Model struct. The per-layer
// PARAM_LAYOUT offsets this file indexes are resolved through the named kQNorm/kKNorm/kW1/etc.
// constants (see "per-layer parameter slot offsets" below) rather than the old fixed "+5 for Wo,
// +6 for W1" arithmetic, so the q_norm/k_norm slots inserted between Wo and the FFN block shift
// every later offset correctly from a single source of truth. The configurator's --qk-norm/--compute
// refusal has been lifted to match. (No static_assert left for this axis -- ternary above remains
// CPU-only and keeps its guard.)


namespace {

// Saved-activation element type. ACT_DTYPE selects the storage precision (configurator-baked):
// F32 today; BF16 halves the per-token train scratch on sm_80+ once the kernels are templated.
// FP32 accumulate is preserved (cuBLAS computeType / kernel math both stay float regardless).
using act_t = std::conditional_t<ACT_DTYPE == Dtype::BF16, __nv_bfloat16, float>;
[[maybe_unused]] __device__ inline float  to_f32(float v)         { return v; }
[[maybe_unused]] __device__ inline float  to_f32(__nv_bfloat16 v) { return __bfloat162float(v); }
[[maybe_unused]] __device__ inline void   st_act(float* p, float v)         { *p = v; }
[[maybe_unused]] __device__ inline void   st_act(__nv_bfloat16* p, float v) { *p = __float2bfloat16(v); }
[[maybe_unused]] __device__ inline void   atomicAdd_act(float* p, float v)         { atomicAdd(p, v); }
// bf16 atomic-add fallback: read-modify-write (no native bf16 atomic). Only the query-scheme attn
// backward uses it; the head scheme (bf16 default) is atomic-free so contention is not exercised.
[[maybe_unused]] __device__ inline void   atomicAdd_act(__nv_bfloat16* p, float v) { *p = __float2bfloat16(__bfloat162float(*p) + v); }

// Block-wide sum reduction, returned to EVERY thread (broadcast): warp-shuffle butterfly within each
// warp, then one more shuffle-reduce across per-warp partials via shared memory. BLOCK must be a
// compile-time power-of-two multiple of 32 (every caller launches with a fixed, template-known block
// size). The trailing __syncthreads() guards a second call reusing warp_sums[] in the same kernel
// (e.g. ce_backward's max-then-sum-of-exp) against a thread racing ahead into the next reduction's
// writes before every thread has finished reading this one's broadcast value.
template <int BLOCK>
[[maybe_unused]] __device__ inline float block_reduce_sum(float val) {
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) val += __shfl_down_sync(0xffffffffu, val, off);
    if constexpr (BLOCK > 32) {
        __shared__ float warp_sums[BLOCK / 32];
        const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
        if (lane == 0) warp_sums[wid] = val;
        __syncthreads();
        if (wid == 0) {
            val = (lane < BLOCK / 32) ? warp_sums[lane] : 0.f;
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1) val += __shfl_down_sync(0xffffffffu, val, off);
            if (lane == 0) warp_sums[0] = val;
        }
        __syncthreads();
        val = warp_sums[0];
        __syncthreads();
    } else {
        val = __shfl_sync(0xffffffffu, val, 0);
    }
    return val;
}
// Same, but returns the block-wide MAX (used by ce_backward's softmax-stabilizing row max).
template <int BLOCK>
[[maybe_unused]] __device__ inline float block_reduce_max(float val) {
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) val = fmaxf(val, __shfl_down_sync(0xffffffffu, val, off));
    if constexpr (BLOCK > 32) {
        __shared__ float warp_maxs[BLOCK / 32];
        const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
        if (lane == 0) warp_maxs[wid] = val;
        __syncthreads();
        if (wid == 0) {
            val = (lane < BLOCK / 32) ? warp_maxs[lane] : -1e30f;
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1) val = fmaxf(val, __shfl_down_sync(0xffffffffu, val, off));
            if (lane == 0) warp_maxs[0] = val;
        }
        __syncthreads();
        val = warp_maxs[0];
        __syncthreads();
    } else {
        val = __shfl_sync(0xffffffffu, val, 0);
    }
    return val;
}
// Down-cast a float buffer into the act_t store type (bf16 weight mirrors + dh16). F32: plain copy.
__global__ void f32_to_act_kernel(const float* __restrict__ x, act_t* __restrict__ y, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) st_act(&y[i], x[i]);
}
// Up-cast the act_t store type back to float (used by the attention naive-vs-tiled self-check to
// bring bf16 device buffers to the host for comparison; F32 build == plain copy).
__global__ void act_to_f32_kernel(const act_t* __restrict__ x, float* __restrict__ y, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = to_f32(x[i]);
}

// CUDA error check: report file:line + the error string and bail out of the calling function with a
// nonzero code. `sub0_cuda_train_step`'s own trailing sync+cudaGetLastError use this to surface any
// async failure from the kernels launched inside it (including the void-returning helpers below,
// whose own errors are stream-ordered and only actually detected here); `GpuTrainer::step()`
// (train_stage.cpp) checks THAT return code and stops the training loop on a device fault instead of
// continuing to train on a corrupted context -- see [[gpu-illegal-access-hardware-fault-not-code-bug]]
// for the real incident this was missing for.
#define SUB0_CUDA_CHECK(expr)                                                          \
    do {                                                                               \
        const cudaError_t _err = (expr);                                              \
        if (_err != cudaSuccess) {                                                     \
            std::fprintf(stderr, "cuda error: %s at %s:%d -> %s\n", #expr, __FILE__,   \
                         __LINE__, cudaGetErrorString(_err));                          \
            return 1;                                                                  \
        }                                                                              \
    } while (0)

// Same check for void-returning helpers (the training forward/backward): report and bail out
// (return;) instead of a code. Stream-ordered failures still surface on the next sync.
#define SUB0_CUDA_CHECK_VOID(expr)                                                     \
    do {                                                                               \
        const cudaError_t _err = (expr);                                              \
        if (_err != cudaSuccess) {                                                     \
            std::fprintf(stderr, "cuda error: %s at %s:%d -> %s\n", #expr, __FILE__,   \
                         __LINE__, cudaGetErrorString(_err));                          \
            return;                                                                    \
        }                                                                              \
    } while (0)

// Trivial device kernel: y = a*x + y over n elements (the smoke-test workload).
__global__ void axpy_kernel(float a, const float* __restrict__ x, float* __restrict__ y, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = a * x[i] + y[i];
}

// Device parameter mirror: the model's flat weight blob (PARAM_FLOATS floats) lives here
// during a GPU run. The host params_ptr()/checkpoint buffers sync to/from it via the
// upload/download entry points below (the device half of the sync_params hooks).
float* g_dev_params = nullptr;

// Optimizer state mirrors (Phase 2d), all PARAM_FLOATS long: the reduced gradient the device
// backward writes and the AdamW first/second moments. Weight decay no longer needs a persistent
// PARAM_FLOATS-long mask buffer at all -- see g_decay_ranges below (a handful of compile-time-known
// ranges in __constant__ memory, derived from PARAM_LAYOUT) -- matches the CPU ParamView.decay
// exactly (verified exhaustively, offset by offset, by layout_tests.cpp's "DECAY_RANGES exactly
// reproduces..." test).
float* g_dev_grad  = nullptr;
float* g_dev_m     = nullptr;
float* g_dev_vel   = nullptr;
double* g_dev_normsq = nullptr;   // [1] global grad sum-of-squares (AdamW clip, double like CPU)
float*  g_dev_gs     = nullptr;   // [1] grad-clip scale, computed on-device from g_dev_normsq (see
                                  // grad_clip_scale_kernel) -- avoids a host readback of normsq just
                                  // to compute this one scalar (device_adam_step).

// cuBLAS handle for the dense GEMMs (every Linear + the lm_head). Created lazily, destroyed
// in sub0_cuda_shutdown.
cublasHandle_t g_cublas = nullptr;

// TF32 tensor-core GEMM math expressed as a sub0::Knob: the baked default is the configured
// CUDA_TF32 (a compile-time constant in a normal build; runtime-mutable under SUB0_TUNING so
// the autotuner can sweep it). MEASURED 2026-06 on sm_120: TF32 gives NO speedup at this
// model's small-K GEMMs (96/384) -- 0.98x vs FP32 -- so CUDA_TF32 defaults off. The benchmark
// and sub0_cuda_set_tf32 drive the cuBLAS handle directly to compare modes regardless of the
// baked value.
using CudaTf32 = sub0::Knob<bool, CUDA_TF32>;

// Dedicated stream for all device work so the forward can be captured into a CUDA graph
// (capturing the legacy default stream is disallowed). Plus the captured executable graph and
// the (batch,T) it was captured for -- recaptured when the shape or the GEMM math mode changes.
cudaStream_t    g_stream      = nullptr;
cudaGraphExec_t g_graph_exec  = nullptr;
int             g_graph_batch = -1;
int             g_graph_T     = -1;

// Captured graph for the per-token decode step (forward_one_device_graphed) -- see g_decode_state and
// capture_decode_graph() below. Unlike g_graph_exec above (recaptured per (batch,T) shape), this one
// has no shape axis to key on: forward_one_device_graphed is always M=1, so a single captured instance
// is reused across every token of every generation session until something invalidates it (KV-cache
// reset, a param upload, or a math-mode change -- see invalidate_decode_graph()'s call sites).
cudaGraphExec_t g_decode_graph_exec = nullptr;

// Lazily cached SM count (queried once via cudaDeviceGetAttribute, reused thereafter -- avoids a
// per-launch device-property query). Used below to size the grid-stride dgamma-reduction kernels'
// block count off the ACTUAL device, not a guessed constant (this project's "derive bounds from known
// scale, not fixed thresholds" convention -- see rmsnorm_backward_act_kernel/qknorm_backward_act_kernel).
int g_sm_count = 0;
inline int sm_count() {
    if (g_sm_count == 0) {
        int dev = 0, n = 0;
        cudaGetDevice(&dev);
        cudaDeviceGetAttribute(&n, cudaDevAttrMultiProcessorCount, dev);
        g_sm_count = (n > 0) ? n : 20;   // sane fallback if the query somehow fails
    }
    return g_sm_count;
}
// dgamma_grid_blocks (grid-size cap for rmsnorm_backward_act_kernel/qknorm_backward_act_kernel) is
// defined further below, right after those two kernels -- it needs their addresses for
// cudaOccupancyMaxActiveBlocksPerMultiprocessor, so it can't live up here with sm_count().

// Broadcast bias add: Y[m,o] += bias[o] over all M rows. Applied after the cuBLAS GEMM
// (the legacy cublasSgemm has no bias epilogue) to add the linear's bias.
// idx/bound computed in 64-bit: M*N (lm_head's bias-add is M=batch*T against N=VOCAB) overflows
// signed 32-bit at ordinary training sizes -- e.g. batch=128, T=512, VOCAB=32873 already exceeds
// INT32_MAX. Verified via this project's own generated config, not a hypothetical: a plain `int`
// product here is silent UB (observed as either a corrupted grid or a bias-add skipped for most
// rows), not a future-proofing nicety.
__global__ void bias_add_kernel(float* __restrict__ Y, const float* __restrict__ bias, int M, int N) {
    const long long idx = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < static_cast<long long>(M) * N) Y[idx] += bias[idx % N];
}

// --- Fast transcendental math (CUDA native intrinsics) ----------------------
// Use the hardware fast-math path -- __expf on the SFU, rsqrtf for reciprocal-sqrt --
// rather than re-deriving the CPU's software polynomial. Both are ~1e-6 approximations,
// so the GPU forward stays close to the CPU FAST_MATH path but is NOT bit-identical; the
// CPU<->GPU parity test uses a tolerance that absorbs the two approximations' difference.
// dev_tanh keeps the tanh-form GELU shape (matching the CPU's GPT-2-style approx) built on
// __expf, so the GELU stays consistent with the CPU op_gelu.
__device__ inline float dev_tanh(float x) {
    // Clamp the exponent exactly like the CPU fast_exp ([-87,88]) BEFORE __expf. Without this a
    // large negative GELU input makes -2x large positive, __expf overflows to +Inf, and the tanh
    // becomes (1-Inf)/(1+Inf) = NaN -- harmless at small init but it poisons the weights once
    // training grows the activations (the cause of GPU-training divergence). Clamping saturates
    // to +-1 like the CPU path instead.
    float a = -2.f * x;
    a = a < -87.f ? -87.f : (a > 88.f ? 88.f : a);
    const float e = __expf(a);
    return (1.f - e) / (1.f + e);
}
__device__ inline float dev_gelu(float v) {
    return 0.5f * v * (1.f + dev_tanh(0.7978845608f * (v + 0.0447150000f * v * v * v)));
}
// GELU derivative, kept consistent with dev_gelu (mirrors the CPU dgelu_fast tanh form) so the
// backward gradient matches the CPU reference. GELU_C = sqrt(2/pi), GELU_A = 0.044715.
__device__ inline float dev_dgelu(float v) {
    const float t  = dev_tanh(0.7978845608f * (v + 0.0447150000f * v * v * v));
    const float du = 0.7978845608f * (1.f + 3.f * 0.0447150000f * v * v);
    return 0.5f * (1.f + t) + 0.5f * v * (1.f - t * t) * du;
}
// SiLU/Swish (silu(v) = v*sigmoid(v)) built on the same __expf basis as dev_tanh/dev_gelu above, for
// consistency with the CPU's FAST_MATH fast_sigmoid (which shares fast_exp with fast_tanh). Used by
// the SwiGLU-gated FFN (USE_GATED_FFN): silu(gate) * up -- see op_swiglu/silu_fast in backend_cpu.cpp.
// Unlike dev_tanh, this needs NO overflow clamp: __expf(-v) saturates to +Inf as v -> -inf, giving
// sigmoid -> 0 and silu -> 0 (IEEE-safe: 1/(1+Inf) = 0, no NaN), the correct limiting value -- no
// poisoned-weights failure mode analogous to the unclamped tanh one dev_tanh's comment describes.
__device__ inline float dev_silu(float v) {
    return v / (1.f + __expf(-v));
}
// SiLU derivative, kept consistent with dev_silu so the backward gradient matches the CPU reference
// (dsilu_fast in backend_cpu.cpp): d/dv[v*sigmoid(v)] = sigmoid(v)*(1 + v*(1-sigmoid(v))).
__device__ inline float dev_dsilu(float v) {
    const float s = 1.f / (1.f + __expf(-v));
    return s * (1.f + v * (1.f - s));
}

// ============================================================================
//  Binding-compose (docs/BACKENDS.md "Design: binding-compose on CUDA") -- device side
// ============================================================================
// The HOST owns every binding table and its semantics (which token follows which sigil, which of the
// three views wins -- see op_embed's dispatch in backend_cpu.cpp); per step it precomputes WHICH flat
// positions embed as a composed row and uploads that as a POD "override table" via
// sub0_cuda_set_window_bindings below. The device never parses bindings: the embed kernels just check
// "is row m overridden?" and, if so, compose the row from fragment tok_emb rows with the entry's
// encoder arm -- the exact math of sub0::encode_slot (scratch_slots.hpp), param-free arms only
// (MeanPool / Hash / HRR; the learned-enc_w arms keep their documented CPU-only limit and are
// REJECTED at install). Naive per-thread compose first (HRR is O(C) per channel = O(C^2) per row):
// parity before performance, per the design doc.
//
// Wire layout across the extern "C" seam (POD ints only -- cross-referenced with the constants block
// in include/sub0/device_backend.hpp, which deliberately doesn't share code with this TU):
//   override_idx[n_positions] : row m composes entry override_idx[m]; -1 = plain tok_emb lookup.
//                               For a training step the row index is m = b*T + t (the flat batch
//                               row); for decode (forward_one) it is the token's POSITION `pos`.
//   entries[3 * n_entries]    : {frag_offset, frag_len, encoding} int triples; frag_offset indexes
//                               into frags[], encoding carries sub0::SlotEncoding's underlying value.
//   frags[n_frags]            : the concatenated fragment token ids (each in [0, VOCAB)).
constexpr int kBindEntryInts   = 3;  // ints per entry triple
constexpr int kBindEncMeanPool = static_cast<int>(sub0::SlotEncoding::MeanPool);
constexpr int kBindEncHash     = static_cast<int>(sub0::SlotEncoding::Hash);
constexpr int kBindEncHRR      = static_cast<int>(sub0::SlotEncoding::HRR);
static_assert(kBindEncMeanPool == 0 && kBindEncHash == 2 && kBindEncHRR == 4,
    "SlotEncoding's underlying values are the seam's wire encoding (documented in "
    "device_backend.hpp's SUB0_DEV_BIND_ENC_* constants) -- if this trips, the enum was reordered "
    "and BOTH sides must move together.");

// The device-resident override-table header. Kernels take a pointer to ONE fixed device-memory
// instance (g_dev_bind below, allocated once and never moved) rather than the four buffers as
// individual kernel arguments: the batched-inference forward and the decode step are CUDA-graph
// captured, and a captured kernel's ARGUMENTS are baked at capture time -- reading the (mutable)
// header CONTENTS through a fixed pointer is the same fixed-buffer/changing-contents pattern as
// g_decode_state, so installing/clearing/growing a table never invalidates a captured graph.
// n_positions == 0 (the cleared state, also the freshly-allocated state) => every row is a plain
// lookup, and the kernels' extra work is one cached 4-byte header read per thread -- the compose
// math itself is bit-for-bit untouched (the inertness parity test pins this).
struct DevBindings {
    const int*   override_idx;   // [n_positions]
    const int*   entries;        // [kBindEntryInts * n_entries]
    const int*   frags;          // [n_frags]
    const float* roles;          // [HRR_MAX_POS, D_MODEL] device copy of sub0::hrr_role_table
    int          n_positions;    // 0 = no table installed
    int          pad_;           // keep the struct 8-byte-aligned end-to-end (POD across H2D memcpy)
};

// Row m's entry index, or -1 for a plain lookup. Null-safe (the diagnostic checks may probe before
// any header exists); production kernels always receive the fixed g_dev_bind instance.
__device__ inline int binding_override_at(const DevBindings* __restrict__ bind, int m) {
    return (bind && m < bind->n_positions) ? bind->override_idx[m] : -1;
}

// Forward compose for ONE channel j of an overridden row -- the device transliteration of
// sub0::encode_slot's MeanPool/Hash/HRR arms (scratch_slots.hpp has the math + provenance comments;
// keep the two in lockstep). Per-fragment accumulation runs in the same p-order as the CPU loop so
// the FP32 sums agree to rounding; cosf/sinf/powf are CUDA's accurate (few-ulp) versions, not the
// fast intrinsics, for the tightest parity with std::cos/sin/pow float.
__device__ inline float compose_bound_channel(const float* __restrict__ tok_emb,
                                              const int* __restrict__ frag_ids, int len, int enc,
                                              const float* __restrict__ roles, int j) {
    constexpr int C = D_MODEL;
    float out = 0.f;
    if (enc == kBindEncMeanPool) {
        // mean of the fragment rows (encode_slot: sum then one multiply by 1/n)
        for (int p = 0; p < len; ++p) out += tok_emb[static_cast<size_t>(frag_ids[p]) * C + j];
        out *= 1.f / static_cast<float>(len);
    } else if (enc == kBindEncHash) {
        // RoPE-style positional binding: rotate fragment p's row pair (2m,2m+1) by a p-dependent
        // angle, then sum -- interleaved-pair convention + HASH_ROPE_THETA exactly as encode_slot.
        constexpr int half = C / 2;
        if (j >= 2 * half) {                       // odd-C tail channel: identity, still summed
            for (int p = 0; p < len; ++p) out += tok_emb[static_cast<size_t>(frag_ids[p]) * C + j];
        } else {
            const int mp = j / 2;                   // this channel's rotation pair index
            for (int p = 0; p < len; ++p) {
                const float* row = tok_emb + static_cast<size_t>(frag_ids[p]) * C;
                const float  ang = static_cast<float>(p) * powf(sub0::HASH_ROPE_THETA, -2.f * mp / C);
                const float  cs = cosf(ang), sn = sinf(ang);
                const float  x0 = row[2 * mp], x1 = row[2 * mp + 1];
                out += ((j & 1) == 0) ? (x0 * cs - x1 * sn) : (x0 * sn + x1 * cs);
            }
        }
    } else {  // kBindEncHRR (install rejects everything else before it can reach a kernel)
        // circular convolution against the fixed role vector: out[j] += sum_k role[k]*filler[(j-k) mod C]
        for (int p = 0; p < len; ++p) {
            const float* filler = tok_emb + static_cast<size_t>(frag_ids[p]) * C;
            const int    pi     = p < sub0::HRR_MAX_POS ? p : sub0::HRR_MAX_POS - 1;   // same clamp as encode_slot
            const float* role   = roles + static_cast<size_t>(pi) * C;
            float s = 0.f;
            for (int k = 0; k < C; ++k) {
                int idx = j - k; if (idx < 0) idx += C;
                s += role[k] * filler[idx];
            }
            out += s;
        }
    }
    return out;
}

// Backward scatter for ONE channel j of an overridden row -- the device transliteration of
// sub0::encode_slot_bwd's MeanPool/Hash/HRR arms: the composed position's row grad dout[C] flows
// into the FRAGMENT rows of dtok (never into the slot id's own row). atomicAdd like the plain
// embedding scatter above it: several positions (or repeated fragment ids within one slot) may
// target the same row.
__device__ inline void compose_bound_scatter(const float* __restrict__ dout,
                                             const int* __restrict__ frag_ids, int len, int enc,
                                             const float* __restrict__ roles,
                                             float* __restrict__ dtok, int j) {
    constexpr int C = D_MODEL;
    if (enc == kBindEncMeanPool) {
        const float inv = 1.f / static_cast<float>(len);
        const float val = inv * dout[j];                       // dout/n into every fragment row
        for (int p = 0; p < len; ++p)
            atomicAdd(&dtok[static_cast<size_t>(frag_ids[p]) * C + j], val);
    } else if (enc == kBindEncHash) {
        // rotation is orthogonal -> adjoint is the INVERSE rotation of dout (no tok_emb read)
        constexpr int half = C / 2;
        if (j >= 2 * half) {                                   // odd-C tail: identity
            const float val = dout[j];
            for (int p = 0; p < len; ++p)
                atomicAdd(&dtok[static_cast<size_t>(frag_ids[p]) * C + j], val);
        } else {
            const int   mp = j / 2;
            const float d0 = dout[2 * mp], d1 = dout[2 * mp + 1];
            for (int p = 0; p < len; ++p) {
                const float ang = static_cast<float>(p) * powf(sub0::HASH_ROPE_THETA, -2.f * mp / C);
                const float cs = cosf(ang), sn = sinf(ang);
                const float val = ((j & 1) == 0) ? (d0 * cs + d1 * sn) : (-d0 * sn + d1 * cs);
                atomicAdd(&dtok[static_cast<size_t>(frag_ids[p]) * C + j], val);
            }
        }
    } else {  // kBindEncHRR
        // adjoint of circular convolution by a fixed role is circular CORRELATION by that role:
        // d(filler)[j] = sum_n dout[n] * role[(n-j) mod C] (no tok_emb read)
        for (int p = 0; p < len; ++p) {
            const int    pi   = p < sub0::HRR_MAX_POS ? p : sub0::HRR_MAX_POS - 1;
            const float* role = roles + static_cast<size_t>(pi) * C;
            float s = 0.f;
            for (int n = 0; n < C; ++n) {
                int idx = n - j; if (idx < 0) idx += C;
                s += dout[n] * role[idx];
            }
            atomicAdd(&dtok[static_cast<size_t>(frag_ids[p]) * C + j], s);
        }
    }
}

// h[m,j] = tok_emb[ids[m], j] + pos_emb[t, j], where m is the global row over batch*T and
// t = m % T is the position WITHIN the window (op_embed + op_add: first forward step). C is D_MODEL
// at every call site below -- constexpr-folded (no loop to unroll here, pure elementwise; the win is
// just the dropped runtime parameter). M/T stay runtime (genuinely vary).
__global__ void embed_add_kernel(const float* __restrict__ tok_emb, const float* __restrict__ pos_emb,
                                 const int* __restrict__ ids, float* __restrict__ h, int M, int T) {
    constexpr int C = D_MODEL;
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    const int m = blockIdx.y * blockDim.y + threadIdx.y;
    if (m < M && j < C) {
        const int t = m % T;
        h[m * C + j] = tok_emb[ids[m] * C + j] + pos_emb[t * C + j];
    }
}

// Token-only embedding (RoPE path): h[m,j] = tok_emb[ids[m], j]. Position is injected later by
// the RoPE rotation inside attention, so there is no pos_emb add here. `bind` is the fixed
// device-resident override-table header (see DevBindings above): a row the host marked overridden
// composes from its bound fragments' rows instead of the plain lookup; with no table installed
// (n_positions == 0, the default) the output is bit-identical to the plain lookup.
__global__ void embed_kernel(const float* __restrict__ tok_emb, const int* __restrict__ ids,
                             float* __restrict__ h, int M, const DevBindings* __restrict__ bind) {
    constexpr int C = D_MODEL;
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    const int m = blockIdx.y * blockDim.y + threadIdx.y;
    if (m >= M || j >= C) return;
    const int ov = binding_override_at(bind, m);
    if (ov >= 0) {
        const int* e = bind->entries + ov * kBindEntryInts;   // {frag_offset, frag_len, encoding}
        h[m * C + j] = compose_bound_channel(tok_emb, bind->frags + e[0], e[1], e[2], bind->roles, j);
    } else {
        h[m * C + j] = tok_emb[ids[m] * C + j];
    }
}

// Activation-typed variants: write into the saved-activation store type (act_t). F32 build keeps
// these identical to the float kernels above; BF16 stores half-width with FP32 source. Same
// override branch as embed_kernel (compose math stays FP32; only the store rounds).
template <class A>
__global__ void embed_act_kernel(const float* __restrict__ tok_emb, const int* __restrict__ ids,
                                 A* __restrict__ h, int M, const DevBindings* __restrict__ bind) {
    constexpr int C = D_MODEL;
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    const int m = blockIdx.y * blockDim.y + threadIdx.y;
    if (m >= M || j >= C) return;
    const int ov = binding_override_at(bind, m);
    if (ov >= 0) {
        const int* e = bind->entries + ov * kBindEntryInts;
        st_act(&h[m * C + j], compose_bound_channel(tok_emb, bind->frags + e[0], e[1], e[2], bind->roles, j));
    } else {
        st_act(&h[m * C + j], tok_emb[ids[m] * C + j]);
    }
}
template <class A>
__global__ void embed_add_act_kernel(const float* __restrict__ tok_emb, const float* __restrict__ pos_emb,
                                     const int* __restrict__ ids, A* __restrict__ h, int M, int T) {
    constexpr int C = D_MODEL;
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    const int m = blockIdx.y * blockDim.y + threadIdx.y;
    if (m < M && j < C) { const int t = m % T; st_act(&h[m * C + j], tok_emb[ids[m] * C + j] + pos_emb[t * C + j]); }
}

// y[m,j] = x[m,j] * (1/sqrt(mean_j x^2 + eps)) * gamma[j]  (op_rmsnorm forward). One BLOCK per row,
// threads striding coalesced over D_MODEL -- was one thread per row (the fix and reasoning are
// identical to rmsnorm_train_act_kernel above, which has the full ncu evidence for the same
// one-thread-per-row anti-pattern; this is the plain non-training sibling used by generation/decode,
// including the per-TOKEN decode hot path where rows==1 -- under the old layout that meant 63 of 64
// threads in the launch sat idle every single token). D_MODEL used directly (constexpr, never a
// runtime size at any call site) instead of as a parameter.
template <int BLOCK>
__global__ void __launch_bounds__(BLOCK)
rmsnorm_kernel(const float* __restrict__ x, const float* __restrict__ gamma,
              float* __restrict__ y, int rows) {
    constexpr int   C    = D_MODEL;
    constexpr float invC = 1.0f / static_cast<float>(C);
    constexpr float eps  = 1e-5f;
    const int m = blockIdx.x;
    if (m >= rows) return;
    const float* xr = x + static_cast<size_t>(m) * C;
    float partial = 0.f;
    #pragma unroll
    for (int j = threadIdx.x; j < C; j += BLOCK) partial += xr[j] * xr[j];
    const float ms = block_reduce_sum<BLOCK>(partial) * invC;
    const float r  = rsqrtf(ms + eps);                  // CUDA fast reciprocal sqrt
    float* yr = y + static_cast<size_t>(m) * C;
    #pragma unroll
    for (int j = threadIdx.x; j < C; j += BLOCK) yr[j] = xr[j] * r * gamma[j];
}

// Elementwise tanh-form GELU (op_gelu, FAST_MATH path).
__global__ void gelu_kernel(const float* __restrict__ x, float* __restrict__ y, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = dev_gelu(x[i]);
}
// act-typed GELU: y = gelu(x), reading/writing the store type (F32 build == gelu_kernel).
template <class A>
__global__ void gelu_act_kernel(const A* __restrict__ x, A* __restrict__ y, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) st_act(&y[i], dev_gelu(to_f32(x[i])));
}

// SwiGLU (op_swiglu, USE_GATED_FFN's FFN nonlinearity): y = silu(gate_pre) * up_pre, elementwise. Each
// thread reads its own gate[i]/up[i] into locals BEFORE writing y[i], so the caller may alias y ==
// up_pre for an in-place update (no cross-thread dependency -- every thread only ever touches index i).
__global__ void swiglu_kernel(const float* __restrict__ gate, const float* __restrict__ up,
                              float* __restrict__ y, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) { const float g = gate[i], u = up[i]; y[i] = dev_silu(g) * u; }
}
// act-typed SwiGLU: y = silu(gate)*up in the store type (F32 build == swiglu_kernel). Same in-place-
// on-up_pre aliasing convention as the plain kernel above.
template <class A>
__global__ void swiglu_act_kernel(const A* __restrict__ gate, const A* __restrict__ up,
                                  A* __restrict__ y, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) { const float g = to_f32(gate[i]), u = to_f32(up[i]); st_act(&y[i], dev_silu(g) * u); }
}

// Elementwise add: c[i] = a[i] + b[i] (residual connections; safe in-place when c == a).
__global__ void add_kernel(const float* __restrict__ a, const float* __restrict__ b,
                           float* __restrict__ c, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}
// act-typed add: c = a + b in the store type (F32 build == add_kernel).
template <class A>
__global__ void add_act_kernel(const A* __restrict__ a, const A* __restrict__ b, A* __restrict__ c, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) st_act(&c[i], to_f32(a[i]) + to_f32(b[i]));
}
// act-typed RMSNorm-train: y = rmsnorm(x)*gamma, saving rinv; in X / out Y store types, FP32 math.
// One BLOCK per row, threads striding coalesced over D_MODEL -- same fix and same reasoning as
// rmsnorm_backward_act_kernel above (that comment has the full ncu evidence): one-thread-per-row put
// a warp's simultaneous loads C floats apart with no coalescing. D_MODEL is compile-time constexpr
// (never a runtime size at any call site), so it's used directly rather than taken as a parameter,
// which also lets the strided loops fully unroll and folds the /C divide into a constant multiply.
template <class X, class Y, int BLOCK>
__global__ void __launch_bounds__(BLOCK)
rmsnorm_train_act_kernel(const X* __restrict__ x, const float* __restrict__ gamma,
                         Y* __restrict__ y, float* __restrict__ rinv, int rows) {
    constexpr int   C    = D_MODEL;
    constexpr float invC = 1.0f / static_cast<float>(C);
    constexpr float eps  = 1e-5f;
    const int m = blockIdx.x;
    if (m >= rows) return;
    const X* xr = x + static_cast<size_t>(m) * C;
    float partial = 0.f;
    #pragma unroll
    for (int j = threadIdx.x; j < C; j += BLOCK) { const float v = to_f32(xr[j]); partial += v * v; }
    const float ms = block_reduce_sum<BLOCK>(partial) * invC;
    const float r = rsqrtf(ms + eps);
    if (threadIdx.x == 0) rinv[m] = r;
    Y* yr = y + static_cast<size_t>(m) * C;
    #pragma unroll
    for (int j = threadIdx.x; j < C; j += BLOCK) st_act(&yr[j], to_f32(xr[j]) * r * gamma[j]);
}

// Causal multi-head attention (op_attn) is served by the flash-style attn_fwd_tiled_kernel below
// (used for BOTH inference and training, via launch_attn / launch_attn_train_t). The naive
// one-thread-per-query variant survives only as attn_train_act_kernel, kept as the on-device parity
// REFERENCE the self-test compares the tiled kernel against (sub0_cuda_attn_check).

// Build the fused QKV weight Wqkv[C, QKV_STRIDE] (row-major) from Wq [C,C] and Wk,Wv [C,D_KV]: row p
// holds [Wq[p] | Wk[p] | Wv[p]]. Materialized ONCE at upload so the three projection GEMMs collapse
// into one a . Wqkv -> [M, QKV_STRIDE] (better-shaped GEMM + fewer launches).
// Under GQA the three sub-blocks have DIFFERENT widths (see layout.hpp's QKV_* constants), so the
// thread mapping is one thread per FUSED-matrix element -- each thread resolving which source matrix
// its column falls in -- rather than the old one-thread-does-all-three-writes over a shared [C,C]
// index. Under MHA the two are the same set of writes. C/D_KV/QKV_STRIDE are constexpr at the one
// call site, so the p=idx/W, c=idx%W split (GPU integer div/mod has no native instruction) still
// resolves at compile time instead of once per thread per layer.
__global__ void build_qkv_kernel(const float* __restrict__ Wq, const float* __restrict__ Wk,
                                 const float* __restrict__ Wv, float* __restrict__ Wqkv) {
    constexpr int C = D_MODEL, KV = sub0::D_KV, W = sub0::QKV_STRIDE;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;   // over the C*W elements of the FUSED matrix
    if (idx >= C * W) return;
    const int p = idx / W, c = idx - p * W;                  // fused row, fused column
    if (c < C)            Wqkv[idx] = Wq[static_cast<size_t>(p) * C  + c];
    else if (c < C + KV)  Wqkv[idx] = Wk[static_cast<size_t>(p) * KV + (c - C)];
    else                  Wqkv[idx] = Wv[static_cast<size_t>(p) * KV + (c - C - KV)];
}

// Same fused layout, writing DIRECTLY to the activation (bf16) mirror instead of staging through an
// F32 intermediate -- what BF16 training uses (build_qkv_weights() below) so it never needs the
// F32 g_fwd.wqkv buffer at all. st_act rounds each element on the store, identical to building the
// F32 buffer above and converting it afterward, minus the extra read+write pass over 3*C*C elements.
template <class A>
__global__ void build_qkv_act_kernel(const float* __restrict__ Wq, const float* __restrict__ Wk,
                                     const float* __restrict__ Wv, A* __restrict__ Wqkv) {
    constexpr int C = D_MODEL, KV = sub0::D_KV, W = sub0::QKV_STRIDE;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= C * W) return;
    const int p = idx / W, c = idx - p * W;
    if (c < C)            st_act(&Wqkv[idx], Wq[static_cast<size_t>(p) * C  + c]);
    else if (c < C + KV)  st_act(&Wqkv[idx], Wk[static_cast<size_t>(p) * KV + (c - C)]);
    else                  st_act(&Wqkv[idx], Wv[static_cast<size_t>(p) * KV + (c - C - KV)]);
}

// RoPE forward: rotate the Q and K sub-blocks (columns [0,D_MODEL) and [QKV_K_OFF, QKV_V_OFF)) of the
// fused [*, QKV_STRIDE] qkv buffer in place, in interleaved pairs per head; V is left alone. One thread
// per (row m, pair pg, sub-block z). pos = m % T (position within the window). Mirrors the CPU
// op_rope; the math (CUDA __sincosf/powf vs CPU std::) agrees to fast-math tolerance.
//
// GQA note: this kernel used to have a SINGLE thread rotate the Q pair and the K pair at the same
// (head, pair) index, which is only valid while Q and K have the same head count. They do not under
// GQA, so the Q and K passes are now separate, selected by blockIdx.z -- the same idiom
// qknorm_act_kernel below already uses for exactly this Q-vs-K split, rather than a new mechanism.
//
// C, H, in_stride are D_MODEL/N_HEADS/3*D_MODEL at every call site (confirmed by grep) -- baked
// constexpr, so used directly instead of as parameters. This also folds `d = C/H` from a per-thread
// runtime division into the already-baked D_HEAD constant (GPU integer div/mod has no native
// instruction; this ran over M*D_MODEL/2 threads every layer, every forward AND backward pass under
// RoPE). T/batch stay runtime (genuinely vary, same as ce_backward_kernel's T).
// Rotate ONE sub-block (blockIdx.z: 0 = Q, 1 = K) of the fused row, in interleaved pairs per head.
// Shared by all three RoPE kernels so the forward, the act-typed forward and the backward cannot drift.
// `sign` is +1 forward, -1 backward (the adjoint of a rotation is the rotation by -angle).
template <class A>
__device__ inline void rope_pair_body(A* __restrict__ row, int pos, float theta, int pg,
                                      int which, float sign) {
    constexpr int d = D_HEAD, half = d / 2;
    const int base = which == 0 ? 0 : sub0::QKV_K_OFF;
    const int h    = pg / half, mi = pg % half;
    const int a0   = base + h * d + 2 * mi;
    const float ang = (static_cast<float>(pos) * sub0::ROPE_POS_SCALE) * powf(theta, -2.0f * mi / d);
    float sn, cs; __sincosf(ang, &sn, &cs);
    sn *= sign;                                     // sign = -1 gives the rotation's transpose (backward)
    const float x0 = to_f32(row[a0]), x1 = to_f32(row[a0 + 1]);
    st_act(&row[a0],     x0 * cs - x1 * sn);
    st_act(&row[a0 + 1], x0 * sn + x1 * cs);
}
// Pairs in the sub-block selected by blockIdx.z. Q is D_MODEL wide, K is D_KV wide -- equal only under
// MHA -- so the grid is sized for the WIDER of the two and the narrower sub-block early-outs.
__device__ inline int rope_sub_pairs(int which) {
    return (which == 0 ? D_MODEL : sub0::D_KV) / 2;
}
__global__ void rope_kernel(float* __restrict__ qkv, int batch, int T, float theta, int which) {
    constexpr int in_stride = sub0::QKV_STRIDE;
    const int pg = blockIdx.x * blockDim.x + threadIdx.x;   // pair within this sub-block
    const int m  = blockIdx.y * blockDim.y + threadIdx.y;   // row over M = batch*T
    if (m >= batch * T || pg >= rope_sub_pairs(which)) return;
    rope_pair_body(qkv + static_cast<size_t>(m) * in_stride, m % T, theta, pg, which, +1.0f);
}

// QK-norm forward (USE_QK_NORM, op_qknorm/qknorm_row in backend_cpu.cpp): RMSNorm applied
// INDEPENDENTLY to each head's D_HEAD-wide slice of the fused [rows,QKV_STRIDE] qkv buffer's Q (or K)
// sub-block, IN PLACE, right after the QKV GEMM and before rope_kernel/rope_act_kernel touch the
// same buffer -- a separate per-head statistic from op_rmsnorm's whole-row one, same reason as the
// CPU op (see its comment): mixing every head into one norm statistic would defeat the point of a
// per-head stabilizer. One block per (row, head, which): blockIdx.x = row, blockIdx.y = head,
// blockIdx.z = 0 selects the Q sub-block / 1 selects K (cols [2C,3C), V, is never touched, same
// convention as rope_kernel above). BLOCK=32 (one warp): D_HEAD is 32-96 at this project's scales, an
// order of magnitude narrower than D_MODEL's own rmsnorm reduction, so a single warp's strided loop +
// shuffle-reduce (block_reduce_sum's BLOCK<=32 path, no cross-warp shared-memory step needed) covers
// it. H/DH are template params (not D_MODEL/N_HEADS/D_HEAD directly) so the dims-independent CUDA
// self-test can instantiate a toy shape in the SAME binary as the baked production shape.
template <class A, int H, int KVH, int DH, int BLOCK>
__global__ void __launch_bounds__(BLOCK)
qknorm_act_kernel(A* __restrict__ qkv, const float* __restrict__ qgamma, const float* __restrict__ kgamma,
                  int which) {
    constexpr int   C     = H * DH, CKV = KVH * DH, in_stride = C + 2 * CKV;
    constexpr float invDH = 1.0f / static_cast<float>(DH);
    constexpr float eps   = 1e-5f;
    const int row = blockIdx.x, h = blockIdx.y;                       // which: 0 = Q, 1 = K (an ARGUMENT,
    const int off = (which == 0) ? (h * DH) : (C + h * DH);           // so each pass gets its own grid)
    A* xr = qkv + static_cast<size_t>(row) * in_stride + off;
    const float* g = (which == 0) ? qgamma : kgamma;
    float partial = 0.f;
    #pragma unroll
    for (int j = threadIdx.x; j < DH; j += BLOCK) { const float v = to_f32(xr[j]); partial += v * v; }
    const float ms = block_reduce_sum<BLOCK>(partial) * invDH;
    const float r  = rsqrtf(ms + eps);
    #pragma unroll
    for (int j = threadIdx.x; j < DH; j += BLOCK) st_act(&xr[j], to_f32(xr[j]) * r * g[j]);
}

// Same as qknorm_act_kernel, but first stashes the PRE-norm x into qk_pre[rows,2C] (same Q|K layout,
// col offset `which*C + h*DH`, just a 2C- instead of 3C-wide row stride) before overwriting qkv in
// place. Used ONLY by backward_device's checkpoint-recompute pass: the forward chain immediately
// applies RoPE to the SAME buffer right after qknorm (rope_kernel/rope_act_kernel), so nothing else
// in this file's resident scratch still holds the pre-norm values by the time qknorm_backward_act_kernel
// needs them (RoPE's own backward runs first, since RoPE sits AFTER qknorm in the forward graph) --
// see the TrainScratch::qk_pre comment in memplan.hpp's train_scratch_bytes for the full reasoning.
template <class A, int H, int KVH, int DH, int BLOCK>
__global__ void __launch_bounds__(BLOCK)
qknorm_save_act_kernel(A* __restrict__ qkv, A* __restrict__ qk_pre,
                       const float* __restrict__ qgamma, const float* __restrict__ kgamma, int which) {
    constexpr int   C        = H * DH, CKV = KVH * DH;
    constexpr int   in_stride = C + 2 * CKV, pre_stride = C + CKV;
    constexpr float invDH    = 1.0f / static_cast<float>(DH);
    constexpr float eps      = 1e-5f;
    const int row = blockIdx.x, h = blockIdx.y;
    const int off = (which == 0) ? (h * DH) : (C + h * DH);
    A* xr = qkv    + static_cast<size_t>(row) * in_stride  + off;
    A* pr = qk_pre + static_cast<size_t>(row) * pre_stride + off;
    const float* g = (which == 0) ? qgamma : kgamma;
    float partial = 0.f;
    #pragma unroll
    for (int j = threadIdx.x; j < DH; j += BLOCK) {
        const float v = to_f32(xr[j]);
        st_act(&pr[j], v);
        partial += v * v;
    }
    const float ms = block_reduce_sum<BLOCK>(partial) * invDH;
    const float r  = rsqrtf(ms + eps);
    #pragma unroll
    for (int j = threadIdx.x; j < DH; j += BLOCK) st_act(&xr[j], to_f32(xr[j]) * r * g[j]);
}

// ============================================================================
//  Backward kernels (Phase 2d) -- mirror src/backend_cpu.cpp backward_node()
// ============================================================================

// act-typed attention forward (flash, P-free) -- q/k/v/out in the store type, FP32 softmax.
template <class A>
__global__ void attn_train_act_kernel(const A* __restrict__ q, const A* __restrict__ k,
                                       const A* __restrict__ v, A* __restrict__ out,
                                       int batch, int T, int C, int H, int in_stride, int kv_group) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x; const int per = H * T;
    const int b = idx / per; if (b >= batch) return;
    const int rem = idx - b * per, h = rem / T, i = rem % T, d = C / H, off = h * d;
    const int kv_off = (h / kv_group) * d;      // GQA: this query head's shared KV head (== off at group 1)
    const float scale = 1.0f / sqrtf(static_cast<float>(d));
    const size_t in_base = static_cast<size_t>(b) * T * in_stride, out_base = static_cast<size_t>(b) * T * C;
    const A* qi = q + in_base + static_cast<size_t>(i) * in_stride + off;
    float m = -1e30f, Z = 0.f, acc[128] = {};
    for (int j = 0; j <= i; ++j) {
        const A* kj = k + in_base + static_cast<size_t>(j) * in_stride + kv_off;
        float s = 0.f; for (int a = 0; a < d; ++a) s += to_f32(qi[a]) * to_f32(kj[a]); s *= scale;
        const float mn = fmaxf(m, s), c = __expf(m - mn), e = __expf(s - mn); Z = Z * c + e;
        const A* vj = v + in_base + static_cast<size_t>(j) * in_stride + kv_off;
        for (int a = 0; a < d; ++a) acc[a] = acc[a] * c + e * to_f32(vj[a]); m = mn;
    }
    A* oi = out + out_base + static_cast<size_t>(i) * C + off; const float invZ = 1.f / Z;
    for (int a = 0; a < d; ++a) st_act(&oi[a], acc[a] * invZ);
}

// ---- Flash-style TILED attention forward (the hot-path replacement for attn_train_act_kernel) ----
// The naive kernel above is L1/TEX-cache-bound: every query thread re-streams ALL of K and V from
// global, so each K/V row is read O(T) times (ncu at d448/T256: L1/TEX 93%, Memory 90%, Compute 19%,
// = 92% of the forward). This tiles the keys through shared memory. A block owns TILE_Q queries of
// ONE (b,h) -- grid = (T-tiles, H, batch) -- stages each K/V tile into shared ONCE, and all TILE_Q
// queries in the block reuse it (the redundant global reads collapse by ~TILE_Q). q and the online-
// softmax accumulator live in registers (HD = D_HEAD is compile-time, so the per-channel loops fully
// unroll and the streamed data never spills to local/L1). Keys are still consumed in INCREASING j,
// exactly like the naive kernel, so the online-softmax arithmetic -- and therefore the result -- is
// bit-identical to attn_train_act_kernel (gated on device by sub0_cuda_attn_check, naive vs tiled).
// HD = head dim, TILE_Q = queries per block, TILE_K = keys staged per shared tile.
//
// TILE_K/TILE_Q at HD<=64 shrunk 64->32 (2026-07, isolated-harness investigation): ncu --set full
// on all five flash-attention kernels at production dims (d448/T256) found occupancy is NOT purely
// register-bound -- Block Limit Shared Mem (2-3 blocks/SM at TILE=64's 32-33.5KB/block) sits AT OR
// BELOW Block Limit Registers (4-6 blocks/SM), so the 32-33.5KB static Ks/Vs shared-memory tile is a
// co-binding occupancy constraint, not just the register footprint. Halving it to 16-16.8KB (paired
// with attn_block_q<HD>() below widening the block 64->128) lets twice as many blocks land per SM.
// Measured (isolated harness, bit-exact vs this file's kernels, verified via cudaFuncGetAttributes +
// ncu, 3 independent process launches to rule out this GPU's thermal drift): achieved occupancy
// ~11.7-12.1%/8.2% -> ~22.7-24.5% at HD=64, wall-clock 1.6-2.3x across all five kernels, 0.0 parity
// diff (pure tile/grid-shape change, no arithmetic touched). HD=32 gains too (1.3-1.5x on fwd/stats/
// dv; smaller on dq/dk, which are separately register-bound there regardless of tile size -- not
// this change's target). The >64 branches (already smem-overflow-driven, see attn_tile_q's own
// comment below) are UNCHANGED -- this only touches the HD<=64 path, which is what was measured.
template <int HD> constexpr int attn_tile_k() { return HD <= 64 ? 32 : (HD <= 128 ? 32 : 16); }
// Block size (threads/block = queries-or-keys per block) for the five flash-attention kernels below.
// HD-templated (not a flat constant) for the same reason attn_tile_k/attn_tile_q already are: the
// widened 128 is only validated for HD<=64 (see above); every other HD keeps the original 64,
// unchanged in every byte of its compiled output. Single source of truth for the five kernels'
// __launch_bounds__ and every launch_attn*/self-test call site's block/grid math below.
template <int HD> constexpr int attn_block_q() { return HD <= 64 ? 128 : 64; }

// Query-tile size for the dk/dv backward kernel, which stages BOTH q and dout (2*TILE_Q*HD floats,
// plus 3 per-query stats). At the fixed 64 that overflows the 48 KB static shared-memory limit once
// HD grows (HD=96 -> 2*64*96*4 + 3*64*4 = 49920 B > 49152): shrink the staged query tile for large HD
// so it fits, exactly as attn_tile_k does for the key tile. The tile is DECOUPLED from the 64-thread
// (64-key) block -- the staging loops stride by blockDim.x and the query loop steps by TILE_Q from the
// key-tile base, so causality and the result are unchanged for any TILE_Q. (Register pressure at large
// HD is a separate, still-open perf item: 4*HD accumulators spill; a DV/DK split would help.)
//
// HD<=64 branch shrunk 64->32, same occupancy reasoning and same measurements as attn_tile_k above
// (this tile is 2*TILE_Q*HD+3*TILE_Q floats vs attn_tile_k's 2*TILE_K*HD -- structurally the same
// shared-memory-per-block story, just for the dv/dk kernels' query tile instead of fwd/stats/dq's key
// tile). The >64 branches stay untouched, same as attn_tile_k.
template <int HD> constexpr int attn_tile_q() { return HD <= 64 ? 32 : (HD <= 128 ? 32 : 16); }

// Warp-cooperative channel split for the dq kernel: LANES threads (a power of 2) share one query,
// each owning HD/LANES channels of qr/dr/dqa, so the per-thread register need drops from 3*HD to
// 3*(HD/LANES). Engaged once the raw 3*HD array footprint APPROACHES the 255-register architectural
// cap (verified via ptxas -v -- neither #pragma unroll nor a smaller __launch_bounds__ hint change
// that cap; 255 is a hardware ceiling, not a launch-config artifact). The trigger is NOT simply
// "3*HD > 255": measured via cudaFuncGetAttributes (sub0_cuda_attn_regcheck), HD=64 (3*HD=192,
// comfortably under 255 by the raw-array count alone) still compiled to 255 regs + 56B spill --
// per-thread state beyond the three raw arrays (softmax stats, shared-memory staging pointers, loop
// induction) adds real, non-negligible register pressure the naive formula doesn't count. HD=32
// (3*HD=96) measured 0 spill, so the threshold is moved down to 3*HD >= 192 (HD >= 64) to also
// cover that case, while leaving HD=32 on the unsplit LANES=1 path. LANES=1 makes every formula in
// the kernel below reduce algebraically to the single-thread-per-query code (lane=0, HALF=HD), so
// small-HD builds are otherwise unaffected in code path or performance.
template <int HD> __host__ __device__ constexpr int attn_dq_lanes() { return (3 * HD >= 192 && HD % 2 == 0) ? 2 : 1; }

// Same channel-split idea, applied to the dk kernel below: LANES threads share one KEY, each owning
// HD/LANES channels of kr/vr/dka. dk_j = sum_i ds_ij q_i needs both the QK score (via kr) and the
// dp = <dout_i, v_j> term (via vr) to form the scalar ds_ij -- structurally the SAME shape as dq's
// s/dp pair, just with q/k roles swapped (dk owns the key, streams q/dout; dq owns the query, streams
// k/v), so the identical warp-shuffle-combine trick applies. Same empirically-grounded threshold as
// attn_dq_lanes (see its comment): kept as a separate function since the two kernels' register
// pressure could diverge in the future even though the formula is identical today.
template <int HD> __host__ __device__ constexpr int attn_dk_lanes() { return (3 * HD >= 192 && HD % 2 == 0) ? 2 : 1; }

template <class A, int HD, int TILE_K>
__global__ void __launch_bounds__(attn_block_q<HD>())
attn_fwd_tiled_kernel(const A* __restrict__ q, const A* __restrict__ k, const A* __restrict__ v,
                      A* __restrict__ out, int T, int C, int in_stride, int kv_group) {
    __shared__ float Ks[TILE_K * HD];
    __shared__ float Vs[TILE_K * HD];
    const int    b  = blockIdx.z, h = blockIdx.y;
    const int    q0 = blockIdx.x * blockDim.x;               // first query of this block
    const int    i  = q0 + threadIdx.x;                      // this thread's query (block owns one (b,h))
    const int    off = h * HD;                               // Q (and out) offset -- per QUERY head
    const int    kv_off = (h / kv_group) * HD;               // K/V offset -- per KV head (== off at group 1)
    const float  scale    = rsqrtf(static_cast<float>(HD));
    const size_t in_base  = static_cast<size_t>(b) * T * in_stride;
    const size_t out_base = static_cast<size_t>(b) * T * C;

    float qr[HD];                                            // this query's row, cached in registers
    if (i < T) {
        const A* qi = q + in_base + static_cast<size_t>(i) * in_stride + off;
        #pragma unroll
        for (int a = 0; a < HD; ++a) qr[a] = to_f32(qi[a]);
    }
    float m = -1e30f, Z = 0.f, acc[HD];
    #pragma unroll
    for (int a = 0; a < HD; ++a) acc[a] = 0.f;

    const int jmax = min(T, q0 + static_cast<int>(blockDim.x));   // no query in this block attends j >= jmax
    for (int j0 = 0; j0 < jmax; j0 += TILE_K) {
        const int tk = min(TILE_K, jmax - j0);
        __syncthreads();
        for (int e = threadIdx.x; e < tk * HD; e += blockDim.x) { // cooperative K/V tile load (all threads)
            const int    jl  = e / HD, a = e - jl * HD;
            const size_t row = in_base + static_cast<size_t>(j0 + jl) * in_stride + kv_off;
            Ks[e] = to_f32(k[row + a]);
            Vs[e] = to_f32(v[row + a]);
        }
        __syncthreads();
        if (i < T) {
            const int jend = min(tk, i - j0 + 1);            // causal: only keys j <= i (<=0 -> skip tile)
            for (int jl = 0; jl < jend; ++jl) {
                const float* ks = Ks + jl * HD;
                float s = 0.f;
                #pragma unroll
                for (int a = 0; a < HD; ++a) s += qr[a] * ks[a];
                s *= scale;
                const float mn = fmaxf(m, s), c = __expf(m - mn), e2 = __expf(s - mn);
                Z = Z * c + e2;
                const float* vs = Vs + jl * HD;
                #pragma unroll
                for (int a = 0; a < HD; ++a) acc[a] = acc[a] * c + e2 * vs[a];
                m = mn;
            }
        }
    }
    if (i < T) {
        A* oi = out + out_base + static_cast<size_t>(i) * C + off;
        const float invZ = 1.f / Z;
        #pragma unroll
        for (int a = 0; a < HD; ++a) st_act(&oi[a], acc[a] * invZ);
    }
}

// act-typed RoPE forward/backward -- both delegate to rope_pair_body above, which is the ONE place
// the angle and the rotation live (the backward is the same rotation with sign = -1, i.e. its transpose).
template <class A> __global__ void rope_act_kernel(A* __restrict__ qkv, int batch, int T, float theta, int which) {
    const int pg = blockIdx.x * blockDim.x + threadIdx.x;
    const int m  = blockIdx.y * blockDim.y + threadIdx.y;
    if (m >= batch * T || pg >= rope_sub_pairs(which)) return;
    rope_pair_body(qkv + static_cast<size_t>(m) * sub0::QKV_STRIDE, m % T, theta, pg, which, +1.0f);
}
template <class A> __global__ void rope_bwd_act_kernel(A* __restrict__ dq, int batch, int T, float theta, int which) {
    const int pg = blockIdx.x * blockDim.x + threadIdx.x;
    const int m  = blockIdx.y * blockDim.y + threadIdx.y;
    if (m >= batch * T || pg >= rope_sub_pairs(which)) return;
    rope_pair_body(dq + static_cast<size_t>(m) * sub0::QKV_STRIDE, m % T, theta, pg, which, -1.0f);
}

// Cross-entropy backward (op_cross_entropy): one BLOCK per row m over [M,V], threads striding
// coalesced over V (was one thread per row doing 3 fully sequential O(V) passes -- ncu at V=16489
// measured 30.3ms/call, 28% occupancy, 4.7% compute throughput, the same uncoalesced-warp pattern as
// the rmsnorm kernels above: adjacent threads owned adjacent ROWS, so a fixed loop iteration touched
// addresses V floats apart). Row m belongs to window b = m/T at position t = m%T. Positions
// t >= lengths[b] are PADDING (a short document padded up to T): they get zero gradient and add no
// loss. Each trained row is weighted 1/(batch*len[b]) so the loss/grad is the per-window mean over its
// real length, then the mean over the batch -- identical to the CPU train_batch reduction, and exactly
// 1/M when every window is full (len == T, lengths == nullptr), which keeps the dense gradient-parity
// gate unchanged.
//
// V is VOCAB, a baked constexpr (this kernel's one call site always passes it), used directly instead
// of as a parameter. UNLIKE D_MODEL in the rmsnorm kernels above, V is NOT unrolled -- it runs into
// the tens of thousands (32873 in the production config), so a full #pragma unroll would be a
// code-size/I-cache disaster for a handful fewer loop-branch instructions; the real win here is
// coalescing + parallelizing the O(V) work across BLOCK threads, same as the reduction itself. T stays
// a runtime parameter (unlike V, it genuinely varies -- benchmark/self-test paths call this with a
// window length shorter than SEQ_LEN).
//
// `row_offset`: this kernel is now called ONCE PER ROW-CHUNK (see head_ce_chunked below), each call
// covering only `M` (the CHUNK's row count, not the full training M) rows starting at absolute row
// `row_offset`. `logits`/`dlogits`/`targets` are already chunk-relative pointers (the caller does that
// arithmetic), so the per-row math indexed by the LOCAL `m` stays correct as-is -- but `b`/`t` (and
// therefore `lengths[b]`) depend on the row's ABSOLUTE position in the full M, which `row_offset`
// supplies. Every row's cross-entropy math is otherwise fully independent of every other row (no
// cross-row reduction inside this kernel), and `loss_acc`'s atomicAdd is already safe across however
// many separate kernel launches touch it, so chunking needs nothing else here.
//
// `dlogits == nullptr` runs the kernel LOSS-ONLY: no gradient is written and no dlogits buffer is
// needed at all. This is the eval path (sub0_cuda_forward_loss). It is deliberately the SAME kernel
// rather than a sibling, because the number it must produce is the same number: the masking rules
// (padding, LOSS_IGNORE_INDEX), the per-window active-count denominator, and the 1/(batch*denom)
// weight all live here exactly once, so a forward-only eval cannot drift from the loss the trainer
// reports. `dlogits` is uniform across the whole launch, so every branch on it below is
// block-uniform -- no warp divergence, and the block-wide reductions stay fully populated.
template <int BLOCK>
__global__ void __launch_bounds__(BLOCK)
ce_backward_kernel(const float* __restrict__ logits, const int* __restrict__ targets,
                   float* __restrict__ dlogits, double* __restrict__ loss_acc,
                   int M, int T, int batch, const int* __restrict__ lengths,
                   const int* __restrict__ active, int row_offset) {
    constexpr int V = VOCAB;
    const int m = blockIdx.x;
    if (m >= M) return;
    const int abs_m = row_offset + m;
    const int b   = abs_m / T;
    const int t   = abs_m - b * T;
    const int len = lengths ? lengths[b] : T;
    float* dl     = dlogits + static_cast<size_t>(m) * V;
    const int tgt = targets[m];
    // Inert rows contribute no loss and no gradient: a PADDING row (t >= len, a short document padded
    // up to T) OR a MASKED row (tgt < 0 == LOSS_IGNORE_INDEX, a loss-masked position -- e.g. the
    // uncombine curriculum's harness-injected spans). Both zero their dlogits and return, matching the
    // CPU op_cross_entropy's skip.
    if (t >= len || tgt < 0) {
        if (dlogits) for (int j = threadIdx.x; j < V; j += BLOCK) dl[j] = 0.f;
        return;
    }
    const float* lr = logits + static_cast<size_t>(m) * V;
    float mx_partial = -1e30f;
    for (int j = threadIdx.x; j < V; j += BLOCK) mx_partial = fmaxf(mx_partial, lr[j]);
    const float mx = block_reduce_max<BLOCK>(mx_partial);
    float Z_partial = 0.f;
    for (int j = threadIdx.x; j < V; j += BLOCK) Z_partial += __expf(lr[j] - mx);
    const float Z = block_reduce_sum<BLOCK>(Z_partial);
    const float invZ = 1.f / Z;
    // Per-window mean over its ACTIVE (non-ignored) positions: active[b] when a loss mask is in play,
    // else the window's trained length. active == nullptr is the dense/length-only path, where every
    // in-length position is active, so this is bit-identical to before the mask existed (dense parity
    // gate unchanged). The (denom<1?1:denom) guards a fully-masked window -- whose rows all returned
    // above, so the divide never actually runs, but it keeps the weight finite regardless.
    const int   denom = active ? active[b] : len;
    const float w     = 1.0f / (static_cast<float>(batch) * static_cast<float>(denom < 1 ? 1 : denom));
    float ptgt;
    if (dlogits) {
        float ptgt_partial = 0.f;                    // capture before any write (dlogits may alias logits)
        for (int j = threadIdx.x; j < V; j += BLOCK) {
            const float p = __expf(lr[j] - mx) * invZ;
            if (j == tgt) ptgt_partial = p;           // exactly one thread across the block owns j==tgt
            dl[j] = w * (p - (j == tgt ? 1.f : 0.f));// safe in-place: reads lr[j] then writes dl[j]
        }
        ptgt = block_reduce_sum<BLOCK>(ptgt_partial); // sum isolates the one nonzero contributor
    } else {
        // Loss-only: the target's probability is a SINGLE element, so the third O(V) pass (which
        // exists solely to write dlogits) is skipped entirely -- the reduction above would have
        // isolated exactly this value. Every thread computes it redundantly; only thread 0 reads it,
        // and no block-wide collective runs here, which is why the branch must stay block-uniform.
        ptgt = __expf(lr[tgt] - mx) * invZ;
    }
    if (threadIdx.x == 0)
        atomicAdd(loss_acc, static_cast<double>(w * -__logf(fmaxf(1e-9f, ptgt))));
}

// Bias gradient: dbias[o] = sum_m dY[m,o] (one thread per output column). `accumulate` mirrors
// gemm()'s beta: false OVERWRITES dbias (every existing call site), true adds this call's column
// sum into it -- not used by any call site yet (see gemm()'s beta comment); default keeps every
// existing call's behavior unchanged.
__global__ void bias_grad_kernel(const float* __restrict__ dY, float* __restrict__ dbias, int M, int N,
                                 bool accumulate = false) {
    const int o = blockIdx.x * blockDim.x + threadIdx.x;
    if (o >= N) return;
    float s = 0.f;
    for (int m = 0; m < M; ++m) s += dY[static_cast<size_t>(m) * N + o];
    dbias[o] = accumulate ? dbias[o] + s : s;
}
// act-typed bias gradient: dbias[o] = sum_m dY_act[m,o], reading the bf16 store with FP32 accum. This
// kernel's memory pattern is already fine (adjacent threads o, o+1 read adjacent columns at each fixed
// m -- coalesced across the warp); unlike the row-reduction kernels above, there's no block-per-row fix
// to apply here. N is D_FF at this kernel's one call site (a baked constexpr, unlike bias_grad_kernel
// below whose N genuinely varies across ITS call sites -- confirmed by checking every one), so it's
// used directly rather than as a parameter; the M-loop stays runtime (M varies) with nothing to unroll.
// ACC: add instead of assign -- LoopSplit (see launch_linear_bwd_t's accumulate_dw). dbias lands in the
// shared param blob, so a re-executed layer's later runs must not erase the earlier ones.
template <class A, bool ACC = false>
__global__ void bias_grad_act_kernel(const A* __restrict__ dY, float* __restrict__ dbias, int M) {
    constexpr int N = D_FF;
    const int o = blockIdx.x * blockDim.x + threadIdx.x;
    if (o >= N) return;
    float s = 0.f;
    for (int m = 0; m < M; ++m) s += to_f32(dY[static_cast<size_t>(m) * N + o]);
    if constexpr (ACC) dbias[o] += s;
    else               dbias[o]  = s;
}

// RMSNorm backward: one BLOCK per row, threads striding coalesced over D_MODEL (was one thread per
// row -- ncu measured that at 26-28% occupancy, 3% compute throughput, 77-78% of warp cycles stalled
// on L1TEX scoreboard dependencies, because at a fixed loop iteration j, adjacent threads (adjacent
// rows) touch addresses D_MODEL floats apart -- no coalescing across the warp at all). Here thread `k`
// of the block reads column k, k+BLOCK, k+2*BLOCK, ... of the SAME row, so a warp's simultaneous loads
// land in one contiguous span; the row's S = sum_j dy[j]*gamma[j]*x[j] is then a block_reduce_sum
// instead of a sequential accumulation.
//
// Flush a block's per-thread partial dgamma accumulators to global memory: one atomicAdd per channel
// the thread owns, called ONCE per block (after a caller-side grid-stride loop has finished summing
// every row/triple that block was assigned into dg_acc, entirely in registers) rather than once per
// row. Shared by rmsnorm_backward_act_kernel and qknorm_backward_act_kernel below -- both hit the
// identical atomicAdd-fan-in problem (many blocks all incrementing the same handful of gamma-gradient
// addresses; see each kernel's own comment for the ncu evidence) and both fix it the same way, just at
// different WIDTHs (D_MODEL vs D_HEAD) -- so the flush itself is the one piece worth sharing (DRY),
// rather than duplicating this loop in both kernel bodies. NJ is deduced from dg_acc's array extent, so
// the caller's `constexpr int NJ = (WIDTH + BLOCK - 1) / BLOCK;` is the single source of truth for how
// many channels each thread owns.
template <int WIDTH, int BLOCK, int NJ>
__device__ inline void flush_dgamma_partial(const float (&dg_acc)[NJ], float* __restrict__ dg) {
    #pragma unroll
    for (int k = 0, j = threadIdx.x; j < WIDTH; j += BLOCK, ++k) atomicAdd(&dg[j], dg_acc[k]);
}
//
// D_MODEL is baked constexpr (one build = one architecture), not a runtime size -- rmsnorm always
// normalizes the full residual stream, never a differently-sized slice, at every call site in this
// file. Taking it as a compile-time constant instead of a parameter lets the strided loops fully
// unroll (their trip count is then also compile time) and folds the /D_MODEL divide into a constant
// multiply. BLOCK=64 (see launch_rmsnorm_bwd_t) evenly divides every D_MODEL this project currently
// bakes (192/448/768), so no thread does a ragged final iteration.
//
// ACCUMULATES dx into the running residual-stream gradient (+=) and dgamma. Residual input x is the
// store type; dy/dx and dgamma stay F32. F32 build: X=float; BF16: reads the half-width residual.
// Mirrors the CPU formula exactly.
//
// dgamma reduction: ncu-profiled on a real training step (production d448/L11/H7, batch=128, this
// project's Phase-2-audit session) at 619-625us/call, 95.67% achieved occupancy (NOT occupancy-limited
// -- near its own ceiling already), 84.4% memory throughput, but 92-94% of the ~83-90 average stalled
// cycles/instruction attributed to an L1TEX long-scoreboard dependency -- the same atomicAdd-fan-in
// signature qknorm_backward_act_kernel below already documents (there: 75.8% of stall cycles, same root
// cause). The ORIGINAL version launched one block per ROW (`t = blockIdx.x`, grid.x == rows), so every
// one of up to ~32K rows issued its own atomicAdd into one of just D_MODEL addresses -- since occupancy
// was already near-ceiling, more warps-in-flight wasn't the lever; fewer total atomicAdds was. Below,
// each block now grid-strides over MULTIPLE rows (gridDim.x capped by dgamma_grid_blocks/sm_count(),
// see that helper's comment) and keeps a PER-THREAD REGISTER accumulator (dg_acc, one slot per channel
// the thread owns -- BLOCK divides D_MODEL evenly at every dims this project bakes, so no ragged tail)
// across the whole stride loop, flushing to global dgamma with flush_dgamma_partial() ONCE per block
// at the very end instead of once per row. This cuts total atomicAdds (and thus the fan-in on each
// address) from `rows` to `gridDim.x` -- typically 1-2 orders of magnitude at production batch sizes,
// while degenerating EXACTLY to the pre-fix one-block-per-row behavior whenever rows is already small
// (dgamma_grid_blocks returns rows unchanged below its cap), so tiny-dims tests are unaffected. The dx
// math itself is untouched -- only the block/grid structure and the dgamma accumulation path changed,
// so results are numerically equivalent to the old version up to floating-point summation-order
// differences (gated by this project's existing relL2/gradient-check tolerances, not bit-exact).
template <class X, int BLOCK>
__global__ void __launch_bounds__(BLOCK)
rmsnorm_backward_act_kernel(const X* __restrict__ x, const float* __restrict__ gamma,
                            const float* __restrict__ rinv, const float* __restrict__ dy,
                            float* __restrict__ dx, float* __restrict__ dgamma, int rows) {
    constexpr int   C    = D_MODEL;
    constexpr float invC = 1.0f / static_cast<float>(C);
    constexpr int   NJ   = (C + BLOCK - 1) / BLOCK;   // dgamma channels owned per thread (ceil)
    float dg_acc[NJ] = {};
    for (int t = blockIdx.x; t < rows; t += gridDim.x) {
        const X*     xr  = x  + static_cast<size_t>(t) * C;
        const float* dyr = dy + static_cast<size_t>(t) * C;
        float*       dxr = dx + static_cast<size_t>(t) * C;
        float partial = 0.f;
        #pragma unroll
        for (int j = threadIdx.x; j < C; j += BLOCK) partial += dyr[j] * gamma[j] * to_f32(xr[j]);
        const float S = block_reduce_sum<BLOCK>(partial);
        const float r = rinv[t], r3 = r * r * r;
        #pragma unroll
        for (int k = 0, j = threadIdx.x; j < C; j += BLOCK, ++k) {
            const float xj = to_f32(xr[j]), dyj = dyr[j], gj = gamma[j];
            dxr[j] += r * dyj * gj - (xj * r3 * invC) * S;
            dg_acc[k] += dyj * xj * r;
        }
    }
    flush_dgamma_partial<C, BLOCK>(dg_acc, dgamma);
}

// QK-norm backward: same math shape as rmsnorm_backward_act_kernel, grouped per (row, head, {q,k})
// over width DH instead of once per row over D_MODEL (mirrors backend_cpu.cpp's Op::QKNorm case --
// see its comment for why this is a separate op from Op::RMSNorm's backward, not a reshaped reuse of
// it). Reads the pre-norm x from qk_pre (stashed by qknorm_save_act_kernel during the recompute
// pass) and OVERWRITES dqkv's Q/K sub-block IN PLACE: `dqkv` arrives holding dy (grad w.r.t. this
// op's OUTPUT, i.e. RoPE's backward has already converted it from "grad w.r.t. roped Q/K" back to
// "grad w.r.t. qknorm's output" -- see backward_device's call ordering) and leaves holding dx (grad
// w.r.t. this op's INPUT, i.e. the raw QKV-GEMM output, exactly what the qkv-GEMM's own backward
// needs next). r is recomputed from the saved pre-norm x (one extra strided pass) rather than a saved
// rinv -- the reduction is small (D_HEAD wide), so a persistent rinv buffer isn't worth the extra
// scratch allocation. NOTE on precision: qk_pre is stored at act_t (bf16 on a BF16 build, matching this
// file's usual training-activation convention -- see TrainScratch's other bf16 fields), so the x this
// kernel's reduction reads is whatever bf16-rounded value qknorm_save_act_kernel wrote. The recomputed
// r is therefore self-consistent with the SAME rounded x the rest of this exact pipeline already
// operates on, in FP32 math -- NOT a bit-identical replay of the forward's own r (that would require x
// to be resident in F32, which it isn't on a BF16 build). This is the same precision budget every other
// BF16 checkpoint-recompute in this file already accepts (a/qkv/att/fbuf/ff1/gact are all bf16-stored
// and recomputed the same way in backward_device).
//
// dgamma reduction: this kernel's fan-in is H times worse than rmsnorm_backward_act_kernel's -- gamma
// is one [1,DH] vector shared across every row AND every head (rmsnorm's gamma is [1,C], written by M
// writers per address; qknorm's gamma is [1,DH], written by M*H writers per address -- at production
// dims, M=4096,H=7, that was ~28,672 writers contending on just 64 floats). ncu measured this on a real
// training step (backward_device, M=4096, D_MODEL=448/N_HEADS=7/D_HEAD=64, BF16): 411us vs 118us
// (qknorm_act_kernel, forward) and 145us (qknorm_save_act_kernel, the backward recompute pass) -- a
// real ~2.8-3.5x slowdown, NOT explained by occupancy (achieved 48.3% here vs 30.8-35.5% for the other
// two -- MORE warps resident, not fewer) or raw throughput (13.1% compute/memory, the lowest of the
// three). The Warp State Statistics section attributed 75.8% of the average 91.6 stalled
// cycles/instruction to an L1TEX long-scoreboard dependency -- consistent with atomic RMW round-trip
// serialization on the handful of hot addresses, not a bandwidth or compute ceiling. That was left as
// plain per-thread atomicAdd at the time ("fix only once ncu shows it's warranted, not preemptively");
// a later fresh Phase-2-audit profile found rmsnorm_backward_act_kernel above has the IDENTICAL root
// cause (92-94% of its own stalls, same L1TEX-long-scoreboard signature) and is called roughly 2x as
// often per step (always-on, vs qknorm-only-when-enabled) -- making a SHARED fix worth doing for both
// at once rather than treating this as one isolated 6-8%-of-a-step cost. Below: the SAME technique as
// rmsnorm_backward_act_kernel (grid-stride over multiple (row,head) pairs per block instead of one
// block per triple, a per-thread register accumulator across the stride loop, one flush_dgamma_partial
// atomicAdd-per-channel at the end instead of one per triple) -- folding `head` into the same
// grid-stride axis as `row` (not a separate grid dimension) is correct here specifically BECAUSE gamma
// has no per-head sub-range: every head's contribution lands on the exact same DH addresses, so there
// is no reason to keep row and head on separate axes once the goal is cutting the address's total
// writer count. `which` (Q vs K) stays a distinct grid.z, since Q and K write to DIFFERENT gamma buffers
// (dqgamma/dkgamma) with no shared contention to fold together.
template <class A, int H, int KVH, int DH, int BLOCK>
__global__ void __launch_bounds__(BLOCK)
qknorm_backward_act_kernel(const A* __restrict__ qk_pre, const float* __restrict__ qgamma,
                           const float* __restrict__ kgamma, A* __restrict__ dqkv,
                           float* __restrict__ dqgamma, float* __restrict__ dkgamma, int rows) {
    constexpr int   C        = H * DH, CKV = KVH * DH;
    constexpr int   in_stride = C + 2 * CKV, pre_stride = C + CKV;
    constexpr float invDH    = 1.0f / static_cast<float>(DH);
    constexpr float eps      = 1e-5f;
    constexpr int   NJ       = (DH + BLOCK - 1) / BLOCK;   // dgamma channels owned per thread (ceil)
    const int which = blockIdx.z;
    const float* g  = (which == 0) ? qgamma  : kgamma;
    float*       dg = (which == 0) ? dqgamma : dkgamma;
    const int NH    = (which == 0) ? H : KVH;              // K covers fewer heads than Q under GQA
    const int total = rows * NH;                           // flattened (row,head) grid-stride extent
    float dg_acc[NJ] = {};
    for (int rh = blockIdx.x; rh < total; rh += gridDim.x) {
        const int row = rh / NH, h = rh - row * NH;
        const int off = (which == 0) ? (h * DH) : (C + h * DH);
        const A* xr  = qk_pre + static_cast<size_t>(row) * pre_stride + off;
        A*       dyr = dqkv   + static_cast<size_t>(row) * in_stride  + off;   // in: dy, out: dx (in place)
        float ms = 0.f, S = 0.f;
        #pragma unroll
        for (int j = threadIdx.x; j < DH; j += BLOCK) {
            const float xj = to_f32(xr[j]);
            ms += xj * xj;
            S  += to_f32(dyr[j]) * g[j] * xj;
        }
        ms = block_reduce_sum<BLOCK>(ms) * invDH;
        S  = block_reduce_sum<BLOCK>(S);
        const float r = rsqrtf(ms + eps), r3 = r * r * r;
        #pragma unroll
        for (int k = 0, j = threadIdx.x; j < DH; j += BLOCK, ++k) {
            const float xj = to_f32(xr[j]), dyj = to_f32(dyr[j]), gj = g[j];
            st_act(&dyr[j], dyj * r * gj - (xj * r3 * invDH) * S);
            dg_acc[k] += dyj * xj * r;
        }
    }
    flush_dgamma_partial<DH, BLOCK>(dg_acc, dg);
}

// Actual max resident blocks/SM for a SPECIFIC compiled kernel + launch config, queried ONCE via CUDA's
// own occupancy API (cudaOccupancyMaxActiveBlocksPerMultiprocessor) and cached -- the answer is a pure
// function of the compiled kernel's register/shared-mem footprint and never changes at runtime, so one
// query per distinct template instantiation is enough. This replaces an EARLIER cut of this fix that
// used a guessed constant multiplier (32) for dgamma_grid_blocks below: ncu showed that guess produced
// "Waves Per SM: 1.33" (grid size not a whole multiple of the REAL block limit) with a flagged partial
// trailing wave costing up to 50% of the kernel's own runtime -- a real, measured regression from
// guessing instead of asking CUDA directly. BLOCK is fixed at each helper's one real call site (64 for
// rmsnorm, 32 for qknorm), so there is no risk of the same cache serving two different block sizes.
template <class X> inline int rmsnorm_bwd_blocks_per_sm() {
    static int cached = 0;
    if (cached == 0) {
        int blocks = 0;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, rmsnorm_backward_act_kernel<X, 64>, 64, 0);
        cached = (blocks > 0) ? blocks : 1;
    }
    return cached;
}
template <class A, int H, int KVH, int DH> inline int qknorm_bwd_blocks_per_sm() {
    static int cached = 0;
    if (cached == 0) {
        int blocks = 0;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, qknorm_backward_act_kernel<A, H, KVH, DH, 32>, 32, 0);
        cached = (blocks > 0) ? blocks : 1;
    }
    return cached;
}
// Grid-size cap for the dgamma-reducing backward kernels (rmsnorm_backward_act_kernel,
// qknorm_backward_act_kernel): `blocks_per_sm` (the REAL occupancy ceiling, from
// rmsnorm_bwd_blocks_per_sm/qknorm_bwd_blocks_per_sm above) times sm_count() times kDgammaWaves whole
// waves -- landing on a WHOLE number of full-occupancy waves (no partial trailing wave -- see the
// occupancy helpers' own comment for why that matters) while keeping the total block count (= the
// atomicAdd fan-in on dgamma's handful of addresses, now ONE atomicAdd per block instead of one per
// row) far below the row count. Degenerates to exactly `total_rows` blocks (the pre-fix
// one-block-per-unit behavior) whenever total_rows is already <= the computed cap, e.g. every
// small-dims unit test.
constexpr int kDgammaWaves = 4;
inline int dgamma_grid_blocks(long long total_rows, int blocks_per_sm) {
    const long long cap = static_cast<long long>(blocks_per_sm) * sm_count() * kDgammaWaves;
    return static_cast<int>(std::min(total_rows, cap));
}

// Naive causal attention backward (op_attn), HEAD-per-thread: one thread per (window b, head h) owns
// the whole head, so dq/dk/dv accumulate with NO atomics. Superseded on the hot path by the flash
// tiled backward below; KEPT as the on-device parity REFERENCE the tiled kernels are checked against
// (sub0_cuda_attn_bwd_check). The dq/dk/dv buffer must be zeroed before launch. q/k/v and dqkv are
// the store type (F32 build == FP32 path; BF16 reads/writes bf16 with FP32 softmax+accum).
template <class A>
__global__ void attn_backward_head_act_kernel(const A* __restrict__ q, const A* __restrict__ k,
                                          const A* __restrict__ v, const A* __restrict__ dout,
                                          A* __restrict__ dq, A* __restrict__ dk, A* __restrict__ dv,
                                          int batch, int T, int C, int H, int in_stride, int kv_group) {
    const int HKV = H / kv_group;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x; if (idx >= batch * HKV) return;
    // One thread per (batch, KV head), looping over that head's query group. Under MHA kv_group == 1
    // and this is exactly the old one-thread-per-(batch,head) reference. Under GQA the group MUST be
    // serialized inside one thread: dk/dv rows are SHARED by the group and this kernel accumulates
    // into them with +=, so splitting the group across threads would be a read-modify-write race.
    const int b = idx / HKV, hk = idx % HKV, d = C / H;
    const int kv_off = hk * d;
    const float scale = 1.0f / sqrtf(static_cast<float>(d));
    const size_t in_base = static_cast<size_t>(b) * T * in_stride, out_base = static_cast<size_t>(b) * T * C;
    for (int g = 0; g < kv_group; ++g) {
    const int off = (hk * kv_group + g) * d;                 // this query head's Q/dout/dq offset
    for (int i = 0; i < T; ++i) {
        const A* qi = q + in_base + static_cast<size_t>(i) * in_stride + off;
        const A* di = dout + out_base + static_cast<size_t>(i) * C + off;
        float m = -1e30f, Z = 0.f;
        for (int j = 0; j <= i; ++j) { const A* kj = k + in_base + static_cast<size_t>(j) * in_stride + kv_off;
            float s = 0.f; for (int a = 0; a < d; ++a) s += to_f32(qi[a]) * to_f32(kj[a]); s *= scale;
            const float mn = fmaxf(m, s); Z = Z * __expf(m - mn) + __expf(s - mn); m = mn; }
        const float invZ = 1.f / Z; float dot = 0.f;
        for (int j = 0; j <= i; ++j) { const A* kj = k + in_base + static_cast<size_t>(j) * in_stride + kv_off;
            float s = 0.f; for (int a = 0; a < d; ++a) s += to_f32(qi[a]) * to_f32(kj[a]);
            const float p = __expf(s * scale - m) * invZ; const A* vj = v + in_base + static_cast<size_t>(j) * in_stride + kv_off;
            A* dvj = dv + in_base + static_cast<size_t>(j) * in_stride + kv_off; float dp = 0.f;
            for (int a = 0; a < d; ++a) { st_act(&dvj[a], to_f32(dvj[a]) + p * to_f32(di[a])); dp += to_f32(di[a]) * to_f32(vj[a]); }
            dot += p * dp; }
        A* dqi = dq + in_base + static_cast<size_t>(i) * in_stride + off;
        for (int j = 0; j <= i; ++j) { const A* kj = k + in_base + static_cast<size_t>(j) * in_stride + kv_off;
            const A* vj = v + in_base + static_cast<size_t>(j) * in_stride + kv_off; float s = 0.f, dp = 0.f;
            for (int a = 0; a < d; ++a) { s += to_f32(qi[a]) * to_f32(kj[a]); dp += to_f32(di[a]) * to_f32(vj[a]); }
            const float p = __expf(s * scale - m) * invZ, ds = p * (dp - dot) * scale;
            A* dkj = dk + in_base + static_cast<size_t>(j) * in_stride + kv_off;
            for (int a = 0; a < d; ++a) { st_act(&dqi[a], to_f32(dqi[a]) + ds * to_f32(kj[a])); st_act(&dkj[a], to_f32(dkj[a]) + ds * to_f32(qi[a])); } }
    }
    }
}
// ---- Flash-style TILED attention BACKWARD (the hot-path replacement for the two naive kernels) ----
// The naive kernels above are either low-parallelism (head-per-thread: only batch*H threads) or
// atomic-heavy (query-per-thread: bf16 RMW races), and both re-stream K/V/Q from global O(T) times.
// This mirrors the FlashAttention backward: THREE atomic-free kernels, each parallel over the full
// (b,h,T) grid and staging the contracted dimension through shared memory. Per-query softmax stats
// (m_i, 1/Z_i, dot_i = <dout_i, out_i>) are precomputed once, then dq is accumulated query-parallel
// (owned per query -> no atomics) and dk/dv key-parallel (owned per key -> no atomics). Uses the SAVED
// attention output `att` for dot_i (= out_i), so no value re-accumulation is needed to get it. The
// arithmetic matches the naive backward to bf16 rounding (gated on device by sub0_cuda_attn_check).
// The flat per-(b,h,i) stats index is ((b*H)+h)*T + i (H = gridDim.y).

// (1) Per-query stats: m_i (max scaled score), 1/Z_i, and dot_i = sum_{j<=i} p_ij <dout_i, v_j>
//     ( = <dout_i, out_i> ). Two tiled key-passes: pass A gets m/Z from shared K; pass B gets dot in
//     FP32 from shared K+V (recomputed p). Computing dot in fp32 -- rather than from the bf16-rounded
//     saved output -- keeps ds = p*(dp - dot) matching the naive backward through near-cancellation.
template <class A, int HD, int TILE_K>
__global__ void __launch_bounds__(attn_block_q<HD>())
attn_bwd_stats_kernel(const A* __restrict__ q, const A* __restrict__ k, const A* __restrict__ v,
                      const A* __restrict__ dout, float* __restrict__ stat_m, float* __restrict__ stat_invZ,
                      float* __restrict__ stat_dot, int T, int C, int in_stride, int kv_group) {
    __shared__ float Ks[TILE_K * HD];
    __shared__ float Vs[TILE_K * HD];
    const int    b = blockIdx.z, h = blockIdx.y;
    const int    q0 = blockIdx.x * blockDim.x, i = q0 + threadIdx.x;
    const int    off = h * HD;                               // Q / dout offset -- per QUERY head
    const int    kv_off = (h / kv_group) * HD;               // K/V offset -- per KV head
    const float  scale    = rsqrtf(static_cast<float>(HD));
    const size_t in_base  = static_cast<size_t>(b) * T * in_stride;
    const size_t out_base = static_cast<size_t>(b) * T * C;
    float qr[HD], dr[HD];
    if (i < T) {
        const A* qi = q + in_base + static_cast<size_t>(i) * in_stride + off;
        const A* di = dout + out_base + static_cast<size_t>(i) * C + off;
        #pragma unroll
        for (int a = 0; a < HD; ++a) { qr[a] = to_f32(qi[a]); dr[a] = to_f32(di[a]); }
    }
    const int jmax = min(T, q0 + static_cast<int>(blockDim.x));
    float m = -1e30f, Z = 0.f;
    for (int j0 = 0; j0 < jmax; j0 += TILE_K) {                        // pass A: m, Z (shared K)
        const int tk = min(TILE_K, jmax - j0);
        __syncthreads();
        for (int e = threadIdx.x; e < tk * HD; e += blockDim.x) {
            const int jl = e / HD, a = e - jl * HD;
            Ks[e] = to_f32(k[in_base + static_cast<size_t>(j0 + jl) * in_stride + kv_off + a]); }
        __syncthreads();
        if (i < T) { const int jend = min(tk, i - j0 + 1);
            for (int jl = 0; jl < jend; ++jl) { const float* ks = Ks + jl * HD;
                float s = 0.f;
                #pragma unroll
                for (int a = 0; a < HD; ++a) s += qr[a] * ks[a];
                s *= scale;
                const float mn = fmaxf(m, s); Z = Z * __expf(m - mn) + __expf(s - mn); m = mn; } }
    }
    const float invZ = (i < T) ? 1.f / Z : 0.f;
    float dot = 0.f;
    for (int j0 = 0; j0 < jmax; j0 += TILE_K) {                        // pass B: dot = sum p_ij <dout_i, v_j>
        const int tk = min(TILE_K, jmax - j0);
        __syncthreads();
        for (int e = threadIdx.x; e < tk * HD; e += blockDim.x) {
            const int jl = e / HD, a = e - jl * HD;
            const size_t row = in_base + static_cast<size_t>(j0 + jl) * in_stride + kv_off;
            Ks[e] = to_f32(k[row + a]); Vs[e] = to_f32(v[row + a]); }
        __syncthreads();
        if (i < T) { const int jend = min(tk, i - j0 + 1);
            for (int jl = 0; jl < jend; ++jl) { const float* ks = Ks + jl * HD; const float* vs = Vs + jl * HD;
                float s = 0.f, dp = 0.f;
                #pragma unroll
                for (int a = 0; a < HD; ++a) { s += qr[a] * ks[a]; dp += dr[a] * vs[a]; }
                dot += __expf(s * scale - m) * invZ * dp; } }
    }
    if (i < T) {
        const size_t sidx = (static_cast<size_t>(b) * gridDim.y + h) * T + i;
        stat_m[sidx] = m; stat_invZ[sidx] = invZ; stat_dot[sidx] = dot;
    }
}

// (2) dq (query-parallel, tiled over keys): dq_i = sum_{j<=i} ds_ij k_j, owned per query (no atomics).
// LANES threads warp-cooperate per query at large HD (attn_dq_lanes<HD>() above): each owns HALF =
// HD/LANES channels of qr/dr/dqa (register need 3*HALF instead of 3*HD), and the two partial dot
// products (s, dp) that need the FULL HD range are combined via a warp-shuffle XOR between the pair.
// LANES=1 makes lane=0, HALF=HD, qi_local=threadIdx.x, and the `if constexpr` shuffle branch compile
// away entirely -- algebraically IDENTICAL to the pre-split single-thread-per-query kernel, so small-
// HD builds are unaffected in code path or performance.
template <class A, int HD, int TILE_K>
__global__ void __launch_bounds__(attn_block_q<HD>())
attn_bwd_dq_kernel(const A* __restrict__ q, const A* __restrict__ k, const A* __restrict__ v,
                   const A* __restrict__ dout, const float* __restrict__ stat_m,
                   const float* __restrict__ stat_invZ, const float* __restrict__ stat_dot,
                   A* __restrict__ dq, int T, int C, int in_stride, int kv_group) {
    constexpr int LANES = attn_dq_lanes<HD>();
    constexpr int HALF  = HD / LANES;
    __shared__ float Ks[TILE_K * HD];
    __shared__ float Vs[TILE_K * HD];
    const int    b = blockIdx.z, h = blockIdx.y;
    const int    lane     = threadIdx.x % LANES;              // which HALF-channel slice this thread owns
    const int    qi_local = threadIdx.x / LANES;               // this pair's query slot within the block
    const int    q0 = blockIdx.x * (static_cast<int>(blockDim.x) / LANES), i = q0 + qi_local;
    const int    kv_stage_off = (h / kv_group) * HD;           // K/V staging -- per KV head, FULL HD
    const int    off = h * HD + lane * HALF;                    // Q/dout/dq -- per QUERY head, own channel half
    const float  scale    = rsqrtf(static_cast<float>(HD));
    const size_t in_base  = static_cast<size_t>(b) * T * in_stride;
    const size_t out_base = static_cast<size_t>(b) * T * C;
    float qr[HALF], dr[HALF], dqa[HALF];
    float mi = 0.f, invZi = 0.f, doti = 0.f;
    if (i < T) {
        const A* qi_p = q + in_base + static_cast<size_t>(i) * in_stride + off;
        const A* di_p = dout + out_base + static_cast<size_t>(i) * C + off;
        #pragma unroll
        for (int a = 0; a < HALF; ++a) { qr[a] = to_f32(qi_p[a]); dr[a] = to_f32(di_p[a]); dqa[a] = 0.f; }
        const size_t sidx = (static_cast<size_t>(b) * gridDim.y + h) * T + i;
        mi = stat_m[sidx]; invZi = stat_invZ[sidx]; doti = stat_dot[sidx];
    }
    const int jmax = min(T, q0 + static_cast<int>(blockDim.x) / LANES);
    for (int j0 = 0; j0 < jmax; j0 += TILE_K) {
        const int tk = min(TILE_K, jmax - j0);
        __syncthreads();
        for (int e = threadIdx.x; e < tk * HD; e += blockDim.x) {
            const int jl = e / HD, a = e - jl * HD;
            const size_t row = in_base + static_cast<size_t>(j0 + jl) * in_stride + kv_stage_off;
            Ks[e] = to_f32(k[row + a]); Vs[e] = to_f32(v[row + a]); }
        __syncthreads();
        if (i < T) { const int jend = min(tk, i - j0 + 1);
            for (int jl = 0; jl < jend; ++jl) {
                const float* ks = Ks + jl * HD + lane * HALF;   // this thread's channel half of key j
                const float* vs = Vs + jl * HD + lane * HALF;
                float s = 0.f, dp = 0.f;
                #pragma unroll
                for (int a = 0; a < HALF; ++a) { s += qr[a] * ks[a]; dp += dr[a] * vs[a]; }
                if constexpr (LANES > 1) {                      // combine the two lanes' partial dot products
                    const unsigned mask = __activemask();       // pair is always both-active or both-inactive
                    s  += __shfl_xor_sync(mask, s,  1);          // (same query i -> same i<T branch outcome)
                    dp += __shfl_xor_sync(mask, dp, 1);
                }
                const float p = __expf(s * scale - mi) * invZi, ds = p * (dp - doti) * scale;
                #pragma unroll
                for (int a = 0; a < HALF; ++a) dqa[a] += ds * ks[a]; } }
    }
    if (i < T) { A* dqi = dq + in_base + static_cast<size_t>(i) * in_stride + off;
        #pragma unroll
        for (int a = 0; a < HALF; ++a) st_act(&dqi[a], dqa[a]); }
}

// (3) dk/dv (key-parallel, tiled over queries), SPLIT into two kernels by register need. The combined
// kernel held kr+vr+dka+dva = 4*HD float registers per thread -- fine at HD<=64 (256 regs, the
// architectural cap) but AT HD=96 (native's d768 config) that is 384, over the 255-register hard limit,
// so ptxas spilled ~50% of them to local memory (measured: 255 regs used, 984B stack frame, 1172B
// spill store/load -- occupancy-limiting AND adds off-register traffic every launch). dv_j = sum p_ij
// dout_i depends only on p (hence kr, for the QK score) -- NOT on v at all, so it splits cleanly into
// its own 2*HD-register kernel. dk_j = sum ds_ij q_i needs BOTH kr (for p) and vr (for dp, feeding ds)
// together, so the 4->3*HD reduction alone isn't enough at large HD -- but it turns out to have the
// SAME shape as dq's s/dp pair (two independent half-width dot products combined into one scalar), so
// it takes the identical warp-cooperative channel split (attn_dk_lanes<HD>() above), dropping the
// per-thread need to 3*(HD/LANES) and eliminating the spill entirely rather than just bounding it (see
// attn_bwd_dk_kernel below). Splitting trades one dk/dv kernel launch for two independent ones (no
// shared state between them -- same causal tiling, same stats reads); dv drops stat_dot/v/Sd entirely
// (dv doesn't need them), so it is also lighter on shared memory and global bandwidth, not just
// registers.
template <class A, int HD, int TILE_Q>
__global__ void __launch_bounds__(attn_block_q<HD>())
attn_bwd_dv_kernel(const A* __restrict__ q, const A* __restrict__ k,
                   const A* __restrict__ dout, const float* __restrict__ stat_m,
                   const float* __restrict__ stat_invZ,
                   A* __restrict__ dv, int T, int C, int in_stride, int kv_group, int H) {
    __shared__ float Qs[TILE_Q * HD];
    __shared__ float Ds[TILE_Q * HD];
    __shared__ float Sm[TILE_Q], Sz[TILE_Q];
    const int    b = blockIdx.z, hk = blockIdx.y;                         // blockIdx.y is the KV head
    const int    k0 = blockIdx.x * blockDim.x, j = k0 + threadIdx.x;      // this thread's KEY
    const int    kv_off = hk * HD;
    const float  scale    = rsqrtf(static_cast<float>(HD));
    const size_t in_base  = static_cast<size_t>(b) * T * in_stride;
    const size_t out_base = static_cast<size_t>(b) * T * C;
    float kr[HD], dva[HD];
    if (j < T) {
        const A* kj = k + in_base + static_cast<size_t>(j) * in_stride + kv_off;
        #pragma unroll
        for (int a = 0; a < HD; ++a) { kr[a] = to_f32(kj[a]); dva[a] = 0.f; }
    }
    // Every query head sharing this KV head contributes to the SAME dv_j, so they are summed here in
    // registers rather than written separately -- that is what keeps the write atomic-free (see the
    // launcher's comment). kv_group is grid-uniform, so the __syncthreads() below stay collective.
    for (int g = 0; g < kv_group; ++g) {
        const int h = hk * kv_group + g;                                  // query head
        const int q_off = h * HD;
        for (int i0 = k0; i0 < T; i0 += TILE_Q) {                         // only queries i >= j = k0..
            const int ti = min(TILE_Q, T - i0);
            __syncthreads();
            for (int e = threadIdx.x; e < ti * HD; e += blockDim.x) {
                const int il = e / HD, a = e - il * HD;
                Qs[e] = to_f32(q[in_base + static_cast<size_t>(i0 + il) * in_stride + q_off + a]);
                Ds[e] = to_f32(dout[out_base + static_cast<size_t>(i0 + il) * C + q_off + a]); }
            for (int il = threadIdx.x; il < ti; il += blockDim.x) {
                const size_t sidx = (static_cast<size_t>(b) * H + h) * T + (i0 + il);   // stats are per QUERY head
                Sm[il] = stat_m[sidx]; Sz[il] = stat_invZ[sidx]; }
            __syncthreads();
            if (j < T) {
                for (int il = 0; il < ti; ++il) { const int i = i0 + il;
                    if (i < j) continue;                                 // causal: key j only seen by i >= j
                    const float* qs = Qs + il * HD; const float* dsr = Ds + il * HD;
                    float s = 0.f;
                    #pragma unroll
                    for (int a = 0; a < HD; ++a) s += qs[a] * kr[a];
                    const float p = __expf(s * scale - Sm[il]) * Sz[il];
                    #pragma unroll
                    for (int a = 0; a < HD; ++a) dva[a] += p * dsr[a]; }
            }
        }
    }
    if (j < T) {
        A* dvj = dv + in_base + static_cast<size_t>(j) * in_stride + kv_off;
        #pragma unroll
        for (int a = 0; a < HD; ++a) st_act(&dvj[a], dva[a]);
    }
}

// LANES threads warp-cooperate per key at large HD (attn_dk_lanes<HD>() above): each owns HALF =
// HD/LANES channels of kr/vr/dka, and the two partial dot products (s, dp) that need the FULL HD
// range are combined via a warp-shuffle XOR between the pair -- the mirror image of attn_bwd_dq_kernel
// above (there the OWNED state is q/dout and the STREAMED tiles are k/v; here the OWNED state is k/v
// and the STREAMED tiles are q/dout). LANES=1 reduces every formula below to the pre-split
// single-thread-per-key kernel (lane=0, HALF=HD), so small-HD builds are unaffected.
template <class A, int HD, int TILE_Q>
__global__ void __launch_bounds__(attn_block_q<HD>())
attn_bwd_dk_kernel(const A* __restrict__ q, const A* __restrict__ k, const A* __restrict__ v,
                   const A* __restrict__ dout, const float* __restrict__ stat_m,
                   const float* __restrict__ stat_invZ, const float* __restrict__ stat_dot,
                   A* __restrict__ dk, int T, int C, int in_stride, int kv_group, int H) {
    constexpr int LANES = attn_dk_lanes<HD>();
    constexpr int HALF  = HD / LANES;
    __shared__ float Qs[TILE_Q * HD];
    __shared__ float Ds[TILE_Q * HD];
    __shared__ float Sm[TILE_Q], Sz[TILE_Q], Sd[TILE_Q];
    const int    b = blockIdx.z, hk = blockIdx.y;             // blockIdx.y is the KV head
    const int    lane    = threadIdx.x % LANES;               // which HALF-channel slice this thread owns
    const int    j_local = threadIdx.x / LANES;                // this pair's key slot within the block
    const int    k0 = blockIdx.x * (static_cast<int>(blockDim.x) / LANES), j = k0 + j_local;  // this thread's KEY
    const int    off = hk * HD + lane * HALF;                  // this thread's own channel half of the KV head
    const float  scale    = rsqrtf(static_cast<float>(HD));
    const size_t in_base  = static_cast<size_t>(b) * T * in_stride;
    const size_t out_base = static_cast<size_t>(b) * T * C;
    float kr[HALF], vr[HALF], dka[HALF];
    if (j < T) {
        const A* kj = k + in_base + static_cast<size_t>(j) * in_stride + off;
        const A* vj = v + in_base + static_cast<size_t>(j) * in_stride + off;
        #pragma unroll
        for (int a = 0; a < HALF; ++a) { kr[a] = to_f32(kj[a]); vr[a] = to_f32(vj[a]); dka[a] = 0.f; }
    }
    // Same grouping as attn_bwd_dv_kernel: every query head sharing this KV head folds into the SAME
    // dk_j, summed in registers so the final write stays exclusively owned (atomic-free).
    for (int g = 0; g < kv_group; ++g) {
        const int h = hk * kv_group + g;                                  // query head
        const int stage_off = h * HD;                          // unshifted: Qs/Ds staging covers the FULL HD
        for (int i0 = k0; i0 < T; i0 += TILE_Q) {                         // only queries i >= j = k0..
            const int ti = min(TILE_Q, T - i0);
            __syncthreads();
            for (int e = threadIdx.x; e < ti * HD; e += blockDim.x) {
                const int il = e / HD, a = e - il * HD;
                Qs[e] = to_f32(q[in_base + static_cast<size_t>(i0 + il) * in_stride + stage_off + a]);
                Ds[e] = to_f32(dout[out_base + static_cast<size_t>(i0 + il) * C + stage_off + a]); }
            for (int il = threadIdx.x; il < ti; il += blockDim.x) {
                const size_t sidx = (static_cast<size_t>(b) * H + h) * T + (i0 + il);   // stats are per QUERY head
                Sm[il] = stat_m[sidx]; Sz[il] = stat_invZ[sidx]; Sd[il] = stat_dot[sidx]; }
            __syncthreads();
            if (j < T) {
                for (int il = 0; il < ti; ++il) { const int i = i0 + il;
                    if (i < j) continue;                                 // causal: key j only seen by i >= j
                    const float* qs = Qs + il * HD + lane * HALF;    // this thread's channel half of query i
                    const float* dsr = Ds + il * HD + lane * HALF;
                    float s = 0.f, dp = 0.f;
                    #pragma unroll
                    for (int a = 0; a < HALF; ++a) { s += qs[a] * kr[a]; dp += dsr[a] * vr[a]; }
                    if constexpr (LANES > 1) {                  // combine the two lanes' partial dot products
                        const unsigned mask = __activemask();   // pair is always both-active or both-inactive
                        s  += __shfl_xor_sync(mask, s,  1);      // (same key j -> same j<T / i<j branch outcome)
                        dp += __shfl_xor_sync(mask, dp, 1);
                    }
                    const float p = __expf(s * scale - Sm[il]) * Sz[il], dsc = p * (dp - Sd[il]) * scale;
                    #pragma unroll
                    for (int a = 0; a < HALF; ++a) dka[a] += dsc * qs[a]; }
            }
        }
    }
    if (j < T) {
        A* dkj = dk + in_base + static_cast<size_t>(j) * in_stride + off;
        #pragma unroll
        for (int a = 0; a < HALF; ++a) st_act(&dkj[a], dka[a]);
    }
}

// Embedding backward (op_embed x2): scatter-add the residual-stream grad into tok_emb / pos_emb
// rows. Multiple rows map to the same token/position, so accumulation is atomic. One thread per
// (row m, channel j). pos = m % T (position within the window).
// C is D_MODEL at this kernel's one call site -- constexpr-folded; T stays runtime (genuinely
// varies, same reasoning as ce_backward_kernel's T).
__global__ void embed_backward_kernel(const float* __restrict__ dh, const int* __restrict__ ids,
                                      float* __restrict__ dtok, float* __restrict__ dpos,
                                      int M, int T) {
    constexpr int C = D_MODEL;
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    const int m = blockIdx.y * blockDim.y + threadIdx.y;
    if (m < M && j < C) {
        const float g = dh[static_cast<size_t>(m) * C + j];
        atomicAdd(&dtok[static_cast<size_t>(ids[m]) * C + j], g);
        atomicAdd(&dpos[static_cast<size_t>(m % T) * C + j], g);
    }
}

// Token-only embedding backward (RoPE path): scatter the residual-stream grad into tok_emb only
// (no pos_emb, which carries no gradient under RoPE).
// C is D_MODEL at this kernel's one call site -- constexpr-folded.
// `bind`: an overridden row's grad flows into its FRAGMENT rows (compose_bound_scatter, the exact
// adjoint of the forward's compose -- mirrors encode_slot_bwd's dispatch in backend_cpu.cpp's
// Op::Embed backward), never into the slot id's own row; plain rows scatter exactly as before.
__global__ void embed_backward_token_kernel(const float* __restrict__ dh, const int* __restrict__ ids,
                                            float* __restrict__ dtok, int M,
                                            const DevBindings* __restrict__ bind) {
    constexpr int C = D_MODEL;
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    const int m = blockIdx.y * blockDim.y + threadIdx.y;
    if (m >= M || j >= C) return;
    const int ov = binding_override_at(bind, m);
    if (ov >= 0) {
        const int* e = bind->entries + ov * kBindEntryInts;
        compose_bound_scatter(dh + static_cast<size_t>(m) * C, bind->frags + e[0], e[1], e[2],
                              bind->roles, dtok, j);
    } else {
        atomicAdd(&dtok[static_cast<size_t>(ids[m]) * C + j], dh[static_cast<size_t>(m) * C + j]);
    }
}

// Split the fused QKV weight gradient dWqkv[C,QKV_STRIDE] back into the per-projection grads dWq [C,C]
// and dWk/dWv [C,D_KV] at their param-blob offsets (inverse of build_qkv_kernel). One thread per FUSED
// element, matching build_qkv_kernel's mapping so the two stay obviously inverse under GQA's unequal
// sub-block widths. Constexpr-folded at its one call site, same div/mod reasoning as above.
// ACC = accumulate into the destination instead of overwriting -- LoopSplit needs every execution of a
// shared layer to contribute (see launch_linear_bwd_t's accumulate_dw). The fused dW GEMM upstream
// still writes g_tr.dwqkv with beta=0: that is a per-execution TEMP, fully rewritten each time. It is
// only this split, which lands in the shared param blob, that must add.
template <bool ACC>
__global__ void split_dqkv_kernel(const float* __restrict__ dWqkv, float* __restrict__ dWq,
                                  float* __restrict__ dWk, float* __restrict__ dWv) {
    constexpr int C = D_MODEL, KV = sub0::D_KV, W = sub0::QKV_STRIDE;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= C * W) return;
    const int p = idx / W, c = idx - p * W;
    float* dst;
    if (c < C)            dst = &dWq[static_cast<size_t>(p) * C  + c];
    else if (c < C + KV)  dst = &dWk[static_cast<size_t>(p) * KV + (c - C)];
    else                  dst = &dWv[static_cast<size_t>(p) * KV + (c - C - KV)];
    if constexpr (ACC) *dst += dWqkv[idx];
    else               *dst  = dWqkv[idx];
}

// AdamW: block-reduce sum of grad^2 into a double accumulator (matches the CPU double-precision
// global-norm clip). Shared-mem reduction per block, one atomicAdd per block.
__global__ void grad_normsq_kernel(const float* __restrict__ grad, double* __restrict__ normsq, int n) {
    __shared__ double sm[256];
    const int tid = threadIdx.x;
    const int i   = blockIdx.x * blockDim.x + tid;
    double v = 0.0;
    if (i < n) { const double g = grad[i]; v = g * g; }
    sm[tid] = v;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sm[tid] += sm[tid + s];
        __syncthreads();
    }
    if (tid == 0) atomicAdd(normsq, sm[0]);
}

// One-thread kernel: reads the grad L2 normsq the previous kernel just accumulated and writes the
// AdamW clip scale gs = (norm > clip) ? clip/(norm+eps) : 1 -- entirely on-device, so adam_step_kernel
// below can read it via pointer instead of the host needing normsq back just to compute this one
// scalar (see device_adam_step: this is what lets the per-step normsq readback sync disappear).
__global__ void grad_clip_scale_kernel(const double* __restrict__ normsq, float clip,
                                       float* __restrict__ gs) {
    const float norm = sqrtf(static_cast<float>(*normsq));
    *gs = (norm > clip) ? clip / (norm + 1e-6f) : 1.0f;
}

// Weight-decay ranges (sub0::DECAY_RANGES, layout.hpp), mirrored into __constant__ memory once at
// opt_alloc time. Small (a few dozen entries even at production scale -- adjacent decay=true
// PARAM_LAYOUT tensors merge into one range) and read identically by every thread every step, so
// __constant__'s broadcast cache serves it far better than a PARAM_FLOATS-long global-memory mask
// ever could -- this REPLACES that mask buffer entirely (was 1B/param persistent, ~157 MiB at
// production scale; now zero persistent bytes, just this tiny compile-time-sized table).
__constant__ sub0::DecayRange g_decay_ranges[sub0::NUM_DECAY_RANGES];

// Linear scan over g_decay_ranges: NUM_DECAY_RANGES is tiny (a couple ranges per layer, verified by
// layout_tests.cpp's exhaustive coverage test), so this is negligible next to the surrounding
// sqrt/div-heavy Adam math -- no persistent-memory cost, unlike a lookup table would be.
__device__ __forceinline__ bool is_decay_param(int i) {
    #pragma unroll
    for (int r = 0; r < sub0::NUM_DECAY_RANGES; ++r)
        if (i >= static_cast<int>(g_decay_ranges[r].start) && i < static_cast<int>(g_decay_ranges[r].end))
            return true;
    return false;
}

// Muon-routed ranges (sub0::MUON_RANGES, layout.hpp), mirrored into __constant__ memory once at
// opt_alloc time -- ALWAYS uploaded (like g_decay_ranges), independent of whether this particular
// run actually uses Muon: a non-Muon run's ranges simply never match any Muon-eligible-kind
// parameter's grad, which is precisely the correct behavior (adam_step_kernel below falls through
// to its ordinary per-element update for every parameter, exactly as before this feature existed).
__constant__ sub0::DecayRange g_muon_ranges[sub0::NUM_MUON_RANGES];

// Linear scan, same shape/cost as is_decay_param above.
__device__ __forceinline__ bool is_muon_param(int i) {
    #pragma unroll
    for (int r = 0; r < sub0::NUM_MUON_RANGES; ++r)
        if (i >= static_cast<int>(g_muon_ranges[r].start) && i < static_cast<int>(g_muon_ranges[r].end))
            return true;
    return false;
}

// AdamW per-parameter update (op-for-op identical to AdamW::step on the CPU). gs = global grad
// clip scale (device-resident scalar -- see grad_clip_scale_kernel above), bc1/bc2 = bias
// corrections; weight decay (matrices only) comes from is_decay_param(i) above, not a mask buffer.
// `muon_active` (== this step's muon_lr > 0, see device_adam_step) gates the Muon skip: Muon
// ELIGIBILITY (is_muon_param/g_muon_ranges) is a compile-time-derived property of a parameter's
// KIND (Wq/Wk/.../Wg) and is uploaded unconditionally regardless of whether any given run actually
// uses Muon, so checking is_muon_param(i) WITHOUT this gate would skip those parameters' ordinary
// AdamW update on every pure-AdamW GPU run too -- silently leaving the majority of the model's
// trainable weights (every Muon-ELIGIBLE-kind matrix) never updated at all whenever muon_lr<=0.
// (Caught by this port's own CUDA test suite: "CUDA AdamW step matches the CPU optimizer update"
// failed hard -- cos 0.32 instead of >0.7 -- before this gate was added; a genuinely load-bearing
// regression test, not a hypothetical.) When `muon_active` IS true, device_adam_step's per-matrix
// host loop (muon_step_matrix, below) already updated these parameters AND their slice of m[]
// (Muon's own momentum buffer -- see that function's comment) before this kernel runs, so running
// the ordinary Adam update here too would double-update p[] and corrupt Muon's momentum.
__global__ void adam_step_kernel(float* __restrict__ p, const float* __restrict__ grad,
                                 float* __restrict__ m, float* __restrict__ vel, int n,
                                 const float* __restrict__ gs_ptr, float lr,
                                 float b1, float b2, float eps, float wd, float bc1, float bc2,
                                 bool muon_active) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    if (muon_active && is_muon_param(i)) return;
    const float g = grad[i] * (*gs_ptr);
    m[i]   = b1 * m[i]   + (1.f - b1) * g;
    vel[i] = b2 * vel[i] + (1.f - b2) * g * g;
    const float mhat = m[i] / bc1;
    const float vhat = vel[i] / bc2;
    p[i] -= lr * mhat / (sqrtf(vhat) + eps);
    p[i] -= lr * (is_decay_param(i) ? wd : 0.f) * p[i];
}

// --- Muon (hybrid optimizer) elementwise device kernels ---------------------------------------
// GPU port of include/sub0/muon.hpp's Newton-Schulz building blocks + src/backend_cpu.cpp's
// muon_step_one hybrid dispatch. The matrix-product pieces (GEMMs) live further down, past the
// cuBLAS wrapper helpers (gemm_muon, muon_newton_schulz_device) -- these four are the purely
// elementwise pieces, kept here alongside adam_step_kernel since they don't need cuBLAS.

// upd[i] = Nesterov-lookahead gradient; m[i] updated to the new momentum EMA IN PLACE (this is
// g_dev_m's own slice for this parameter, reused as Muon's momentum buffer exactly like the CPU's
// muon_step_one -- g_dev_vel stays untouched/zero for these params, so the checkpoint format needs
// no new field). gs_ptr is the SAME global grad-clip scale the plain-AdamW path uses (device
// pointer, see grad_clip_scale_kernel) -- clipping stays uniform across routings.
__global__ void muon_ema_nesterov_kernel(float* __restrict__ m, const float* __restrict__ grad,
                                         float* __restrict__ upd, int n, float beta,
                                         const float* __restrict__ gs_ptr) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float g = grad[i] * (*gs_ptr);
    const float mnew = beta * m[i] + (1.f - beta) * g;
    m[i]   = mnew;
    upd[i] = (1.f - beta) * g + beta * mnew;
}

// x[i] /= norm, norm = sqrt(*ss_ptr) + 1e-7f -- same epsilon as the CPU reference (muon.hpp), same
// division (not the reciprocal-multiply the CPU file's own TODO comment flags as a possible future
// speedup -- kept as literal division here too, matching the reference exactly rather than
// introducing a new algebraic variant this port wasn't asked to validate). *ss_ptr is accumulated
// in DOUBLE by grad_normsq_kernel, reused verbatim (not a new reduction kernel) -- Phase 1 audit's
// precision recommendation: double for the Frobenius norm only, float everywhere else.
__global__ void muon_normalize_kernel(float* __restrict__ x, int n, const double* __restrict__ ss_ptr) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float norm = sqrtf(static_cast<float>(*ss_ptr)) + 1e-7f;
    x[i] /= norm;
}

// out[i] = alpha*x[i] + beta*y[i]. Used for both of Newton-Schulz's per-iteration elementwise
// combines: B = b*A + c*AA (out==x==A, y==AA, beta=c) and X = a*X + BX (out==x==X, y==BX, beta=1).
// Safe with out aliasing x (each thread only ever reads its own index before writing it).
__global__ void axpby_kernel(float* __restrict__ out, const float* __restrict__ x,
                             const float* __restrict__ y, float alpha, float beta, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = alpha * x[i] + beta * y[i];
}

// Final apply: decoupled weight decay then the orthogonalized-update step, same order as the CPU's
// muon_step_one. `upd` here is the post-Newton-Schulz result, already in p's own native
// [rows,cols] row-major orientation (see muon_newton_schulz_device -- there is no un-transpose step).
__global__ void muon_apply_kernel(float* __restrict__ p, const float* __restrict__ upd, int n,
                                  float lr, float wd, float scale) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    p[i] -= lr * wd * p[i];
    p[i] -= lr * scale * upd[i];
}

// --- Device-pointer launchers (no host alloc; drive the resident forward chain) ---
inline void ensure_stream() { if (!g_stream) cudaStreamCreate(&g_stream); }

// Invalidate the captured CUDA graphs (defined below) -- a changed forward shape or GEMM math
// mode makes them stale, so they must be recaptured. Two graphs share this trigger: g_graph_exec (the
// batched forward, keyed by (batch,T)) and g_decode_graph_exec (the per-token decode step).
void invalidate_graph();
void invalidate_decode_graph();

// Set the cuBLAS handle's math mode directly -- used to compare modes in the benchmark and to
// force a mode in the parity tests, independent of the baked knob. The mode is baked into a
// captured graph's algo, so CHANGING it invalidates both captured graphs.
//
// Re-requesting the mode the handle is ALREADY in is a no-op, graphs included. This is not just an
// optimization: sub0_cuda_forward_loss (the eval path) asks for FP32 math on every call and then
// replays a captured graph, so an unconditional invalidate would force a full recapture per eval
// call and make graph capture pointless there. It is also why run_fwd_bwd's per-step
// set_handle_tf32(CudaTf32::get()) no longer drops the decode graph on every training step.
// g_handle_tf32 tracks what was last successfully pushed to the handle; -1 means "unknown", which
// is the state before the handle exists and after it is destroyed, so the next call always applies.
int g_handle_tf32 = -1;
inline void set_handle_tf32(bool on) {
    const int want = on ? 1 : 0;
    if (g_cublas && g_handle_tf32 == want) return;
    if (g_cublas) {
        cublasSetMathMode(g_cublas, on ? CUBLAS_TF32_TENSOR_OP_MATH : CUBLAS_DEFAULT_MATH);
        g_handle_tf32 = want;
    }
    invalidate_graph();
    invalidate_decode_graph();
}
inline void apply_math_mode() { set_handle_tf32(CudaTf32::get()); }   // production default = baked knob
inline void ensure_cublas() {
    if (!g_cublas) {
        cublasCreate(&g_cublas);
        ensure_stream();
        cublasSetStream(g_cublas, g_stream);   // all GEMMs on the capture stream
        apply_math_mode();
    }
}

// cuBLAS compute type for the FP32-storage GEMMs: when the config bakes GEMM BF16 AND the tensor
// path is enabled (CudaTf32 knob; parity tests force it off), use 32F_FAST_16BF -- inputs/outputs
// stay FP32 in memory, cuBLAS down-converts to BF16 for the tensor cores and accumulates in FP32.
// Otherwise pure FP32. Keeps the parity gate FP32-exact while making "GEMM BF16" genuinely active.
// `force_tc` requests bf16-tensor-core compute (FAST_16BF) whenever GEMM_DTYPE==BF16, independent
// of the CudaTf32 knob -- for GEMMs where cuBLAS's own algorithm heuristic (under the knob's default
// plain CUBLAS_COMPUTE_32F) does not reliably land on a tensor-core kernel by itself. Measured (nsys,
// 2026-07-02): the lm_head GEMM's awkward VOCAB-wide N (e.g. 32873) falls back to a CUDA-core SGEMM
// (magma_sgemmEx_kernel, ~60-100ms/call) while every other GEMM in the same forward/backward pass
// (QKV/attn-out/FFN, all "nicer" N like D_MODEL/D_FF) already lands on a bf16 tensor-op CUTLASS
// kernel (<1ms/call) under that SAME default compute type -- so the rest of the network is already
// running at this precision level; only the lm_head wasn't getting it. Explicit request only for
// that call site; every other caller's already-working behavior is untouched.
inline cublasComputeType_t gemm_compute(bool force_tc = false) {
    return (GEMM_DTYPE == Dtype::BF16 && (force_tc || CudaTf32::get())) ? CUBLAS_COMPUTE_32F_FAST_16BF
                                                                        : CUBLAS_COMPUTE_32F;
}
// Thin cublasGemmEx wrapper (FP32 A/B/C) honoring gemm_compute(); replaces cublasSgemm everywhere.
// beta defaults to 0 (OVERWRITE C) -- the behavior of every pre-existing call site. beta=1
// ACCUMULATES C += A.B; not used by any call site yet (added groundwork for a future row-chunked
// GEMM, e.g. splitting the lm_head backward over M -- see Wave 8 in memory), but exercised directly
// by its own dedicated test so it's verified correct before anything depends on it.
inline void gemm(cublasOperation_t opA, cublasOperation_t opB, int m, int n, int k,
                 const float* A, int lda, const float* B, int ldb, float* C, int ldc,
                 bool force_tc = false, float beta = 0.0f) {
    const float alpha = 1.0f;
    cublasGemmEx(g_cublas, opA, opB, m, n, k, &alpha, A, CUDA_R_32F, lda, B, CUDA_R_32F, ldb,
                 &beta, C, CUDA_R_32F, ldc, gemm_compute(force_tc), CUBLAS_GEMM_DEFAULT);
}
// Muon-dedicated GEMM: same row-major-via-cuBLAS calling convention as gemm() above, but ALWAYS
// CUBLAS_COMPUTE_32F_PEDANTIC, independent of gemm_compute()'s knob-driven TF32/BF16 choice.
// Why this can't just reuse gemm(): run_fwd_bwd sets the SHARED cuBLAS handle to TF32 tensor-op
// math on EVERY training step (set_handle_tf32/apply_math_mode), so a Muon GEMM going through the
// ordinary gemm()/gemm_compute() path could silently inherit reduced-precision math from an
// unrelated forward-pass knob -- a real, project-flagged hazard (a wrong precision choice here
// "silently produces a worse optimizer, not a crash", AGENTS.md). PEDANTIC is the cuBLAS compute
// type documented to disable tensor-core/reduced-precision paths regardless of the handle's math
// mode, so this makes Newton-Schulz's GEMMs immune to that knob by construction, not by convention
// -- see cuda_tests.cpp's TF32-forced-on regression test, the direct proof case for this hazard.
inline void gemm_muon(cublasOperation_t opA, cublasOperation_t opB, int m, int n, int k,
                      const float* A, int lda, const float* B, int ldb, float* C, int ldc) {
    const float alpha = 1.0f, beta = 0.0f;
    cublasGemmEx(g_cublas, opA, opB, m, n, k, &alpha, A, CUDA_R_32F, lda, B, CUDA_R_32F, ldb,
                 &beta, C, CUDA_R_32F, ldc, CUBLAS_COMPUTE_32F_PEDANTIC, CUBLAS_GEMM_DEFAULT);
}
template <class T> constexpr cudaDataType_t cu_type() {
    return std::is_same_v<T, __nv_bfloat16> ? CUDA_R_16BF : CUDA_R_32F;
}
// Mixed-type GemmEx: A/B share type IN, output type OUT; FP32 accumulate always. Used for bf16
// activation GEMMs (IN=bf16 weights+acts) writing either bf16 (chained) or f32 (residual) output.
// beta defaults to 0 (OVERWRITE C), the behavior of every pre-existing call site; beta=1 ACCUMULATES
// C += A.B -- same convention as the FP32 gemm()'s own beta parameter above, added for
// launch_linear_bwd_t's accumulate_dx mode (USE_GATED_FFN's backward needs dfbuf to receive TWO
// contributions, from Wg's and W1's dX GEMMs -- see backward_device's gated FFN branch).
template <class IN, class OUT>
inline void gemm_t(cublasOperation_t opA, cublasOperation_t opB, int m, int n, int k,
                   const IN* A, int lda, const IN* B, int ldb, OUT* C, int ldc, float beta = 0.0f) {
    const float alpha = 1.0f;
    cublasGemmEx(g_cublas, opA, opB, m, n, k, &alpha, A, cu_type<IN>(), lda, B, cu_type<IN>(), ldb,
                 &beta, C, cu_type<OUT>(), ldc, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
}
// Y[M,out] = X[M,in] . W[in,out] (+ bias) via cuBLAS. cublasSgemm is column-major, so we
// compute the column-major Y^T = W^T . X^T -- which, read back row-major, IS Y = X . W:
//   cublasSgemm(N,N, out,M,in, a, W,out, X,in, b, Y,out). Bias (no epilogue in the legacy
// API) is a cheap broadcast-add kernel afterwards. Same default stream as the kernels, so
// ordering is preserved.
// TODO(perf): migrate to cublasLt (cublasLtMatmul) with a fused bias+GELU epilogue to drop the
// separate bias_add/gelu launches. TODO(perf): at K=D_MODEL=96 these GEMMs are launch/occupancy
// bound -- FP16/BF16 storage + tensor cores (or batched-strided GEMM across layers) is the real
// lever; QKV fusion measured neutral at training scale (M=4096).
inline void launch_linear(const float* dX, const float* dW, const float* dB, float* dY,
                          int M, int in, int out, bool force_tc = false) {
    ensure_cublas();
    gemm(CUBLAS_OP_N, CUBLAS_OP_N, out, M, in, dW, out, dX, in, dY, out, force_tc);
    if (dB) {
        // 64-bit: M*out (lm_head: out=VOCAB) overflows int32 well within reachable batch sizes --
        // see bias_add_kernel's comment. grid.x itself stays far under the 2^31-1 hardware cap.
        const long long n = static_cast<long long>(M) * out;
        const int block = 256;
        const int grid = static_cast<int>((n + block - 1) / block);
        bias_add_kernel<<<grid, block, 0, g_stream>>>(dY, dB, M, out);
    }
}
// act-typed linear: X(IN) . W(IN) -> Y(OUT), bias OUT. IN=act_t for bf16 GEMMs (W is the bf16
// mirror); OUT chains bf16 or lands f32 at a residual boundary. No bias on the bf16 chain GEMMs.
template <class IN, class OUT>
inline void launch_linear_t(const IN* dX, const IN* dW, OUT* dY, int M, int in, int out) {
    ensure_cublas();
    gemm_t<IN, OUT>(CUBLAS_OP_N, CUBLAS_OP_N, out, M, in, dW, out, dX, in, dY, out);
}
inline void launch_rmsnorm(const float* dX, const float* dG, float* dY, int T) {
    constexpr int kNormBlock = 64;   // one block per row; divides every D_MODEL this project bakes
    rmsnorm_kernel<kNormBlock><<<T, kNormBlock, 0, g_stream>>>(dX, dG, dY, T);
}
inline void launch_gelu(const float* dX, float* dY, int n) {
    const int block = 256;
    gelu_kernel<<<(n + block - 1) / block, block, 0, g_stream>>>(dX, dY, n);
}
// SwiGLU forward (USE_GATED_FFN): y = silu(gate)*up, F32 buffers (the F32 inference paths -- see
// gelu_kernel/launch_gelu above for the same F32-vs-act_t split). `y` may alias `up` (see swiglu_kernel).
inline void launch_swiglu(const float* dGate, const float* dUp, float* dY, int n) {
    const int block = 256;
    swiglu_kernel<<<(n + block - 1) / block, block, 0, g_stream>>>(dGate, dUp, dY, n);
}
inline void launch_add(const float* dA, const float* dB, float* dC, int n) {
    const int block = 256;
    add_kernel<<<(n + block - 1) / block, block, 0, g_stream>>>(dA, dB, dC, n);
}
template <class A> inline void launch_add_t(const A* dA, const A* dB, A* dC, int n) {
    const int block = 256;
    add_act_kernel<A><<<(n + block - 1) / block, block, 0, g_stream>>>(dA, dB, dC, n);
}
// act-typed FFN launchers (bf16 storage): rmsnorm f32->act, bias-add on act, gelu/gelu-bwd act.
template <class X, class Y>
inline void launch_rmsnorm_train_t(const X* dX, const float* dG, Y* dY, float* dRinv, int T) {
    constexpr int kNormBlock = 64;   // one block per row; divides every D_MODEL this project bakes
    rmsnorm_train_act_kernel<X, Y, kNormBlock><<<T, kNormBlock, 0, g_stream>>>(dX, dG, dY, dRinv, T);
}
// 64-bit index/bound -- same overflow reasoning as bias_add_kernel above (this one is reached at a
// higher batch, N=D_FF scale, but the same UB applies once it is).
template <class A>
__global__ void bias_add_act_kernel(A* __restrict__ Y, const float* __restrict__ bias, int M, int N) {
    const long long i = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < static_cast<long long>(M) * N) st_act(&Y[i], to_f32(Y[i]) + bias[i % N]);
}
template <class A> inline void launch_bias_act(A* dY, const float* dB, int M, int N) {
    const long long n = static_cast<long long>(M) * N;
    const int block = 256;
    const int grid = static_cast<int>((n + block - 1) / block);
    bias_add_act_kernel<A><<<grid, block, 0, g_stream>>>(dY, dB, M, N);
}
template <class A> inline void launch_gelu_t(const A* dX, A* dY, int n) {
    const int block = 256; gelu_act_kernel<A><<<(n + block - 1) / block, block, 0, g_stream>>>(dX, dY, n);
}
template <class A>
__global__ void gelu_backward_act_kernel(const A* __restrict__ x, const A* __restrict__ dy, A* __restrict__ dx, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) st_act(&dx[i], to_f32(dy[i]) * dev_dgelu(to_f32(x[i])));
}
template <class A> inline void launch_gelu_bwd_t(const A* x, const A* dy, A* dx, int n) {
    const int block = 256; gelu_backward_act_kernel<A><<<(n + block - 1) / block, block, 0, g_stream>>>(x, dy, dx, n);
}
// act-typed SwiGLU forward (USE_GATED_FFN): y = silu(gate)*up. `y` may alias `up` (swiglu_act_kernel).
template <class A> inline void launch_swiglu_t(const A* dGate, const A* dUp, A* dY, int n) {
    const int block = 256; swiglu_act_kernel<A><<<(n + block - 1) / block, block, 0, g_stream>>>(dGate, dUp, dY, n);
}
// SwiGLU backward: y = silu(gate)*up -> dgate = dy*up*dsilu(gate), dup = dy*silu(gate). Each thread
// reads gate[i]/up[i]/dy[i] into locals before writing dgate[i]/dup[i], so the caller may alias
// dup == dy for an in-place update (same reasoning as the forward kernel's y==up aliasing) -- dgate
// must NOT alias dy or up (it is written from a fresh read of both, but a caller writing dgate over
// gate/up itself would corrupt the OTHER output's read within the same thread; dgate may safely alias
// a buffer no longer needed by this thread, e.g. the plain-FFN ff1/dff1 checkpoint scratch).
template <class A>
__global__ void swiglu_backward_act_kernel(const A* __restrict__ gate, const A* __restrict__ up,
                                           const A* __restrict__ dy, A* __restrict__ dgate,
                                           A* __restrict__ dup, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float g = to_f32(gate[i]), u = to_f32(up[i]), d = to_f32(dy[i]);
    const float silu  = dev_silu(g);
    const float dsilu = dev_dsilu(g);
    st_act(&dgate[i], d * u * dsilu);
    st_act(&dup[i],   d * silu);
}
template <class A>
inline void launch_swiglu_bwd_t(const A* gate, const A* up, const A* dy, A* dgate, A* dup, int n) {
    const int block = 256;
    swiglu_backward_act_kernel<A><<<(n + block - 1) / block, block, 0, g_stream>>>(gate, up, dy, dgate, dup, n);
}
inline void launch_attn(const float* dQ, const float* dK, const float* dV, float* dOut,
                        int batch, int T, int C, int H, int in_stride, int kv_group) {
    constexpr int HD = D_HEAD, TK = attn_tile_k<HD>();       // head dim baked -> tiled kernel (see above)
    constexpr int BQ = attn_block_q<HD>();
    const dim3 block(BQ);
    const dim3 grid((T + BQ - 1) / BQ, H, batch);
    attn_fwd_tiled_kernel<float, HD, TK><<<grid, block, 0, g_stream>>>(dQ, dK, dV, dOut, T, C, in_stride, kv_group);
}
// RoPE: rotate Q/K (forward) or dQ/dK (backward) in place over the fused [*, in_stride] buffer.
// One thread per (row, pair); grid.x over C/2 pairs, grid.y over batch*T rows. C/H/in_stride dropped
// as parameters -- constexpr-folded inside the kernels themselves, see rope_kernel above.
// ONE LAUNCH PER SUB-BLOCK, each grid.x sized to that sub-block's own pair count -- `which` is a kernel
// argument, not blockIdx.z. A single fused grid.z=2 launch has to size grid.x for the WIDER sub-block
// (Q), so under GQA the K pass launches whole x-blocks that do nothing but return: at D_MODEL=384 with
// D_KV=96 that is 4 of 6 x-blocks on the K pass, i.e. 33% of ALL rope blocks. Measured on qknorm's
// identical shape, such blocks cost very nearly what real ones do (-37.5% blocks gave -36% time), so
// this is not the free early-out it looks like. Identical shape under MHA, where the sub-blocks match.
inline dim3 rope_grid(const dim3& block, int pairs, int batch, int T) {
    return dim3((pairs + block.x - 1) / block.x, (batch * T + block.y - 1) / block.y);
}
inline void launch_rope(float* qkv, int batch, int T) {
    const dim3 block(32, 8);
    rope_kernel<<<rope_grid(block, D_MODEL / 2, batch, T), block, 0, g_stream>>>(qkv, batch, T, ROPE_THETA, 0);
    rope_kernel<<<rope_grid(block, sub0::D_KV / 2, batch, T), block, 0, g_stream>>>(qkv, batch, T, ROPE_THETA, 1);
}
template <class A> inline void launch_rope_t(A* qkv, int batch, int T) {
    const dim3 block(32, 8);
    rope_act_kernel<A><<<rope_grid(block, D_MODEL / 2, batch, T), block, 0, g_stream>>>(qkv, batch, T, ROPE_THETA, 0);
    rope_act_kernel<A><<<rope_grid(block, sub0::D_KV / 2, batch, T), block, 0, g_stream>>>(qkv, batch, T, ROPE_THETA, 1);
}
template <class A> inline void launch_rope_bwd_t(A* dqkv, int batch, int T) {
    const dim3 block(32, 8);
    rope_bwd_act_kernel<A><<<rope_grid(block, D_MODEL / 2, batch, T), block, 0, g_stream>>>(dqkv, batch, T, ROPE_THETA, 0);
    rope_bwd_act_kernel<A><<<rope_grid(block, sub0::D_KV / 2, batch, T), block, 0, g_stream>>>(dqkv, batch, T, ROPE_THETA, 1);
}

// QK-norm: per-head RMSNorm on the Q/K sub-blocks of the fused qkv buffer, in place, right after the
// QKV GEMM and before RoPE (see qknorm_act_kernel above). Grid = (rows, N_HEADS, 2): z=0 covers the Q
// sub-block, z=1 covers K (V, cols [2C,3C), is never touched). N_HEADS/D_HEAD are baked constexpr
// (one build = one architecture), used directly rather than as parameters, same convention as
// rope_kernel's C/H. `rows` (M = batch*T, or 1 for the per-token decode step) rides grid.x, which has
// a far larger CUDA limit than grid.y/z -- same placement rope_kernel gives its own M-sized axis.
template <class A> inline void launch_qknorm_t(A* qkv, const float* qgamma, const float* kgamma, int rows) {
    constexpr int kQkBlock = 32;   // one warp: D_HEAD (32-96 at this project's scales) is an order of
                                    // magnitude under D_MODEL, so block_reduce_sum's BLOCK<=32 path
                                    // (no cross-warp shared-memory step) already covers the reduction.
    // TWO launches, each with grid.y = that pass's OWN head count, rather than one launch at
    // grid.y = N_HEADS covering both and letting the K pass early-out. Under GQA the K pass needs only
    // N_KV_HEADS blocks, so the fused shape launched (N_HEADS - N_KV_HEADS) * rows blocks that did
    // nothing but return -- 37.5% of all blocks at H=8/KV=2, measured 130,000 launched vs 81,250 useful.
    // One extra launch is far cheaper than that. Identical block count under MHA.
    qknorm_act_kernel<A, N_HEADS, N_KV_HEADS, D_HEAD, kQkBlock>
        <<<dim3(rows, N_HEADS),    kQkBlock, 0, g_stream>>>(qkv, qgamma, kgamma, 0);
    qknorm_act_kernel<A, N_HEADS, N_KV_HEADS, D_HEAD, kQkBlock>
        <<<dim3(rows, N_KV_HEADS), kQkBlock, 0, g_stream>>>(qkv, qgamma, kgamma, 1);
}
// Same, but also stashes the pre-norm Q/K into qk_pre[rows,2C] -- see qknorm_save_act_kernel above.
// Used only by backward_device's checkpoint-recompute pass.
template <class A>
inline void launch_qknorm_save_t(A* qkv, A* qk_pre, const float* qgamma, const float* kgamma, int rows) {
    constexpr int kQkBlock = 32;
    qknorm_save_act_kernel<A, N_HEADS, N_KV_HEADS, D_HEAD, kQkBlock>
        <<<dim3(rows, N_HEADS),    kQkBlock, 0, g_stream>>>(qkv, qk_pre, qgamma, kgamma, 0);
    qknorm_save_act_kernel<A, N_HEADS, N_KV_HEADS, D_HEAD, kQkBlock>
        <<<dim3(rows, N_KV_HEADS), kQkBlock, 0, g_stream>>>(qkv, qk_pre, qgamma, kgamma, 1);
}
// QK-norm backward: converts dqkv's Q/K sub-blocks in place from "grad w.r.t. qknorm's output" to
// "grad w.r.t. qknorm's input", reading the pre-norm x from qk_pre -- see qknorm_backward_act_kernel
// above for the full call-ordering reasoning. grid.x is capped via dgamma_grid_blocks (row*head folded
// into one grid-stride axis -- see that kernel's own comment for why); grid.z stays 2 (Q vs K, distinct
// output buffers).
template <class A>
inline void launch_qknorm_bwd_t(const A* qk_pre, const float* qgamma, const float* kgamma,
                                A* dqkv, float* dqgamma, float* dkgamma, int rows) {
    constexpr int kQkBlock = 32;
    const dim3 grid(dgamma_grid_blocks(static_cast<long long>(rows) * N_HEADS,
                                       qknorm_bwd_blocks_per_sm<A, N_HEADS, N_KV_HEADS, D_HEAD>()), 1, 2);
    qknorm_backward_act_kernel<A, N_HEADS, N_KV_HEADS, D_HEAD, kQkBlock><<<grid, kQkBlock, 0, g_stream>>>(
        qk_pre, qgamma, kgamma, dqkv, dqgamma, dkgamma, rows);
}

// Linear backward (mirrors the Op::Linear case): given the forward input X[M,in], weight W[in,out]
// and upstream grad dY[M,out], produce dX[M,in] = dY.W^T and dW[in,out] = X^T.dY via two cuBLAS
// GEMMs (same column-major trick as launch_linear), plus the bias column-sum. With the default
// accumulate=false, dW/dbias write straight into the param grad blob (beta=0; each weight is used
// once per forward). accumulate=true switches ONLY the dW GEMM and the dbias column-sum to +=
// (their M dimension is a reduction axis, so a row-chunked caller must sum partial contributions
// into a pre-zeroed output); not used by any call site yet -- see gemm()'s beta comment. dX stays
// an overwrite either way -- its rows are per-chunk DISJOINT, each written exactly once. dX may be
// null for the bottom of the graph (no input grad needed).
inline void launch_linear_bwd(const float* dX_in, const float* dW_in, const float* dY,
                              float* dX, float* dW, float* dbias, int M, int in, int out,
                              bool force_tc = false, bool accumulate = false) {
    if (dX)
        gemm(CUBLAS_OP_T, CUBLAS_OP_N, in, M, out, dW_in, out, dY, out, dX, in, force_tc);   // dX = dY . W^T
    gemm(CUBLAS_OP_N, CUBLAS_OP_T, out, in, M, dY, out, dX_in, in, dW, out, force_tc,
         accumulate ? 1.0f : 0.0f);                                                          // dW (+)= X^T . dY
    if (dbias) {
        const int block = 128;
        bias_grad_kernel<<<(out + block - 1) / block, block, 0, g_stream>>>(dY, dbias, M, out, accumulate);
    }
}

// Tied-embedding head (USE_TIED_EMBEDDINGS): the LM head reuses tok_emb [V,C] (V=VOCAB rows, C=D_MODEL
// cols -- the SAME table op_embed reads for lookup) instead of a separate [C,V] lm_head matrix + bias.
// Mirrors backend_cpu.cpp's op_tied_head/Op::TiedHead exactly (see that file's comment for why the
// forward pays a transposed-access cost -- inherent to weight tying -- while both backward passes stay
// GEMM-efficient). C/V are runtime parameters (not baked D_MODEL/VOCAB) so a single instantiation
// serves every call site AND a dims-independent self-test at any scale, matching launch_linear's own
// in/out convention.
//
// Forward: logits[M,V] = a[M,C] . tok_emb[V,C]^T. tok_emb is stored row-major [V,C] (needed for
// op_embed's row-major lookup); reading it into cuBLAS with opA=CUBLAS_OP_T recovers exactly tok_emb^T
// as the [C,V] forward "weight" -- the SAME row-major-via-column-major transpose-recovery trick
// launch_linear_bwd's dX GEMM above already relies on, so this is the same gemm() primitive, just a
// new operand-role combination (not new cuBLAS usage). No bias (tied heads drop the head bias too).
inline void launch_tied_head(const float* dA, const float* dTok, float* dLogits,
                              int M, int C, int V, bool force_tc = false) {
    ensure_cublas();
    gemm(CUBLAS_OP_T, CUBLAS_OP_N, V, M, C, dTok, C, dA, C, dLogits, V, force_tc);
}

// Tied-embedding head backward. Two independent GEMMs (no bias):
//   dA[M,C]      = dY[M,V] . tok_emb[V,C]        -- grad into `a` (the rmsnorm_f output). The SAME
//                  forward-shaped GEMM as launch_linear's (tok_emb used UNTRANSPOSED, as a plain
//                  [in=V,out=C] weight) -- axpy-efficient, no transpose needed.
//   dTok[V,C]   += dY[M,V]^T . dA_in[M,C]        -- grad into tok_emb, ACCUMULATING (beta=1.0
//                  unconditionally, never a plain overwrite): this is one of TWO contributions
//                  tok_emb's gradient receives this step, the other being the ordinary embedding-
//                  lookup scatter-add (embed_backward_kernel / embed_backward_token_kernel elsewhere
//                  in backward_device). Matches backend_cpu.cpp's Op::TiedHead / Op::Embed exactly:
//                  both are plain += there too (see backend_cpu.cpp:636-661 and :522-528), so this
//                  stays correct regardless of which of the two writers runs first -- unlike
//                  launch_linear_bwd's own dW (used once per forward, so it defaults to overwrite),
//                  tok_emb here is used TWICE per forward (lookup + head), so it must always add.
// dA may be null if the caller has no use for it (no such caller today; kept for symmetry with
// launch_linear_bwd's own dX-may-be-null convention).
inline void launch_tied_head_bwd(const float* dA_in, const float* dTok, const float* dY,
                                  float* dA, float* dTokGrad, int M, int C, int V, bool force_tc = false) {
    ensure_cublas();
    if (dA) gemm(CUBLAS_OP_N, CUBLAS_OP_N, C, M, V, dTok, C, dY, V, dA, C, force_tc);           // dA = dY . tok_emb
    gemm(CUBLAS_OP_N, CUBLAS_OP_T, C, V, M, dA_in, C, dY, V, dTokGrad, C, force_tc, 1.0f);       // dTok += a^T . dY
}

// act-typed linear backward: bf16 inputs/grads, dW lands F32 in the grad blob, dbias optional F32.
// accumulate_dx (default false, every pre-existing call site's behavior unchanged) switches ONLY the
// dX GEMM to beta=1 (dX += dY.W^T instead of overwrite) -- for a caller whose dX slot receives more
// than one contribution in the same backward pass (USE_GATED_FFN: dfbuf accumulates from both Wg's and
// W1's dX GEMM, mirroring launch_linear_bwd's own FP32 accumulate parameter and its dedicated
// sub0_cuda_test_accumulate_check coverage).
// DwWrite: whether the dW GEMM adds into its destination or overwrites it. Named rather than a second
// defaulted bool because the two modes are NOT interchangeable and the choice is invisible at a
// positional call site -- see kDefaultDwWrite for why a plain default would have been worse still.
enum class DwWrite { Overwrite, Accumulate };
// LoopSplit runs a weight-shared layer more than once per forward, so each execution's dW must survive
// the next. Accumulating is safe unconditionally (backward_device zeroes the grad blob at step start,
// so beta=1 into zeroed memory equals beta=0 for a single write), but the non-looped build keeps
// Overwrite to avoid paying for a read it does not need.
inline constexpr DwWrite kDefaultDwWrite = sub0::LOOP_SPLIT_ON ? DwWrite::Accumulate : DwWrite::Overwrite;
template <class IN, class DX>
inline void launch_linear_bwd_t(const IN* dX_in, const IN* dW16, const IN* dY,
                                DX* dX, float* dW, int M, int in, int out, bool accumulate_dx = false,
                                DwWrite dw = kDefaultDwWrite) {
    if (dX) gemm_t<IN, DX>(CUBLAS_OP_T, CUBLAS_OP_N, in, M, out, dW16, out, dY, out, dX, in,
                           accumulate_dx ? 1.0f : 0.0f);                                        // dX (+)= dY.W^T
    gemm_t<IN, float>(CUBLAS_OP_N, CUBLAS_OP_T, out, in, M, dY, out, dX_in, in, dW, out,
                      dw == DwWrite::Accumulate ? 1.0f : 0.0f);                                 // dW (+)= X^T.dY
}
template <class X> inline void launch_rmsnorm_bwd_t(const X* x, const float* gamma, const float* rinv,
                               const float* dy, float* dx, float* dgamma, int rows) {
    constexpr int kNormBlock = 64;   // divides every D_MODEL this project bakes
    // grid.x capped via dgamma_grid_blocks -- see rmsnorm_backward_act_kernel's own comment for why
    // (bounds the dgamma atomicAdd fan-in to gridDim.x instead of `rows`); degenerates to `rows` blocks
    // (the pre-fix one-block-per-row shape) whenever rows is already small.
    const int blocks = dgamma_grid_blocks(rows, rmsnorm_bwd_blocks_per_sm<X>());
    rmsnorm_backward_act_kernel<X, kNormBlock><<<blocks, kNormBlock, 0, g_stream>>>(
        x, gamma, rinv, dy, dx, dgamma, rows);
}
// act-typed attention forward/backward (bf16 store): same launch shape, store-typed q/k/v/att.
template <class A> inline void launch_attn_train_t(const A* dQ, const A* dK, const A* dV, A* dOut,
                              int batch, int T, int C, int H, int in_stride, int kv_group) {
    constexpr int HD = D_HEAD, TK = attn_tile_k<HD>();       // head dim baked -> tiled kernel (see above)
    constexpr int BQ = attn_block_q<HD>();
    const dim3 block(BQ);
    const dim3 grid((T + BQ - 1) / BQ, H, batch);
    attn_fwd_tiled_kernel<A, HD, TK><<<grid, block, 0, g_stream>>>(dQ, dK, dV, dOut, T, C, in_stride, kv_group);
}

//TODO: We should be able to know the upper lilmit of the scratch size needed for backward, and allocate it once.
// this risks OOM during flight and overhead of alloc/fragmentation.
// Per-query softmax-stats scratch for the flash backward (m_i, 1/Z_i, dot_i), each batch*H*T floats.
// Grown on demand (like the fwd/train scratch) and reused across steps; freed in train_free().
float* g_bwd_m = nullptr; float* g_bwd_invZ = nullptr; float* g_bwd_dot = nullptr;
size_t g_bwd_stats_cap = 0;
inline int ensure_bwd_stats(size_t n) {
    if (n <= g_bwd_stats_cap) return 0;
    if (g_bwd_m)    cudaFree(g_bwd_m);
    if (g_bwd_invZ) cudaFree(g_bwd_invZ);
    if (g_bwd_dot)  cudaFree(g_bwd_dot);
    g_bwd_stats_cap = 0;
    if (cudaMalloc(&g_bwd_m,    n * sizeof(float)) != cudaSuccess) return 1;
    if (cudaMalloc(&g_bwd_invZ, n * sizeof(float)) != cudaSuccess) return 1;
    if (cudaMalloc(&g_bwd_dot,  n * sizeof(float)) != cudaSuccess) return 1;
    g_bwd_stats_cap = n;
    return 0;
}
inline void free_bwd_stats() {
    if (g_bwd_m)    cudaFree(g_bwd_m);
    if (g_bwd_invZ) cudaFree(g_bwd_invZ);
    if (g_bwd_dot)  cudaFree(g_bwd_dot);
    g_bwd_m = g_bwd_invZ = g_bwd_dot = nullptr; g_bwd_stats_cap = 0;
}

// Flash attention backward: stats -> dq (query-parallel) -> dv, dk (key-parallel, register-split; see
// the comment above attn_bwd_dv_kernel), all atomic-free. dqkv need NOT be pre-zeroed (each dq_i /
// dk_j / dv_j is written exactly once, in full).
template <class A> inline void launch_attn_bwd_t(const A* qkv, const A* dout, A* dqkv,
                            int batch, int T, int C, int H, int in_stride, int kv_group) {
    constexpr int HD = D_HEAD, TK = attn_tile_k<HD>(), TQ = attn_tile_q<HD>();
    constexpr int BQ = attn_block_q<HD>();
    constexpr int DQ_LANES = attn_dq_lanes<HD>();
    constexpr int DK_LANES = attn_dk_lanes<HD>();
    if (ensure_bwd_stats(static_cast<size_t>(batch) * H * T)) return;    // OOM -> next sync surfaces it
    const int HKV = H / kv_group;                // KV heads; == H under MHA
    const int KVW = C / kv_group;                // D_KV, the K and V sub-block width
    const dim3 block(BQ);
    const dim3 grid((T + BQ - 1) / BQ, H, batch);
    // dq's block covers BQ/DQ_LANES queries (LANES threads warp-cooperate per query at large HD; see
    // attn_dq_lanes), so its grid.x scales inversely -- more, smaller-coverage blocks. dk is the same
    // idea over keys instead of queries (attn_dk_lanes).
    const dim3 grid_dq((T + BQ / DQ_LANES - 1) / (BQ / DQ_LANES), H, batch);
    // dv/dk are KV-head-parallel, NOT query-head-parallel: under GQA one dk_j/dv_j is shared by
    // kv_group query heads, so a query-head grid would have kv_group blocks racing on the same address.
    // Regrouping to HKV (and summing the group inside the kernel) keeps each output exclusively owned
    // by one block, preserving the atomic-free write. Total work is unchanged -- kv_group times fewer
    // blocks each doing kv_group times the query passes. Under MHA HKV == H and this is the old grid.
    const dim3 grid_dv((T + BQ - 1) / BQ, HKV, batch);
    const dim3 grid_dk((T + BQ / DK_LANES - 1) / (BQ / DK_LANES), HKV, batch);
    const A* q = qkv; const A* k = qkv + C; const A* v = qkv + C + KVW;
    A* dq = dqkv; A* dk = dqkv + C; A* dv = dqkv + C + KVW;
    attn_bwd_stats_kernel<A, HD, TK><<<grid, block, 0, g_stream>>>(q, k, v, dout, g_bwd_m, g_bwd_invZ, g_bwd_dot, T, C, in_stride, kv_group);
    attn_bwd_dq_kernel<A, HD, TK><<<grid_dq, block, 0, g_stream>>>(q, k, v, dout, g_bwd_m, g_bwd_invZ, g_bwd_dot, dq, T, C, in_stride, kv_group);
    attn_bwd_dv_kernel<A, HD, TQ><<<grid_dv, block, 0, g_stream>>>(q, k, dout, g_bwd_m, g_bwd_invZ, dv, T, C, in_stride, kv_group, H);
    attn_bwd_dk_kernel<A, HD, TQ><<<grid_dk, block, 0, g_stream>>>(q, k, v, dout, g_bwd_m, g_bwd_invZ, g_bwd_dot, dk, T, C, in_stride, kv_group, H);
}

// ============================================================================
//  Binding-compose (docs/BACKENDS.md) -- host-side override-table residency
// ============================================================================
// One fixed header (g_dev_bind, what every embed kernel dereferences -- see DevBindings' comment for
// why it must never move) + three grow-on-demand data buffers (the fwd_alloc/train_alloc pattern:
// capacity in ELEMENTS, grown by free+malloc, monotonic grow counter for observability) + the HRR
// role table, uploaded ONCE from the host-built sub0::hrr_role_table (same mt19937/normal_distribution
// sequence as the CPU -- the design doc forbids reproducing the RNG in device code, so the bits are
// identical by construction). Freed by sub0_cuda_shutdown.
DevBindings  g_bind_host = {};          // host mirror of the device header (uploaded on install/clear)
DevBindings* g_dev_bind  = nullptr;     // the fixed device-resident header instance
int*      g_bind_idx         = nullptr; size_t g_bind_idx_cap     = 0;
int*      g_bind_entries     = nullptr; size_t g_bind_entries_cap = 0;   // capacity in INTS (3/entry)
int*      g_bind_frags       = nullptr; size_t g_bind_frags_cap   = 0;
float*    g_bind_roles       = nullptr;
long long g_bind_grows       = 0;       // monotonic (re)allocation count -- test observability

// Allocate the fixed header (+ the role table) once; upload the cleared state so kernels captured
// into a graph before any install read a valid, empty table. Idempotent and cheap after the first
// call -- fwd_alloc runs it on every path that can lead to a kernel launch or a graph capture.
int ensure_bind_hdr() {
    if (g_dev_bind) return 0;
    ensure_stream();
    SUB0_CUDA_CHECK(cudaMalloc(&g_dev_bind, sizeof(DevBindings)));
    const size_t role_n = static_cast<size_t>(sub0::HRR_MAX_POS) * D_MODEL;
    SUB0_CUDA_CHECK(cudaMalloc(&g_bind_roles, role_n * sizeof(float)));
    const std::vector<float>& roles = sub0::hrr_role_table(D_MODEL);   // host-built, program-lifetime
    SUB0_CUDA_CHECK(cudaMemcpyAsync(g_bind_roles, roles.data(), role_n * sizeof(float),
                                    cudaMemcpyHostToDevice, g_stream));
    g_bind_host = DevBindings{ nullptr, nullptr, nullptr, g_bind_roles, 0, 0 };
    SUB0_CUDA_CHECK(cudaMemcpyAsync(g_dev_bind, &g_bind_host, sizeof(DevBindings),
                                    cudaMemcpyHostToDevice, g_stream));
    SUB0_CUDA_CHECK(cudaStreamSynchronize(g_stream));
    return 0;
}

// Grow one binding data buffer to >= need elements (never shrinks -- same policy as fwd/train
// scratch). The header is re-uploaded by the caller after ANY reserve, so a moved buffer pointer
// can never be observed stale by a kernel.
int bind_reserve_one(int** buf, size_t* cap, size_t need) {
    if (need <= *cap) return 0;
    if (*buf) cudaFree(*buf);
    *buf = nullptr; *cap = 0;
    SUB0_CUDA_CHECK(cudaMalloc(buf, need * sizeof(int)));
    *cap = need;
    ++g_bind_grows;
    return 0;
}

void bind_free() {
    if (g_dev_bind)     cudaFree(g_dev_bind);
    if (g_bind_idx)     cudaFree(g_bind_idx);
    if (g_bind_entries) cudaFree(g_bind_entries);
    if (g_bind_frags)   cudaFree(g_bind_frags);
    if (g_bind_roles)   cudaFree(g_bind_roles);
    g_dev_bind = nullptr; g_bind_idx = g_bind_entries = g_bind_frags = nullptr; g_bind_roles = nullptr;
    g_bind_idx_cap = g_bind_entries_cap = g_bind_frags_cap = 0;
    g_bind_host = DevBindings{};
}

// Hard cap on the minibatch the resident device scratch will size to. Decoupled from the CPU's
// data-parallel width: the GPU wants a larger batch (bigger GEMM M = better utilization). The
// scratch is sized to the ACTUAL batch requested (grow-on-demand, see fwd_alloc/train_alloc), so a
// small-batch run does not over-allocate; this is just the ceiling a request is validated against.
// MEASURED 2026-06 on sm_120 (train tok/s): 64->62k 128->105k 256->157k 384->176k 512->191k
// 768->195k 1024->209k -- throughput keeps rising but with diminishing returns past ~512. cudaMalloc
// failure (too big for VRAM) is handled gracefully (the alloc returns nonzero, the caller errors).
// This is just a sanity ceiling: the real per-run upper bound is set by VRAM (memplan::max_batch_for_vram),
// computed at tune time, so bf16's halved footprint can reach batches the old 1024 list could not.
// One number shared with the footprint model and the train stage's token-budget batch scheduler.
constexpr int MAX_FWD_BATCH = sub0::memplan::MAX_DEVICE_BATCH;

//TODO: As with other cases - the sizes could be determiend ahead of time for the execution atleast?
// Resident forward scratch. The batch-dependent buffers are sized to the largest batch seen so far
// (g_fwd_cap, grown on demand); the fused-QKV weight buffers (wqkv) are batch-independent and built
// once at upload. Freed by sub0_cuda_shutdown.
struct FwdScratch {
    int*   dids   = nullptr;
    float* h      = nullptr;
    float* a      = nullptr;
    float* qkv    = nullptr;            // fused [M, QKV_STRIDE] projections (q|k|v sub-blocks)
    float* att    = nullptr;
    float* proj   = nullptr;
    float* fbuf   = nullptr;
    float* ff1    = nullptr;
    float* gact   = nullptr;
    float* ff2    = nullptr;
    float* logits = nullptr;
    float* wqkv[N_LAYERS] = {};         // per-layer fused QKV weight [C, QKV_STRIDE], built once at upload
};
FwdScratch g_fwd;
size_t     g_fwd_rows = 0;             // total [M] ROW capacity the batch-dependent buffers are sized for
                                       // (batch*T row-product, NOT a batch count -- so a varied-T training
                                       // step can trade batch against T inside one reserved budget)
int        g_fwd_full = 0;             // 1 = full inference scratch allocated; 0 = dids-only (training)
long long  g_fwd_grows = 0;            // monotonic (re)allocation count -- test observability
// BF16 builds only: true when the lazily-built F32 wqkv (ensure_wqkv_f32, for sub0_cuda_forward /
// sub0_cuda_forward_one) is stale relative to the current weights -- set by build_qkv_weights() on
// every weight change, cleared once ensure_wqkv_f32() rebuilds it.
bool g_wqkv_f32_dirty = true;

// Batch-independent fused-QKV weight buffer (F32, [C,QKV_STRIDE] per layer). BF16 builds no longer keep
// this resident during training -- build_qkv_weights() below writes the bf16 mirror (g_wqkv16)
// DIRECTLY from Wq/Wk/Wv instead of staging through this F32 buffer, so it's dead weight for a pure
// training run (~108 MiB at production dims). The F32 CUDA inference paths (sub0_cuda_forward,
// sub0_cuda_forward_one) still need it -- see ensure_wqkv_f32() below, which allocates+builds it
// lazily, on demand, only in a process that actually calls them. F32-activation builds are
// unaffected: g_wqkv16 there ALIASES this buffer directly (build_qkv_weights()'s F32 branch), so it
// stays eagerly built exactly as before.
int wqkv_alloc() {
    if constexpr (ACT_DTYPE == Dtype::BF16) return 0;   // handled by ensure_wqkv_f32() / build_qkv_weights()
    if (g_fwd.wqkv[0]) return 0;
    ensure_stream();
    for (int l = 0; l < N_LAYERS; ++l)
        SUB0_CUDA_CHECK(cudaMalloc(&g_fwd.wqkv[l], static_cast<size_t>(D_MODEL) * sub0::QKV_STRIDE * sizeof(float)));
    return 0;
}

// Free only the batch-dependent forward buffers (for a grow-realloc); leaves wqkv intact.
void invalidate_graph();           // fwd: a grow-realloc frees buffers the captured graph references
void invalidate_decode_graph();    // ...and the decode graph, which references the SAME g_fwd buffers
void fwd_free_batch() {
    invalidate_graph();            // the captured forward graph references these buffers -> drop it
    invalidate_decode_graph();     // so does the decode graph (forward_one_device_graphed uses g_fwd too)
    cudaFree(g_fwd.dids); cudaFree(g_fwd.h);    cudaFree(g_fwd.a);    cudaFree(g_fwd.qkv);
    cudaFree(g_fwd.att);  cudaFree(g_fwd.proj); cudaFree(g_fwd.fbuf); cudaFree(g_fwd.ff1);
    cudaFree(g_fwd.gact); cudaFree(g_fwd.ff2);  cudaFree(g_fwd.logits);
    g_fwd.dids = nullptr; g_fwd.h = g_fwd.a = g_fwd.qkv = g_fwd.att = g_fwd.proj = g_fwd.fbuf =
        g_fwd.ff1 = g_fwd.gact = g_fwd.ff2 = g_fwd.logits = nullptr;
    g_fwd_rows = 0;
    g_fwd_full = 0;
}

// Ensure the forward scratch covers batch*T rows (grow-on-demand) plus the wqkv buffers. Capacity is
// the ROW PRODUCT, not the batch: any (batch, T) pair whose batch*T fits the reserved rows is served
// without reallocating, so the training loop can grow its effective batch as T shrinks inside one
// fixed token budget (batch_t*T ~= tuned_batch*SEQ_LEN). T defaults to SEQ_LEN, which reproduces the
// old "sized for `batch` full-length windows" behavior exactly for every caller that passes a plain
// batch. `full` allocates the inference activation scratch (h..logits); training passes full=false
// (it reads from g_tr, only needs dids + wqkv) -- skipping ~6 [M,C]/[M,F]/[M,V] buffers saves
// hundreds of MiB during training.
int fwd_alloc(int batch, bool full = true, int T = SEQ_LEN) {
    if (wqkv_alloc()) return 1;
    if (ensure_bind_hdr()) return 1;   // the embed kernels dereference g_dev_bind unconditionally --
                                       // it must exist (cleared) before any launch or graph capture
    const size_t need = static_cast<size_t>(batch) * static_cast<size_t>(T);
    if (g_fwd_rows >= need && g_fwd.dids && (!full || g_fwd_full)) return 0;   // already big enough
    fwd_free_batch();                                      // grow: drop the old (smaller) buffers
    ++g_fwd_grows;
    const size_t Mm = need;
    const size_t MC = Mm * D_MODEL, MF = Mm * D_FF, MV = Mm * VOCAB;
    SUB0_CUDA_CHECK(cudaMalloc(&g_fwd.dids,   Mm * sizeof(int)));
    if (full) {
        SUB0_CUDA_CHECK(cudaMalloc(&g_fwd.h,      MC * sizeof(float)));
        SUB0_CUDA_CHECK(cudaMalloc(&g_fwd.a,      MC * sizeof(float)));
        SUB0_CUDA_CHECK(cudaMalloc(&g_fwd.qkv,    Mm * sub0::QKV_STRIDE * sizeof(float)));
        SUB0_CUDA_CHECK(cudaMalloc(&g_fwd.att,    MC * sizeof(float)));
        SUB0_CUDA_CHECK(cudaMalloc(&g_fwd.proj,   MC * sizeof(float)));
        SUB0_CUDA_CHECK(cudaMalloc(&g_fwd.fbuf,   MC * sizeof(float)));
        SUB0_CUDA_CHECK(cudaMalloc(&g_fwd.ff1,    MF * sizeof(float)));
        SUB0_CUDA_CHECK(cudaMalloc(&g_fwd.gact,   MF * sizeof(float)));
        SUB0_CUDA_CHECK(cudaMalloc(&g_fwd.ff2,    MC * sizeof(float)));
        SUB0_CUDA_CHECK(cudaMalloc(&g_fwd.logits, MV * sizeof(float)));
    }
    g_fwd_rows = Mm;
    g_fwd_full = full ? 1 : 0;
    return 0;
}

void fwd_free() {
    fwd_free_batch();
    for (int l = 0; l < N_LAYERS; ++l) { cudaFree(g_fwd.wqkv[l]); g_fwd.wqkv[l] = nullptr; }
    g_wqkv_f32_dirty = true;   // BF16: force ensure_wqkv_f32() to rebuild after the next (re)alloc
}

// Resident TRAINING scratch (Phase 2d): the forward saves every activation the backward needs
// (no recompute), plus the gradient temporaries that thread the reverse pass. Sized to a ROW budget
// (batch*T product, grown on demand -- see train_alloc). Freed by sub0_cuda_shutdown.
struct TrainScratch {
    // per-layer saved forward activations (residual stream stored bf16; transient scratch is act_t)
    // Indexed by EXECUTION, not layer: under LoopSplit a layer's weights run more than once and each
    // run has its OWN input/statistics that its own backward step needs. LOOP_EXEC_COUNT == N_LAYERS
    // when LoopSplit is off, so the non-looped build allocates exactly what it always did.
    act_t* h_in [sub0::LOOP_EXEC_COUNT] = {};   // [M,C] layer input (= rmsnorm1 input)
    float* rinv1[sub0::LOOP_EXEC_COUNT] = {};   // [M]   rmsnorm1 reciprocal-rms
    act_t* a    = nullptr;         // [M,C] rmsnorm1 output scratch (bf16) -- recomputed from h_in in backward (not per-layer)
    act_t* qkv  = nullptr;         // [M,QKV_STRIDE] fused q|k|v scratch (bf16) -- recomputed from a in backward (not per-layer)
    act_t* att  = nullptr;         // [M,C] attention output scratch (bf16) -- recomputed from qkv in backward (not per-layer)
    act_t* h_mid[sub0::LOOP_EXEC_COUNT] = {};   // [M,C] after residual-1 (= rmsnorm2 input)
    float* rinv2[sub0::LOOP_EXEC_COUNT] = {};   // [M]   rmsnorm2 reciprocal-rms
    act_t* fbuf = nullptr;         // [M,C] rmsnorm2 output scratch -- recomputed from h_mid in backward (= W1 input)
    act_t* ff1  = nullptr;         // [M,F] pre-GELU scratch -- recomputed from fbuf in backward (not per-layer)
    act_t* gact = nullptr;         // [M,F] GELU output scratch -- recomputed from fbuf in backward (not per-layer)
    // final block
    act_t* h_final = nullptr;      // [M,C] last residual stream (= rmsnorm_f input)
    float* rinv_f  = nullptr;      // [M]
    float* a_final = nullptr;      // [M,C] rmsnorm_f output (= lm_head input)
    float* logits  = nullptr;      // [chunk_rows,V] (chunked -- see head_ce_chunked/g_tr_logits_chunk)
    // gradient temporaries (reused across layers); dh threads the residual stream in F32
    float* dh      = nullptr;      // [M,C] running residual-stream grad
    float* da      = nullptr;      // [M,C] f32 (feeds rmsnorm1 bwd dy)
    act_t* dqkv    = nullptr;      // [M,QKV_STRIDE] bf16
    act_t* datt    = nullptr;      // [M,C] bf16
    float* dfbuf   = nullptr;      // [M,C]
    act_t* dff1    = nullptr;      // [M,F]
    act_t* dgact   = nullptr;      // [M,F]
    float* dlogits = nullptr;      // [chunk_rows,V] (chunked -- see head_ce_chunked/g_tr_logits_chunk)
    float* dwqkv   = nullptr;      // [C,QKV_STRIDE] fused QKV weight-grad temp (per-EXECUTION, overwritten)
    act_t* dh16    = nullptr;      // [M,C] bf16 cast of dh (FFN W2 backward operand)
    double* loss   = nullptr;      // [1] accumulated cross-entropy (device)
    int*   dtargets = nullptr;     // [M] next-token targets for cross-entropy
    int*   lengths  = nullptr;     // [MAX_FWD_BATCH] per-window trained length (padding mask for short
                                   // docs) -- constant-sized (16 KiB) so it covers ANY effective batch
                                   // the row budget admits, decoupled from the rows the scratch grows to
    int*   active   = nullptr;     // [MAX_FWD_BATCH] per-window count of non-ignored targets (the CE
                                   // loss-mask normalizer) -- only uploaded when a target is masked,
                                   // else the CE kernel falls back to `lengths` (see ce_backward_kernel)
    act_t* qk_pre   = nullptr;     // [M,2C] pre-norm Q|K stash (USE_QK_NORM only, else stays null and
                                   // costs nothing) -- see qknorm_save_act_kernel's comment and
                                   // memplan.hpp's train_scratch_bytes qk_pre term for why this can't
                                   // just be recomputed or aliased onto an existing checkpoint buffer:
                                   // qkv itself is overwritten by RoPE right after qknorm in the
                                   // forward graph, so nothing else survives to hold the pre-norm value
                                   // by the time qknorm's own backward (which runs AFTER RoPE's
                                   // backward, since RoPE sits after qknorm going forward) needs it.
};
TrainScratch g_tr;
size_t       g_tr_rows = 0;        // total [M] ROW capacity (batch*T product) the buffers are sized for
size_t       g_tr_logits_chunk = 0; // row CAPACITY of g_tr.logits/dlogits (<= g_tr_rows) -- see
                                    // sub0::memplan::logits_chunk_rows / head_ce_chunked below
long long    g_tr_grows = 0;       // monotonic (re)allocation count -- test observability
// Per-layer bf16 weight mirrors for the FFN GEMMs (built from the F32 master on upload/step).
act_t* g_w1_16[N_LAYERS] = {};     // [C,F]
act_t* g_w2_16[N_LAYERS] = {};     // [F,C]
act_t* g_wg16[N_LAYERS]  = {};     // [C,F] gate matrix mirror -- USE_GATED_FFN only (else stays all-null, unused)
act_t* g_wo16[N_LAYERS]  = {};     // [C,C] attention output proj mirror
act_t* g_wqkv16[N_LAYERS] = {};    // [C,QKV_STRIDE] fused QKV mirror (bf16 acts) / alias under f32

void train_free();                 // fwd: train_alloc frees the old buffers before a grow-realloc
// Row-product capacity, exactly like fwd_alloc above: any (batch, T) with batch*T <= g_tr_rows is
// served without reallocating; T defaults to SEQ_LEN so existing plain-batch callers keep their old
// sizing bit-for-bit. Callers that vary (batch, T) per step should reserve the full budget up front
// (sub0_cuda_train_reserve) so the varying products never trigger a grow-realloc mid-run.
int train_alloc(int batch, int T = SEQ_LEN) {
    const size_t need = static_cast<size_t>(batch) * static_cast<size_t>(T);
    if (g_tr_rows >= need && g_tr.h_in[0]) return 0;       // already big enough
    if (g_tr.h_in[0]) train_free();                        // grow: drop the old (smaller) buffers
    ensure_stream();
    ++g_tr_grows;
    const size_t Mm = need;
    const size_t MC = Mm * D_MODEL, MF = Mm * D_FF;
    // logits/dlogits are CHUNKED (head_ce_chunked processes M in row-chunks of at most this many rows
    // instead of one [M,V] shot -- V dwarfs every other per-row scratch term at this project's typical
    // vocab scale, see sub0::memplan::logits_n_chunks' comment for the derivation) -- allocate only
    // [g_tr_logits_chunk, V], not [Mm, V].
    g_tr_logits_chunk = sub0::memplan::logits_chunk_rows(VOCAB, D_FF, Mm);
    const size_t MV = g_tr_logits_chunk * VOCAB;
    // TODO(mem): BF16 activation storage (sm>=80) would roughly halve the per-layer/final scratch -- the
    // biggest remaining lever for larger batches now that qkv/att/ff1/gact are checkpointed to singles.
    for (int e = 0; e < sub0::LOOP_EXEC_COUNT; ++e) {   // per EXECUTION -- see TrainScratch::h_in
        SUB0_CUDA_CHECK(cudaMalloc(&g_tr.h_in[e],  MC * sizeof(act_t)));
        SUB0_CUDA_CHECK(cudaMalloc(&g_tr.rinv1[e], Mm * sizeof(float)));
        SUB0_CUDA_CHECK(cudaMalloc(&g_tr.h_mid[e], MC * sizeof(act_t)));
        SUB0_CUDA_CHECK(cudaMalloc(&g_tr.rinv2[e], Mm * sizeof(float)));
    }
    SUB0_CUDA_CHECK(cudaMalloc(&g_tr.a,      MC * sizeof(act_t)));   // single (checkpoint scratch, bf16)
    SUB0_CUDA_CHECK(cudaMalloc(&g_tr.fbuf,   MC * sizeof(act_t)));  // single (checkpoint scratch, bf16)
    SUB0_CUDA_CHECK(cudaMalloc(&g_tr.ff1,    MF * sizeof(act_t)));  // single (checkpoint scratch, bf16)
    SUB0_CUDA_CHECK(cudaMalloc(&g_tr.gact,   MF * sizeof(act_t)));  // single (checkpoint scratch, bf16)
    SUB0_CUDA_CHECK(cudaMalloc(&g_tr.qkv,    Mm * sub0::QKV_STRIDE * sizeof(act_t)));  // single (checkpoint scratch, bf16)
    SUB0_CUDA_CHECK(cudaMalloc(&g_tr.att,    MC * sizeof(act_t)));   // single (checkpoint scratch, bf16)
    SUB0_CUDA_CHECK(cudaMalloc(&g_tr.h_final, MC * sizeof(act_t)));
    SUB0_CUDA_CHECK(cudaMalloc(&g_tr.rinv_f,  Mm * sizeof(float)));
    SUB0_CUDA_CHECK(cudaMalloc(&g_tr.a_final, MC * sizeof(float)));
    SUB0_CUDA_CHECK(cudaMalloc(&g_tr.logits,  MV * sizeof(float)));  // [g_tr_logits_chunk,V], chunked
    SUB0_CUDA_CHECK(cudaMalloc(&g_tr.dh,      MC * sizeof(float)));
    SUB0_CUDA_CHECK(cudaMalloc(&g_tr.da,      MC * sizeof(float)));
    SUB0_CUDA_CHECK(cudaMalloc(&g_tr.dqkv,    Mm * sub0::QKV_STRIDE * sizeof(act_t)));
    SUB0_CUDA_CHECK(cudaMalloc(&g_tr.datt,    MC * sizeof(act_t)));
    SUB0_CUDA_CHECK(cudaMalloc(&g_tr.dfbuf,   MC * sizeof(float)));
    SUB0_CUDA_CHECK(cudaMalloc(&g_tr.dff1,    MF * sizeof(act_t)));
    SUB0_CUDA_CHECK(cudaMalloc(&g_tr.dgact,   MF * sizeof(act_t)));
    g_tr.dlogits = g_tr.logits;                 // CE backward is in-place: dlogits overwrites logits (chunked)
    SUB0_CUDA_CHECK(cudaMalloc(&g_tr.dwqkv,   static_cast<size_t>(D_MODEL) * sub0::QKV_STRIDE * sizeof(float)));
    SUB0_CUDA_CHECK(cudaMalloc(&g_tr.dh16,    MC * sizeof(act_t)));   // bf16 cast of dh for FFN W2 bwd
    SUB0_CUDA_CHECK(cudaMalloc(&g_tr.loss,    sizeof(double)));
    SUB0_CUDA_CHECK(cudaMalloc(&g_tr.dtargets, Mm * sizeof(int)));
    // lengths is indexed [0, batch) at step time, where batch can be any value up to MAX_FWD_BATCH
    // that the row budget admits (a short-T step runs MORE windows). Constant-size it to the ceiling
    // (16 KiB) instead of the reserving batch, so no legal (batch, T) pair can overrun it.
    SUB0_CUDA_CHECK(cudaMalloc(&g_tr.lengths,  static_cast<size_t>(MAX_FWD_BATCH) * sizeof(int)));
    SUB0_CUDA_CHECK(cudaMalloc(&g_tr.active,   static_cast<size_t>(MAX_FWD_BATCH) * sizeof(int)));  // CE mask normalizer
    // qk_pre [M,2C]: only when USE_QK_NORM (see the TrainScratch::qk_pre comment) -- zero bytes and
    // stays nullptr on the default (qk-norm off) build.
    if constexpr (USE_QK_NORM) SUB0_CUDA_CHECK(cudaMalloc(&g_tr.qk_pre, Mm * sub0::QK_PRE_STRIDE * sizeof(act_t)));
    g_tr_rows = Mm;
    return 0;
}

void train_free() {
    for (int e = 0; e < sub0::LOOP_EXEC_COUNT; ++e) {   // per EXECUTION -- matches train_alloc above
        cudaFree(g_tr.h_in[e]);  cudaFree(g_tr.rinv1[e]);
        cudaFree(g_tr.h_mid[e]); cudaFree(g_tr.rinv2[e]);
    }
    cudaFree(g_tr.a); cudaFree(g_tr.fbuf); cudaFree(g_tr.ff1); cudaFree(g_tr.gact); cudaFree(g_tr.qkv); cudaFree(g_tr.att);
    cudaFree(g_tr.h_final); cudaFree(g_tr.rinv_f); cudaFree(g_tr.a_final); cudaFree(g_tr.logits);
    cudaFree(g_tr.dh);   cudaFree(g_tr.da);   cudaFree(g_tr.dqkv); cudaFree(g_tr.datt);
    cudaFree(g_tr.dfbuf); cudaFree(g_tr.dff1); cudaFree(g_tr.dgact); // dlogits aliases logits (freed above)
    cudaFree(g_tr.dwqkv); cudaFree(g_tr.dh16); cudaFree(g_tr.loss); cudaFree(g_tr.dtargets); cudaFree(g_tr.lengths); cudaFree(g_tr.active);
    if constexpr (USE_QK_NORM) cudaFree(g_tr.qk_pre);   // no-op (cudaFree(nullptr) is well-defined) when off
    free_bwd_stats();                                   // flash-backward per-query stats scratch
    g_tr = TrainScratch{};
    g_tr_rows = 0;
}

// Resident EVAL scratch (sub0_cuda_forward_loss). Deliberately its OWN small allocation rather than a
// reuse of TrainScratch's dtargets/lengths/active/loss: eval is a forward-only path that must be
// usable without ever calling train_alloc, whose per-execution saved-activation buffers are the
// single largest device allocation this backend makes and are pure waste when no backward will run.
// Everything here is O(rows) ints plus one double -- ~64 KiB at a typical eval shape.
struct EvalScratch {
    int*    targets = nullptr;   // [rows]           next-token ids (LOSS_IGNORE_INDEX-aware, like training)
    int*    lengths = nullptr;   // [MAX_FWD_BATCH]  per-window trained length (padding mask)
    int*    active  = nullptr;   // [MAX_FWD_BATCH]  per-window non-ignored target count (CE normalizer)
    double* loss    = nullptr;   // scalar accumulator
};
EvalScratch g_ev;
size_t      g_ev_rows = 0;

void eval_free() {
    cudaFree(g_ev.targets); cudaFree(g_ev.lengths); cudaFree(g_ev.active); cudaFree(g_ev.loss);
    g_ev = EvalScratch{};
    g_ev_rows = 0;
}

// Grow-on-demand to cover `rows` (= batch*T) target ids. lengths/active are constant-sized to the
// batch ceiling for the same reason train_alloc does it: a short-T call runs MORE windows, so sizing
// them to any single call's batch would let a later, wider call overrun them.
int eval_alloc(size_t rows) {
    if (g_ev.targets && g_ev_rows >= rows) return 0;
    eval_free();
    SUB0_CUDA_CHECK(cudaMalloc(&g_ev.targets, rows * sizeof(int)));
    SUB0_CUDA_CHECK(cudaMalloc(&g_ev.lengths, static_cast<size_t>(MAX_FWD_BATCH) * sizeof(int)));
    SUB0_CUDA_CHECK(cudaMalloc(&g_ev.active,  static_cast<size_t>(MAX_FWD_BATCH) * sizeof(int)));
    SUB0_CUDA_CHECK(cudaMalloc(&g_ev.loss,    sizeof(double)));
    g_ev_rows = rows;
    return 0;
}


// ============================================================================
//  GPU incremental single-token inference (device KV-cache decode)
// ============================================================================
// The device counterpart of the CPU forward_one: a resident per-layer K/V cache + a per-token forward
// over a SINGLE row (reusing the M=1 rmsnorm/linear/gelu/add launches), so autoregressive gen runs on
// the device at O(T) per token instead of re-forwarding the whole context. Positions must stay
// < SEQ_LEN. Dense FP32 params (g_dev_params + the fused g_fwd.wqkv), same weights as sub0_cuda_forward.
float* g_kv_k = nullptr;   // [N_LAYERS * SEQ_LEN * D_KV] -- roped K per (layer, position)
float* g_kv_v = nullptr;   // [N_LAYERS * SEQ_LEN * D_KV] -- V per (layer, position). D_KV, not D_MODEL:
                           // under GQA the cache holds only the SHARED KV heads, which is where GQA's
                           // memory saving actually lands (it is the decode-time KV cache that shrinks).

// [2] = {id, pos} for the current decode token, device-resident. Written by ONE small H2D memcpy
// before each captured-graph replay (see replay_decode_graph below) instead of baking id/pos as
// kernel-launch VALUE arguments -- a graph's node arguments are fixed at capture time, so a per-token
// value would need re-capturing (or per-node patching) every call; reading id/pos from this fixed
// buffer's CONTENTS instead means the graph's structure (and every node's arguments) is IDENTICAL
// across every token, exactly like this file's existing batched-forward capture_graph()/cudaGraphLaunch
// pattern (fixed buffers, changing contents) -- see capture_decode_graph()'s own comment.
int* g_decode_state = nullptr;

inline int kv_alloc() {
    if (g_kv_k) return 0;
    const size_t n = static_cast<size_t>(sub0::LOOP_EXEC_COUNT) * SEQ_LEN * sub0::D_KV;
    if (cudaMalloc(&g_kv_k, n * sizeof(float)) != cudaSuccess) return 1;
    if (cudaMalloc(&g_kv_v, n * sizeof(float)) != cudaSuccess) return 1;
    if (cudaMalloc(&g_decode_state, 2 * sizeof(int)) != cudaSuccess) return 1;
    return 0;
}
inline void kv_free() {
    if (g_kv_k) cudaFree(g_kv_k);
    if (g_kv_v) cudaFree(g_kv_v);
    if (g_decode_state) cudaFree(g_decode_state);
    g_kv_k = g_kv_v = nullptr;
    g_decode_state = nullptr;
}

// Embed one token into h[C] (+ pos_emb[pos] under Absolute; pos_emb == nullptr under RoPE).
// C is D_MODEL at this kernel's one call site (the per-token decode path) -- constexpr-folded.
// Body factored out so the plain (id/pos by VALUE, used by the eager eager forward_one_device path
// below) and the _g graphed variant (id/pos read from a device pointer -- see g_decode_state's own
// comment for why) share one implementation instead of two copies of the same math.
// `bind`: decode reads the SAME installed override table as the batched paths, indexed by the token's
// POSITION `pos` (the decode counterpart of the batched kernels' flat row m) -- so a full forward and
// a token-by-token decode over the same ids compose identically. The header pointer is fixed
// (g_dev_bind), so the graphed variant below stays capture-safe exactly like id_pos.
__device__ inline void embed_one_body(const float* __restrict__ tok_emb, const float* __restrict__ pos_emb,
                                      int id, int pos, float* __restrict__ h, int j,
                                      const DevBindings* __restrict__ bind) {
    constexpr int C = D_MODEL;
    const int ov = binding_override_at(bind, pos);
    float v;
    if (ov >= 0) {
        const int* e = bind->entries + ov * kBindEntryInts;
        v = compose_bound_channel(tok_emb, bind->frags + e[0], e[1], e[2], bind->roles, j);
    } else {
        v = tok_emb[static_cast<size_t>(id) * C + j];
    }
    if (pos_emb) v += pos_emb[static_cast<size_t>(pos) * C + j];
    h[j] = v;
}
__global__ void embed_one_kernel(const float* __restrict__ tok_emb, const float* __restrict__ pos_emb,
                                 int id, int pos, float* __restrict__ h,
                                 const DevBindings* __restrict__ bind) {
    constexpr int C = D_MODEL;
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= C) return;
    embed_one_body(tok_emb, pos_emb, id, pos, h, j, bind);
}
// Graphed-decode variant: id/pos come from g_decode_state[0]/[1] instead of by-value kernel arguments,
// so this node's captured arguments never change between tokens (only the buffer's CONTENTS do) -- see
// g_decode_state's own comment and capture_decode_graph() below.
__global__ void embed_one_kernel_g(const float* __restrict__ tok_emb, const float* __restrict__ pos_emb,
                                   const int* __restrict__ id_pos, float* __restrict__ h,
                                   const DevBindings* __restrict__ bind) {
    constexpr int C = D_MODEL;
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= C) return;
    embed_one_body(tok_emb, pos_emb, id_pos[0], id_pos[1], h, j, bind);
}
// RoPE one row: rotate q (cols [0,D_MODEL)) then k (cols [QKV_K_OFF,QKV_V_OFF)) of a fused qkv row at pos
// (mirrors rope_kernel with t = pos). V (cols [2C,3C)) untouched. C/H constexpr-folded (D_MODEL/
// N_HEADS, its one call site) -- see rope_kernel above; pos stays runtime (the decode position). Body
// factored out for the same reason as embed_one_body above.
// blockIdx.y selects the sub-block (0 = Q, 1 = K), exactly as rope_kernel's blockIdx.z does -- Q and K
// no longer have the same head count under GQA, so one thread can no longer rotate both.
__device__ inline void rope_one_body(float* __restrict__ qkv, int pos, float theta, int pg, int which) {
    rope_pair_body(qkv, pos, theta, pg, which, +1.0f);
}
__global__ void rope_one_kernel(float* __restrict__ qkv, int pos, float theta, int which) {
    const int pg = blockIdx.x * blockDim.x + threadIdx.x;
    if (pg >= rope_sub_pairs(which)) return;
    rope_one_body(qkv, pos, theta, pg, which);
}
// Graphed-decode variant: pos read from g_decode_state[1] -- see embed_one_kernel_g's own comment.
__global__ void rope_one_kernel_g(float* __restrict__ qkv, const int* __restrict__ pos_ptr, float theta, int which) {
    const int pg = blockIdx.x * blockDim.x + threadIdx.x;
    if (pg >= rope_sub_pairs(which)) return;
    rope_one_body(qkv, *pos_ptr, theta, pg, which);
}
// Appends this token's K/V (from the just-computed qkv row) into the per-layer KV-cache at position
// *pos_ptr -- the GRAPHED-decode counterpart of forward_one_device's two raw cudaMemcpyAsync D2D calls.
// A captured memcpy node's destination ADDRESS is fixed at capture time, so appending at a NEW position
// every token (a genuinely different device address per call, not just a different scalar value) needs
// the offset computed INSIDE a kernel that reads pos from a device pointer, rather than a host-computed
// destination pointer baked into the node at capture time. kcache_base/vcache_base are this LAYER's
// cache base (g_kv_k/g_kv_v + l*SEQ_LEN*D_KV) -- fixed per node, since a given layer's node is always that
// same layer across every replay; only `pos` (read from device memory) varies.
__global__ void kv_append_kernel(const float* __restrict__ qkv, float* __restrict__ kcache_base,
                                 float* __restrict__ vcache_base, const int* __restrict__ pos_ptr) {
    constexpr int KV = sub0::D_KV;                       // cache rows are D_KV wide, not D_MODEL
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= KV) return;
    const int pos = *pos_ptr;
    kcache_base[static_cast<size_t>(pos) * KV + j] = qkv[sub0::QKV_K_OFF + j];
    vcache_base[static_cast<size_t>(pos) * KV + j] = qkv[sub0::QKV_V_OFF + j];
}
// Decode attention: the single (roped) query q[C] attends the cached K/V (rows over [SEQ_LEN,C]) for
// j=0..pos -> att[C]. One block per head; blockDim=128. Shared: the query head + scores[pos+1]. The
// math is the last-query row of op_attn, so att matches the full forward's final row to fp tolerance.
// C is D_MODEL at its one call site -- constexpr-folded. H was a dead parameter (never read in the
// body; the caller uses it only to size the launch's grid.x, one block per head, not passed in here).
// Body factored out for the same reason as embed_one_body above.
template <int HD>
__device__ inline void attn_decode_body(const float* __restrict__ q, const float* __restrict__ kcache,
                                        const float* __restrict__ vcache, float* __restrict__ att, int pos) {
    extern __shared__ float sh[];                 // qh[HD] then sc[pos+1]
    float* qh = sh;
    float* sc = sh + HD;
    __shared__ float red[128];
    const int hh = blockIdx.x, off = hh * HD, tid = threadIdx.x, nt = blockDim.x;
    const int kv_off = (hh / sub0::GQA_GROUP) * HD;    // this query head's shared KV head in the cache
    const float scale = rsqrtf(static_cast<float>(HD));
    for (int a = tid; a < HD; a += nt) qh[a] = q[off + a];
    __syncthreads();
    for (int j = tid; j <= pos; j += nt) {         // scaled scores q.k_j
        const float* kj = kcache + static_cast<size_t>(j) * sub0::D_KV + kv_off;
        float s = 0.f;
        #pragma unroll
        for (int a = 0; a < HD; ++a) s += qh[a] * kj[a];
        sc[j] = s * scale;
    }
    __syncthreads();
    float lm = -1e30f;                             // block-reduce max
    for (int j = tid; j <= pos; j += nt) lm = fmaxf(lm, sc[j]);
    red[tid] = lm; __syncthreads();
    for (int s = nt / 2; s > 0; s >>= 1) { if (tid < s) red[tid] = fmaxf(red[tid], red[tid + s]); __syncthreads(); }
    const float mx = red[0]; __syncthreads();
    float lz = 0.f;                                // block-reduce sum(exp)
    for (int j = tid; j <= pos; j += nt) lz += __expf(sc[j] - mx);
    red[tid] = lz; __syncthreads();
    for (int s = nt / 2; s > 0; s >>= 1) { if (tid < s) red[tid] += red[tid + s]; __syncthreads(); }
    const float invZ = 1.f / red[0]; __syncthreads();
    for (int j = tid; j <= pos; j += nt) sc[j] = __expf(sc[j] - mx) * invZ;   // p_j
    __syncthreads();
    for (int a = tid; a < HD; a += nt) {           // att = sum_j p_j v_j
        float acc = 0.f;
        for (int j = 0; j <= pos; ++j) acc += sc[j] * vcache[static_cast<size_t>(j) * sub0::D_KV + kv_off + a];
        att[off + a] = acc;
    }
}
template <int HD>
__global__ void attn_decode_kernel(const float* __restrict__ q, const float* __restrict__ kcache,
                                   const float* __restrict__ vcache, float* __restrict__ att, int pos) {
    attn_decode_body<HD>(q, kcache, vcache, att, pos);
}
// Graphed-decode variant: pos read from g_decode_state[1] -- see embed_one_kernel_g's own comment.
template <int HD>
__global__ void attn_decode_kernel_g(const float* __restrict__ q, const float* __restrict__ kcache,
                                     const float* __restrict__ vcache, float* __restrict__ att,
                                     const int* __restrict__ pos_ptr) {
    attn_decode_body<HD>(q, kcache, vcache, att, *pos_ptr);
}

// ============================================================================
//  Per-layer parameter slot offsets (PARAM_LAYOUT indices) -- single source of truth
// ============================================================================
// Every layer occupies a FIXED run of consecutive PARAM_LAYOUT slots: 10 without QK-norm/gated-FFN
// (ln1,ln2,Wq,Wk,Wv,Wo,W1,b1,W2,b2), +2 with QK-norm (q_norm,k_norm inserted between Wo and the FFN
// block), and the FFN block itself is 4 slots (W1,b1,W2,b2) normally or 3 (Wg,W1,W2, no biases) when
// USE_GATED_FFN -- see layout.hpp's make_param_layout. The functions below used to compute "+5 for Wo",
// "+6 for W1", etc. independently at every call site (forward_one_device, forward_device,
// build_qkv_weights x2, ensure_wqkv_f32, forward_train, head_ce_chunked, backward_device -- 12 sites
// total) -- adding USE_QK_NORM's 2 new slots (or USE_GATED_FFN's shrunk FFN block) would have needed
// every one of those updated in lock-step, with no compiler help if one was missed (a wrong offset
// silently reads/writes the wrong tensor's data; it doesn't fail to compile or even necessarily fail at
// runtime). These named constants are now the ONE place that arithmetic lives; every call site indexes
// `L[layer_base(l) + kWo]` etc. instead. check_layer_offsets()'s static_assert below cross-checks
// them against the REAL PARAM_LAYOUT table (not just against each other), so a mistake here is a
// compile error, not a silent wrong-tensor read.
constexpr int kLn1 = 0, kLn2 = 1, kWq = 2, kWk = 3, kWv = 4, kWo = 5;
constexpr int kQNorm = 6, kKNorm = 7;                        // present only when USE_QK_NORM (else unused)
constexpr int kFfnBase = USE_QK_NORM ? 8 : 6;
constexpr int kWg  = kFfnBase;                               // gate matrix -- USE_GATED_FFN only (else unused)
constexpr int kW1  = kFfnBase + (USE_GATED_FFN ? 1 : 0);     // gated: "up" projection; plain: the only FFN-in weight
constexpr int kW2  = kW1 + (USE_GATED_FFN ? 1 : 2);          // gated: Wg,W1,W2 (3 slots); plain: W1,b1,W2 (3 slots to here)
constexpr int kB1  = kW1 + 1, kB2 = kW1 + 3;                 // plain (non-gated) FFN bias slots only -- MEANINGLESS
                                                              // (and, on the last layer of a tied+gated build,
                                                              // OUT OF BOUNDS) when USE_GATED_FFN. Every call site
                                                              // reading L[.. + kB1/kB2] MUST be guarded
                                                              // `if constexpr (!USE_GATED_FFN)` -- see the b1/b2
                                                              // pointer inits below, none may run unconditionally.
constexpr int kLPer = 2 + 4 + (USE_QK_NORM ? 2 : 0) + (USE_GATED_FFN ? 3 : 4);  // params per transformer block
// Leading params before the first block: tok_emb always, pos_emb only under absolute positions
// (layout.hpp's HAS_POS_EMB -- RoPE omits the table entirely, which is what decouples SEQ_LEN from
// the checkpoint). check_layer_offsets()'s static_assert below is the guard against this drifting.
constexpr int kHead = sub0::HAS_POS_EMB ? 2 : 1;
constexpr int layer_base(int l) { return kHead + kLPer * l; }
constexpr int kFinalBase = kHead + kLPer * N_LAYERS;         // index of ln_f (tail block start)
constexpr int kLnF = 0, kLmHead = 1, kLmBias = 2;            // offsets within the tail block

consteval bool check_layer_offsets() {
    using sub0::PKind;
    const auto& Lc = sub0::PARAM_LAYOUT;
    for (int l = 0; l < N_LAYERS; ++l) {
        const int b = layer_base(l);
        if (Lc[b + kLn1].kind != PKind::Ln1 || Lc[b + kLn2].kind != PKind::Ln2 ||
            Lc[b + kWq].kind  != PKind::Wq  || Lc[b + kWk].kind  != PKind::Wk  ||
            Lc[b + kWv].kind  != PKind::Wv  || Lc[b + kWo].kind  != PKind::Wo)
            return false;
        if constexpr (USE_QK_NORM) {
            if (Lc[b + kQNorm].kind != PKind::QNorm || Lc[b + kKNorm].kind != PKind::KNorm) return false;
        }
        if constexpr (USE_GATED_FFN) {
            if (Lc[b + kWg].kind != PKind::Wg || Lc[b + kW1].kind != PKind::W1 ||
                Lc[b + kW2].kind != PKind::W2)
                return false;
        } else {
            if (Lc[b + kW1].kind != PKind::W1 || Lc[b + kB1].kind != PKind::B1 ||
                Lc[b + kW2].kind != PKind::W2 || Lc[b + kB2].kind != PKind::B2)
                return false;
        }
    }
    return Lc[kFinalBase + kLnF].kind == PKind::LnF;
}
static_assert(check_layer_offsets(),
    "layer_base()/kLn1../kFinalBase drifted from the real PARAM_LAYOUT table -- update the constants "
    "above to match make_param_layout() in layout.hpp.");

// The fused-buffer kernels above are templated on their OWN head counts (H/KVH/DH) so the dims-
// independent self-test can instantiate toy shapes in this same binary, which means they re-derive
// `in_stride = C + 2*CKV` and `pre_stride = C + CKV` locally instead of using layout.hpp's constants.
// That is deliberate, but it leaves two independent derivations of one layout with nothing forcing
// them to agree -- exactly the shape that already produced two real bugs in this feature set
// (memplan::param_floats kept counting a removed pos_emb; kFootprintDims under-predicted VRAM). Pin
// the PRODUCTION instantiation to the constants so a future divergence is a compile error.
static_assert(N_HEADS * D_HEAD + 2 * (N_KV_HEADS * D_HEAD) == sub0::QKV_STRIDE,
    "the attention kernels' locally-derived fused QKV row width disagrees with layout.hpp's QKV_STRIDE");
static_assert(N_HEADS * D_HEAD + (N_KV_HEADS * D_HEAD) == sub0::QK_PRE_STRIDE,
    "the qknorm kernels' locally-derived qk_pre row width disagrees with layout.hpp's QK_PRE_STRIDE");

// Grouped-query attention IS supported here (the guard that used to sit at this spot is gone). Two
// things it required, both worth knowing before touching the attention kernels:
//   1. The fused Q/K/V buffer is [M, QKV_STRIDE] with UNEQUAL sub-blocks (Q is D_MODEL, K and V are
//      D_KV) -- see layout.hpp's QKV_* constants. Nothing may assume a 3*D_MODEL stride or a
//      sub-block at s*D_MODEL any more.
//   2. dk/dv are still atomic-free, but for a NEW reason. They used to be safe because each dk_j/dv_j
//      was written by exactly one block when grid.y ran over query heads. Under GQA a KV head is
//      shared by GQA_GROUP query heads, so those kernels are now KV-HEAD-parallel (grid.y = KV heads)
//      and sum the group in registers before a single write -- ownership preserved without atomics.
//      Note the consequence: in those two kernels gridDim.y is the KV head count, NOT the query head
//      count, so the per-(b,h,i) softmax-stats index takes H as an explicit argument instead of
//      reading gridDim.y (which is what it used to do, and which would now silently mis-index).

// LoopSplit IS supported here (the guard that used to sit at this spot is gone). What it required:
//   1. Every PARAM-GRAD write accumulates instead of overwriting, because a weight-shared layer runs
//      more than once per forward and each execution contributes. See launch_linear_bwd_t's
//      accumulate_dw, split_dqkv_kernel's ACC and bias_grad_*_kernel's ACC -- all gated on
//      LOOP_SPLIT_ON, so the non-looped path keeps its cheaper write-without-read. Grad writes that
//      already went through atomicAdd (dgamma, embeddings) needed nothing: backward_device zeroes the
//      whole grad blob at step start, so they were always accumulating into a clean buffer.
//   2. Activation checkpoints and decode KV slots are indexed by EXECUTION, not layer (TrainScratch's
//      h_in/rinv1/h_mid/rinv2 and g_kv_k/g_kv_v), since each run of a shared layer has its own input
//      and its own statistics that its own backward step needs.
//   3. All four forward paths plus the backward walk sub0::LAYER_EXEC_ORDER rather than 0..N_LAYERS.
// The failure mode this replaced was NOT a crash: it was a quietly wrong gradient with only the last
// execution's contribution surviving, which looks like a run that merely learns worse.

// One decode step on the device: token `id` at window position `pos`, reusing the M=1 dense launches
// and the K/V cache. Writes logits into g_fwd.logits[0..VOCAB). Requires uploaded params + wqkv +
// fwd_alloc(1) + kv_alloc(); the caller resets pos to 0 at the start of a sequence.
void forward_one_device(int id, int pos) {
    const auto&  L  = sub0::PARAM_LAYOUT;
    const int    C  = D_MODEL, F = D_FF, V = VOCAB, H = N_HEADS;
    float* const base = g_dev_params;
    const float* tok_emb = base + L[0].off;
    const float* pos_emb = sub0::HAS_POS_EMB ? base + L[1].off : nullptr;   // absent under RoPE
    const int    fi      = kFinalBase;
    const float* ln_f    = base + L[fi + kLnF].off;
    [[maybe_unused]] const float* lm_head = nullptr;
    [[maybe_unused]] const float* lm_bias = nullptr;
    if constexpr (!USE_TIED_EMBEDDINGS) { lm_head = base + L[fi + kLmHead].off; lm_bias = base + L[fi + kLmBias].off; }
    float* h = g_fwd.h; float* a = g_fwd.a; float* qkv = g_fwd.qkv; float* att = g_fwd.att;
    float* proj = g_fwd.proj; float* fbuf = g_fwd.fbuf; float* ff1 = g_fwd.ff1; float* gact = g_fwd.gact; float* ff2 = g_fwd.ff2;

    { const int blk = 256; const float* pe = (POS_ENCODING == PosEncoding::Absolute) ? pos_emb : nullptr;
      embed_one_kernel<<<(C + blk - 1) / blk, blk, 0, g_stream>>>(tok_emb, pe, id, pos, h, g_dev_bind); }
    for (int e = 0; e < sub0::LOOP_EXEC_COUNT; ++e) {   // EXECUTIONS, not layers (LoopSplit)
        const int    l   = sub0::LAYER_EXEC_ORDER[e];
        const int    b0  = layer_base(l);
        const float* ln1 = base + L[b0 + kLn1].off, *ln2 = base + L[b0 + kLn2].off;
        const float* Wo  = base + L[b0 + kWo].off, *W1 = base + L[b0 + kW1].off;
        const float* W2  = base + L[b0 + kW2].off;
        [[maybe_unused]] const float* Wg = nullptr;
        [[maybe_unused]] const float* b1 = nullptr;
        [[maybe_unused]] const float* b2 = nullptr;
        if constexpr (USE_GATED_FFN) Wg = base + L[b0 + kWg].off;
        else                         { b1 = base + L[b0 + kB1].off; b2 = base + L[b0 + kB2].off; }
        [[maybe_unused]] const float* qgamma = nullptr;
        [[maybe_unused]] const float* kgamma = nullptr;
        if constexpr (USE_QK_NORM) { qgamma = base + L[b0 + kQNorm].off; kgamma = base + L[b0 + kKNorm].off; }
        launch_rmsnorm(h, ln1, a, 1);
        launch_linear(a, g_fwd.wqkv[l], nullptr, qkv, 1, C, sub0::QKV_STRIDE);      // fused q|k|v (one row)
        if constexpr (USE_QK_NORM) launch_qknorm_t<float>(qkv, qgamma, kgamma, 1);   // per-head RMSNorm, before RoPE
        if constexpr (POS_ENCODING == PosEncoding::Rope) {
            const int blk = 128;
            rope_one_kernel<<<(C / 2 + blk - 1) / blk, blk, 0, g_stream>>>(qkv, pos, ROPE_THETA, 0);
            rope_one_kernel<<<(sub0::D_KV / 2 + blk - 1) / blk, blk, 0, g_stream>>>(qkv, pos, ROPE_THETA, 1);
        }
        float* kc = g_kv_k + (static_cast<size_t>(e) * SEQ_LEN + pos) * sub0::D_KV;  // append this token's K/V
        float* vc = g_kv_v + (static_cast<size_t>(e) * SEQ_LEN + pos) * sub0::D_KV;   // slot per EXECUTION
        cudaMemcpyAsync(kc, qkv + sub0::QKV_K_OFF, sub0::D_KV * sizeof(float), cudaMemcpyDeviceToDevice, g_stream);
        cudaMemcpyAsync(vc, qkv + sub0::QKV_V_OFF, sub0::D_KV * sizeof(float), cudaMemcpyDeviceToDevice, g_stream);
        { constexpr int HD = D_HEAD; const size_t shb = (HD + SEQ_LEN) * sizeof(float);
          attn_decode_kernel<HD><<<H, 128, shb, g_stream>>>(qkv, g_kv_k + static_cast<size_t>(e) * SEQ_LEN * sub0::D_KV,
                                                            g_kv_v + static_cast<size_t>(e) * SEQ_LEN * sub0::D_KV, att, pos); }
        launch_linear(att, Wo, nullptr, proj, 1, C, C);
        launch_add(h, proj, h, C);
        launch_rmsnorm(h, ln2, fbuf, 1);
        if constexpr (USE_GATED_FFN) {
            launch_linear(fbuf, Wg, nullptr, ff1, 1, C, F);    // ff1 = gate_pre = fbuf . Wg
            launch_linear(fbuf, W1, nullptr, gact, 1, C, F);   // gact = up_pre   = fbuf . W1
            launch_swiglu(ff1, gact, gact, F);                 // gact = silu(gate_pre) * up_pre (in place)
            launch_linear(gact, W2, nullptr, ff2, 1, F, C);
        } else {
            launch_linear(fbuf, W1, b1, ff1, 1, C, F);
            launch_gelu(ff1, gact, F);
            launch_linear(gact, W2, b2, ff2, 1, F, C);
        }
        launch_add(h, ff2, h, C);
    }
    launch_rmsnorm(h, ln_f, a, 1);
    if constexpr (USE_TIED_EMBEDDINGS) launch_tied_head(a, tok_emb, g_fwd.logits, 1, C, V, /*force_tc=*/true);
    else                               launch_linear(a, lm_head, lm_bias, g_fwd.logits, 1, C, V, /*force_tc=*/true);
}

// Graphed counterpart of forward_one_device: IDENTICAL op sequence and math, but reads id/pos from
// g_decode_state instead of taking them as arguments (embed_one_kernel_g/rope_one_kernel_g/
// attn_decode_kernel_g) and appends K/V via kv_append_kernel instead of a raw pos-addressed
// cudaMemcpyAsync (see g_decode_state's and kv_append_kernel's own comments for why). Every OTHER
// launch here (rmsnorm/linear/gelu/add/qknorm) is already pos/id-independent -- pure fixed-buffer-to-
// fixed-buffer ops -- so this function only differs from forward_one_device at the 4 spots that
// actually touch id/pos. Called from exactly ONE place: capture_decode_graph()'s stream-capture block,
// never eagerly -- see that function for why (this is the CAPTURE TEMPLATE, not a callable decode step).
void forward_one_device_graphed() {
    const auto&  L  = sub0::PARAM_LAYOUT;
    const int    C  = D_MODEL, F = D_FF, V = VOCAB, H = N_HEADS;
    float* const base = g_dev_params;
    const float* tok_emb = base + L[0].off;
    const float* pos_emb = sub0::HAS_POS_EMB ? base + L[1].off : nullptr;   // absent under RoPE
    const int    fi      = kFinalBase;
    const float* ln_f    = base + L[fi + kLnF].off;
    [[maybe_unused]] const float* lm_head = nullptr;
    [[maybe_unused]] const float* lm_bias = nullptr;
    if constexpr (!USE_TIED_EMBEDDINGS) { lm_head = base + L[fi + kLmHead].off; lm_bias = base + L[fi + kLmBias].off; }
    float* h = g_fwd.h; float* a = g_fwd.a; float* qkv = g_fwd.qkv; float* att = g_fwd.att;
    float* proj = g_fwd.proj; float* fbuf = g_fwd.fbuf; float* ff1 = g_fwd.ff1; float* gact = g_fwd.gact; float* ff2 = g_fwd.ff2;
    const int* id_ptr  = g_decode_state;
    const int* pos_ptr = g_decode_state + 1;

    { const int blk = 256; const float* pe = (POS_ENCODING == PosEncoding::Absolute) ? pos_emb : nullptr;
      embed_one_kernel_g<<<(C + blk - 1) / blk, blk, 0, g_stream>>>(tok_emb, pe, id_ptr, h, g_dev_bind); }
    for (int e = 0; e < sub0::LOOP_EXEC_COUNT; ++e) {   // EXECUTIONS, not layers (LoopSplit)
        const int    l   = sub0::LAYER_EXEC_ORDER[e];
        const int    b0  = layer_base(l);
        const float* ln1 = base + L[b0 + kLn1].off, *ln2 = base + L[b0 + kLn2].off;
        const float* Wo  = base + L[b0 + kWo].off, *W1 = base + L[b0 + kW1].off;
        const float* W2  = base + L[b0 + kW2].off;
        [[maybe_unused]] const float* Wg = nullptr;
        [[maybe_unused]] const float* b1 = nullptr;
        [[maybe_unused]] const float* b2 = nullptr;
        if constexpr (USE_GATED_FFN) Wg = base + L[b0 + kWg].off;
        else                         { b1 = base + L[b0 + kB1].off; b2 = base + L[b0 + kB2].off; }
        [[maybe_unused]] const float* qgamma = nullptr;
        [[maybe_unused]] const float* kgamma = nullptr;
        if constexpr (USE_QK_NORM) { qgamma = base + L[b0 + kQNorm].off; kgamma = base + L[b0 + kKNorm].off; }
        launch_rmsnorm(h, ln1, a, 1);
        launch_linear(a, g_fwd.wqkv[l], nullptr, qkv, 1, C, sub0::QKV_STRIDE);      // fused q|k|v (one row)
        if constexpr (USE_QK_NORM) launch_qknorm_t<float>(qkv, qgamma, kgamma, 1);   // per-head RMSNorm, before RoPE
        if constexpr (POS_ENCODING == PosEncoding::Rope) {
            const int blk = 128;
            rope_one_kernel_g<<<(C / 2 + blk - 1) / blk, blk, 0, g_stream>>>(qkv, pos_ptr, ROPE_THETA, 0);
            rope_one_kernel_g<<<(sub0::D_KV / 2 + blk - 1) / blk, blk, 0, g_stream>>>(qkv, pos_ptr, ROPE_THETA, 1);
        }
        float* kc_base = g_kv_k + static_cast<size_t>(e) * SEQ_LEN * sub0::D_KV;   // this EXECUTION's cache
        float* vc_base = g_kv_v + static_cast<size_t>(e) * SEQ_LEN * sub0::D_KV;   // base (fixed per node -- pos
                                                                                   //  is read on-device)
        { const int blk = 256; kv_append_kernel<<<(sub0::D_KV + blk - 1) / blk, blk, 0, g_stream>>>(qkv, kc_base, vc_base, pos_ptr); }
        { constexpr int HD = D_HEAD; const size_t shb = (HD + SEQ_LEN) * sizeof(float);
          attn_decode_kernel_g<HD><<<H, 128, shb, g_stream>>>(qkv, kc_base, vc_base, att, pos_ptr); }
        launch_linear(att, Wo, nullptr, proj, 1, C, C);
        launch_add(h, proj, h, C);
        launch_rmsnorm(h, ln2, fbuf, 1);
        if constexpr (USE_GATED_FFN) {
            launch_linear(fbuf, Wg, nullptr, ff1, 1, C, F);    // ff1 = gate_pre = fbuf . Wg
            launch_linear(fbuf, W1, nullptr, gact, 1, C, F);   // gact = up_pre   = fbuf . W1
            launch_swiglu(ff1, gact, gact, F);                 // gact = silu(gate_pre) * up_pre (in place)
            launch_linear(gact, W2, nullptr, ff2, 1, F, C);
        } else {
            launch_linear(fbuf, W1, b1, ff1, 1, C, F);
            launch_gelu(ff1, gact, F);
            launch_linear(gact, W2, b2, ff2, 1, F, C);
        }
        launch_add(h, ff2, h, C);
    }
    launch_rmsnorm(h, ln_f, a, 1);
    if constexpr (USE_TIED_EMBEDDINGS) launch_tied_head(a, tok_emb, g_fwd.logits, 1, C, V, /*force_tc=*/true);
    else                               launch_linear(a, lm_head, lm_bias, g_fwd.logits, 1, C, V, /*force_tc=*/true);
}

// Device-side forward chain (no host transfers): assumes g_fwd.dids is populated and the
// params uploaded; writes logits into g_fwd.logits over M = batch*T rows. Shared by
// sub0_cuda_forward and the benchmark so both run the IDENTICAL kernel sequence.
void forward_device(int batch, int T) {
    const auto&  L  = sub0::PARAM_LAYOUT;
    const int    C  = D_MODEL, F = D_FF, V = VOCAB, H = N_HEADS;
    const int    M  = batch * T;
    const int    MC = M * C, MF = M * F;
    float* const base = g_dev_params;

    const float* tok_emb = base + L[0].off;
    [[maybe_unused]] const float* pos_emb = sub0::HAS_POS_EMB ? base + L[1].off : nullptr;   // absent under RoPE
    const int    fi      = kFinalBase;          // index of ln_f
    const float* ln_f    = base + L[fi + kLnF].off;
    [[maybe_unused]] const float* lm_head = nullptr;
    [[maybe_unused]] const float* lm_bias = nullptr;
    if constexpr (!USE_TIED_EMBEDDINGS) { lm_head = base + L[fi + kLmHead].off; lm_bias = base + L[fi + kLmBias].off; }

    float* h    = g_fwd.h;
    float* a    = g_fwd.a;
    float* qkv  = g_fwd.qkv;
    float* att  = g_fwd.att;
    float* proj = g_fwd.proj;
    float* fbuf = g_fwd.fbuf;
    float* ff1  = g_fwd.ff1;
    float* gact = g_fwd.gact;
    float* ff2  = g_fwd.ff2;

    // h = tok_emb[ids] (+ pos_emb under Absolute); position = row % T
    {
        const dim3 block(16, 16);
        const dim3 grid((C + block.x - 1) / block.x, (M + block.y - 1) / block.y);
        if constexpr (POS_ENCODING == PosEncoding::Absolute)
            embed_add_kernel<<<grid, block, 0, g_stream>>>(tok_emb, pos_emb, g_fwd.dids, h, M, T);
        else
            embed_kernel<<<grid, block, 0, g_stream>>>(tok_emb, g_fwd.dids, h, M, g_dev_bind);
    }
    for (int e = 0; e < sub0::LOOP_EXEC_COUNT; ++e) {   // EXECUTIONS, not layers (LoopSplit)
        const int    l   = sub0::LAYER_EXEC_ORDER[e];
        const int    b0  = layer_base(l);
        const float* ln1 = base + L[b0 + kLn1].off;
        const float* ln2 = base + L[b0 + kLn2].off;
        const float* Wo  = base + L[b0 + kWo].off;
        const float* W1  = base + L[b0 + kW1].off;
        const float* W2  = base + L[b0 + kW2].off;
        [[maybe_unused]] const float* Wg = nullptr;
        [[maybe_unused]] const float* b1 = nullptr;
        [[maybe_unused]] const float* b2 = nullptr;
        if constexpr (USE_GATED_FFN) Wg = base + L[b0 + kWg].off;
        else                         { b1 = base + L[b0 + kB1].off; b2 = base + L[b0 + kB2].off; }
        [[maybe_unused]] const float* qgamma = nullptr;
        [[maybe_unused]] const float* kgamma = nullptr;
        if constexpr (USE_QK_NORM) { qgamma = base + L[b0 + kQNorm].off; kgamma = base + L[b0 + kKNorm].off; }

        launch_rmsnorm(h, ln1, a, M);                         // a = rmsnorm(h, ln1)
        launch_linear(a, g_fwd.wqkv[l], nullptr, qkv, M, C, sub0::QKV_STRIDE);  // fused qkv = a . [Wq|Wk|Wv]
        if constexpr (USE_QK_NORM) launch_qknorm_t<float>(qkv, qgamma, kgamma, M);   // per-head RMSNorm, before RoPE
        if constexpr (POS_ENCODING == PosEncoding::Rope) launch_rope(qkv, batch, T);
        launch_attn(qkv, qkv + sub0::QKV_K_OFF, qkv + sub0::QKV_V_OFF, att, batch, T, C, H,
                    sub0::QKV_STRIDE, sub0::GQA_GROUP);   // q/k/v sub-blocks of the fused row
        launch_linear(att, Wo, nullptr, proj, M, C, C);      // proj = att . Wo
        launch_add(h, proj, h, MC);                          // h = h + proj
        launch_rmsnorm(h, ln2, fbuf, M);                     // f = rmsnorm(h, ln2)
        if constexpr (USE_GATED_FFN) {
            launch_linear(fbuf, Wg, nullptr, ff1, M, C, F);   // ff1 = gate_pre = f . Wg
            launch_linear(fbuf, W1, nullptr, gact, M, C, F);  // gact = up_pre   = f . W1
            launch_swiglu(ff1, gact, gact, MF);               // gact = silu(gate_pre) * up_pre (in place)
            launch_linear(gact, W2, nullptr, ff2, M, F, C);
        } else {
            launch_linear(fbuf, W1, b1, ff1, M, C, F);        // ff1 = f . W1 + b1
            launch_gelu(ff1, gact, MF);                       // gelu
            launch_linear(gact, W2, b2, ff2, M, F, C);        // ff2 = gelu . W2 + b2
        }
        launch_add(h, ff2, h, MC);                        // h = h + ff2
    }
    launch_rmsnorm(h, ln_f, a, M);                        // a = rmsnorm(h, ln_f)
    if constexpr (USE_TIED_EMBEDDINGS)                    // logits = a . tok_emb^T (tied head, no bias)
        launch_tied_head(a, tok_emb, g_fwd.logits, M, C, V, /*force_tc=*/true);
    else                                                   // logits = a . lm_head + lm_bias
        launch_linear(a, lm_head, lm_bias, g_fwd.logits, M, C, V, /*force_tc=*/true);
}

// Drop any captured graph (shape or math-mode change -> recapture on the next forward).
void invalidate_graph() {
    if (g_graph_exec) { cudaGraphExecDestroy(g_graph_exec); g_graph_exec = nullptr; }
    g_graph_batch = -1;
    g_graph_T     = -1;
}

// Capture forward_device(batch,T) into an executable CUDA graph (once per shape + math mode),
// collapsing the ~100 per-forward kernel launches into a single graph launch. The stream must
// be idle when called (callers streamSync between forwards). A warmup pass first lets cuBLAS do
// any lazy allocation OUTSIDE capture (cudaMalloc during capture is illegal). Returns 0 on ok.
int capture_graph(int batch, int T) {
    if (g_graph_exec && g_graph_batch == batch && g_graph_T == T) return 0;
    invalidate_graph();
    ensure_cublas();
    SUB0_CUDA_CHECK(cudaMemsetAsync(g_fwd.dids, 0, static_cast<size_t>(batch) * T * sizeof(int), g_stream));
    forward_device(batch, T);                            // warmup (safe ids=0): cuBLAS lazy alloc
    SUB0_CUDA_CHECK(cudaStreamSynchronize(g_stream));
    cudaGraph_t graph = nullptr;
    SUB0_CUDA_CHECK(cudaStreamBeginCapture(g_stream, cudaStreamCaptureModeThreadLocal));
    forward_device(batch, T);
    SUB0_CUDA_CHECK(cudaStreamEndCapture(g_stream, &graph));
    SUB0_CUDA_CHECK(cudaGraphInstantiate(&g_graph_exec, graph, 0));
    cudaGraphDestroy(graph);
    g_graph_batch = batch;
    g_graph_T     = T;
    return 0;
}

// Drop the captured decode graph (see g_decode_state / g_decode_graph_exec's own comments) -- called
// everywhere invalidate_graph() above is (KV-cache reset, a param upload/download-triggered rebuild, or
// a math-mode change), since forward_one_device_graphed reads the SAME g_dev_params/g_fwd/g_kv_k/g_kv_v
// buffers the batched graph does and is just as stale under any of those events.
void invalidate_decode_graph() {
    if (g_decode_graph_exec) { cudaGraphExecDestroy(g_decode_graph_exec); g_decode_graph_exec = nullptr; }
}

// Capture forward_one_device_graphed() into an executable CUDA graph (once per KV-cache session --
// see invalidate_decode_graph()'s call sites for what forces a recapture), collapsing decode's
// ~147 per-token kernel launches into a single graph launch. Mirrors capture_graph() above exactly:
// a warmup pass OUTSIDE capture lets cuBLAS's GEMV path do any lazy allocation (cudaMalloc during
// capture is illegal), then the real capture/instantiate. The warmup's actual id/pos value is
// irrelevant (0/0, same convention as capture_graph's ids=0 warmup) -- only the STRUCTURE is captured;
// every real token's id/pos arrives afterward via g_decode_state's contents, never by recapturing.
// Requires g_decode_state to already be allocated (kv_alloc, called by sub0_cuda_kv_reset before this)
// and g_fwd.wqkv/ensure_wqkv_f32 already built (same precondition sub0_cuda_forward_one already has).
// Returns 0 on ok.
int capture_decode_graph() {
    if (g_decode_graph_exec) return 0;
    ensure_cublas();
    const int warmup_id_pos[2] = { 0, 0 };
    SUB0_CUDA_CHECK(cudaMemcpyAsync(g_decode_state, warmup_id_pos, 2 * sizeof(int), cudaMemcpyHostToDevice, g_stream));
    forward_one_device_graphed();                         // warmup (safe id=pos=0): cuBLAS lazy alloc
    SUB0_CUDA_CHECK(cudaStreamSynchronize(g_stream));
    cudaGraph_t graph = nullptr;
    SUB0_CUDA_CHECK(cudaStreamBeginCapture(g_stream, cudaStreamCaptureModeThreadLocal));
    forward_one_device_graphed();
    SUB0_CUDA_CHECK(cudaStreamEndCapture(g_stream, &graph));
    SUB0_CUDA_CHECK(cudaGraphInstantiate(&g_decode_graph_exec, graph, 0));
    cudaGraphDestroy(graph);
    return 0;
}

// Materialize the per-layer fused QKV weight from the uploaded param blob. Called once whenever the
// weights change (upload/load/every AdamW step). TODO(qkv-train): this rebuild runs inside every
// device training step (cheap: N_LAYERS*C*C threads) -- or store QKV fused natively and slice it
// back for the layout-table serialization.
//
// BF16 builds write g_wqkv16 DIRECTLY from Wq/Wk/Wv (build_qkv_act_kernel) -- no F32 g_fwd.wqkv
// staging buffer at all, since training never reads it (only the bf16 mirror feeds the training
// GEMMs). That buffer is dead weight during training (~108 MiB at production dims): marks
// g_wqkv_f32_dirty so the F32 CUDA inference paths (ensure_wqkv_f32, sub0_cuda_forward /
// sub0_cuda_forward_one) rebuild their own lazily-allocated copy on next use instead of reading a
// stale one -- this is what keeps an interleaved train-then-infer call (e.g. the forward_one/full-
// forward parity check) correct.
//
// F32 builds are UNCHANGED from before this split: g_fwd.wqkv is eagerly built here (same
// build_qkv_kernel as always) and every mirror aliases the master weights directly (no copy).
void build_qkv_weights() {
    const auto& L = sub0::PARAM_LAYOUT;
    const int   C = D_MODEL;
    // One thread per FUSED [C, QKV_STRIDE] element -- build_qkv_kernel's mapping, NOT one thread per
    // [C,C] source element doing three writes (which under GQA's unequal sub-block widths could not
    // address all three matrices from a single index).
    const int   n = C * sub0::QKV_STRIDE, block = 256, grid = (n + block - 1) / block;
    if constexpr (ACT_DTYPE == Dtype::BF16) {
        g_wqkv_f32_dirty = true;
        const int F = D_FF;
        const int nce = C * F, gce = (nce + 255) / 256;
        for (int l = 0; l < N_LAYERS; ++l) {
            const int    b0 = layer_base(l);
            const float* Wq = g_dev_params + L[b0 + kWq].off;
            const float* Wk = g_dev_params + L[b0 + kWk].off;
            const float* Wv = g_dev_params + L[b0 + kWv].off;
            if (!g_w1_16[l])   cudaMalloc(&g_w1_16[l],   static_cast<size_t>(C) * F * sizeof(act_t));
            if (!g_w2_16[l])   cudaMalloc(&g_w2_16[l],   static_cast<size_t>(F) * C * sizeof(act_t));
            if (!g_wo16[l])    cudaMalloc(&g_wo16[l],    static_cast<size_t>(C) * C * sizeof(act_t));
            if (!g_wqkv16[l])  cudaMalloc(&g_wqkv16[l],  static_cast<size_t>(C) * sub0::QKV_STRIDE * sizeof(act_t));
            f32_to_act_kernel<<<gce, 256, 0, g_stream>>>(g_dev_params + L[b0 + kW1].off, g_w1_16[l], nce);
            f32_to_act_kernel<<<gce, 256, 0, g_stream>>>(g_dev_params + L[b0 + kW2].off, g_w2_16[l], nce);
            if constexpr (USE_GATED_FFN) {   // gate matrix mirror -- same [C,F] shape as W1's
                if (!g_wg16[l]) cudaMalloc(&g_wg16[l], static_cast<size_t>(C) * F * sizeof(act_t));
                f32_to_act_kernel<<<gce, 256, 0, g_stream>>>(g_dev_params + L[b0 + kWg].off, g_wg16[l], nce);
            }
            { const int ncc = C * C, gcc = (ncc + 255) / 256;
              f32_to_act_kernel<<<gcc, 256, 0, g_stream>>>(g_dev_params + L[b0 + kWo].off, g_wo16[l], ncc); }
            build_qkv_act_kernel<act_t><<<grid, block, 0, g_stream>>>(Wq, Wk, Wv, g_wqkv16[l]);
        }
    } else {                                             // F32: mirrors alias the master weights (no copy)
        for (int l = 0; l < N_LAYERS; ++l) {
            const int    b0 = layer_base(l);
            const float* Wq = g_dev_params + L[b0 + kWq].off;
            const float* Wk = g_dev_params + L[b0 + kWk].off;
            const float* Wv = g_dev_params + L[b0 + kWv].off;
            build_qkv_kernel<<<grid, block, 0, g_stream>>>(Wq, Wk, Wv, g_fwd.wqkv[l]);
            g_w1_16[l] = reinterpret_cast<act_t*>(g_dev_params + L[b0 + kW1].off);
            g_w2_16[l] = reinterpret_cast<act_t*>(g_dev_params + L[b0 + kW2].off);
            if constexpr (USE_GATED_FFN) g_wg16[l] = reinterpret_cast<act_t*>(g_dev_params + L[b0 + kWg].off);
            g_wo16[l]  = reinterpret_cast<act_t*>(g_dev_params + L[b0 + kWo].off);
            g_wqkv16[l] = reinterpret_cast<act_t*>(g_fwd.wqkv[l]);
        }
    }
}

// Lazily (re)builds the F32 fused QKV weight for the CUDA inference-only paths (sub0_cuda_forward,
// sub0_cuda_forward_one) on a BF16 build, which otherwise keeps only the bf16 mirror resident (see
// build_qkv_weights() above). No-op on an F32 build: g_fwd.wqkv there is already kept current by
// build_qkv_weights() directly, exactly as before this split.
int ensure_wqkv_f32() {
    if constexpr (ACT_DTYPE != Dtype::BF16) return 0;
    if (!g_fwd.wqkv[0]) {
        ensure_stream();
        for (int l = 0; l < N_LAYERS; ++l)
            SUB0_CUDA_CHECK(cudaMalloc(&g_fwd.wqkv[l], static_cast<size_t>(D_MODEL) * sub0::QKV_STRIDE * sizeof(float)));
        g_wqkv_f32_dirty = true;         // freshly (re)allocated -- garbage until populated below
    }
    if (g_wqkv_f32_dirty) {
        const auto& L = sub0::PARAM_LAYOUT;
        const int C = D_MODEL, n = C * sub0::QKV_STRIDE, block = 256, grid = (n + block - 1) / block;
        for (int l = 0; l < N_LAYERS; ++l) {
            const int b0 = layer_base(l);
            build_qkv_kernel<<<grid, block, 0, g_stream>>>(g_dev_params + L[b0 + kWq].off,
                g_dev_params + L[b0 + kWk].off, g_dev_params + L[b0 + kWv].off, g_fwd.wqkv[l]);
        }
        g_wqkv_f32_dirty = false;
    }
    return 0;
}

// ============================================================================
//  Training forward + backward + AdamW (Phase 2d) -- mirrors backend_cpu.cpp
// ============================================================================

// Forward pass that SAVES every activation the backward needs into g_tr (no recompute). Same op
// sequence as forward_device but writes to the per-layer training buffers and uses the train
// variants of rmsnorm/attn (which also save rinv / the softmax weights P). Assumes g_fwd.dids is
// populated, params uploaded and g_wqkv16[]/g_w1_16[]/g_w2_16[]/g_wo16[] built (build_qkv_weights) --
// training reads ONLY the bf16 mirrors, never the F32 g_fwd.wqkv (that buffer is inference-only, see
// ensure_wqkv_f32). Eager (not graph-captured): the backward reads these buffers, so the whole step
// runs as plain stream-ordered launches.
// NOTE(perf): deliberately NOT graph-captured. A step graph is an EFFICIENCY change -- it removes
// per-launch CPU/driver management overhead (~200 launches -> 1), not a compute change. The inference
// forward IS captured (capture_graph) because gen is launch-bound; training is COMPUTE-bound (measured
// graph 1.00x at M=4096), so removing that CPU overhead buys nothing, and the AdamW step has a host-in-
// the-loop break that prevents single-graph capture anyway (D2H grad-norm -> host clip scale + pow bias-
// correction). Revisit only if a device-side global-norm clip removes that sync AND a launch-bound regime appears.
// TODO(mem): saved-activation scratch dominates at large batch; a/qkv/att are already checkpointed
// (recomputed in backward) and acts + residual stream are bf16, so batch 512 now fits 8GB. Further
// cuts would need checkpointing the per-layer residual (h_in/h_mid) too, recomputed from the input.
// NOTE: this no longer computes logits -- the lm_head/tied-head forward now lives in head_ce_chunked
// (called from backward_device, row-chunked over M along with the cross-entropy fwd+bwd and the
// lm_head/tied-head backward -- see that function's comment). forward_train ends at a_final; every
// call site (run_fwd_bwd, the training benchmarks/profiler below) already calls backward_device
// unconditionally on the very next line, so this is a safe split, not a behavior change.
void forward_train(int batch, int T) {
    const auto&  L  = sub0::PARAM_LAYOUT;
    const int    C  = D_MODEL, F = D_FF, H = N_HEADS;
    const int    M  = batch * T;
    const int    MC = M * C, MF = M * F;
    float* const base = g_dev_params;

    const float* tok_emb = base + L[0].off;
    [[maybe_unused]] const float* pos_emb = sub0::HAS_POS_EMB ? base + L[1].off : nullptr;   // absent under RoPE
    const int    fi      = kFinalBase;
    const float* ln_f    = base + L[fi + kLnF].off;

    {   // h_in[0] = tok_emb[ids] (+ pos_emb under Absolute)
        const dim3 block(16, 16);
        const dim3 grid((C + block.x - 1) / block.x, (M + block.y - 1) / block.y);
        if constexpr (POS_ENCODING == PosEncoding::Absolute)
            embed_add_act_kernel<act_t><<<grid, block, 0, g_stream>>>(tok_emb, pos_emb, g_fwd.dids, g_tr.h_in[0], M, T);
        else
            embed_act_kernel<act_t><<<grid, block, 0, g_stream>>>(tok_emb, g_fwd.dids, g_tr.h_in[0], M, g_dev_bind);
    }
    for (int e = 0; e < sub0::LOOP_EXEC_COUNT; ++e) {   // EXECUTIONS, not layers (LoopSplit)
        const int    l   = sub0::LAYER_EXEC_ORDER[e];
        const int    b0  = layer_base(l);
        const float* ln1 = base + L[b0 + kLn1].off;
        const float* ln2 = base + L[b0 + kLn2].off;
        const float* W1  = base + L[b0 + kW1].off;
        const float* W2  = base + L[b0 + kW2].off;
        [[maybe_unused]] const float* Wg = nullptr;
        [[maybe_unused]] const float* b1 = nullptr;
        [[maybe_unused]] const float* b2 = nullptr;
        if constexpr (USE_GATED_FFN) Wg = base + L[b0 + kWg].off;
        else                         { b1 = base + L[b0 + kB1].off; b2 = base + L[b0 + kB2].off; }
        [[maybe_unused]] const float* qgamma = nullptr;
        [[maybe_unused]] const float* kgamma = nullptr;
        if constexpr (USE_QK_NORM) { qgamma = base + L[b0 + kQNorm].off; kgamma = base + L[b0 + kKNorm].off; }
        act_t* const hin  = g_tr.h_in[e];
        act_t* const hmid = g_tr.h_mid[e];
        act_t* const next = (e + 1 < sub0::LOOP_EXEC_COUNT) ? g_tr.h_in[e + 1] : g_tr.h_final;

        launch_rmsnorm_train_t<act_t, act_t>(hin, ln1, g_tr.a, g_tr.rinv1[e], M);          // a = rmsnorm(hin,ln1) bf16
        launch_linear_t<act_t, act_t>(g_tr.a, g_wqkv16[l], g_tr.qkv, M, C, sub0::QKV_STRIDE);  // fused qkv bf16
        // Plain (non-save) variant here: this qkv buffer is a CHECKPOINT (recomputed from scratch in
        // backward_device -- see that function's own qknorm call, which uses the SAVE variant because
        // its recompute is where the pre-norm x actually needs to survive for qknorm's own backward).
        if constexpr (USE_QK_NORM) launch_qknorm_t<act_t>(g_tr.qkv, qgamma, kgamma, M);
        if constexpr (POS_ENCODING == PosEncoding::Rope) launch_rope_t<act_t>(g_tr.qkv, batch, T);
        launch_attn_train_t<act_t>(g_tr.qkv, g_tr.qkv + sub0::QKV_K_OFF, g_tr.qkv + sub0::QKV_V_OFF,
                          g_tr.att, batch, T, C, H, sub0::QKV_STRIDE, sub0::GQA_GROUP);  // attention (P-free)
        launch_linear_t<act_t, act_t>(g_tr.att, g_wo16[l], hmid, M, C, C);           // hmid = proj (bf16)
        launch_add_t<act_t>(hin, hmid, hmid, MC);                                   // hmid = hin + proj
        launch_rmsnorm_train_t<act_t, act_t>(hmid, ln2, g_tr.fbuf, g_tr.rinv2[e], M);      // fbuf bf16
        if constexpr (USE_GATED_FFN) {
            launch_linear_t<act_t, act_t>(g_tr.fbuf, g_wg16[l], g_tr.ff1, M, C, F);   // ff1 = gate_pre
            launch_linear_t<act_t, act_t>(g_tr.fbuf, g_w1_16[l], g_tr.gact, M, C, F); // gact = up_pre
            launch_swiglu_t<act_t>(g_tr.ff1, g_tr.gact, g_tr.gact, MF);               // gact = silu(gate_pre)*up_pre (in place)
            launch_linear_t<act_t, act_t>(g_tr.gact, g_w2_16[l], next, M, F, C);      // next = ff2 bf16
        } else {
            launch_linear_t<act_t, act_t>(g_tr.fbuf, g_w1_16[l], g_tr.ff1, M, C, F);
            launch_bias_act(g_tr.ff1, b1, M, F);                                       // ff1 = f.W1 + b1
            launch_gelu_t(g_tr.ff1, g_tr.gact, MF);                                     // gelu
            launch_linear_t<act_t, act_t>(g_tr.gact, g_w2_16[l], next, M, F, C);        // next = ff2 bf16
            launch_bias_act(next, b2, M, C);
        }
        launch_add_t<act_t>(hmid, next, next, MC);                                  // next = hmid + ff2
    }
    launch_rmsnorm_train_t<act_t, float>(g_tr.h_final, ln_f, g_tr.a_final, g_tr.rinv_f, M);     // a_final f32
}

// Chunked lm_head/tied-head forward -> cross-entropy fwd+bwd -> lm_head/tied-head backward. Processes
// M rows in row-chunks of at most g_tr_logits_chunk (the [chunk_rows,V] g_tr.logits/dlogits scratch
// buffer is allocated ONCE at that capacity in train_alloc and reused across every chunk) instead of
// one [M,V] shot -- V (~16.5k at this project's production scale) makes a full [M,V] buffer the
// single largest per-window TRAINING scratch buffer (~55-60% of it), so chunking it is the main lever
// for raising the VRAM-fit training batch (see sub0::memplan::logits_n_chunks for the derivation).
// Mathematically IDENTICAL to the old unchunked path: g_tr_logits_chunk >= M degenerates to exactly
// one chunk covering the whole M (today's behavior verbatim), zero extra overhead in that case.
//
// Called from backward_device in place of its old single CE+head-backward block. Per chunk:
//   * head forward OVERWRITES the same reused [chunk_rows,V] g_tr.logits (a fresh value every chunk,
//     no accumulation -- mirrors op_linear/op_tied_head's forward, each row used once).
//   * ce_backward_kernel needs the chunk's ABSOLUTE row offset (its row_offset param) to compute the
//     right window/position (b=abs_m/T, t=abs_m%T) and lengths[b] -- passing chunk-LOCAL indices here
//     would silently corrupt every row past the first chunk boundary. Its loss_acc atomicAdd is
//     already safe across however many chunks call it (accumulates regardless of launch count).
//   * head backward: dA (grad into a_final) is an OVERWRITE per chunk -- each chunk's rows are
//     disjoint, exactly like launch_linear_bwd's own dX/dA convention for a row-chunked caller (see
//     its doc comment). dW/dbias (untied) or dtok_emb (tied) ACCUMULATE (beta=1) across every chunk
//     into the SAME grad slots backward_device already zeroed once before this runs -- untied passes
//     accumulate=true explicitly (launch_linear_bwd defaults to false, the single-shot-per-forward
//     case its other call sites want); tied's launch_tied_head_bwd already always accumulates
//     unconditionally (added for the tied-embeddings two-contribution case), reused here verbatim.
void head_ce_chunked(int batch, int T, const int* d_lengths, const int* d_active) {
    const auto&  L  = sub0::PARAM_LAYOUT;
    const int    C  = D_MODEL, V = VOCAB;
    const int    M  = batch * T;
    const int    fi = kFinalBase;
    float* const pb = g_dev_params;
    float* const gb = g_dev_grad;
    const float* tok_emb = pb + L[0].off;
    [[maybe_unused]] const float* lm_head = nullptr;
    [[maybe_unused]] const float* lm_bias = nullptr;
    if constexpr (!USE_TIED_EMBEDDINGS) { lm_head = pb + L[fi + kLmHead].off; lm_bias = pb + L[fi + kLmBias].off; }

    constexpr int kCeBlock = 256;   // one block per row; see ce_backward_kernel above
    const int chunk_cap = static_cast<int>(g_tr_logits_chunk);
    for (int m0 = 0; m0 < M; m0 += chunk_cap) {
        const int rows = (M - m0 < chunk_cap) ? (M - m0) : chunk_cap;
        const float* a_chunk   = g_tr.a_final  + static_cast<size_t>(m0) * C;
        float*       da_chunk  = g_tr.da       + static_cast<size_t>(m0) * C;
        const int*   tgt_chunk = g_tr.dtargets + m0;

        if constexpr (USE_TIED_EMBEDDINGS)
            launch_tied_head(a_chunk, tok_emb, g_tr.logits, rows, C, V, /*force_tc=*/true);
        else
            launch_linear(a_chunk, lm_head, lm_bias, g_tr.logits, rows, C, V, /*force_tc=*/true);

        ce_backward_kernel<kCeBlock><<<rows, kCeBlock, 0, g_stream>>>(
            g_tr.logits, tgt_chunk, g_tr.dlogits, g_tr.loss, rows, T, batch, d_lengths, d_active, m0);

        if constexpr (USE_TIED_EMBEDDINGS)
            launch_tied_head_bwd(a_chunk, tok_emb, g_tr.dlogits, da_chunk, gb + L[0].off, rows, C, V, /*force_tc=*/true);
        else
            launch_linear_bwd(a_chunk, lm_head, g_tr.dlogits, da_chunk,
                              gb + L[fi + kLmHead].off, gb + L[fi + kLmBias].off, rows, C, V,
                              /*force_tc=*/true, /*accumulate=*/true);
    }
}

// Reverse pass: consumes the saved activations, writes the reduced gradient into g_dev_grad and
// accumulates the mean cross-entropy into g_tr.loss. dh threads the residual stream; the rmsnorm
// backward kernels accumulate into it (residual skip + through-norm paths), exactly like the CPU
// tape walk. Weight/bias grads are written straight to their PARAM_LAYOUT offsets (each weight is
// used once per forward). Loss scaling invM = 1/M makes the result equal the CPU train_batch grad.
void backward_device(int batch, int T, const int* d_lengths, const int* d_active) {
    const auto&  L  = sub0::PARAM_LAYOUT;
    const int    C  = D_MODEL, F = D_FF, H = N_HEADS;
    const int    M  = batch * T;
    const int    MC = M * C, MF = M * F;
    float* const pb = g_dev_params;
    float* const gb = g_dev_grad;
    const int    fi = kFinalBase;

    SUB0_CUDA_CHECK_VOID(cudaMemsetAsync(gb, 0, sub0::PARAM_FLOATS * sizeof(float), g_stream));
    SUB0_CUDA_CHECK_VOID(cudaMemsetAsync(g_tr.loss, 0, sizeof(double), g_stream));

    // chunked cross-entropy + lm_head/tied-head fwd+bwd -- see head_ce_chunked's own comment.
    head_ce_chunked(batch, T, d_lengths, d_active);
    // rmsnorm_f: dh starts here (grad into h_final)
    SUB0_CUDA_CHECK_VOID(cudaMemsetAsync(g_tr.dh, 0, static_cast<size_t>(MC) * sizeof(float), g_stream));
    launch_rmsnorm_bwd_t<act_t>(g_tr.h_final, pb + L[fi + kLnF].off, g_tr.rinv_f, g_tr.da, g_tr.dh,
                       gb + L[fi + kLnF].off, M);

    // Reverse EXECUTION order (mirrors forward_train): `e` indexes this run's saved activations, `l`
    // says whose weights/grads they belong to. Under LoopSplit several `e` share one `l`, which is
    // exactly why every param-grad write below accumulates -- see launch_linear_bwd_t's accumulate_dw.
    for (int e = sub0::LOOP_EXEC_COUNT - 1; e >= 0; --e) {
        const int l  = sub0::LAYER_EXEC_ORDER[e];
        const int b0 = layer_base(l);
        [[maybe_unused]] const float* b1 = nullptr;
        if constexpr (!USE_GATED_FFN) b1 = pb + L[b0 + kB1].off;
        [[maybe_unused]] const float* qgamma = nullptr;
        [[maybe_unused]] const float* kgamma = nullptr;
        [[maybe_unused]] float*       dqgamma = nullptr;
        [[maybe_unused]] float*       dkgamma = nullptr;
        if constexpr (USE_QK_NORM) {
            qgamma  = pb + L[b0 + kQNorm].off; kgamma  = pb + L[b0 + kKNorm].off;
            dqgamma = gb + L[b0 + kQNorm].off; dkgamma = gb + L[b0 + kKNorm].off;
        }
        // checkpoint: fbuf/ff1/gact not saved -- recompute fbuf from h_mid, then the FFN-in
        // intermediate(s), before use. USE_GATED_FFN recomputes THREE intermediates (gate_pre, up_pre,
        // and their SwiGLU combination hswi) where the plain path recomputes only two (ff1 pre-GELU,
        // gact post-GELU) -- see the top-of-file "SwiGLU-gated FFN" comment for why. Buffer roles are
        // REMAPPED under gated (no new [M,F] buffers): ff1 <- gate_pre, gact <- up_pre, dff1 <- hswi
        // (the W2 GEMM's input) then, after W2's backward, dff1 <- dgate and dgact <- dup.
        launch_rmsnorm_train_t<act_t, act_t>(g_tr.h_mid[e], pb + L[b0 + kLn2].off, g_tr.fbuf, g_tr.rinv2[e], M);
        if constexpr (USE_GATED_FFN) {
            launch_linear_t<act_t, act_t>(g_tr.fbuf, g_wg16[l], g_tr.ff1,  M, C, F);   // ff1  = gate_pre
            launch_linear_t<act_t, act_t>(g_tr.fbuf, g_w1_16[l], g_tr.gact, M, C, F);  // gact = up_pre
            launch_swiglu_t<act_t>(g_tr.ff1, g_tr.gact, g_tr.dff1, MF);                // dff1 = hswi = silu(gate_pre)*up_pre
            // ff residual: dh = grad into layer output = d(hswi @ W2). W2 backward (input hswi=dff1); dh->bf16
            { const int n2 = M * C, bk = 256; f32_to_act_kernel<<<(n2 + bk - 1) / bk, bk, 0, g_stream>>>(g_tr.dh, g_tr.dh16, n2); }
            launch_linear_bwd_t<act_t, act_t>(g_tr.dff1, g_w2_16[l], g_tr.dh16, g_tr.dgact, gb + L[b0 + kW2].off, M, F, C);
            // dgact currently holds d(hswi); SwiGLU backward turns it into dgate (-> dff1, overwrite) and
            // dup (-> dgact, in place over its own dy input -- see swiglu_backward_act_kernel's aliasing note).
            launch_swiglu_bwd_t<act_t>(g_tr.ff1, g_tr.gact, g_tr.dgact, g_tr.dff1, g_tr.dgact, MF);
            // Wg backward (dX = dfbuf, OVERWRITE -- the first of dfbuf's two contributions this layer).
            launch_linear_bwd_t<act_t, float>(g_tr.fbuf, g_wg16[l], g_tr.dff1, g_tr.dfbuf, gb + L[b0 + kWg].off, M, C, F);
            // W1 (up) backward (dX = dfbuf, ACCUMULATE -- the second contribution; must run AFTER the
            // Wg backward above so the overwrite lands first, in the same stream order).
            launch_linear_bwd_t<act_t, float>(g_tr.fbuf, g_w1_16[l], g_tr.dgact, g_tr.dfbuf, gb + L[b0 + kW1].off,
                                              M, C, F, /*accumulate_dx=*/true);
        } else {
            launch_linear_t<act_t, act_t>(g_tr.fbuf, g_w1_16[l], g_tr.ff1, M, C, F);
            launch_bias_act(g_tr.ff1, b1, M, F);
            launch_gelu_t(g_tr.ff1, g_tr.gact, MF);
            // ff residual: dh = grad into layer output = d(ff2). W2 backward (input gact); dh->bf16
            { const int n2 = M * C, bk = 256; f32_to_act_kernel<<<(n2 + bk - 1) / bk, bk, 0, g_stream>>>(g_tr.dh, g_tr.dh16, n2); }
            launch_linear_bwd_t<act_t, act_t>(g_tr.gact, g_w2_16[l], g_tr.dh16, g_tr.dgact, gb + L[b0 + kW2].off, M, F, C);
            { const int bk = 128; bias_grad_kernel<<<(C + bk - 1) / bk, bk, 0, g_stream>>>(g_tr.dh, gb + L[b0 + kB2].off, M, C,
                                                                          sub0::LOOP_SPLIT_ON); }
            launch_gelu_bwd_t(g_tr.ff1, g_tr.dgact, g_tr.dff1, MF);                     // dff1
            launch_linear_bwd_t<act_t, float>(g_tr.fbuf, g_w1_16[l], g_tr.dff1, g_tr.dfbuf, gb + L[b0 + kW1].off, M, C, F);
            { const int bk = 128; bias_grad_act_kernel<act_t, sub0::LOOP_SPLIT_ON><<<(F + bk - 1) / bk, bk, 0, g_stream>>>(g_tr.dff1, gb + L[b0 + kB1].off, M); }
        }
        launch_rmsnorm_bwd_t<act_t>(g_tr.h_mid[e], pb + L[b0 + kLn2].off, g_tr.rinv2[e], g_tr.dfbuf,
                           g_tr.dh, gb + L[b0 + kLn2].off, M);                          // dh += -> d(h_mid)
        // proj residual: dh = grad into h_mid = d(proj). Wo backward (input att, bf16)
        // checkpoint: a/qkv/att not saved -- recompute a from h_in, then qkv (+qknorm+rope), then attention
        launch_rmsnorm_train_t<act_t, act_t>(g_tr.h_in[e], pb + L[b0 + kLn1].off, g_tr.a, g_tr.rinv1[e], M);
        launch_linear_t<act_t, act_t>(g_tr.a, g_wqkv16[l], g_tr.qkv, M, C, sub0::QKV_STRIDE);
        // QK-norm recompute: the SAVE variant (unlike forward_train's plain launch_qknorm_t) because
        // this pre-norm x must survive until qknorm's own backward runs below, AFTER attention's and
        // RoPE's backward -- see qknorm_save_act_kernel's comment and the TrainScratch::qk_pre note in
        // memplan.hpp's train_scratch_bytes.
        if constexpr (USE_QK_NORM) launch_qknorm_save_t<act_t>(g_tr.qkv, g_tr.qk_pre, qgamma, kgamma, M);
        if constexpr (POS_ENCODING == PosEncoding::Rope) launch_rope_t<act_t>(g_tr.qkv, batch, T);
        launch_attn_train_t<act_t>(g_tr.qkv, g_tr.qkv + sub0::QKV_K_OFF, g_tr.qkv + sub0::QKV_V_OFF,
                                   g_tr.att, batch, T, C, H, sub0::QKV_STRIDE, sub0::GQA_GROUP);
        { const int n2 = M * C, bk = 256; f32_to_act_kernel<<<(n2 + bk - 1) / bk, bk, 0, g_stream>>>(g_tr.dh, g_tr.dh16, n2); }
        launch_linear_bwd_t<act_t, act_t>(g_tr.att, g_wo16[l], g_tr.dh16, g_tr.datt, gb + L[b0 + kWo].off, M, C, C);
        // flash backward writes each dq/dk/dv exactly once (no atomics), so no pre-zero is needed.
        launch_attn_bwd_t<act_t>(g_tr.qkv, g_tr.datt, g_tr.dqkv, batch, T, C, H, sub0::QKV_STRIDE, sub0::GQA_GROUP);
        // RoPE: convert dQ/dK (w.r.t. the rotated q/k attention used) back to grad w.r.t. the
        // qknorm output (or the projected q/k directly, without QK-norm) before qknorm's own backward
        // (if enabled) and the qkv-GEMM backward (inverse rotation). dV is untouched.
        if constexpr (POS_ENCODING == PosEncoding::Rope) launch_rope_bwd_t<act_t>(g_tr.dqkv, batch, T);
        // QK-norm backward: converts dqkv's Q/K sub-blocks in place from "grad w.r.t. qknorm's output"
        // (what RoPE's backward just produced) to "grad w.r.t. qknorm's input" (= the raw QKV-GEMM
        // output), which the qkv-GEMM backward below needs. Must run AFTER RoPE's backward and BEFORE
        // the qkv-GEMM backward, matching the forward order (qknorm then RoPE) run in reverse.
        if constexpr (USE_QK_NORM)
            launch_qknorm_bwd_t<act_t>(g_tr.qk_pre, qgamma, kgamma, g_tr.dqkv, dqgamma, dkgamma, M);
        // qkv backward (input a): da = grad into a (f32), dWqkv -> split into dWq/dWk/dWv
        // dW target here is g_tr.dwqkv, a per-execution TEMP -- NOT the shared param blob. It must be
        // overwritten (accumulate_dw=false), or under LoopSplit it would accumulate here AND again in
        // split_dqkv_kernel below, double-counting every repeat. This is the one call site of the seven
        // whose dW is not `gb + ...`; the other six correctly take the accumulating default.
        launch_linear_bwd_t<act_t, float>(g_tr.a, g_wqkv16[l], g_tr.dqkv, g_tr.da, g_tr.dwqkv, M, C,
                                          sub0::QKV_STRIDE, /*accumulate_dx=*/false, DwWrite::Overwrite);
        {
            const int n = C * sub0::QKV_STRIDE, block = 256;
            split_dqkv_kernel<sub0::LOOP_SPLIT_ON><<<(n + block - 1) / block, block, 0, g_stream>>>(
                g_tr.dwqkv, gb + L[b0 + kWq].off, gb + L[b0 + kWk].off, gb + L[b0 + kWv].off);
        }
        launch_rmsnorm_bwd_t<act_t>(g_tr.h_in[e], pb + L[b0 + kLn1].off, g_tr.rinv1[e], g_tr.da,
                           g_tr.dh, gb + L[b0 + kLn1].off, M);                          // dh += -> d(h_in)
    }
    // embed backward: scatter dh into tok_emb (+ pos_emb under Absolute) grads
    {
        const dim3 block(16, 16);
        const dim3 grid((C + block.x - 1) / block.x, (M + block.y - 1) / block.y);
        if constexpr (POS_ENCODING == PosEncoding::Absolute)
            embed_backward_kernel<<<grid, block, 0, g_stream>>>(
                g_tr.dh, g_fwd.dids, gb + L[0].off, gb + L[1].off, M, T);
        else
            embed_backward_token_kernel<<<grid, block, 0, g_stream>>>(
                g_tr.dh, g_fwd.dids, gb + L[0].off, M, g_dev_bind);
    }
}

// --- Muon (hybrid optimizer) matrix-level pipeline --------------------------------------------
// Scratch for GPU Newton-Schulz, sized ONCE to the largest Muon-eligible matrix (sub0::MUON_MAX_MN
// / sub0::MUON_MAX_MM, layout.hpp) and reused for every eligible matrix, every step (AGENTS.md #1
// -- every call below is bounded by ITS OWN current rows*cols/m*m, never by these buffers' own
// capacity). `upd`/`bx` are always in a matrix's NATIVE [rows,cols] row-major orientation; `a`/`aa`
// are the [m,m] (m=min(rows,cols)) Gram-matrix scratch. Allocated LAZILY (not in opt_alloc) so an
// AdamW-only GPU run pays zero extra VRAM for this feature (AGENTS.md #4) -- ~8 MiB at this
// project's production d448/D_FF=1792 shape once a Muon run actually touches it.
float*  g_dev_muon_upd = nullptr;
float*  g_dev_muon_bx  = nullptr;
float*  g_dev_muon_a   = nullptr;
float*  g_dev_muon_aa  = nullptr;
double* g_dev_muon_ss  = nullptr;   // [1] Frobenius norm-sq accumulator (double, like grad clip)

int ensure_muon_scratch() {
    if (g_dev_muon_upd) return 0;
    SUB0_CUDA_CHECK(cudaMalloc(&g_dev_muon_upd, sub0::MUON_MAX_MN * sizeof(float)));
    SUB0_CUDA_CHECK(cudaMalloc(&g_dev_muon_bx,  sub0::MUON_MAX_MN * sizeof(float)));
    SUB0_CUDA_CHECK(cudaMalloc(&g_dev_muon_a,   sub0::MUON_MAX_MM * sizeof(float)));
    SUB0_CUDA_CHECK(cudaMalloc(&g_dev_muon_aa,  sub0::MUON_MAX_MM * sizeof(float)));
    SUB0_CUDA_CHECK(cudaMalloc(&g_dev_muon_ss,  sizeof(double)));
    return 0;
}
void free_muon_scratch() {
    cudaFree(g_dev_muon_upd); cudaFree(g_dev_muon_bx);
    cudaFree(g_dev_muon_a);   cudaFree(g_dev_muon_aa);
    cudaFree(g_dev_muon_ss);
    g_dev_muon_upd = g_dev_muon_bx = g_dev_muon_a = g_dev_muon_aa = nullptr;
    g_dev_muon_ss = nullptr;
}

// GPU Newton-Schulz orthogonalization (5 default iterations) -- the GPU counterpart of
// sub0::muon::newton_schulz5 (include/sub0/muon.hpp). `upd` (device, rows*cols floats) holds the
// Nesterov-lookahead gradient on entry, in its NATIVE row-major [rows,cols] layout (the same
// layout the parameter/gradient blob itself uses -- NOT the CPU reference's internal [m,n]
// (m=min(rows,cols)) "working orientation" buffer). On return, `upd` holds the orthogonalized
// result, STILL in that same native [rows,cols] layout, ready for muon_apply_kernel to subtract
// directly from p[] -- there is no un-transpose step anywhere in this function.
//
// Why no transpose is needed at all (the CPU reference's physical transpose-for-compute-
// efficiency step, muon.hpp lines ~66-73/133-140, is a pure compute-cost optimization for a
// row-major CPU loop; it has no GPU equivalent because cuBLAS already gives GEMM op-flag control
// over which operand is read transposed, for free, from the SAME buffer):
//
//   cuBLAS is column-major. A row-major buffer M of logical shape [P,Q] (element (i,j) at offset
//   i*Q+j), handed to cuBLAS with ld=Q, is READ by cuBLAS as a column-major matrix M_cm[Q,P] with
//   M_cm(j,i) = M(i,j) -- i.e. M_cm == M^T, unconditionally. This project's launch_linear (see its
//   own comment above) establishes the standard consequence: to compute a ROW-MAJOR product
//   C[p,q] = L[p,k] . R[k,q] via cuBLAS, call cublasGemmEx(opR, opL, q, p, k, R_buf, ldR, L_buf,
//   ldL, C_buf, q) -- operands SWAPPED (R first), each op flag N/T selecting whether that operand
//   is used as its OWN native row-major shape (N) or that shape's transpose (T).
//
//   Applying this to Newton-Schulz's three per-iteration products, where U (== `upd`, native
//   [rows,cols], m=min(rows,cols), n=max(rows,cols)) stands in for the CPU reference's Xp buffer
//   (Xp==U when rows<=cols; Xp==U^T when rows>cols):
//
//   1) A[m,m] = Xp @ Xp^T (the Gram matrix, ALWAYS contracting over n, the large dimension):
//        rows<=cols: A = U @ U^T   -> gemm_muon(OP_T, OP_N, rows, rows, cols, U, cols, U, cols, A, rows)
//        rows>cols:  A = U^T @ U   -> gemm_muon(OP_N, OP_T, cols, cols, rows, U, cols, U, cols, A, cols)
//      (U supplied as BOTH operands -- legal and safe: read-only, matches the op flags derived
//      above, no data race since neither call writes U.)
//   2) AA[m,m] = A @ A -- square, no transpose subtlety either way:
//        gemm_muon(OP_N, OP_N, m, m, m, A, m, A, m, AA, m)
//   3) BX_native[rows,cols] = "B @ Xp" re-expressed so the OUTPUT lands directly in U's native
//      orientation (never Xp's [m,n] working orientation) -- the crux of avoiding a final
//      un-transpose entirely:
//        rows<=cols (Xp==U, so B@Xp is already native [rows,cols]):
//              gemm_muon(OP_N, OP_N, cols, rows, rows, U, cols, B, m, BX, cols)
//        rows>cols (Xp==U^T, so BX=B@Xp -> BX^T = Xp^T@B^T = U@B^T, which IS what we want since we
//                   need BX's TRANSPOSE to land in U's native orientation for the next
//                   iteration/the final apply):
//              gemm_muon(OP_T, OP_N, cols, rows, cols, B, m, U, cols, BX, cols)
//      (B lives in the `a` scratch buffer after the axpby combine below overwrites it in place.)
//   4) B = b*A + c*AA (elementwise, in place into A -- axpby_kernel)
//   5) U = a*U + BX (elementwise, native [rows,cols] -- axpby_kernel), which becomes the NEXT
//      iteration's U with no reorientation needed, and IS the final result after the last iteration.
//
// This derivation is the single riskiest, most novel piece of this GPU port (Phase 1 audit) --
// verified against the CPU double-accumulated reference AND an independent Gram-property check
// (off-diagonal collapse, bounded diagonal) at BOTH rows>cols and rows<cols shapes, toy and
// production, by cuda_tests.cpp's "CUDA Muon Newton-Schulz" test.
inline void muon_newton_schulz_device(float* upd, int rows, int cols, int steps = 5) {
    constexpr float a = 3.4445f, b = -4.7750f, c = 2.0315f;
    const bool transposed = rows > cols;
    const int m = transposed ? cols : rows;
    const int mn = rows * cols;
    const int mm = m * m;
    const int block = 256;

    SUB0_CUDA_CHECK_VOID(cudaMemsetAsync(g_dev_muon_ss, 0, sizeof(double), g_stream));
    grad_normsq_kernel<<<(mn + block - 1) / block, block, 0, g_stream>>>(upd, g_dev_muon_ss, mn);
    muon_normalize_kernel<<<(mn + block - 1) / block, block, 0, g_stream>>>(upd, mn, g_dev_muon_ss);

    float* A  = g_dev_muon_a;
    float* AA = g_dev_muon_aa;
    float* BX = g_dev_muon_bx;
    for (int it = 0; it < steps; ++it) {
        if (!transposed) gemm_muon(CUBLAS_OP_T, CUBLAS_OP_N, rows, rows, cols, upd, cols, upd, cols, A, rows);
        else              gemm_muon(CUBLAS_OP_N, CUBLAS_OP_T, cols, cols, rows, upd, cols, upd, cols, A, cols);
        gemm_muon(CUBLAS_OP_N, CUBLAS_OP_N, m, m, m, A, m, A, m, AA, m);
        axpby_kernel<<<(mm + block - 1) / block, block, 0, g_stream>>>(A, A, AA, b, c, mm);
        if (!transposed) gemm_muon(CUBLAS_OP_N, CUBLAS_OP_N, cols, rows, rows, upd, cols, A, m, BX, cols);
        else              gemm_muon(CUBLAS_OP_T, CUBLAS_OP_N, cols, rows, cols, A, m, upd, cols, BX, cols);
        axpby_kernel<<<(mn + block - 1) / block, block, 0, g_stream>>>(upd, upd, BX, a, 1.0f, mn);
    }
}

// One Muon-routed weight matrix's GPU update: momentum EMA + Nesterov lookahead ->
// muon_newton_schulz_device -> fan-ratio scale -> decoupled weight decay -> apply. Mirrors
// src/backend_cpu.cpp's muon_step_one exactly (same op order, same reuse of the AdamW momentum
// arena, same scale_factor call) -- see that function's own comment for the checkpoint-
// compatibility reasoning (g_dev_vel stays untouched/zero for these params). `beta` is Muon's own
// momentum EMA coefficient -- hardcoded to 0.95f at the one call site below, matching
// include/sub0/core.hpp's AdamW::muon_beta_ default (never independently configurable on the CPU
// side either, so no new surface here -- AGENTS.md #8).
void muon_step_matrix(std::size_t off, int rows, int cols, float lr, float beta, float wd) {
    ensure_muon_scratch();
    const int n = rows * cols, block = 256;
    muon_ema_nesterov_kernel<<<(n + block - 1) / block, block, 0, g_stream>>>(
        g_dev_m + off, g_dev_grad + off, g_dev_muon_upd, n, beta, g_dev_gs);
    muon_newton_schulz_device(g_dev_muon_upd, rows, cols, 5);
    const float scale = sub0::muon::scale_factor(rows, cols);
    muon_apply_kernel<<<(n + block - 1) / block, block, 0, g_stream>>>(
        g_dev_params + off, g_dev_muon_upd, n, lr, wd, scale);
}

// AdamW state: the reduced grad, the two moments (zeroed), and the double norm accumulator.
// Allocated lazily alongside the train scratch. The weight-decay ranges are compile-time-known
// (sub0::DECAY_RANGES, layout.hpp) -- copied once into __constant__ memory here, not a persistent
// per-parameter mask. g_muon_ranges (sub0::MUON_RANGES) is uploaded here too, ALWAYS (not
// conditionally on Muon being used) -- same reasoning as g_decay_ranges: it's a tiny compile-time
// table, cheap to always upload, and correct-by-construction for a non-Muon run (see is_muon_param).
int opt_alloc() {
    if (g_dev_grad) return 0;
    ensure_stream();
    const size_t n = sub0::PARAM_FLOATS;
    SUB0_CUDA_CHECK(cudaMalloc(&g_dev_grad,  n * sizeof(float)));
    SUB0_CUDA_CHECK(cudaMalloc(&g_dev_m,     n * sizeof(float)));
    SUB0_CUDA_CHECK(cudaMalloc(&g_dev_vel,   n * sizeof(float)));
    SUB0_CUDA_CHECK(cudaMalloc(&g_dev_normsq, sizeof(double)));
    SUB0_CUDA_CHECK(cudaMalloc(&g_dev_gs, sizeof(float)));
    SUB0_CUDA_CHECK(cudaMemset(g_dev_m,   0, n * sizeof(float)));
    SUB0_CUDA_CHECK(cudaMemset(g_dev_vel, 0, n * sizeof(float)));
    SUB0_CUDA_CHECK(cudaMemcpyToSymbol(g_decay_ranges, sub0::DECAY_RANGES.data(),
                                       sizeof(sub0::DecayRange) * sub0::NUM_DECAY_RANGES));
    SUB0_CUDA_CHECK(cudaMemcpyToSymbol(g_muon_ranges, sub0::MUON_RANGES.data(),
                                       sizeof(sub0::DecayRange) * sub0::NUM_MUON_RANGES));
    return 0;
}

void opt_free() {
    cudaFree(g_dev_grad);  cudaFree(g_dev_m);    cudaFree(g_dev_vel);
    cudaFree(g_dev_normsq); cudaFree(g_dev_gs);
    g_dev_grad = g_dev_m = g_dev_vel = nullptr;
    g_dev_normsq = nullptr;
    g_dev_gs = nullptr;
    free_muon_scratch();
}

// One AdamW step over the whole param blob, op-for-op identical to AdamW::step on the CPU: global
// grad L2 clip (double accumulation), then the per-parameter moment update with the decay mask.
// `t` is the post-increment step counter (>=1) the host optimizer drives. Rebuilds the fused QKV
// weights (Wq/Wk/Wv just changed) and invalidates the inference graph afterwards.
//
// NO host sync here (unlike the pre-2026-07 version): the clip scale gs is computed ON DEVICE by
// grad_clip_scale_kernel from g_dev_normsq and read by adam_step_kernel via pointer, so nothing here
// needs normsq back on the host at all. bc1/bc2 are already host-computable from `t` alone (no
// device dependency). This is the second of two per-step syncs removed 2026-07 -- run_fwd_bwd's loss
// readback (below) was the other -- leaving a single sync at the end of sub0_cuda_train_step.
//
// `muon_lr` <= 0 means pure AdamW (bit-identical to this function's pre-Muon behavior -- the
// per-matrix loop below never runs, adam_step_kernel's is_muon_param check always evaluates false
// against zero-touched-but-still-valid ranges the same way it would for any non-Muon-eligible
// index). `muon_lr` > 0 routes the Muon-eligible matrices (sub0::is_muon_kind) through
// muon_step_matrix FIRST (host loop over PARAM_LAYOUT, matrix by matrix -- mirrors the CPU's
// AdamW::step loop structure, unlike the flat elementwise adam_step_kernel below), then the flat
// AdamW kernel runs over everything else, skipping the ranges Muon just updated
// (is_muon_param/g_muon_ranges) so nothing is double-updated. Order between the two doesn't affect
// correctness (disjoint parameter ranges, same stream g_stream serializes both regardless) --
// matrix-level work first here purely as the more natural reading order.
void device_adam_step(float lr, long t, float b1, float b2, float eps, float wd, float clip,
                      float muon_lr) {
    const int n = static_cast<int>(sub0::PARAM_FLOATS);
    const int block = 256, grid = (n + block - 1) / block;
    SUB0_CUDA_CHECK_VOID(cudaMemsetAsync(g_dev_normsq, 0, sizeof(double), g_stream));
    grad_normsq_kernel<<<grid, block, 0, g_stream>>>(g_dev_grad, g_dev_normsq, n);
    grad_clip_scale_kernel<<<1, 1, 0, g_stream>>>(g_dev_normsq, clip, g_dev_gs);
    const float bc1  = 1.0f - std::pow(b1, static_cast<float>(t));
    const float bc2  = 1.0f - std::pow(b2, static_cast<float>(t));
    if (muon_lr > 0.f) {
        constexpr float MUON_BETA = 0.95f;   // core.hpp's AdamW::muon_beta_ default -- see muon_step_matrix
        for (const sub0::ParamDesc& pd : sub0::PARAM_LAYOUT)
            if (sub0::is_muon_kind(pd.kind))
                muon_step_matrix(pd.off, pd.rows, pd.cols, muon_lr, MUON_BETA, wd);
    }
    adam_step_kernel<<<grid, block, 0, g_stream>>>(g_dev_params, g_dev_grad, g_dev_m, g_dev_vel,
                                                   n, g_dev_gs, lr, b1, b2, eps, wd, bc1, bc2,
                                                   muon_lr > 0.f);
    build_qkv_weights();      // Wq/Wk/Wv changed -> refresh the fused inference/train weight
    invalidate_graph();       // params changed -> recapture the forward graph on next inference
    invalidate_decode_graph();  // ...and the decode graph (defensive: training and decode don't share a
                                // process in this project's normal usage, but cuda_tests.cpp DOES train
                                // then decode in-process in at least one parity test)
}

}  // namespace

// Self-test: confirm the toolchain built device code for THIS GPU and that H2D, kernel
// launch and D2H all work end to end on a buffer the size of the model's parameter blob
// (PARAM_FLOATS). Returns 0 on success. Exposed via the C ABI so the clang-built host
// driver can call it across the DLL boundary.
SUB0_CUDA_API int sub0_cuda_selftest() {
    int dev = 0;
    SUB0_CUDA_CHECK(cudaGetDevice(&dev));
    cudaDeviceProp prop{};
    SUB0_CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
    std::printf("cuda selftest: device %d = %s, sm_%d%d, %.1f GB VRAM (built for sm_%d)\n",
                dev, prop.name, prop.major, prop.minor,
                static_cast<double>(prop.totalGlobalMem) / (1024.0 * 1024.0 * 1024.0),
                CUDA_ARCH);

    const int    n     = static_cast<int>(sub0::PARAM_FLOATS);
    const size_t bytes = static_cast<size_t>(n) * sizeof(float);
    std::vector<float> hx(static_cast<size_t>(n), 1.0f);
    std::vector<float> hy(static_cast<size_t>(n), 2.0f);

    float* dx = nullptr;
    float* dy = nullptr;
    SUB0_CUDA_CHECK(cudaMalloc(&dx, bytes));
    SUB0_CUDA_CHECK(cudaMalloc(&dy, bytes));
    SUB0_CUDA_CHECK(cudaMemcpy(dx, hx.data(), bytes, cudaMemcpyHostToDevice));
    SUB0_CUDA_CHECK(cudaMemcpy(dy, hy.data(), bytes, cudaMemcpyHostToDevice));

    const int threads = 256;
    const int blocks  = (n + threads - 1) / threads;
    axpy_kernel<<<blocks, threads>>>(3.0f, dx, dy, n);   // y = 3*x + y -> 5
    SUB0_CUDA_CHECK(cudaGetLastError());
    SUB0_CUDA_CHECK(cudaDeviceSynchronize());

    SUB0_CUDA_CHECK(cudaMemcpy(hy.data(), dy, bytes, cudaMemcpyDeviceToHost));
    cudaFree(dx);
    cudaFree(dy);

    int bad = 0;
    for (int i = 0; i < n; ++i)
        if (hy[static_cast<size_t>(i)] != 5.0f) ++bad;
    if (bad) {
        std::fprintf(stderr, "cuda selftest: FAIL (%d/%d elements mismatched)\n", bad, n);
        return 2;
    }
    std::printf("cuda selftest: OK (%d elements, y = 3*x + y == 5)\n", n);
    return 0;
}

// ============================================================================
//  Device parameter mirror (the device half of the sync_params hooks)
// ============================================================================
// Phase 2b: allocate the device weight blob and move it host<->device. When this file
// becomes the active backend these back sync_params_to_device/host directly; for now the
// parity tests drive them. Idempotent init; upload auto-inits.

SUB0_CUDA_API [[nodiscard]] int sub0_cuda_init() {
    if (g_dev_params) return 0;
    // Block (sleep) the host thread on device syncs instead of spinning a core at 100%: the resident
    // training step is GPU-bound, so busy-waiting the per-step sync just wastes a CPU core (and its
    // power/heat) for no throughput. Must precede the first runtime call that creates the context; a
    // non-zero return (context already active) is harmless, so it is intentionally not checked.
    cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);
    SUB0_CUDA_CHECK(cudaMalloc(&g_dev_params, sub0::PARAM_FLOATS * sizeof(float)));
    return 0;
}

SUB0_CUDA_API void sub0_cuda_shutdown() {
    invalidate_graph();
    invalidate_decode_graph();
    if (g_dev_params) cudaFree(g_dev_params);
    g_dev_params = nullptr;
    fwd_free();                                  // release the resident forward scratch too
    kv_free();                                   // and the decode KV-cache
    train_free();                                // and the training scratch (Phase 2d)
    eval_free();                                 // and the forward-only eval scratch
    opt_free();                                  // and the optimizer state
    bind_free();                                 // and the binding-compose override table
    if (g_cublas) { cublasDestroy(g_cublas); g_cublas = nullptr; g_handle_tf32 = -1; }
    if (g_stream) { cudaStreamDestroy(g_stream); g_stream = nullptr; }
}

SUB0_CUDA_API [[nodiscard]] int sub0_cuda_upload_params(const float* host) {
    if (!g_dev_params) { const int r = sub0_cuda_init(); if (r) return r; }
    if (wqkv_alloc()) return 1;                  // ensure the fused-QKV weight buffers exist
    SUB0_CUDA_CHECK(cudaMemcpyAsync(g_dev_params, host, sub0::PARAM_FLOATS * sizeof(float),
                                    cudaMemcpyHostToDevice, g_stream));
    build_qkv_weights();                         // rebuild the fused [Wq|Wk|Wv] from the new weights
    invalidate_graph();                          // weights changed -> recapture the forward graph
    invalidate_decode_graph();                   // ...and the decode graph
    SUB0_CUDA_CHECK(cudaStreamSynchronize(g_stream));
    return 0;
}

SUB0_CUDA_API [[nodiscard]] int sub0_cuda_download_params(float* host) {
    if (!g_dev_params) return 1;
    SUB0_CUDA_CHECK(cudaMemcpy(host, g_dev_params, sub0::PARAM_FLOATS * sizeof(float),
                               cudaMemcpyDeviceToHost));
    return 0;
}

// AdamW moment sync (for crash-safe checkpoint / resume): the optimizer state lives on the device
// during a GPU run, so the train stage round-trips it through the host adam_m/adam_v buffers
// around save/load -- mirroring the param sync above. Both are PARAM_FLOATS long.
SUB0_CUDA_API [[nodiscard]] int sub0_cuda_download_opt(float* host_m, float* host_v) {
    if (!g_dev_m || !g_dev_vel) return 1;
    SUB0_CUDA_CHECK(cudaMemcpy(host_m, g_dev_m,   sub0::PARAM_FLOATS * sizeof(float), cudaMemcpyDeviceToHost));
    SUB0_CUDA_CHECK(cudaMemcpy(host_v, g_dev_vel, sub0::PARAM_FLOATS * sizeof(float), cudaMemcpyDeviceToHost));
    return 0;
}

SUB0_CUDA_API [[nodiscard]] int sub0_cuda_upload_opt(const float* host_m, const float* host_v) {
    if (opt_alloc()) return 1;                   // ensure the moment buffers exist
    SUB0_CUDA_CHECK(cudaMemcpy(g_dev_m,   host_m, sub0::PARAM_FLOATS * sizeof(float), cudaMemcpyHostToDevice));
    SUB0_CUDA_CHECK(cudaMemcpy(g_dev_vel, host_v, sub0::PARAM_FLOATS * sizeof(float), cudaMemcpyHostToDevice));
    return 0;
}

// ============================================================================
//  Dense linear (parity building block)
// ============================================================================
// Y[T,out] = X[T,in] . W[in,out] (+ bias). Takes host pointers, runs on the device and
// returns the result -- a self-contained kernel the CPU-parity test drives. Per-call
// device allocations are fine for the test harness; the real forward keeps them resident.
// TODO(robustness): the SUB0_CUDA_CHECK early-returns leak any already-allocated dX/dW/dY/dB.
// Acceptable for the test harness (process exits); add RAII/goto-cleanup if this ever ships.
SUB0_CUDA_API int sub0_cuda_linear(const float* X, int T, int in, int out,
                                   const float* W, const float* bias, float* Y) {
    const size_t xb = static_cast<size_t>(T) * in * sizeof(float);
    const size_t wb = static_cast<size_t>(in) * out * sizeof(float);
    const size_t yb = static_cast<size_t>(T) * out * sizeof(float);
    const size_t bb = static_cast<size_t>(out) * sizeof(float);

    float* dX = nullptr;
    float* dW = nullptr;
    float* dY = nullptr;
    float* dB = nullptr;
    SUB0_CUDA_CHECK(cudaMalloc(&dX, xb));
    SUB0_CUDA_CHECK(cudaMalloc(&dW, wb));
    SUB0_CUDA_CHECK(cudaMalloc(&dY, yb));
    if (bias) SUB0_CUDA_CHECK(cudaMalloc(&dB, bb));
    SUB0_CUDA_CHECK(cudaMemcpy(dX, X, xb, cudaMemcpyHostToDevice));
    SUB0_CUDA_CHECK(cudaMemcpy(dW, W, wb, cudaMemcpyHostToDevice));
    if (bias) SUB0_CUDA_CHECK(cudaMemcpy(dB, bias, bb, cudaMemcpyHostToDevice));

    launch_linear(dX, dW, dB, dY, T, in, out);   // cuBLAS GEMM (+ bias) -- same path as the forward
    SUB0_CUDA_CHECK(cudaGetLastError());
    SUB0_CUDA_CHECK(cudaDeviceSynchronize());

    SUB0_CUDA_CHECK(cudaMemcpy(Y, dY, yb, cudaMemcpyDeviceToHost));
    cudaFree(dX);
    cudaFree(dW);
    cudaFree(dY);
    if (dB) cudaFree(dB);
    return 0;
}

// ============================================================================
//  Full forward pass (single window) -> logits, CPU-parity gated
// ============================================================================
// Runs the same op sequence as Model::forward over a BATCH of windows: M = batch*T rows.
// Every Linear / RMSNorm / GELU / residual runs over all M rows at once; attention is
// per-window causal (windows independent). Requires sub0_cuda_upload_params() first. ids is
// [batch*T]; out_logits receives [batch*T, VOCAB]. Scratch is resident.
// NOTE: this is the inference (graph-captured) path. The training path -- forward_train (saves
// activations) + backward_device + device_adam_step -- lives below (sub0_cuda_train_step), gated
// by the grad/AdamW parity tests. TODO(phase-2e): wire sub0_cuda_train_step into the train stage
// (implement the sub0:: API in a GPU engine) so SUB0_COMPUTE=GPU trains end-to-end on device;
// today the train stage still runs the CPU backend.
SUB0_CUDA_API int sub0_cuda_forward(const int* ids, int batch, int T, float* out_logits) {
    if (!g_dev_params) return 1;
    if (batch < 1 || batch > MAX_FWD_BATCH || T < 1 || T > SEQ_LEN) return 1;
    if (fwd_alloc(batch)) return 1;
    if (ensure_wqkv_f32()) return 1;             // BF16: (re)build the F32 mirror this path reads
    ensure_cublas();
    if (capture_graph(batch, T)) return 1;       // capture once per shape (replay thereafter)
    const int M = batch * T;
    // H2D ids -> graph replay -> D2H logits, all ordered on the capture stream.
    SUB0_CUDA_CHECK(cudaMemcpyAsync(g_fwd.dids, ids, static_cast<size_t>(M) * sizeof(int),
                                    cudaMemcpyHostToDevice, g_stream));
    SUB0_CUDA_CHECK(cudaGraphLaunch(g_graph_exec, g_stream));
    SUB0_CUDA_CHECK(cudaMemcpyAsync(out_logits, g_fwd.logits, static_cast<size_t>(M) * VOCAB * sizeof(float),
                                    cudaMemcpyDeviceToHost, g_stream));
    SUB0_CUDA_CHECK(cudaStreamSynchronize(g_stream));
    return 0;
}

// ============================================================================
//  Batched forward-only cross-entropy: the EVAL path (val_nelbo, report)
// ============================================================================
// Returns the mean cross-entropy over `batch` windows of `T` tokens -- the same reduction
// sub0_cuda_train_step reports as its loss, with no backward pass, no gradient buffer, and no saved
// activations. This is what makes evaluating a model on device cheap: the training entry point would
// compute and immediately discard a full backward and needs the whole TrainScratch allocation to do it.
//
// Built on the INFERENCE forward (forward_device via the captured graph -- the identical path
// sub0_cuda_forward drives), NOT on forward_train. That choice is deliberate and load-bearing:
//   * it is FP32 end to end, matching the CPU sub0::forward() this number is compared against, where
//     forward_train stores its residual stream in bf16;
//   * it is the path already gated against the CPU forward by sub0_cuda_forward_check, so eval
//     inherits an existing, tested parity claim instead of asserting a new one;
//   * it allocates no per-execution activation buffers, which is the whole point.
// The cost is that it materializes the full [batch*T, VOCAB] logits buffer (fwd_alloc's `full` half)
// rather than chunking it the way training's head_ce_chunked does, so the CALLER picks a batch whose
// logits fit -- see evaluate()'s eval_device_batch() in train_stage.cpp, which derives it from a
// buffer budget and splits its window set accordingly.
//
// `lengths` (optional) masks short-window padding exactly as the training step does; targets < 0
// (LOSS_IGNORE_INDEX) are excluded from both the loss and the per-window normalizer, same as training.
// Requires sub0_cuda_upload_params() first. FP32 math mode is forced (parity with the CPU eval); the
// tracked set_handle_tf32 makes that free on every call after the first.
SUB0_CUDA_API [[nodiscard]] int sub0_cuda_forward_loss(const int* ids, const int* targets, int batch, int T,
                                                       double* out_loss, const int* lengths) {
    if (!g_dev_params || !ids || !targets || !out_loss) return 1;
    if (batch < 1 || batch > MAX_FWD_BATCH || T < 1 || T > SEQ_LEN) return 1;
    const int    M  = batch * T;
    const size_t Mz = static_cast<size_t>(M);
    if (fwd_alloc(batch, true, T)) return 1;
    if (ensure_wqkv_f32()) return 1;             // BF16 builds: (re)build the F32 mirror this path reads
    if (eval_alloc(Mz)) return 1;
    ensure_cublas();
    set_handle_tf32(false);                      // tight FP32 inference math -- parity with the CPU eval
    if (capture_graph(batch, T)) return 1;       // capture once per shape (replay thereafter)

    SUB0_CUDA_CHECK(cudaMemcpyAsync(g_fwd.dids, ids, Mz * sizeof(int),
                                    cudaMemcpyHostToDevice, g_stream));
    SUB0_CUDA_CHECK(cudaMemcpyAsync(g_ev.targets, targets, Mz * sizeof(int),
                                    cudaMemcpyHostToDevice, g_stream));
    const int* d_lengths = nullptr;              // null => every window is full T (dense path)
    if (lengths) {
        SUB0_CUDA_CHECK(cudaMemcpyAsync(g_ev.lengths, lengths, static_cast<size_t>(batch) * sizeof(int),
                                        cudaMemcpyHostToDevice, g_stream));
        d_lengths = g_ev.lengths;
    }
    // Per-window count of non-ignored targets, computed host-side from the targets we already hold --
    // the same derivation run_fwd_bwd does, and for the same reason: with a loss mask in play the
    // per-window mean must normalize over ACTIVE positions, not raw length. Stays nullptr (and the CE
    // kernel uses `lengths` verbatim) when nothing is masked, which is the ordinary eval case.
    const int* d_active = nullptr;
    bool any_masked = false;
    for (int i = 0; i < M; ++i) if (targets[i] < 0) { any_masked = true; break; }
    if (any_masked) {
        std::vector<int> active(static_cast<size_t>(batch), 0);
        for (int b = 0; b < batch; ++b) {
            const int len = lengths ? lengths[b] : T;
            int a = 0;
            for (int tt = 0; tt < len; ++tt) if (targets[static_cast<size_t>(b) * T + tt] >= 0) ++a;
            active[static_cast<size_t>(b)] = a;
        }
        SUB0_CUDA_CHECK(cudaMemcpyAsync(g_ev.active, active.data(), static_cast<size_t>(batch) * sizeof(int),
                                        cudaMemcpyHostToDevice, g_stream));
        d_active = g_ev.active;
    }

    SUB0_CUDA_CHECK(cudaMemsetAsync(g_ev.loss, 0, sizeof(double), g_stream));
    SUB0_CUDA_CHECK(cudaGraphLaunch(g_graph_exec, g_stream));      // logits -> g_fwd.logits [M,V]
    constexpr int kCeBlock = 256;                                  // one block per row, as in training
    ce_backward_kernel<kCeBlock><<<M, kCeBlock, 0, g_stream>>>(
        g_fwd.logits, g_ev.targets, /*dlogits=*/nullptr, g_ev.loss, M, T, batch, d_lengths, d_active,
        /*row_offset=*/0);
    double loss = 0.0;
    SUB0_CUDA_CHECK(cudaMemcpyAsync(&loss, g_ev.loss, sizeof(double), cudaMemcpyDeviceToHost, g_stream));
    SUB0_CUDA_CHECK(cudaStreamSynchronize(g_stream));
    SUB0_CUDA_CHECK(cudaGetLastError());
    *out_loss = loss;
    return 0;
}

// Reset the decode KV-cache for a new sequence: ensure the M=1 forward scratch + the K/V cache exist.
// (Positions are supplied per call, so "reset" just guarantees the buffers.) Requires uploaded params.
//
// set_handle_tf32(false) (tight FP32 inference math, parity with the CPU path) lives HERE, once per
// generation session, rather than inside sub0_cuda_forward_one below -- it used to run on EVERY decode
// token, but the handle's math mode doesn't change between tokens within one generation, so repeating
// it ~150-200x/session (once per sampled token) was pure waste on an already launch-count-dominated
// hot path (see the Phase-1 decode-loop audit: GPU busy only ~31% of decode wall-time, dominated by
// per-token kernel-launch dispatch, not by this kind of redundant call -- still, free to remove).
SUB0_CUDA_API [[nodiscard]] int sub0_cuda_kv_reset() {
    if (!g_dev_params) return 1;
    if (fwd_alloc(1) || kv_alloc()) return 1;
    ensure_cublas();
    set_handle_tf32(false);
    return 0;
}
// One decode step: token `id` at window position `pos` (0-based, < SEQ_LEN). Runs the CAPTURED decode
// graph (forward_one_device_graphed, one cudaGraphLaunch instead of forward_one_device's ~147
// individual kernel launches -- see the Phase-2 decode-loop audit: GPU busy was only ~31% of decode
// wall-time, cudaLaunchKernel alone was 68.6% of it, dominated by per-token launch dispatch, not actual
// GPU work) and copies the logits [VOCAB] to the host. Requires sub0_cuda_upload_params +
// sub0_cuda_kv_reset (which also sets the FP32 math mode this path needs, and allocates g_decode_state
// -- see that function's comment). Captures the graph lazily on first use per session (or after any
// invalidate_decode_graph() trigger); every call after that is just the {id,pos} memcpy + graph replay.
SUB0_CUDA_API [[nodiscard]] int sub0_cuda_forward_one(int id, int pos, float* out_logits) {
    if (!g_dev_params || !g_kv_k) return 1;
    if (id < 0 || id >= VOCAB || pos < 0 || pos >= SEQ_LEN) return 1;
    if (ensure_wqkv_f32()) return 1;               // BF16: (re)build the F32 mirror this path reads
    if (capture_decode_graph()) return 1;          // no-op if already captured for this session
    const int h_id_pos[2] = { id, pos };
    SUB0_CUDA_CHECK(cudaMemcpyAsync(g_decode_state, h_id_pos, 2 * sizeof(int), cudaMemcpyHostToDevice, g_stream));
    SUB0_CUDA_CHECK(cudaGraphLaunch(g_decode_graph_exec, g_stream));
    SUB0_CUDA_CHECK(cudaMemcpyAsync(out_logits, g_fwd.logits, VOCAB * sizeof(float), cudaMemcpyDeviceToHost, g_stream));
    SUB0_CUDA_CHECK(cudaStreamSynchronize(g_stream));
    SUB0_CUDA_CHECK(cudaGetLastError());
    return 0;
}

// On-device correctness + speed guard for the decode path: with random params, the incremental
// forward_one (KV-cache) must reproduce the FULL forward's per-position logits -- forward_one(id_t, t)
// attends the same causal context as sub0_cuda_forward()'s row t -- and then time the decode tok/s.
// Returns nonzero on a parity failure. Self-contained (uploads its own random weights).
SUB0_CUDA_API int sub0_cuda_forward_one_check(int T, int iters, double* out_maxreldiff, double* out_toks) {
    if (T < 2 || T > SEQ_LEN) return 1;
    if (iters < 1) iters = 200;
    if (sub0_cuda_init()) return 1;
    std::vector<float> hp(sub0::PARAM_FLOATS);                          // small random weights (finite logits)
    unsigned s = 0x243f6a88u;
    for (auto& x : hp) { s = s * 1664525u + 1013904223u; x = (static_cast<float>(s >> 9) / 8388608.0f - 1.0f) * 0.02f; }
    if (sub0_cuda_upload_params(hp.data())) return 1;                   // also builds the fused wqkv
    std::vector<int> ids(static_cast<size_t>(T));
    for (int& x : ids) { s = s * 1664525u + 1013904223u; x = static_cast<int>(s % static_cast<unsigned>(VOCAB)); }

    std::vector<float> full(static_cast<size_t>(T) * VOCAB);            // full-forward reference [T, V]
    if (sub0_cuda_forward(ids.data(), 1, T, full.data())) return 1;
    if (sub0_cuda_kv_reset()) return 1;
    std::vector<float> one(static_cast<size_t>(VOCAB));
    double maxrel = 0.0;
    int top1_agree = 0;                                                // do both pick the same argmax token?
    auto argmax = [V = VOCAB](const float* p) { int b = 0; for (int j = 1; j < V; ++j) if (p[j] > p[b]) b = j; return b; };
    for (int pos = 0; pos < T; ++pos) {
        if (sub0_cuda_forward_one(ids[pos], pos, one.data())) return 1;
        const float* ref = full.data() + static_cast<size_t>(pos) * VOCAB;
        double maxabs = 0.0, maxmag = 1e-30;
        for (int j = 0; j < VOCAB; ++j) {
            maxabs = std::fmax(maxabs, std::fabs(static_cast<double>(one[j]) - ref[j]));
            maxmag = std::fmax(maxmag, std::fabs(static_cast<double>(ref[j])));
        }
        maxrel = std::fmax(maxrel, maxabs / maxmag);
        if (argmax(one.data()) == argmax(ref)) ++top1_agree;
    }
    const double top1_pct = 100.0 * top1_agree / T;

    const int mid = T / 2;                                              // time decode steps at a mid position
    for (int w = 0; w < 5; ++w) forward_one_device(ids[mid], mid);
    cudaStreamSynchronize(g_stream);
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    cudaEventRecord(e0, g_stream);
    for (int it = 0; it < iters; ++it) forward_one_device(ids[mid], mid);
    cudaEventRecord(e1, g_stream); cudaEventSynchronize(e1);
    float ms = 0.f; cudaEventElapsedTime(&ms, e0, e1); cudaEventDestroy(e0); cudaEventDestroy(e1);
    const double per_ms = static_cast<double>(ms) / iters, toks = per_ms > 0 ? 1000.0 / per_ms : 0.0;

    // Under RANDOM weights the two-pass decode softmax vs the flash online-softmax diverge and amplify
    // over the layers, so the raw-logit rel diff is a poor gate; top-1 agreement (would gen pick the
    // same token?) is the gen-relevant signal, verified against a trained model separately.
    const bool ok = top1_pct >= 95.0;
    std::printf("cuda forward_one check: T=%d | top-1 agree %.1f%% (max rel diff %.2e)  %s | decode %.3f ms/tok (%.0f tok/s)\n",
                T, top1_pct, maxrel, ok ? "OK" : "FAIL", per_ms, toks);
    if (out_maxreldiff) *out_maxreldiff = maxrel;
    if (out_toks)       *out_toks       = toks;
    return ok ? 0 : 2;
}

// ============================================================================
//  Training step: forward + backward + AdamW on device (Phase 2d), parity-gated
// ============================================================================
// Shared driver: upload ids+targets, run the activation-saving forward and the reverse pass into
// g_dev_grad, sync, and read back the mean cross-entropy. Requires upload_params() first.
static int run_fwd_bwd(const int* ids, const int* targets, int batch, int T, double* out_loss,
                       const int* lengths = nullptr) {
    if (!g_dev_params) return 1;
    if (batch < 1 || batch > MAX_FWD_BATCH || T < 1 || T > SEQ_LEN) return 1;
    // Row-product sizing: a step only needs batch*T rows, so a varied-T caller that keeps
    // batch*T inside a previously reserved budget (sub0_cuda_train_reserve) never reallocates here.
    if (fwd_alloc(batch, false, T) || train_alloc(batch, T) || opt_alloc()) return 1;
    ensure_cublas();
    set_handle_tf32(CudaTf32::get());            // training uses the baked math mode
    const int M = batch * T;
    SUB0_CUDA_CHECK(cudaMemcpyAsync(g_fwd.dids, ids, static_cast<size_t>(M) * sizeof(int),
                                    cudaMemcpyHostToDevice, g_stream));
    SUB0_CUDA_CHECK(cudaMemcpyAsync(g_tr.dtargets, targets, static_cast<size_t>(M) * sizeof(int),
                                    cudaMemcpyHostToDevice, g_stream));
    const int* d_lengths = nullptr;              // null => every window is full T (dense path)
    if (lengths) {
        SUB0_CUDA_CHECK(cudaMemcpyAsync(g_tr.lengths, lengths, static_cast<size_t>(batch) * sizeof(int),
                                        cudaMemcpyHostToDevice, g_stream));
        d_lengths = g_tr.lengths;
    }
    // Loss masking (ignore-index): a target < 0 (LOSS_IGNORE_INDEX) is a position the model must NOT be
    // graded on -- e.g. the uncombine curriculum's harness-injected spans (see train_batch's loss_mask /
    // sub0::LOSS_IGNORE_INDEX). When any are present, the per-window loss/grad normalizes over the ACTIVE
    // (non-ignored) count, not the raw length, to match the CPU op_cross_entropy. Compute that count per
    // window here (host-side, from the targets we already hold) and upload it. With no masked target this
    // stays nullptr and the CE kernel uses `lengths` exactly as before -- the dense path is untouched.
    const int* d_active = nullptr;
    bool any_masked = false;
    for (int i = 0; i < M; ++i) if (targets[i] < 0) { any_masked = true; break; }
    if (any_masked) {
        std::vector<int> active(static_cast<size_t>(batch), 0);
        for (int b = 0; b < batch; ++b) {
            const int len = lengths ? lengths[b] : T;
            int a = 0;
            for (int tt = 0; tt < len; ++tt) if (targets[static_cast<size_t>(b) * T + tt] >= 0) ++a;
            active[static_cast<size_t>(b)] = a;
        }
        SUB0_CUDA_CHECK(cudaMemcpyAsync(g_tr.active, active.data(), static_cast<size_t>(batch) * sizeof(int),
                                        cudaMemcpyHostToDevice, g_stream));
        d_active = g_tr.active;
    }
    forward_train(batch, T);
    backward_device(batch, T, d_lengths, d_active);
    double loss = 0.0;
    SUB0_CUDA_CHECK(cudaMemcpyAsync(&loss, g_tr.loss, sizeof(double), cudaMemcpyDeviceToHost, g_stream));
    SUB0_CUDA_CHECK(cudaStreamSynchronize(g_stream));
    SUB0_CUDA_CHECK(cudaGetLastError());
    if (out_loss) *out_loss = loss;              // the CE kernel already produced the mean
    return 0;
}

// Forward+backward only: fills out_grad[PARAM_FLOATS] with the reduced gradient (for the
// gradient-parity test against the CPU train_batch grad). Optionally returns the mean loss.
// `lengths` (optional) masks short-window padding exactly as the training step does.
SUB0_CUDA_API int sub0_cuda_backward(const int* ids, const int* targets, int batch, int T,
                                     float* out_grad, double* out_loss, const int* lengths) {
    if (run_fwd_bwd(ids, targets, batch, T, out_loss, lengths)) return 1;
    SUB0_CUDA_CHECK(cudaMemcpy(out_grad, g_dev_grad, sub0::PARAM_FLOATS * sizeof(float),
                               cudaMemcpyDeviceToHost));
    return 0;
}

// AdamW step over the device gradient (after sub0_cuda_backward). `t` is the post-increment step
// counter the host optimizer maintains. Uses the CPU AdamW defaults so the two stay in lockstep.
// `muon_lr` <= 0 is pure AdamW (the pre-Muon behavior, bit-identical); > 0 routes the Muon-eligible
// matrices through Newton-Schulz at that (separate, typically much larger) learning rate -- see
// device_adam_step's own comment.
SUB0_CUDA_API int sub0_cuda_adam_step(float lr, long t, float muon_lr) {
    if (!g_dev_grad) return 1;
    ensure_cublas();
    device_adam_step(lr, t, 0.9f, 0.95f, 1e-8f, 0.01f, 1.0f, muon_lr);
    SUB0_CUDA_CHECK(cudaStreamSynchronize(g_stream));
    SUB0_CUDA_CHECK(cudaGetLastError());
    return 0;
}

// Full device training step: forward + backward + AdamW, returning the mean loss. The end-to-end
// path the GPU train stage will drive once it replaces the CPU backend. `lengths` (optional) gives
// each window's trained length so short documents padded up to T train only on their real tokens.
// `muon_lr` <= 0 is pure AdamW; > 0 hybridizes with Muon on the Muon-eligible matrices (see
// device_adam_step). The CPU train stage computes this the same way opt.use_muon() gates the CPU
// AdamW path's own Muon branch (see train_stage.cpp's GpuTrainer::step call site).
SUB0_CUDA_API [[nodiscard]] int sub0_cuda_train_step(const int* ids, const int* targets, int batch, int T,
                                       float lr, long t, double* out_loss, const int* lengths,
                                       float muon_lr) {
    if (run_fwd_bwd(ids, targets, batch, T, out_loss, lengths)) return 1;
    device_adam_step(lr, t, 0.9f, 0.95f, 1e-8f, 0.01f, 1.0f, muon_lr);
    SUB0_CUDA_CHECK(cudaStreamSynchronize(g_stream));
    SUB0_CUDA_CHECK(cudaGetLastError());
    return 0;
}

// Reserve the full training row budget (batch * SEQ_LEN rows) up front. A varied-T training loop
// calls this ONCE with its nominal/tuned batch, then every per-step (batch_t, T) whose product fits
// the budget reuses the same buffers -- without this, the first short-T step would size the scratch
// to ITS product and each new per-step maximum would trigger a full grow-realloc (~30 cudaMallocs +
// graph invalidation) mid-run. Also pre-sizes the flash-backward stats scratch to its own ceiling
// (batch * N_HEADS * SEQ_LEN >= batch_t * N_HEADS * T for every admitted pair) for the same reason.
// Failure (VRAM) is a clean nonzero so the caller can fall back to the CPU path BEFORE training
// starts, instead of discovering the OOM on step 1.
SUB0_CUDA_API [[nodiscard]] int sub0_cuda_train_reserve(int batch) {
    if (batch < 1 || batch > MAX_FWD_BATCH) return 1;
    // On ANY failure below, unconditionally release whatever partially succeeded before returning --
    // fwd_alloc/train_alloc/ensure_bwd_stats each fail mid-sequence (e.g. train_alloc's ~20-buffer
    // cudaMalloc chain can succeed on the first dozen and fail on a later, larger one) leaving real,
    // sizeable VRAM resident with capacity still marked 0. A caller measuring free VRAM right after a
    // failed reserve (to size a smaller fallback attempt, see GpuTrainer::enable) would otherwise see
    // that leftover partial allocation as "in use", undercounting what's really available once this
    // function's own next call (or anyone else's) frees it anyway.
    if (fwd_alloc(batch, false) || train_alloc(batch) || opt_alloc() ||
        ensure_bwd_stats(static_cast<size_t>(batch) * N_HEADS * SEQ_LEN)) {
        fwd_free_batch(); train_free(); opt_free(); free_bwd_stats();
        return 1;
    }
    return 0;
}

// Scratch observability for the row-budget tests: current row capacities and the monotonic
// (re)allocation counts (bumped every time fwd_alloc/train_alloc actually allocates, NOT on the
// capacity-hit fast path). Any pointer may be null.
SUB0_CUDA_API int sub0_cuda_scratch_stats(long long* fwd_rows, long long* tr_rows,
                                          long long* fwd_grows, long long* tr_grows) {
    if (fwd_rows)  *fwd_rows  = static_cast<long long>(g_fwd_rows);
    if (tr_rows)   *tr_rows   = static_cast<long long>(g_tr_rows);
    if (fwd_grows) *fwd_grows = g_fwd_grows;
    if (tr_grows)  *tr_grows  = g_tr_grows;
    return 0;
}

// Install (or clear) the binding-compose override table the embed kernels read -- the device half of
// docs/BACKENDS.md's "Design: binding-compose on CUDA". The HOST precomputes which flat positions of
// the NEXT step (or decode session) embed as composed rows; the device only composes what it is told
// to. Wire layout: see the kBindEntryInts block above (and the cross-referenced SUB0_DEV_BIND_*
// constants in include/sub0/device_backend.hpp). Semantics:
//   * override_idx == nullptr or n_positions <= 0  =>  CLEAR the installed table (the per-step
//     default -- the trainer clears after each step). Always succeeds, even before init.
//   * A table stays installed across calls until cleared/replaced, and is read by EVERY embed path
//     (train forward/backward, batched inference forward, decode) -- callers that only want it for
//     one step must clear it afterward.
//   * Entries are VALIDATED here, host-side, and the whole install is REJECTED (nonzero, nothing
//     changes) on: an unsupported encoding (only the param-free MeanPool/Hash/HRR arms have device
//     kernels -- the learned-enc_w arms and Scalar keep their documented CPU-only limit), an
//     out-of-range override/frag index, a frag id outside [0, VOCAB), or an Absolute-positional
//     build (only the RoPE-path embed kernels carry the override branch; production is RoPE, and
//     silently ignoring a table would mis-train).
//   * Host contract for PERSISTENT-range ids: a position whose token id is >= VOCAB MUST be marked
//     overridden (an overridden row never reads ids[m], so any id is safe there) -- the plain-lookup
//     branch indexes tok_emb[id] directly, the same OOB the CPU's unconditional is_persistent_slot
//     guard exists to close (see op_embed's comment, backend_cpu.cpp). The device deliberately does
//     NOT re-guard per row: the host walks every window's ids anyway to build this table, and the
//     no-table fast path must stay bit-identical to the pre-binding kernels.
// Returns 0 ok; 1 = CUDA/alloc failure; 2 = rejected input. Synchronizes before returning, so the
// caller's host arrays may be freed/reused immediately (same contract shape as upload_params).
SUB0_CUDA_API [[nodiscard]] int sub0_cuda_set_window_bindings(const int* override_idx, int n_positions,
                                                              const int* entries, int n_entries,
                                                              const int* frags, int n_frags) {
    if (ensure_bind_hdr()) return 1;
    if (!override_idx || n_positions <= 0) {                   // clear
        g_bind_host.n_positions = 0;
        SUB0_CUDA_CHECK(cudaMemcpyAsync(g_dev_bind, &g_bind_host, sizeof(DevBindings),
                                        cudaMemcpyHostToDevice, g_stream));
        SUB0_CUDA_CHECK(cudaStreamSynchronize(g_stream));
        return 0;
    }
    if constexpr (POS_ENCODING != PosEncoding::Rope) {
        std::fprintf(stderr, "sub0_cuda_set_window_bindings: rejected -- only the RoPE-path embed "
                             "kernels carry the override branch (Absolute-positional build)\n");
        return 2;
    }
    // --- host-side validation (reject the whole install rather than silently mis-composing) -------
    if (n_positions > MAX_FWD_BATCH * SEQ_LEN) return 2;       // beyond any admissible step's rows
    if (n_entries < 0 || n_frags < 0) return 2;
    if ((n_entries > 0 && !entries) || (n_frags > 0 && !frags)) return 2;
    for (int m = 0; m < n_positions; ++m)
        if (override_idx[m] < -1 || override_idx[m] >= n_entries) return 2;
    for (int e = 0; e < n_entries; ++e) {
        const int off = entries[e * kBindEntryInts + 0];
        const int len = entries[e * kBindEntryInts + 1];
        const int enc = entries[e * kBindEntryInts + 2];
        if (len < 1 || off < 0 || off > n_frags - len) return 2;
        if (enc != kBindEncMeanPool && enc != kBindEncHash && enc != kBindEncHRR) {
            std::fprintf(stderr, "sub0_cuda_set_window_bindings: rejected -- entry %d encoding %d has "
                                 "no device kernel (param-free MeanPool/Hash/HRR only)\n", e, enc);
            return 2;
        }
    }
    for (int f = 0; f < n_frags; ++f)
        if (frags[f] < 0 || frags[f] >= VOCAB) return 2;       // fragment rows index tok_emb [VOCAB,C]
    // --- upload (grow-on-demand buffers; the header re-upload publishes any moved pointer) --------
    if (bind_reserve_one(&g_bind_idx, &g_bind_idx_cap, static_cast<size_t>(n_positions)) ||
        bind_reserve_one(&g_bind_entries, &g_bind_entries_cap,
                         static_cast<size_t>(n_entries) * kBindEntryInts) ||
        bind_reserve_one(&g_bind_frags, &g_bind_frags_cap, static_cast<size_t>(n_frags)))
        return 1;
    SUB0_CUDA_CHECK(cudaMemcpyAsync(g_bind_idx, override_idx,
                                    static_cast<size_t>(n_positions) * sizeof(int),
                                    cudaMemcpyHostToDevice, g_stream));
    if (n_entries > 0)
        SUB0_CUDA_CHECK(cudaMemcpyAsync(g_bind_entries, entries,
                                        static_cast<size_t>(n_entries) * kBindEntryInts * sizeof(int),
                                        cudaMemcpyHostToDevice, g_stream));
    if (n_frags > 0)
        SUB0_CUDA_CHECK(cudaMemcpyAsync(g_bind_frags, frags,
                                        static_cast<size_t>(n_frags) * sizeof(int),
                                        cudaMemcpyHostToDevice, g_stream));
    g_bind_host = DevBindings{ g_bind_idx, g_bind_entries, g_bind_frags, g_bind_roles, n_positions, 0 };
    SUB0_CUDA_CHECK(cudaMemcpyAsync(g_dev_bind, &g_bind_host, sizeof(DevBindings),
                                    cudaMemcpyHostToDevice, g_stream));
    SUB0_CUDA_CHECK(cudaStreamSynchronize(g_stream));
    return 0;
}

// Select TF32 tensor-core GEMM math (on != 0) vs full FP32 for subsequent forwards. Runtime
// knob -- the optimum depends on the host + GEMM shapes, so it is intended to feed the
// autotuner. Parity tests force FP32 here for a tight gate.
SUB0_CUDA_API void sub0_cuda_set_tf32(int on) {
    ensure_cublas();
    CudaTf32::set(on != 0);        // sweep the knob (a no-op in a non-tuning build)
    set_handle_tf32(on != 0);      // force the handle mode now (for the bench / parity tests)
}

// Profile the device forward: time `iters` runs of the resident kernel chain (no host
// transfers) under FP32 then TF32 math and print avg ms + the speedup. Self-contained
// (allocates the param mirror + scratch, ids = token 0; GEMM timing is data-independent).
// batch is clamped to MAX_FWD_BATCH.
SUB0_CUDA_API int sub0_cuda_benchmark(int batch, int T, int iters) {
    if (T < 1 || T > SEQ_LEN || iters < 1) return 1;
    if (batch < 1) batch = 1;
    if (batch > MAX_FWD_BATCH) batch = MAX_FWD_BATCH;
    if (sub0_cuda_init()) return 1;
    if (fwd_alloc(batch)) return 1;
    const int M = batch * T;
    cudaMemset(g_fwd.dids, 0, static_cast<size_t>(M) * sizeof(int));   // ids = token 0

    auto time_eager = [&](bool tf32) -> double {
        set_handle_tf32(tf32);                                         // also invalidates the graph
        for (int w = 0; w < 5; ++w) forward_device(batch, T);         // warmup
        cudaStreamSynchronize(g_stream);
        cudaEvent_t s, e; cudaEventCreate(&s); cudaEventCreate(&e);
        cudaEventRecord(s, g_stream);
        for (int it = 0; it < iters; ++it) forward_device(batch, T);
        cudaEventRecord(e, g_stream); cudaEventSynchronize(e);
        float ms = 0.f; cudaEventElapsedTime(&ms, s, e);
        cudaEventDestroy(s); cudaEventDestroy(e);
        return static_cast<double>(ms) / iters;
    };
    auto time_graph = [&]() -> double {
        capture_graph(batch, T);                                      // FP32 (set just below)
        for (int w = 0; w < 5; ++w) cudaGraphLaunch(g_graph_exec, g_stream);
        cudaStreamSynchronize(g_stream);
        cudaEvent_t s, e; cudaEventCreate(&s); cudaEventCreate(&e);
        cudaEventRecord(s, g_stream);
        for (int it = 0; it < iters; ++it) cudaGraphLaunch(g_graph_exec, g_stream);
        cudaEventRecord(e, g_stream); cudaEventSynchronize(e);
        float ms = 0.f; cudaEventElapsedTime(&ms, s, e);
        cudaEventDestroy(s); cudaEventDestroy(e);
        return static_cast<double>(ms) / iters;
    };
    const double fp32 = time_eager(false);
    const double tf32 = time_eager(true);
    set_handle_tf32(false);                                           // capture the graph in FP32
    const double graph = time_graph();
    std::printf("cuda bench forward: batch=%d T=%d M=%d iters=%d | eager FP32 %.3f ms | eager TF32 %.3f ms | "
                "graph FP32 %.3f ms | graph %.2fx vs eager\n",
                batch, T, M, iters, fp32, tf32, graph, graph > 0.0 ? fp32 / graph : 0.0);
    return 0;
}

// On-device correctness + speed guard for the flash attention FORWARD. Runs the naive reference
// (attn_train_act_kernel) and the tiled kernel (attn_fwd_tiled_kernel) on the SAME random q/k/v at
// the training config, then (1) compares outputs -- both consume keys in increasing j with the same
// online-softmax recurrence, so they must agree to ~bf16 rounding -- and (2) times both for a
// dimensionless speedup RATIO. The ratio is the regression guard the Catch2 test asserts on: unlike
// an absolute-ms floor it does NOT drift with GPU clocks or host load (both kernels ride the same
// conditions), so it is a stable structural check that the tiling stayed intact. Returns nonzero
// only on a PARITY failure (a real bug); the caller chooses the minimum speedup to require.
SUB0_CUDA_API int sub0_cuda_attn_check(int batch, int T, int iters,
                                       double* out_maxreldiff, double* out_speedup) {
    if (T < 1 || T > SEQ_LEN) return 1;
    if (batch < 1) batch = 1;
    if (batch > MAX_FWD_BATCH) batch = MAX_FWD_BATCH;
    if (iters < 1) iters = 50;
    if (sub0_cuda_init()) return 1;
    const int    C = D_MODEL, H = N_HEADS, M = batch * T;
    const size_t nqkv = static_cast<size_t>(M) * sub0::QKV_STRIDE;      // fused q|k|v
    const size_t natt = static_cast<size_t>(M) * C;

    std::vector<float> h(nqkv);                                         // random q/k/v (host -> f32 -> act_t)
    unsigned s = 0x9e3779b9u;
    for (size_t i = 0; i < nqkv; ++i) { s = s * 1664525u + 1013904223u; h[i] = static_cast<float>(s >> 8) / 8388608.0f - 1.0f; }
    float* dqf = nullptr; act_t* dqkv = nullptr; act_t* dref = nullptr; act_t* dtes = nullptr;
    float* fref = nullptr; float* ftes = nullptr;
    SUB0_CUDA_CHECK(cudaMalloc(&dqf,  nqkv * sizeof(float)));
    SUB0_CUDA_CHECK(cudaMalloc(&dqkv, nqkv * sizeof(act_t)));
    SUB0_CUDA_CHECK(cudaMalloc(&dref, natt * sizeof(act_t)));
    SUB0_CUDA_CHECK(cudaMalloc(&dtes, natt * sizeof(act_t)));
    SUB0_CUDA_CHECK(cudaMalloc(&fref, natt * sizeof(float)));
    SUB0_CUDA_CHECK(cudaMalloc(&ftes, natt * sizeof(float)));
    SUB0_CUDA_CHECK(cudaMemcpy(dqf, h.data(), nqkv * sizeof(float), cudaMemcpyHostToDevice));
    { const int bk = 256; f32_to_act_kernel<<<static_cast<int>((nqkv + bk - 1) / bk), bk, 0, g_stream>>>(dqf, dqkv, static_cast<int>(nqkv)); }

    constexpr int HD = D_HEAD, TK = attn_tile_k<HD>();
    auto run_naive = [&] {
        const int blk = 64, total = batch * H * T;
        attn_train_act_kernel<act_t><<<(total + blk - 1) / blk, blk, 0, g_stream>>>(
            dqkv, dqkv + sub0::QKV_K_OFF, dqkv + sub0::QKV_V_OFF, dref, batch, T, C, H,
            sub0::QKV_STRIDE, sub0::GQA_GROUP);
    };
    auto run_tiled = [&] {
        constexpr int BQ = attn_block_q<HD>();
        const dim3 block(BQ), grid((T + BQ - 1) / BQ, H, batch);
        attn_fwd_tiled_kernel<act_t, HD, TK><<<grid, block, 0, g_stream>>>(
            dqkv, dqkv + sub0::QKV_K_OFF, dqkv + sub0::QKV_V_OFF, dtes, T, C,
            sub0::QKV_STRIDE, sub0::GQA_GROUP);
    };
    run_naive(); run_tiled();
    SUB0_CUDA_CHECK(cudaStreamSynchronize(g_stream));
    SUB0_CUDA_CHECK(cudaGetLastError());

    { const int bk = 256, n = static_cast<int>(natt);                  // bring both outputs to f32 for the host
      act_to_f32_kernel<<<(n + bk - 1) / bk, bk, 0, g_stream>>>(dref, fref, n);
      act_to_f32_kernel<<<(n + bk - 1) / bk, bk, 0, g_stream>>>(dtes, ftes, n); }
    std::vector<float> a(natt), bvals(natt);
    SUB0_CUDA_CHECK(cudaMemcpy(a.data(),     fref, natt * sizeof(float), cudaMemcpyDeviceToHost));
    SUB0_CUDA_CHECK(cudaMemcpy(bvals.data(), ftes, natt * sizeof(float), cudaMemcpyDeviceToHost));
    double maxabs = 0.0, maxmag = 1e-30;
    for (size_t i = 0; i < natt; ++i) {
        const double d = std::fabs(static_cast<double>(a[i]) - bvals[i]), mg = std::fabs(static_cast<double>(a[i]));
        if (d > maxabs) maxabs = d;
        if (mg > maxmag) maxmag = mg;
    }
    const double reldiff = maxabs / maxmag;

    auto timed = [&](auto&& fn) -> double {                            // GPU-event time; the RATIO is the guard
        for (int w = 0; w < 5; ++w) fn();
        cudaStreamSynchronize(g_stream);
        cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
        cudaEventRecord(e0, g_stream);
        for (int it = 0; it < iters; ++it) fn();
        cudaEventRecord(e1, g_stream); cudaEventSynchronize(e1);
        float ms = 0.f; cudaEventElapsedTime(&ms, e0, e1); cudaEventDestroy(e0); cudaEventDestroy(e1);
        return static_cast<double>(ms) / iters;
    };
    const double naive_ms = timed(run_naive), tiled_ms = timed(run_tiled);
    const double speedup  = tiled_ms > 0.0 ? naive_ms / tiled_ms : 0.0;

    cudaFree(dqf); cudaFree(dqkv); cudaFree(dref); cudaFree(dtes); cudaFree(fref); cudaFree(ftes);
    const bool ok = reldiff < 5e-2;
    std::printf("cuda attn check: batch=%d T=%d HD=%d | naive %.3f ms  tiled %.3f ms = %.1fx | "
                "max rel diff %.2e  %s\n", batch, T, HD, naive_ms, tiled_ms, speedup, reldiff, ok ? "OK" : "FAIL");
    if (out_maxreldiff) *out_maxreldiff = reldiff;
    if (out_speedup)    *out_speedup    = speedup;
    return ok ? 0 : 2;
}

// On-device correctness + speed guard for the flash attention BACKWARD (the three atomic-free tiled
// kernels vs the naive head-per-thread reference, which is itself atomic-free and CPU-grad-gated).
// Runs both on the same random q/k/v/dout at the training config, compares the full dq|dk|dv grad
// buffer (must agree to bf16 rounding -- the tiled path takes dot_i from the bf16-rounded saved
// output, so it is not bit-exact but very close), and reports a dimensionless speedup RATIO (the
// stable regression guard). Returns nonzero only on a PARITY failure.
SUB0_CUDA_API int sub0_cuda_attn_bwd_check(int batch, int T, int iters,
                                           double* out_maxreldiff, double* out_speedup) {
    if (T < 1 || T > SEQ_LEN) return 1;
    if (batch < 1) batch = 1;
    if (batch > MAX_FWD_BATCH) batch = MAX_FWD_BATCH;
    if (iters < 1) iters = 30;
    if (sub0_cuda_init()) return 1;
    const int    C = D_MODEL, H = N_HEADS, M = batch * T;
    const size_t nqkv = static_cast<size_t>(M) * sub0::QKV_STRIDE, natt = static_cast<size_t>(M) * C;
    constexpr int HD = D_HEAD;

    std::vector<float> h(nqkv);                                         // random q/k/v
    unsigned s = 0x85ebca6bu;
    for (size_t i = 0; i < nqkv; ++i) { s = s * 1664525u + 1013904223u; h[i] = static_cast<float>(s >> 8) / 8388608.0f - 1.0f; }
    std::vector<float> hd(natt);                                        // random dout
    for (size_t i = 0; i < natt; ++i) { s = s * 1664525u + 1013904223u; hd[i] = static_cast<float>(s >> 8) / 8388608.0f - 1.0f; }

    float* dqf = nullptr; float* ddf = nullptr; act_t* dqkv = nullptr; act_t* ddout = nullptr;
    act_t* dref = nullptr; act_t* dtes = nullptr; float* fref = nullptr; float* ftes = nullptr;
    SUB0_CUDA_CHECK(cudaMalloc(&dqf,  nqkv * sizeof(float)));
    SUB0_CUDA_CHECK(cudaMalloc(&ddf,  natt * sizeof(float)));
    SUB0_CUDA_CHECK(cudaMalloc(&dqkv, nqkv * sizeof(act_t)));
    SUB0_CUDA_CHECK(cudaMalloc(&ddout, natt * sizeof(act_t)));
    SUB0_CUDA_CHECK(cudaMalloc(&dref, nqkv * sizeof(act_t)));
    SUB0_CUDA_CHECK(cudaMalloc(&dtes, nqkv * sizeof(act_t)));
    SUB0_CUDA_CHECK(cudaMalloc(&fref, nqkv * sizeof(float)));
    SUB0_CUDA_CHECK(cudaMalloc(&ftes, nqkv * sizeof(float)));
    SUB0_CUDA_CHECK(cudaMemcpy(dqf, h.data(),  nqkv * sizeof(float), cudaMemcpyHostToDevice));
    SUB0_CUDA_CHECK(cudaMemcpy(ddf, hd.data(), natt * sizeof(float), cudaMemcpyHostToDevice));
    { const int bk = 256;
      f32_to_act_kernel<<<static_cast<int>((nqkv + bk - 1) / bk), bk, 0, g_stream>>>(dqf, dqkv, static_cast<int>(nqkv));
      f32_to_act_kernel<<<static_cast<int>((natt + bk - 1) / bk), bk, 0, g_stream>>>(ddf, ddout, static_cast<int>(natt)); }

    auto run_naive = [&] {                                             // atomic-free head-per-thread reference
        cudaMemsetAsync(dref, 0, nqkv * sizeof(act_t), g_stream);      // it accumulates with +=
        const int blk = 64, total = batch * (H / sub0::GQA_GROUP);   // one thread per (batch, KV head)
        attn_backward_head_act_kernel<act_t><<<(total + blk - 1) / blk, blk, 0, g_stream>>>(
            dqkv, dqkv + sub0::QKV_K_OFF, dqkv + sub0::QKV_V_OFF, ddout,
            dref, dref + sub0::QKV_K_OFF, dref + sub0::QKV_V_OFF, batch, T, C, H,
            sub0::QKV_STRIDE, sub0::GQA_GROUP);
    };
    auto run_tiled = [&] { launch_attn_bwd_t<act_t>(dqkv, ddout, dtes, batch, T, C, H,
                                                    sub0::QKV_STRIDE, sub0::GQA_GROUP); };
    run_naive(); run_tiled();
    SUB0_CUDA_CHECK(cudaStreamSynchronize(g_stream));
    SUB0_CUDA_CHECK(cudaGetLastError());

    { const int bk = 256, n = static_cast<int>(nqkv);
      act_to_f32_kernel<<<(n + bk - 1) / bk, bk, 0, g_stream>>>(dref, fref, n);
      act_to_f32_kernel<<<(n + bk - 1) / bk, bk, 0, g_stream>>>(dtes, ftes, n); }
    std::vector<float> a(nqkv), bvals(nqkv);
    SUB0_CUDA_CHECK(cudaMemcpy(a.data(),     fref, nqkv * sizeof(float), cudaMemcpyDeviceToHost));
    SUB0_CUDA_CHECK(cudaMemcpy(bvals.data(), ftes, nqkv * sizeof(float), cudaMemcpyDeviceToHost));
    double maxabs = 0.0, maxmag = 1e-30;
    for (size_t i = 0; i < nqkv; ++i) {
        const double d = std::fabs(static_cast<double>(a[i]) - bvals[i]), mg = std::fabs(static_cast<double>(a[i]));
        if (d > maxabs) maxabs = d;
        if (mg > maxmag) maxmag = mg;
    }
    const double reldiff = maxabs / maxmag;

    auto timed = [&](auto&& fn) -> double {
        for (int w = 0; w < 5; ++w) fn();
        cudaStreamSynchronize(g_stream);
        cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
        cudaEventRecord(e0, g_stream);
        for (int it = 0; it < iters; ++it) fn();
        cudaEventRecord(e1, g_stream); cudaEventSynchronize(e1);
        float ms = 0.f; cudaEventElapsedTime(&ms, e0, e1); cudaEventDestroy(e0); cudaEventDestroy(e1);
        return static_cast<double>(ms) / iters;
    };
    const double naive_ms = timed(run_naive), tiled_ms = timed(run_tiled);
    const double speedup  = tiled_ms > 0.0 ? naive_ms / tiled_ms : 0.0;

    cudaFree(dqf); cudaFree(ddf); cudaFree(dqkv); cudaFree(ddout);
    cudaFree(dref); cudaFree(dtes); cudaFree(fref); cudaFree(ftes);
    // NOTE: the naive reference ACCUMULATES dq/dk/dv in bf16 (read-modify-write per step), so it
    // carries ~sqrt(T)*bf16_eps (~6% at T=256) of accumulation noise; the tiled kernels accumulate
    // in FP32 and round once, so they are the MORE accurate of the two. This coarse gate only catches
    // gross breakage -- the tight correctness gate is the CPU-fp32 gradient parity test (cuda_tests).
    const bool ok = reldiff < 1.5e-1;
    std::printf("cuda attn-bwd check: batch=%d T=%d HD=%d | naive %.3f ms  tiled %.3f ms = %.1fx | "
                "max rel diff %.2e  %s\n", batch, T, HD, naive_ms, tiled_ms, speedup, reldiff, ok ? "OK" : "FAIL");
    if (out_maxreldiff) *out_maxreldiff = reldiff;
    if (out_speedup)    *out_speedup    = speedup;
    return ok ? 0 : 2;
}

// Verifies gemm()'s beta=1 accumulate mode and bias_grad_kernel's/launch_linear_bwd's accumulate
// flag that ride on it -- groundwork added for a future row-chunked GEMM (e.g. splitting the
// lm_head backward over M) but not yet used by any real call site, so this is its only coverage
// today. Splits a [M,in]x[M,out] backward into two row-chunks (chunk 1 overwrites dW/dbias, chunk 2
// accumulates into the SAME buffers) and compares against a single full-M reference call -- if
// dW/dbias's math correctly splits as a sum over disjoint row ranges (it does: dW[i,o] =
// sum_m X[m,i]*dY[m,o] splits cleanly at any row boundary), the two must match up to GEMM
// reassociation noise, not the wrong-shape/wrong-scale error a real accumulate-mode bug would cause.
SUB0_CUDA_API int sub0_cuda_test_accumulate_check(int M, int in, int out, double* out_maxreldiff) {
    if (M < 2 || in < 1 || out < 1) return 1;
    if (sub0_cuda_init()) return 1;
    ensure_cublas();
    const int M1 = M / 2, M2 = M - M1;

    std::vector<float> hX(static_cast<size_t>(M) * in), hdY(static_cast<size_t>(M) * out);
    unsigned s = 0x9e3779b9u;
    auto randf = [&] { s = s * 1664525u + 1013904223u; return static_cast<float>(s >> 8) / 8388608.0f - 1.0f; };
    for (auto& v : hX)  v = randf();
    for (auto& v : hdY) v = randf();

    float *dX_in = nullptr, *dY = nullptr;
    float *dW_ref = nullptr, *dbias_ref = nullptr, *dW_chunk = nullptr, *dbias_chunk = nullptr;
    SUB0_CUDA_CHECK(cudaMalloc(&dX_in, hX.size() * sizeof(float)));
    SUB0_CUDA_CHECK(cudaMalloc(&dY,    hdY.size() * sizeof(float)));
    SUB0_CUDA_CHECK(cudaMalloc(&dW_ref,      static_cast<size_t>(in) * out * sizeof(float)));
    SUB0_CUDA_CHECK(cudaMalloc(&dbias_ref,   static_cast<size_t>(out) * sizeof(float)));
    SUB0_CUDA_CHECK(cudaMalloc(&dW_chunk,    static_cast<size_t>(in) * out * sizeof(float)));
    SUB0_CUDA_CHECK(cudaMalloc(&dbias_chunk, static_cast<size_t>(out) * sizeof(float)));
    SUB0_CUDA_CHECK(cudaMemcpy(dX_in, hX.data(),  hX.size()  * sizeof(float), cudaMemcpyHostToDevice));
    SUB0_CUDA_CHECK(cudaMemcpy(dY,    hdY.data(), hdY.size() * sizeof(float), cudaMemcpyHostToDevice));

    // Reference: single full-M call, accumulate=false (the existing, already-trusted behavior).
    launch_linear_bwd(dX_in, nullptr, dY, nullptr, dW_ref, dbias_ref, M, in, out);

    // Chunked: same X/dY split at row M1 -- chunk 1 overwrites dW_chunk/dbias_chunk, chunk 2
    // accumulates into the SAME buffers. [M,in]/[M,out] are row-major, so row M1 starts at offset
    // M1*in / M1*out floats.
    launch_linear_bwd(dX_in, nullptr, dY, nullptr, dW_chunk, dbias_chunk, M1, in, out, false, false);
    launch_linear_bwd(dX_in + static_cast<size_t>(M1) * in, nullptr, dY + static_cast<size_t>(M1) * out,
                      nullptr, dW_chunk, dbias_chunk, M2, in, out, false, true);
    SUB0_CUDA_CHECK(cudaStreamSynchronize(g_stream));
    SUB0_CUDA_CHECK(cudaGetLastError());

    std::vector<float> hW_ref(static_cast<size_t>(in) * out), hW_chunk(static_cast<size_t>(in) * out);
    std::vector<float> hb_ref(out), hb_chunk(out);
    SUB0_CUDA_CHECK(cudaMemcpy(hW_ref.data(),   dW_ref,      hW_ref.size()   * sizeof(float), cudaMemcpyDeviceToHost));
    SUB0_CUDA_CHECK(cudaMemcpy(hW_chunk.data(), dW_chunk,    hW_chunk.size() * sizeof(float), cudaMemcpyDeviceToHost));
    SUB0_CUDA_CHECK(cudaMemcpy(hb_ref.data(),   dbias_ref,   hb_ref.size()   * sizeof(float), cudaMemcpyDeviceToHost));
    SUB0_CUDA_CHECK(cudaMemcpy(hb_chunk.data(), dbias_chunk, hb_chunk.size() * sizeof(float), cudaMemcpyDeviceToHost));

    double maxrel = 0.0;
    for (size_t i = 0; i < hW_ref.size(); ++i) {
        const double d = std::fabs(static_cast<double>(hW_chunk[i]) - hW_ref[i]);
        const double denom = std::max(1e-6, static_cast<double>(std::fabs(hW_ref[i])));
        maxrel = std::max(maxrel, d / denom);
    }
    for (size_t i = 0; i < hb_ref.size(); ++i) {
        const double d = std::fabs(static_cast<double>(hb_chunk[i]) - hb_ref[i]);
        const double denom = std::max(1e-6, static_cast<double>(std::fabs(hb_ref[i])));
        maxrel = std::max(maxrel, d / denom);
    }
    if (out_maxreldiff) *out_maxreldiff = maxrel;

    cudaFree(dX_in); cudaFree(dY);
    cudaFree(dW_ref); cudaFree(dbias_ref); cudaFree(dW_chunk); cudaFree(dbias_chunk);
    return 0;
}

// Dims-independent GEMM self-test for the tied-embedding head (launch_tied_head /
// launch_tied_head_bwd, USE_TIED_EMBEDDINGS's GPU forward+backward). Parameterized by runtime
// M/C/V (not the baked D_MODEL/VOCAB), same convention as sub0_cuda_test_accumulate_check above, so
// ONE binary exercises both a toy scale and a production-like scale (e.g. C=448, V=4306 -- this
// project's actual production d448/vocab config) with no rebuild. Builds random
// table[V,C]/a[M,C]/dY[M,V] on host, computes the CPU reference forward+backward in double
// precision (mirrors backend_cpu.cpp's op_tied_head / Op::TiedHead exactly: logits[m,v] =
// dot(a[m,:],table[v,:]); dA[m,c] = sum_v dY[m,v]*table[v,c]; dTable[v,c] += sum_m dY[m,v]*a[m,c]),
// then compares against the GPU helpers. The backward check pre-seeds the device dTable buffer with
// a random NONZERO pattern before calling launch_tied_head_bwd, so it directly exercises the
// ACCUMULATE (not overwrite-from-zero) semantics backward_device relies on -- by the time the tied
// head's own backward runs, tok_emb's grad slot may already carry the embedding-lookup scatter-add's
// contribution (or vice versa; launch_tied_head_bwd must add correctly either way).
SUB0_CUDA_API int sub0_cuda_tied_head_check(int M, int C, int V,
                                            double* out_relL2_fwd, double* out_relL2_bwd) {
    if (M < 1 || C < 1 || V < 1) return 1;
    if (sub0_cuda_init()) return 1;
    sub0_cuda_set_tf32(0);   // tight FP32 gate, same convention as the other CPU-parity checks
    ensure_cublas();

    std::vector<float> hTable(static_cast<size_t>(V) * C), hA(static_cast<size_t>(M) * C),
                        hDY(static_cast<size_t>(M) * V), hDTableSeed(static_cast<size_t>(V) * C);
    unsigned s = 0x1b873593u;
    auto randf = [&] { s = s * 1664525u + 1013904223u; return static_cast<float>(s >> 8) / 8388608.0f - 1.0f; };
    for (auto& v : hTable)      v = randf() * 0.1f;
    for (auto& v : hA)          v = randf() * 0.1f;
    for (auto& v : hDY)         v = randf() * 0.1f;
    for (auto& v : hDTableSeed) v = randf() * 0.1f;

    float *dTable = nullptr, *dA = nullptr, *dLogits = nullptr;
    float *dDY = nullptr, *dDA = nullptr, *dDTable = nullptr;
    SUB0_CUDA_CHECK(cudaMalloc(&dTable,  hTable.size() * sizeof(float)));
    SUB0_CUDA_CHECK(cudaMalloc(&dA,      hA.size() * sizeof(float)));
    SUB0_CUDA_CHECK(cudaMalloc(&dLogits, static_cast<size_t>(M) * V * sizeof(float)));
    SUB0_CUDA_CHECK(cudaMalloc(&dDY,     hDY.size() * sizeof(float)));
    SUB0_CUDA_CHECK(cudaMalloc(&dDA,     hA.size() * sizeof(float)));
    SUB0_CUDA_CHECK(cudaMalloc(&dDTable, hTable.size() * sizeof(float)));
    SUB0_CUDA_CHECK(cudaMemcpy(dTable, hTable.data(), hTable.size() * sizeof(float), cudaMemcpyHostToDevice));
    SUB0_CUDA_CHECK(cudaMemcpy(dA,     hA.data(),     hA.size()     * sizeof(float), cudaMemcpyHostToDevice));
    SUB0_CUDA_CHECK(cudaMemcpy(dDY,    hDY.data(),    hDY.size()    * sizeof(float), cudaMemcpyHostToDevice));
    SUB0_CUDA_CHECK(cudaMemcpy(dDTable, hDTableSeed.data(), hDTableSeed.size() * sizeof(float), cudaMemcpyHostToDevice));

    launch_tied_head(dA, dTable, dLogits, M, C, V, /*force_tc=*/false);                    // forward
    launch_tied_head_bwd(dA, dTable, dDY, dDA, dDTable, M, C, V, /*force_tc=*/false);       // backward (dDTable pre-seeded)
    SUB0_CUDA_CHECK(cudaStreamSynchronize(g_stream));
    SUB0_CUDA_CHECK(cudaGetLastError());

    std::vector<float> hLogits(static_cast<size_t>(M) * V), hDA(hA.size()), hDTable(hTable.size());
    SUB0_CUDA_CHECK(cudaMemcpy(hLogits.data(), dLogits, hLogits.size() * sizeof(float), cudaMemcpyDeviceToHost));
    SUB0_CUDA_CHECK(cudaMemcpy(hDA.data(),     dDA,     hDA.size()     * sizeof(float), cudaMemcpyDeviceToHost));
    SUB0_CUDA_CHECK(cudaMemcpy(hDTable.data(), dDTable, hDTable.size() * sizeof(float), cudaMemcpyDeviceToHost));

    cudaFree(dTable); cudaFree(dA); cudaFree(dLogits); cudaFree(dDY); cudaFree(dDA); cudaFree(dDTable);

    // CPU reference, double precision, mirrors op_tied_head / Op::TiedHead exactly. Compared via a
    // WHOLE-ARRAY relative-L2 norm (sqrt(sum(d^2)/sum(ref^2))), the same metric the whole-model
    // "CUDA backward matches the CPU reduced gradient" test uses -- NOT a per-element max-relative
    // diff: at this scale (M*V up to ~275k dot products for the production-like case) a per-element
    // max-relative metric is dominated by whichever entry happens to land near a zero-crossing by
    // chance (a near-zero reference denominator inflates an otherwise-tiny absolute rounding
    // difference into a large ratio) -- an artifact of the METRIC, not a real GEMM bug. The whole-
    // array L2 norm is robust to that: a genuine axis-order/transpose bug shows up as a large L2
    // relative error (order 1 or worse), while ordinary FP32 accumulation rounding stays at the
    // 1e-6..1e-5 level regardless of how many near-zero individual entries exist.
    double num_fwd = 0.0, den_fwd = 0.0;
    for (int m = 0; m < M; ++m) {
        for (int v = 0; v < V; ++v) {
            double acc = 0.0;
            for (int c = 0; c < C; ++c)
                acc += static_cast<double>(hA[static_cast<size_t>(m) * C + c]) * hTable[static_cast<size_t>(v) * C + c];
            const double gpu = hLogits[static_cast<size_t>(m) * V + v];
            const double d = gpu - acc;
            num_fwd += d * d;
            den_fwd += acc * acc;
        }
    }
    double num_bwd = 0.0, den_bwd = 0.0;
    for (int m = 0; m < M; ++m) {                                            // dA[m,c] = sum_v dY[m,v]*table[v,c]
        for (int c = 0; c < C; ++c) {
            double acc = 0.0;
            for (int v = 0; v < V; ++v)
                acc += static_cast<double>(hDY[static_cast<size_t>(m) * V + v]) * hTable[static_cast<size_t>(v) * C + c];
            const double gpu = hDA[static_cast<size_t>(m) * C + c];
            const double d = gpu - acc;
            num_bwd += d * d;
            den_bwd += acc * acc;
        }
    }
    for (int v = 0; v < V; ++v) {                        // dTable[v,c] = seed[v,c] + sum_m dY[m,v]*a[m,c]
        for (int c = 0; c < C; ++c) {
            double acc = static_cast<double>(hDTableSeed[static_cast<size_t>(v) * C + c]);
            for (int m = 0; m < M; ++m)
                acc += static_cast<double>(hDY[static_cast<size_t>(m) * V + v]) * hA[static_cast<size_t>(m) * C + c];
            const double gpu = hDTable[static_cast<size_t>(v) * C + c];
            const double d = gpu - acc;
            num_bwd += d * d;
            den_bwd += acc * acc;
        }
    }

    if (out_relL2_fwd) *out_relL2_fwd = std::sqrt(num_fwd / std::max(den_fwd, 1e-30));
    if (out_relL2_bwd) *out_relL2_bwd = std::sqrt(num_bwd / std::max(den_bwd, 1e-30));
    return 0;
}

// Dims-independent CPU-vs-GPU parity self-test for SwiGLU (swiglu_kernel/swiglu_act_kernel/
// swiglu_backward_act_kernel, USE_GATED_FFN's FFN nonlinearity -- see op_swiglu/Op::SwiGLU in
// backend_cpu.cpp for the reference semantics). M/F are runtime parameters (plain elementwise kernels,
// no template shape dependence unlike QK-norm's per-head kernels), so ONE instantiation serves both a
// toy scale and a production-like scale (M=64,F=1792) with no rebuild, matching sub0_cuda_tied_head_
// check's own convention. Always F32 (act_t=float): the test's job is to prove the MATH, independent
// of whatever ACT_DTYPE the CURRENT build bakes.
//
// Forward is checked TWICE: once into a fresh (non-aliased) output buffer, and once with the output
// aliased onto up_pre (y == up), the in-place convention every real call site uses (forward_one_device/
// forward_device/forward_train/backward_device all write the SwiGLU output back over their "up_pre"
// buffer) -- proving the aliasing is actually safe, not just assumed safe from the kernel's per-thread
// read-before-write structure. Backward is checked with dup ALIASED onto dy (the same in-place
// convention backward_device's role-remapped dgact buffer relies on -- see that function's own SwiGLU
// backward comment), verified against a CPU reference in double precision.
SUB0_CUDA_API int sub0_cuda_swiglu_check(int M, int F, double* out_relL2_fwd, double* out_relL2_bwd) {
    if (M < 1 || F < 1) return 1;
    if (sub0_cuda_init()) return 1;
    ensure_stream();

    const size_t n = static_cast<size_t>(M) * F;
    std::vector<float> hGate(n), hUp(n), hDy(n);
    unsigned s = 0x2545f491u;
    auto randf = [&] { s = s * 1664525u + 1013904223u; return static_cast<float>(s >> 8) / 8388608.0f - 1.0f; };
    for (auto& v : hGate) v = randf() * 2.0f;   // wider range than tied-head's 0.1x: exercise SiLU's
    for (auto& v : hUp)   v = randf() * 2.0f;   // curvature away from the near-linear origin region
    for (auto& v : hDy)   v = randf() * 0.5f;

    float *dGate = nullptr, *dUp = nullptr, *dY = nullptr, *dUpAliased = nullptr;
    float *dDy = nullptr, *dDgate = nullptr, *dDup = nullptr, *dDyAliased = nullptr;
    SUB0_CUDA_CHECK(cudaMalloc(&dGate,      n * sizeof(float)));
    SUB0_CUDA_CHECK(cudaMalloc(&dUp,        n * sizeof(float)));
    SUB0_CUDA_CHECK(cudaMalloc(&dY,         n * sizeof(float)));
    SUB0_CUDA_CHECK(cudaMalloc(&dUpAliased, n * sizeof(float)));
    SUB0_CUDA_CHECK(cudaMalloc(&dDy,        n * sizeof(float)));
    SUB0_CUDA_CHECK(cudaMalloc(&dDgate,     n * sizeof(float)));
    SUB0_CUDA_CHECK(cudaMalloc(&dDup,       n * sizeof(float)));
    SUB0_CUDA_CHECK(cudaMalloc(&dDyAliased, n * sizeof(float)));
    SUB0_CUDA_CHECK(cudaMemcpy(dGate, hGate.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    SUB0_CUDA_CHECK(cudaMemcpy(dUp,   hUp.data(),   n * sizeof(float), cudaMemcpyHostToDevice));
    SUB0_CUDA_CHECK(cudaMemcpy(dUpAliased, hUp.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    SUB0_CUDA_CHECK(cudaMemcpy(dDy,   hDy.data(),   n * sizeof(float), cudaMemcpyHostToDevice));
    SUB0_CUDA_CHECK(cudaMemcpy(dDyAliased, hDy.data(), n * sizeof(float), cudaMemcpyHostToDevice));

    launch_swiglu(dGate, dUp, dY, static_cast<int>(n));                    // non-aliased forward
    launch_swiglu(dGate, dUpAliased, dUpAliased, static_cast<int>(n));     // in-place (y == up) forward
    launch_swiglu_bwd_t<float>(dGate, dUp, dDy, dDgate, dDup, static_cast<int>(n));                 // non-aliased backward
    launch_swiglu_bwd_t<float>(dGate, dUp, dDyAliased, dDgate, dDyAliased, static_cast<int>(n));    // dup aliased onto dy
    SUB0_CUDA_CHECK(cudaStreamSynchronize(g_stream));
    SUB0_CUDA_CHECK(cudaGetLastError());

    std::vector<float> hY(n), hYAliased(n), hDgate(n), hDup(n), hDupAliased(n);
    SUB0_CUDA_CHECK(cudaMemcpy(hY.data(),         dY,         n * sizeof(float), cudaMemcpyDeviceToHost));
    SUB0_CUDA_CHECK(cudaMemcpy(hYAliased.data(),  dUpAliased, n * sizeof(float), cudaMemcpyDeviceToHost));
    SUB0_CUDA_CHECK(cudaMemcpy(hDgate.data(),     dDgate,     n * sizeof(float), cudaMemcpyDeviceToHost));
    SUB0_CUDA_CHECK(cudaMemcpy(hDup.data(),       dDup,       n * sizeof(float), cudaMemcpyDeviceToHost));
    SUB0_CUDA_CHECK(cudaMemcpy(hDupAliased.data(), dDyAliased, n * sizeof(float), cudaMemcpyDeviceToHost));

    cudaFree(dGate); cudaFree(dUp); cudaFree(dY); cudaFree(dUpAliased);
    cudaFree(dDy); cudaFree(dDgate); cudaFree(dDup); cudaFree(dDyAliased);

    // CPU reference, double precision, mirrors op_swiglu/Op::SwiGLU exactly.
    double num_fwd = 0.0, den_fwd = 0.0, num_bwd = 0.0, den_bwd = 0.0;
    for (size_t i = 0; i < n; ++i) {
        const double g = hGate[i], u = hUp[i], dy = hDy[i];
        const double sg = 1.0 / (1.0 + std::exp(-g));
        const double silu = g * sg;
        const double dsilu = sg * (1.0 + g * (1.0 - sg));
        const double y_ref = silu * u;
        const double dgate_ref = dy * u * dsilu;
        const double dup_ref   = dy * silu;

        const double dy0 = static_cast<double>(hY[i]) - y_ref;
        const double dy1 = static_cast<double>(hYAliased[i]) - y_ref;         // in-place forward must match too
        num_fwd += dy0 * dy0 + dy1 * dy1;
        den_fwd += 2.0 * y_ref * y_ref;

        const double dg0 = static_cast<double>(hDgate[i]) - dgate_ref;
        const double du0 = static_cast<double>(hDup[i]) - dup_ref;
        const double du1 = static_cast<double>(hDupAliased[i]) - dup_ref;     // dup-aliased-onto-dy must match too
        num_bwd += dg0 * dg0 + du0 * du0 + du1 * du1;
        den_bwd += dgate_ref * dgate_ref + 2.0 * dup_ref * dup_ref;
    }
    if (out_relL2_fwd) *out_relL2_fwd = std::sqrt(num_fwd / std::max(den_fwd, 1e-30));
    if (out_relL2_bwd) *out_relL2_bwd = std::sqrt(num_bwd / std::max(den_bwd, 1e-30));
    return 0;
}

// Dims-independent CPU-vs-GPU parity self-test for QK-norm (qknorm_act_kernel / qknorm_save_act_kernel
// / qknorm_backward_act_kernel). H/DH are TEMPLATE parameters on the kernels themselves (baked
// constexpr, this file's own convention for per-head-shape kernels -- unlike the tied-head check's
// runtime M/C/V, which are genuine cuBLAS GEMM dims), so this helper is itself a template and the
// public entry point below dispatches to a small fixed set of instantiations -- a toy shape and a
// production-like shape (H=7,DH=64: this project's actual production d448/N_HEADS=7 config) -- rather
// than one dims-independent runtime-parameterized function. Always F32 (act_t=float): the test's job
// is to prove the MATH, independent of whatever ACT_DTYPE the CURRENT build happens to bake, same
// reasoning as sub0_cuda_tied_head_check's tf32-off/F32 convention for its own GEMM check.
//
// Forward: builds a random qkv[rows,QKV_STRIDE] (C=H*DH, near-1 gamma so it exercises a realistic operating
// point) on host, runs qknorm_act_kernel, and compares against a CPU reference that mirrors
// backend_cpu.cpp's op_qknorm exactly (per-(row,head) mean-square over the DH-wide slice, r =
// 1/sqrt(ms+eps), y = x*r*gamma) in double precision. The V sub-block (cols [2C,3C)) is checked
// BYTE-IDENTICAL to the input, since qknorm must never touch it.
//
// Backward: re-derives qk_pre via qknorm_save_act_kernel (the same recompute path backward_device
// itself uses) from a FRESH copy of the pre-norm qkv, then runs qknorm_backward_act_kernel with a
// random upstream dy and dgamma buffers SEEDED with a random NONZERO pattern beforehand (same
// technique as sub0_cuda_tied_head_check's dTable seeding) to directly exercise the atomicAdd
// ACCUMULATE semantics the kernel relies on. Compared against a CPU reference mirroring
// backend_cpu.cpp's Op::QKNorm backward case exactly (double precision), including the in-place
// dy->dx overwrite and the dgamma += convention.
template <int H, int DH>
static int qknorm_check_impl(int rows, double* out_relL2_fwd, double* out_relL2_bwd) {
    constexpr int C = H * DH, kQkBlock = 32;
    ensure_stream();
    std::vector<float> hQkv(static_cast<size_t>(rows) * 3 * C), hQgamma(DH), hKgamma(DH);
    unsigned s = 0x9e3779b9u;
    auto randf = [&] { s = s * 1664525u + 1013904223u; return static_cast<float>(s >> 8) / 8388608.0f - 1.0f; };
    for (auto& v : hQkv)    v = randf() * 0.5f;
    for (auto& v : hQgamma) v = 1.0f + randf() * 0.1f;   // near-1, like a trained gamma
    for (auto& v : hKgamma) v = 1.0f + randf() * 0.1f;

    float *dQkv = nullptr, *dQgamma = nullptr, *dKgamma = nullptr;
    SUB0_CUDA_CHECK(cudaMalloc(&dQkv,    hQkv.size() * sizeof(float)));
    SUB0_CUDA_CHECK(cudaMalloc(&dQgamma, static_cast<size_t>(DH) * sizeof(float)));
    SUB0_CUDA_CHECK(cudaMalloc(&dKgamma, static_cast<size_t>(DH) * sizeof(float)));
    SUB0_CUDA_CHECK(cudaMemcpy(dQkv,    hQkv.data(),    hQkv.size() * sizeof(float), cudaMemcpyHostToDevice));
    SUB0_CUDA_CHECK(cudaMemcpy(dQgamma, hQgamma.data(), static_cast<size_t>(DH) * sizeof(float), cudaMemcpyHostToDevice));
    SUB0_CUDA_CHECK(cudaMemcpy(dKgamma, hKgamma.data(), static_cast<size_t>(DH) * sizeof(float), cudaMemcpyHostToDevice));

    const dim3 grid(rows, H, 2);
    qknorm_act_kernel<float, H, H, DH, kQkBlock><<<dim3(rows, H), kQkBlock, 0, g_stream>>>(dQkv, dQgamma, dKgamma, 0);
    qknorm_act_kernel<float, H, H, DH, kQkBlock><<<dim3(rows, H), kQkBlock, 0, g_stream>>>(dQkv, dQgamma, dKgamma, 1);
    SUB0_CUDA_CHECK(cudaStreamSynchronize(g_stream));
    SUB0_CUDA_CHECK(cudaGetLastError());

    std::vector<float> hQkvOut(hQkv.size());
    SUB0_CUDA_CHECK(cudaMemcpy(hQkvOut.data(), dQkv, hQkv.size() * sizeof(float), cudaMemcpyDeviceToHost));

    // CPU reference forward, mirrors op_qknorm exactly (double precision).
    std::vector<double> refQkv(hQkv.begin(), hQkv.end());
    auto qknorm_ref_fwd = [&](std::vector<double>& buf, const std::vector<float>& gamma, int which) {
        for (int r = 0; r < rows; ++r) {
            for (int h = 0; h < H; ++h) {
                const size_t base = static_cast<size_t>(r) * 3 * C + static_cast<size_t>(which) * C + static_cast<size_t>(h) * DH;
                double ms = 0.0;
                for (int j = 0; j < DH; ++j) ms += buf[base + j] * buf[base + j];
                ms /= DH;
                const double rr = 1.0 / std::sqrt(ms + 1e-5);
                for (int j = 0; j < DH; ++j) buf[base + j] = buf[base + j] * rr * gamma[j];
            }
        }
    };
    qknorm_ref_fwd(refQkv, hQgamma, 0);
    qknorm_ref_fwd(refQkv, hKgamma, 1);

    double numF = 0.0, denF = 0.0;
    bool v_untouched = true;
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < 3 * C; ++c) {
            const size_t idx = static_cast<size_t>(r) * 3 * C + c;
            if (c >= 2 * C) { if (hQkvOut[idx] != hQkv[idx]) v_untouched = false; continue; }   // V: untouched
            const double d = static_cast<double>(hQkvOut[idx]) - refQkv[idx];
            numF += d * d;
            denF += refQkv[idx] * refQkv[idx];
        }
    }
    if (out_relL2_fwd) *out_relL2_fwd = v_untouched ? std::sqrt(numF / std::max(denF, 1e-30)) : 1e30;

    // Backward: recompute qk_pre via the SAVE variant from a FRESH copy of the pre-norm qkv (mirrors
    // backward_device's own recompute path), then run the backward kernel with a random upstream dy
    // and NONZERO-seeded dgamma buffers.
    // NOTE: qk_pre is `float*` here (matching this test's A=float instantiation throughout), NOT the
    // production act_t -- the self-test always runs the kernels' F32 instantiation regardless of the
    // current build's ACT_DTYPE (see this function's own header comment), so every buffer must agree
    // on A=float, including qk_pre.
    float* dQkv2 = nullptr; float* dQkPre = nullptr;
    SUB0_CUDA_CHECK(cudaMalloc(&dQkv2, hQkv.size() * sizeof(float)));
    SUB0_CUDA_CHECK(cudaMemcpy(dQkv2, hQkv.data(), hQkv.size() * sizeof(float), cudaMemcpyHostToDevice));
    SUB0_CUDA_CHECK(cudaMalloc(&dQkPre, static_cast<size_t>(rows) * 2 * C * sizeof(float)));
    qknorm_save_act_kernel<float, H, H, DH, kQkBlock><<<dim3(rows, H), kQkBlock, 0, g_stream>>>(dQkv2, dQkPre, dQgamma, dKgamma, 0);
    qknorm_save_act_kernel<float, H, H, DH, kQkBlock><<<dim3(rows, H), kQkBlock, 0, g_stream>>>(dQkv2, dQkPre, dQgamma, dKgamma, 1);

    std::vector<float> hDy(hQkv.size());
    for (auto& v : hDy) v = randf() * 0.3f;
    float* dDqkv = nullptr;
    SUB0_CUDA_CHECK(cudaMalloc(&dDqkv, hDy.size() * sizeof(float)));
    SUB0_CUDA_CHECK(cudaMemcpy(dDqkv, hDy.data(), hDy.size() * sizeof(float), cudaMemcpyHostToDevice));

    std::vector<float> hDqgammaSeed(DH), hDkgammaSeed(DH);
    for (auto& v : hDqgammaSeed) v = randf();
    for (auto& v : hDkgammaSeed) v = randf();
    float *dDqgamma = nullptr, *dDkgamma = nullptr;
    SUB0_CUDA_CHECK(cudaMalloc(&dDqgamma, static_cast<size_t>(DH) * sizeof(float)));
    SUB0_CUDA_CHECK(cudaMalloc(&dDkgamma, static_cast<size_t>(DH) * sizeof(float)));
    SUB0_CUDA_CHECK(cudaMemcpy(dDqgamma, hDqgammaSeed.data(), static_cast<size_t>(DH) * sizeof(float), cudaMemcpyHostToDevice));
    SUB0_CUDA_CHECK(cudaMemcpy(dDkgamma, hDkgammaSeed.data(), static_cast<size_t>(DH) * sizeof(float), cudaMemcpyHostToDevice));

    // Backward kernel's grid differs from the forward's `grid` above -- row*head folded into one
    // grid-stride axis, capped via dgamma_grid_blocks (see qknorm_backward_act_kernel's own comment).
    const dim3 bwd_grid(dgamma_grid_blocks(static_cast<long long>(rows) * H,
                                           qknorm_bwd_blocks_per_sm<float, H, H, DH>()), 1, 2);
    qknorm_backward_act_kernel<float, H, H, DH, kQkBlock><<<bwd_grid, kQkBlock, 0, g_stream>>>(
        dQkPre, dQgamma, dKgamma, dDqkv, dDqgamma, dDkgamma, rows);
    SUB0_CUDA_CHECK(cudaStreamSynchronize(g_stream));
    SUB0_CUDA_CHECK(cudaGetLastError());

    std::vector<float> hDx(hDy.size()), hDqgamma(DH), hDkgamma(DH);
    SUB0_CUDA_CHECK(cudaMemcpy(hDx.data(), dDqkv, hDx.size() * sizeof(float), cudaMemcpyDeviceToHost));
    SUB0_CUDA_CHECK(cudaMemcpy(hDqgamma.data(), dDqgamma, static_cast<size_t>(DH) * sizeof(float), cudaMemcpyDeviceToHost));
    SUB0_CUDA_CHECK(cudaMemcpy(hDkgamma.data(), dDkgamma, static_cast<size_t>(DH) * sizeof(float), cudaMemcpyDeviceToHost));

    cudaFree(dQkv); cudaFree(dQgamma); cudaFree(dKgamma); cudaFree(dQkv2); cudaFree(dQkPre);
    cudaFree(dDqkv); cudaFree(dDqgamma); cudaFree(dDkgamma);

    // CPU reference backward, mirrors Op::QKNorm's backward exactly (double precision): reads the SAME
    // pre-norm x (hQkv) and gamma the GPU path used, overwrites refDx IN PLACE (dy -> dx) exactly like
    // the kernel does, and accumulates (+=) into the seeded dgamma.
    std::vector<double> refDx(hDy.begin(), hDy.end());
    std::vector<double> refDqgamma(hDqgammaSeed.begin(), hDqgammaSeed.end());
    std::vector<double> refDkgamma(hDkgammaSeed.begin(), hDkgammaSeed.end());
    auto qknorm_ref_bwd = [&](const std::vector<float>& gamma, std::vector<double>& dgamma, int which) {
        for (int r = 0; r < rows; ++r) {
            for (int h = 0; h < H; ++h) {
                const size_t base = static_cast<size_t>(r) * 3 * C + static_cast<size_t>(which) * C + static_cast<size_t>(h) * DH;
                double ms = 0.0, S = 0.0;
                for (int j = 0; j < DH; ++j) {
                    const double xj = hQkv[base + j];
                    ms += xj * xj;
                    S  += refDx[base + j] * gamma[j] * xj;
                }
                ms /= DH;
                const double rr = 1.0 / std::sqrt(ms + 1e-5), r3 = rr * rr * rr;
                for (int j = 0; j < DH; ++j) {
                    const double xj = hQkv[base + j], dyj = refDx[base + j], gj = gamma[j];
                    const double dx = dyj * rr * gj - (xj * r3 / DH) * S;
                    dgamma[j] += dyj * xj * rr;
                    refDx[base + j] = dx;   // in place, matches the kernel's own in-place dy->dx overwrite
                }
            }
        }
    };
    qknorm_ref_bwd(hQgamma, refDqgamma, 0);
    qknorm_ref_bwd(hKgamma, refDkgamma, 1);

    double numB = 0.0, denB = 0.0;
    for (int r = 0; r < rows; ++r) {
        for (int which = 0; which < 2; ++which) {
            for (int c = 0; c < C; ++c) {
                const size_t idx = static_cast<size_t>(r) * 3 * C + static_cast<size_t>(which) * C + c;
                const double d = static_cast<double>(hDx[idx]) - refDx[idx];
                numB += d * d; denB += refDx[idx] * refDx[idx];
            }
        }
    }
    for (int j = 0; j < DH; ++j) {
        double d = static_cast<double>(hDqgamma[j]) - refDqgamma[j]; numB += d * d; denB += refDqgamma[j] * refDqgamma[j];
        d = static_cast<double>(hDkgamma[j]) - refDkgamma[j];        numB += d * d; denB += refDkgamma[j] * refDkgamma[j];
    }
    if (out_relL2_bwd) *out_relL2_bwd = std::sqrt(numB / std::max(denB, 1e-30));
    return 0;
}

// Public entry point: dispatches to qknorm_check_impl<H,DH> for a small fixed set of (H,DH) shapes
// (H/DH are template params on the kernels, not runtime dims -- see qknorm_check_impl's own comment).
// shape_sel: 0 = toy (H=2,DH=8), 1 = production-like (H=7,DH=64 -- this project's actual production
// d448/N_HEADS=7 config). Returns 2 for an unrecognized shape_sel, 1 on a bad `rows`/device failure.
SUB0_CUDA_API int sub0_cuda_qknorm_check(int shape_sel, int rows, double* out_relL2_fwd, double* out_relL2_bwd) {
    if (rows < 1) return 1;
    if (sub0_cuda_init()) return 1;
    switch (shape_sel) {
        case 0: return qknorm_check_impl<2, 8>(rows, out_relL2_fwd, out_relL2_bwd);
        case 1: return qknorm_check_impl<7, 64>(rows, out_relL2_fwd, out_relL2_bwd);
        default: return 2;
    }
}

// Binding-compose parity hook (docs/BACKENDS.md "first caps flip"): drives the PRODUCTION embed
// kernels (embed_kernel, embed_act_kernel<act_t>, embed_backward_token_kernel -- the exact
// launches forward_device/forward_train/backward_device make, same launch geometry) over a
// synthetic token table with a mix of overridden and plain rows, through the REAL install path
// (sub0_cuda_set_window_bindings), and compares against sub0::encode_slot / encode_slot_bwd -- the
// in-process CPU reference (scratch_slots.hpp), computed on the identical host data. Fragment
// lengths cover 1 (degenerate single-fragment), mid lengths, and 20 (> HRR_MAX_POS=16, so the
// role-table position clamp is exercised, and > any production slot's span).
//
// `enc_sel` is the wire encoding (sub0::SlotEncoding underlying value; MeanPool/Hash/HRR only).
// out_fwd_maxabs/out_fwd_maxrel: embed_kernel (FP32 store) max |gpu-ref| and that scaled by the
// reference's own max magnitude. out_fwd_act_maxrel: embed_act_kernel<act_t> vs the same FP32
// reference, scaled the same way (a BF16 build rounds only at the store, so this is the
// storage-rounding envelope -- ~2^-8; an F32 build matches the plain kernel).
// out_bwd_maxabs/out_bwd_maxrel: the backward scatter into a zeroed grad table. Returns 0 ok,
// 1 = CUDA failure, 2 = bad enc_sel. Clears the installed table before returning.
SUB0_CUDA_API int sub0_cuda_binding_compose_check(int enc_sel, unsigned seed,
                                                  double* out_fwd_maxabs, double* out_fwd_maxrel,
                                                  double* out_fwd_act_maxrel,
                                                  double* out_bwd_maxabs, double* out_bwd_maxrel) {
    if (enc_sel != kBindEncMeanPool && enc_sel != kBindEncHash && enc_sel != kBindEncHRR) return 2;
    const sub0::SlotEncoding enc = static_cast<sub0::SlotEncoding>(enc_sel);
    constexpr int C = D_MODEL;
    constexpr int R = 96;                        // synthetic tok-table rows (< VOCAB, always)
    static_assert(R < VOCAB, "frag ids must index real tok_emb rows");
    constexpr int M = 24;                        // positions (every other one overridden)
    ensure_stream();

    // Host data: a random table + ids + upstream grad, and an override set with frag lens
    // {1,2,3,4,6,9,12,20} cycling across the overridden positions.
    std::mt19937 rng(seed);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> tab(static_cast<size_t>(R) * C);
    std::vector<float> dh(static_cast<size_t>(M) * C);
    for (float& v : tab) v = nd(rng);
    for (float& v : dh)  v = nd(rng);
    std::uniform_int_distribution<int> tok(0, R - 1);
    std::vector<int> ids(M);
    for (int& v : ids) v = tok(rng);
    constexpr int kLens[] = { 1, 2, 3, 4, 6, 9, 12, 20 };
    std::vector<int> ovr(M, -1), entries, frags;
    int ne = 0;
    for (int m = 0; m < M; m += 2) {
        const int len = kLens[(m / 2) % (sizeof(kLens) / sizeof(kLens[0]))];
        entries.push_back(static_cast<int>(frags.size()));
        entries.push_back(len);
        entries.push_back(enc_sel);
        for (int p = 0; p < len; ++p) frags.push_back(tok(rng));
        ovr[m] = ne++;
    }
    if (sub0_cuda_set_window_bindings(ovr.data(), M, entries.data(), ne,
                                      frags.data(), static_cast<int>(frags.size())) != 0) return 1;

    // Device buffers (per-call allocations are fine in a test hook -- see sub0_cuda_linear's note).
    float* dtab = nullptr; int* dids = nullptr; float* dfwd = nullptr; float* dactf = nullptr;
    act_t* dact = nullptr; float* ddh = nullptr; float* dgrad = nullptr;
    SUB0_CUDA_CHECK(cudaMalloc(&dtab, tab.size() * sizeof(float)));
    SUB0_CUDA_CHECK(cudaMalloc(&dids, ids.size() * sizeof(int)));
    SUB0_CUDA_CHECK(cudaMalloc(&dfwd, static_cast<size_t>(M) * C * sizeof(float)));
    SUB0_CUDA_CHECK(cudaMalloc(&dact, static_cast<size_t>(M) * C * sizeof(act_t)));
    SUB0_CUDA_CHECK(cudaMalloc(&dactf, static_cast<size_t>(M) * C * sizeof(float)));
    SUB0_CUDA_CHECK(cudaMalloc(&ddh, dh.size() * sizeof(float)));
    SUB0_CUDA_CHECK(cudaMalloc(&dgrad, tab.size() * sizeof(float)));
    SUB0_CUDA_CHECK(cudaMemcpyAsync(dtab, tab.data(), tab.size() * sizeof(float), cudaMemcpyHostToDevice, g_stream));
    SUB0_CUDA_CHECK(cudaMemcpyAsync(dids, ids.data(), ids.size() * sizeof(int), cudaMemcpyHostToDevice, g_stream));
    SUB0_CUDA_CHECK(cudaMemcpyAsync(ddh, dh.data(), dh.size() * sizeof(float), cudaMemcpyHostToDevice, g_stream));
    SUB0_CUDA_CHECK(cudaMemsetAsync(dgrad, 0, tab.size() * sizeof(float), g_stream));

    // The REAL kernels, at the production launch geometry (dim3(16,16) over (C, M)).
    {
        const dim3 block(16, 16);
        const dim3 grid((C + block.x - 1) / block.x, (M + block.y - 1) / block.y);
        embed_kernel<<<grid, block, 0, g_stream>>>(dtab, dids, dfwd, M, g_dev_bind);
        embed_act_kernel<act_t><<<grid, block, 0, g_stream>>>(dtab, dids, dact, M, g_dev_bind);
        embed_backward_token_kernel<<<grid, block, 0, g_stream>>>(ddh, dids, dgrad, M, g_dev_bind);
    }
    { const int n = M * C, bk = 256;
      act_to_f32_kernel<<<(n + bk - 1) / bk, bk, 0, g_stream>>>(dact, dactf, n); }
    std::vector<float> gfwd(static_cast<size_t>(M) * C), gact(static_cast<size_t>(M) * C);
    std::vector<float> ggrad(tab.size());
    SUB0_CUDA_CHECK(cudaMemcpyAsync(gfwd.data(), dfwd, gfwd.size() * sizeof(float), cudaMemcpyDeviceToHost, g_stream));
    SUB0_CUDA_CHECK(cudaMemcpyAsync(gact.data(), dactf, gact.size() * sizeof(float), cudaMemcpyDeviceToHost, g_stream));
    SUB0_CUDA_CHECK(cudaMemcpyAsync(ggrad.data(), dgrad, ggrad.size() * sizeof(float), cudaMemcpyDeviceToHost, g_stream));
    SUB0_CUDA_CHECK(cudaStreamSynchronize(g_stream));
    SUB0_CUDA_CHECK(cudaGetLastError());
    cudaFree(dtab); cudaFree(dids); cudaFree(dfwd); cudaFree(dact); cudaFree(dactf);
    cudaFree(ddh); cudaFree(dgrad);

    // CPU reference: encode_slot / encode_slot_bwd for overridden rows, the plain lookup/scatter
    // for the rest -- mirroring op_embed / backward_node's Op::Embed dispatch exactly.
    std::vector<float> ref(static_cast<size_t>(M) * C, 0.f), refg(tab.size(), 0.f);
    for (int m = 0; m < M; ++m) {
        float* rrow = ref.data() + static_cast<size_t>(m) * C;
        const float* drow = dh.data() + static_cast<size_t>(m) * C;
        if (ovr[m] >= 0) {
            const int off = entries[static_cast<size_t>(ovr[m]) * kBindEntryInts + 0];
            const int len = entries[static_cast<size_t>(ovr[m]) * kBindEntryInts + 1];
            const std::span<const int> fr(frags.data() + off, static_cast<size_t>(len));
            sub0::encode_slot(tab.data(), C, fr, enc, rrow);
            sub0::encode_slot_bwd(drow, C, fr, enc, refg.data(), tab.data());
        } else {
            for (int j = 0; j < C; ++j) rrow[j] = tab[static_cast<size_t>(ids[m]) * C + j];
            for (int j = 0; j < C; ++j) refg[static_cast<size_t>(ids[m]) * C + j] += drow[j];
        }
    }
    auto max_diff = [](const std::vector<float>& got, const std::vector<float>& want,
                       double& maxabs, double& maxrel) {
        double ma = 0.0, mag = 1e-30;
        for (size_t i = 0; i < want.size(); ++i) {
            ma  = std::fmax(ma, std::fabs(static_cast<double>(got[i]) - want[i]));
            mag = std::fmax(mag, std::fabs(static_cast<double>(want[i])));
        }
        maxabs = ma; maxrel = ma / mag;
    };
    double fa = 0, fr_ = 0, aa = 0, ar = 0, ba = 0, br = 0;
    max_diff(gfwd, ref, fa, fr_);
    max_diff(gact, ref, aa, ar);
    max_diff(ggrad, refg, ba, br);
    (void)aa;   // the act path's absolute diff scales with the (arbitrary) synthetic magnitudes;
                // the scaled value below is the meaningful storage-rounding gate
    if (out_fwd_maxabs)     *out_fwd_maxabs     = fa;
    if (out_fwd_maxrel)     *out_fwd_maxrel     = fr_;
    if (out_fwd_act_maxrel) *out_fwd_act_maxrel = ar;
    if (out_bwd_maxabs)     *out_bwd_maxabs     = ba;
    if (out_bwd_maxrel)     *out_bwd_maxrel     = br;
    // Leave no table installed for whatever runs next in this process.
    return sub0_cuda_set_window_bindings(nullptr, 0, nullptr, 0, nullptr, 0);
}

// GPU Newton-Schulz self-test hook: runs muon_newton_schulz_device (the EXACT device pipeline
// device_adam_step's per-matrix Muon loop uses) directly on a caller-supplied [rows,cols] matrix,
// bypassing the surrounding momentum/Nesterov/apply steps -- so cuda_tests.cpp can compare its
// output DIRECTLY against sub0::muon::newton_schulz5 (the CPU reference, include/sub0/muon.hpp) at
// both toy and production shapes, both rows>cols and rows<cols (the cuBLAS op-flag-algebra
// derivation this port's Phase 1 audit flagged as its single riskiest, most novel piece -- see
// muon_newton_schulz_device's own comment for that derivation).
//
// `force_tf32` (nonzero) sets the SHARED cuBLAS handle to TF32 tensor-op math BEFORE running,
// restoring the baked/knob default afterward -- the direct reproduction case for "does a Muon GEMM
// silently inherit the training-step TF32 knob" (run_fwd_bwd calls set_handle_tf32 every training
// step); gemm_muon's CUBLAS_COMPUTE_32F_PEDANTIC compute type exists specifically to be immune to
// this, and the test asserts the result is bit-identical whether this is 0 or 1.
SUB0_CUDA_API int sub0_cuda_muon_ns_check(const float* in, int rows, int cols, int force_tf32,
                                          float* out) {
    if (rows < 1 || cols < 1) return 1;
    const size_t n = static_cast<size_t>(rows) * static_cast<size_t>(cols);
    const size_t m = static_cast<size_t>(std::min(rows, cols));
    if (n > sub0::MUON_MAX_MN || m * m > sub0::MUON_MAX_MM) return 1;   // scratch too small for this shape
    if (sub0_cuda_init()) return 1;
    ensure_cublas();
    // Through set_handle_tf32, NOT a raw cublasSetMathMode: the handle's mode is tracked
    // (g_handle_tf32) so that re-requesting the current mode can skip a graph invalidate. A raw
    // push here would desync that tracker and make the apply_math_mode() restore below a silent
    // no-op, leaving the handle in TF32 for whatever ran next.
    if (force_tf32) set_handle_tf32(true);
    if (ensure_muon_scratch()) return 1;

    SUB0_CUDA_CHECK(cudaMemcpyAsync(g_dev_muon_upd, in, n * sizeof(float), cudaMemcpyHostToDevice, g_stream));
    muon_newton_schulz_device(g_dev_muon_upd, rows, cols, 5);
    SUB0_CUDA_CHECK(cudaMemcpyAsync(out, g_dev_muon_upd, n * sizeof(float), cudaMemcpyDeviceToHost, g_stream));
    SUB0_CUDA_CHECK(cudaStreamSynchronize(g_stream));
    SUB0_CUDA_CHECK(cudaGetLastError());

    if (force_tf32) apply_math_mode();   // restore the baked/knob default math mode for whatever runs next
    return 0;
}

// Verifies ce_backward_kernel's row_offset parameter directly: chunking a [M,V] logits/dlogits/loss
// computation into row-chunks (an explicit, test-controlled chunk_cap, independent of the production
// N_CHUNKS derivation in memplan.hpp -- head_ce_chunked in this file is the real caller) must produce
// IDENTICAL dlogits and loss to a single full-M call. This is the direct correctness proof for the
// exact bug risk row_offset exists to prevent: a chunk after the first computing the wrong window/
// position index (b=abs_m/T) would silently corrupt every row past the first chunk boundary (wrong
// lengths[b] lookup -> wrong padding/real classification -> wrong loss weight), not fail loudly.
//
// Builds random logits/targets/lengths on host (lengths so both the "padding" and "real" branches get
// exercised across chunks), runs ce_backward_kernel ONCE at full M (row_offset=0) as the reference,
// then in an explicit chunked loop (row_offset=m0 per call; chunk_cap need not divide M evenly --
// callers are expected to force a non-full last chunk to stress exactly that boundary), and compares
// dlogits (whole-array rel-L2, same reasoning as sub0_cuda_tied_head_check -- robust to a few near-
// zero individual entries) and loss (plain relative diff, one scalar; atomicAdd already accumulates
// correctly across separate kernel launches, this just confirms end-to-end).
SUB0_CUDA_API int sub0_cuda_ce_chunk_check(int M, int T, int batch, int chunk_cap,
                                           double* out_relL2_dlogits, double* out_reldiff_loss) {
    if (M < 1 || T < 1 || batch < 1 || chunk_cap < 1 || batch * T != M) return 1;
    if (sub0_cuda_init()) return 1;
    constexpr int V = VOCAB;

    std::vector<float> hLogits(static_cast<size_t>(M) * V);
    std::vector<int>   hTargets(static_cast<size_t>(M));
    std::vector<int>   hLengths(static_cast<size_t>(batch));
    unsigned s = 0x2545f491u;
    auto randf = [&] { s = s * 1664525u + 1013904223u; return static_cast<float>(s >> 8) / 8388608.0f - 1.0f; };
    auto randi = [&](int n) { s = s * 1664525u + 1013904223u; return static_cast<int>(s % static_cast<unsigned>(n)); };
    for (auto& v : hLogits)  v = randf();
    for (auto& x : hTargets) x = randi(V);
    for (auto& x : hLengths) x = 1 + randi(T);   // [1,T] -- exercises both padding and real rows

    float* dLogits = nullptr; int* dTargets = nullptr; int* dLengths = nullptr;
    float* dDlogitsRef = nullptr; float* dDlogitsChunk = nullptr;
    double* dLossRef = nullptr; double* dLossChunk = nullptr;
    SUB0_CUDA_CHECK(cudaMalloc(&dLogits,  hLogits.size() * sizeof(float)));
    SUB0_CUDA_CHECK(cudaMalloc(&dTargets, hTargets.size() * sizeof(int)));
    SUB0_CUDA_CHECK(cudaMalloc(&dLengths, hLengths.size() * sizeof(int)));
    SUB0_CUDA_CHECK(cudaMalloc(&dDlogitsRef,   hLogits.size() * sizeof(float)));
    SUB0_CUDA_CHECK(cudaMalloc(&dDlogitsChunk, hLogits.size() * sizeof(float)));
    SUB0_CUDA_CHECK(cudaMalloc(&dLossRef,   sizeof(double)));
    SUB0_CUDA_CHECK(cudaMalloc(&dLossChunk, sizeof(double)));
    SUB0_CUDA_CHECK(cudaMemcpy(dLogits,  hLogits.data(),  hLogits.size()  * sizeof(float), cudaMemcpyHostToDevice));
    SUB0_CUDA_CHECK(cudaMemcpy(dTargets, hTargets.data(), hTargets.size() * sizeof(int),   cudaMemcpyHostToDevice));
    SUB0_CUDA_CHECK(cudaMemcpy(dLengths, hLengths.data(), hLengths.size() * sizeof(int),   cudaMemcpyHostToDevice));
    SUB0_CUDA_CHECK(cudaMemsetAsync(dLossRef,   0, sizeof(double), g_stream));
    SUB0_CUDA_CHECK(cudaMemsetAsync(dLossChunk, 0, sizeof(double), g_stream));

    constexpr int kCeBlock = 256;
    // Reference: one call over the whole M, row_offset=0 (today's unchunked shape).
    ce_backward_kernel<kCeBlock><<<M, kCeBlock, 0, g_stream>>>(
        dLogits, dTargets, dDlogitsRef, dLossRef, M, T, batch, dLengths, /*active=*/nullptr, 0);
    // Chunked: identical inputs, split at chunk_cap (may not divide M evenly -- exercises a non-full
    // last chunk), each call given its own absolute row_offset.
    for (int m0 = 0; m0 < M; m0 += chunk_cap) {
        const int rows = (M - m0 < chunk_cap) ? (M - m0) : chunk_cap;
        ce_backward_kernel<kCeBlock><<<rows, kCeBlock, 0, g_stream>>>(
            dLogits + static_cast<size_t>(m0) * V, dTargets + m0,
            dDlogitsChunk + static_cast<size_t>(m0) * V, dLossChunk, rows, T, batch, dLengths, /*active=*/nullptr, m0);
    }
    SUB0_CUDA_CHECK(cudaStreamSynchronize(g_stream));
    SUB0_CUDA_CHECK(cudaGetLastError());

    std::vector<float> hDlogitsRef(hLogits.size()), hDlogitsChunk(hLogits.size());
    double hLossRef = 0.0, hLossChunk = 0.0;
    SUB0_CUDA_CHECK(cudaMemcpy(hDlogitsRef.data(),   dDlogitsRef,   hDlogitsRef.size()   * sizeof(float), cudaMemcpyDeviceToHost));
    SUB0_CUDA_CHECK(cudaMemcpy(hDlogitsChunk.data(), dDlogitsChunk, hDlogitsChunk.size() * sizeof(float), cudaMemcpyDeviceToHost));
    SUB0_CUDA_CHECK(cudaMemcpy(&hLossRef,   dLossRef,   sizeof(double), cudaMemcpyDeviceToHost));
    SUB0_CUDA_CHECK(cudaMemcpy(&hLossChunk, dLossChunk, sizeof(double), cudaMemcpyDeviceToHost));

    cudaFree(dLogits); cudaFree(dTargets); cudaFree(dLengths);
    cudaFree(dDlogitsRef); cudaFree(dDlogitsChunk); cudaFree(dLossRef); cudaFree(dLossChunk);

    double num = 0.0, den = 0.0;
    for (size_t i = 0; i < hDlogitsRef.size(); ++i) {
        const double d = static_cast<double>(hDlogitsChunk[i]) - hDlogitsRef[i];
        num += d * d;
        den += static_cast<double>(hDlogitsRef[i]) * hDlogitsRef[i];
    }
    if (out_relL2_dlogits) *out_relL2_dlogits = std::sqrt(num / std::max(den, 1e-30));
    if (out_reldiff_loss)  *out_reldiff_loss  = std::fabs(hLossChunk - hLossRef) / std::max(1e-12, std::fabs(hLossRef));
    return 0;
}

// Register/local-memory-spill regression guard for the five flash-attention kernels: query the
// ACTUAL compiled kernel attributes via the CUDA runtime (cudaFuncGetAttributes), not by parsing
// ptxas -v text output -- deterministic and load-independent, same philosophy as the speedup RATIO
// above (not wall-clock). cudaFuncAttributes::localSizeBytes > 0 means the kernel is spilling
// per-thread state to local memory. See the dq/dv/dk register-split work: stats/dq/dv/fwd must stay
// at ZERO spill (each was driven there deliberately); dk keeps a small, bounded spill by design (its
// 3*HD register need has no further clean split -- see the comment above attn_bwd_dv_kernel). This
// function only MEASURES; a Catch2 test applies the actual pass/fail thresholds, so a future change
// to any of these kernels' resident state gets caught here before it silently re-spills at scale.
// Any out-pointer may be null. Returns nonzero only if a query itself fails (e.g. a bad device).
SUB0_CUDA_API int sub0_cuda_attn_regcheck(int* stats_regs, int* stats_spill,
                                          int* dq_regs, int* dq_spill,
                                          int* dv_regs, int* dv_spill,
                                          int* dk_regs, int* dk_spill,
                                          int* fwd_regs, int* fwd_spill) {
    if (sub0_cuda_init()) return 1;
    constexpr int HD = D_HEAD, TK = attn_tile_k<HD>(), TQ = attn_tile_q<HD>();
    cudaFuncAttributes attr;
    auto query = [&](const void* fn, int* regs, int* spill) -> bool {
        if (cudaFuncGetAttributes(&attr, fn) != cudaSuccess) return false;
        if (regs)  *regs  = attr.numRegs;
        if (spill) *spill = static_cast<int>(attr.localSizeBytes);
        return true;
    };
    bool ok = true;
    ok &= query(reinterpret_cast<const void*>(&attn_bwd_stats_kernel<act_t, HD, TK>), stats_regs, stats_spill);
    ok &= query(reinterpret_cast<const void*>(&attn_bwd_dq_kernel<act_t, HD, TK>),    dq_regs,    dq_spill);
    ok &= query(reinterpret_cast<const void*>(&attn_bwd_dv_kernel<act_t, HD, TQ>),    dv_regs,    dv_spill);
    ok &= query(reinterpret_cast<const void*>(&attn_bwd_dk_kernel<act_t, HD, TQ>),    dk_regs,    dk_spill);
    ok &= query(reinterpret_cast<const void*>(&attn_fwd_tiled_kernel<act_t, HD, TK>), fwd_regs,   fwd_spill);
    return ok ? 0 : 1;
}

// Adaptive timing bounds (shared with the CPU tuner via sub0/bench.hpp). Rather than a fixed
// iteration count -- which makes profiling time balloon as the per-step cost grows with batch --
// we size the timed run to a wall-time BUDGET so every measurement costs roughly the same. These
// are the GPU-flavoured defaults (a couple of warmups cover clock ramp + graph capture).
static constexpr sub0::bench::Budget TUNE_BUDGET{
    .budget_ms = 1200.0, .warmup = 2, .min_iters = 3, .max_iters = 60,
};

// Time a full TRAINING step (forward_train + backward_device + AdamW) at a given batch: synthetic
// finite params + ids/targets (timing is data-independent), warmups, then timed steps. Writes the
// mean step time (ms) to *out_ms. Quiet -- callers print. batch clamped to MAX_FWD_BATCH.
//   iters > 0  : run exactly that many timed steps (explicit benchmark).
//   iters <= 0 : auto-size the run to `budget_ms` of wall time -- probe once to estimate the
//                per-step cost, then run clamp(budget/est, MIN, MAX) steps. Keeps the profiling
//                wall-time ~constant across batch sizes instead of growing with the workload.
static int time_train_step(int batch, int T, int iters, double budget_ms, double* out_ms) {
    if (T < 1 || T > SEQ_LEN) return 1;
    if (batch < 1) batch = 1;
    if (batch > MAX_FWD_BATCH) batch = MAX_FWD_BATCH;
    if (sub0_cuda_init() || fwd_alloc(batch, false) || train_alloc(batch) || opt_alloc()) return 1;
    ensure_cublas();
    set_handle_tf32(CudaTf32::get());
    const int M = batch * T;

    // Synthetic, finite weights so nothing overflows; build the fused QKV from them once.
    std::vector<float> hp(sub0::PARAM_FLOATS, 0.02f);
    cudaMemcpy(g_dev_params, hp.data(), sub0::PARAM_FLOATS * sizeof(float), cudaMemcpyHostToDevice);
    build_qkv_weights();
    cudaMemset(g_dev_m, 0, sub0::PARAM_FLOATS * sizeof(float));
    cudaMemset(g_dev_vel, 0, sub0::PARAM_FLOATS * sizeof(float));
    std::vector<int> hid(static_cast<size_t>(M)), htg(static_cast<size_t>(M));
    for (int i = 0; i < M; ++i) { hid[i] = i % VOCAB; htg[i] = (i + 1) % VOCAB; }
    cudaMemcpy(g_fwd.dids,     hid.data(), static_cast<size_t>(M) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(g_tr.dtargets,  htg.data(), static_cast<size_t>(M) * sizeof(int), cudaMemcpyHostToDevice);

    long step = 0;
    auto one_step = [&]() {
        forward_train(batch, T);
        backward_device(batch, T, nullptr, nullptr);
        device_adam_step(0.001f, ++step, 0.9f, 0.95f, 1e-8f, 0.01f, 1.0f, 0.0f);   // pure AdamW (this benchmark isn't Muon-specific)
    };
    // Time `n` steps on the stream and return the elapsed milliseconds (GPU event timing is precise).
    auto run_timed = [&](int n) -> double {
        cudaEvent_t s, e; cudaEventCreate(&s); cudaEventCreate(&e);
        cudaEventRecord(s, g_stream);
        for (int i = 0; i < n; ++i) one_step();
        cudaEventRecord(e, g_stream); cudaEventSynchronize(e);
        float ms = 0.f; cudaEventElapsedTime(&ms, s, e);
        cudaEventDestroy(s); cudaEventDestroy(e);
        return static_cast<double>(ms);
    };

    if (iters > 0) {                                                 // explicit fixed-iteration benchmark
        for (int w = 0; w < TUNE_BUDGET.warmup; ++w) one_step();     // warmup
        cudaStreamSynchronize(g_stream);
        const double total = run_timed(iters);
        if (out_ms) *out_ms = total / iters;
        return 0;
    }
    // Budget mode: shared adaptive sizing (warmup + probe + clamp(budget/est, min, max)).
    sub0::bench::Budget b = TUNE_BUDGET;
    if (budget_ms > 0.0) b.budget_ms = budget_ms;
    const sub0::bench::Timing t = sub0::bench::adaptive_time(one_step, run_timed, b);
    if (out_ms) *out_ms = t.per_step_ms;
    return 0;
}

// Profile a full TRAINING step at a given batch and print ms/step + throughput (tok/s). The device
// measurement primitive the GPU autotuner sweeps over batch / kernel-config knobs. Returns the step
// time in `out_ms` (nullable).
SUB0_CUDA_API int sub0_cuda_train_benchmark(int batch, int T, int iters, double* out_ms) {
    double per = 0.0;
    if (time_train_step(batch, T, iters, 0.0, &per)) return 1;
    const int M = (batch < 1 ? 1 : (batch > MAX_FWD_BATCH ? MAX_FWD_BATCH : batch)) * T;
    const double toks = per > 0.0 ? (static_cast<double>(M) * 1000.0) / per : 0.0;
    std::printf("cuda bench train: batch=%d T=%d M=%d iters=%d | step %.3f ms | %.0f tok/s\n",
                (M / T), T, M, iters, per, toks);
    if (out_ms) *out_ms = per;
    return 0;
}

// Quiet training-step timer for the autotuner: measures with the CURRENT knob state (TF32 /
// attn-backward, set via sub0_cuda_set_*), sizing the timed run to `budget_ms` of wall time so
// every batch profiles in roughly the same time. Writes mean ms/step to *out_ms, prints nothing.
SUB0_CUDA_API [[nodiscard]] int sub0_cuda_time_train_step(int batch, int T, double budget_ms, double* out_ms) {
    return time_train_step(batch, T, 0, budget_ms, out_ms);
}

// Per-phase profiler: times forward_train / backward_device / device_adam_step SEPARATELY via CUDA
// events so the step cost can be attributed -- the small-GEMM forward vs the checkpoint-RECOMPUTE
// backward vs the AdamW update (which carries the grad-norm host sync). Data-independent like
// time_train_step; warms up (clock + cuBLAS) then averages `iters` timed steps. Any out ptr may be null.
SUB0_CUDA_API int sub0_cuda_train_profile(int batch, int T, int iters,
                                          double* fwd_ms, double* bwd_ms, double* adam_ms) {
    if (T < 1 || T > SEQ_LEN || iters < 1) return 1;
    if (batch < 1) batch = 1;
    if (batch > MAX_FWD_BATCH) batch = MAX_FWD_BATCH;
    if (sub0_cuda_init() || fwd_alloc(batch, false) || train_alloc(batch) || opt_alloc()) return 1;
    ensure_cublas();
    set_handle_tf32(CudaTf32::get());
    const int M = batch * T;
    std::vector<float> hp(sub0::PARAM_FLOATS, 0.02f);
    cudaMemcpy(g_dev_params, hp.data(), sub0::PARAM_FLOATS * sizeof(float), cudaMemcpyHostToDevice);
    build_qkv_weights();
    cudaMemset(g_dev_m,   0, sub0::PARAM_FLOATS * sizeof(float));
    cudaMemset(g_dev_vel, 0, sub0::PARAM_FLOATS * sizeof(float));
    std::vector<int> hid(static_cast<size_t>(M)), htg(static_cast<size_t>(M));
    for (int i = 0; i < M; ++i) { hid[i] = i % VOCAB; htg[i] = (i + 1) % VOCAB; }
    cudaMemcpy(g_fwd.dids,    hid.data(), static_cast<size_t>(M) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(g_tr.dtargets, htg.data(), static_cast<size_t>(M) * sizeof(int), cudaMemcpyHostToDevice);

    long t = 0;
    auto one = [&] {
        forward_train(batch, T);
        backward_device(batch, T, nullptr, nullptr);
        device_adam_step(0.001f, ++t, 0.9f, 0.95f, 1e-8f, 0.01f, 1.0f, 0.0f);   // pure AdamW (this benchmark isn't Muon-specific)
    };
    for (int w = 0; w < std::max(4, iters); ++w) one();        // warm the clock + cuBLAS before timing
    cudaStreamSynchronize(g_stream);

    cudaEvent_t e0, e1, e2, e3;
    cudaEventCreate(&e0); cudaEventCreate(&e1); cudaEventCreate(&e2); cudaEventCreate(&e3);
    double fsum = 0.0, bsum = 0.0, asum = 0.0;
    for (int i = 0; i < iters; ++i) {
        cudaEventRecord(e0, g_stream); forward_train(batch, T);
        cudaEventRecord(e1, g_stream); backward_device(batch, T, nullptr, nullptr);
        cudaEventRecord(e2, g_stream); device_adam_step(0.001f, ++t, 0.9f, 0.95f, 1e-8f, 0.01f, 1.0f, 0.0f);
        cudaEventRecord(e3, g_stream); cudaEventSynchronize(e3);
        float f = 0.f, b = 0.f, a = 0.f;
        cudaEventElapsedTime(&f, e0, e1); cudaEventElapsedTime(&b, e1, e2); cudaEventElapsedTime(&a, e2, e3);
        fsum += f; bsum += b; asum += a;
    }
    cudaEventDestroy(e0); cudaEventDestroy(e1); cudaEventDestroy(e2); cudaEventDestroy(e3);
    if (fwd_ms)  *fwd_ms  = fsum / iters;
    if (bwd_ms)  *bwd_ms  = bsum / iters;
    if (adam_ms) *adam_ms = asum / iters;
    return 0;
}

// This build's model dimensions, packaged for the pure footprint model (sub0/memplan.hpp).
// `tied` must match USE_TIED_EMBEDDINGS: param_floats()'s head term (and every persistent-byte
// prediction downstream of it) differs by VOCAB*(D_MODEL+1) floats otherwise -- missed on the first
// pass (this Dims instance is separate from cuda_tests.cpp's own kTestDims and the configurator's
// own Dims construction; all three needed the same fix independently), caught by "memplan prediction
// matches measured device usage" failing at a real tied+GPU production build. `qk_norm` must match
// USE_QK_NORM the same way (adds the q_norm/k_norm gamma floats + the qk_pre training scratch term).
// `gated` must match USE_GATED_FFN the same way (Wg replaces the b1/b2 bias floats and adds a third
// bf16 GEMM-weight mirror -- see memplan.hpp's Dims/param_floats/persistent_bytes comments).
static constexpr sub0::memplan::Dims kFootprintDims = sub0::current_build_dims();

// Predicted resident training footprint (MiB) for `batch`, straight from the pure model. No device
// work -- callers that only need the prediction (the runtime guard, the configurator) use this.
SUB0_CUDA_API int sub0_cuda_train_predicted_mb(int batch) {
    if (batch < 1) batch = 1;
    if (batch > MAX_FWD_BATCH) batch = MAX_FWD_BATCH;
    return sub0::memplan::train_resident_mb(kFootprintDims, batch, sizeof(act_t));
}

// Currently-free dedicated VRAM in MiB. The GPU tuner budgets its batch ladder against THIS rather
// than the baked GPU_VRAM_MB spec, because the usable budget is the card total minus the CUDA
// context/driver reservation (~1 GB) the pure footprint model cannot see -- which is why a batch the
// model says "fits" the spec can still spill. We force ONLY the context (cudaFree(nullptr) is the
// canonical no-op that triggers lazy context creation) so that reservation is already subtracted from
// the figure, but we do NOT allocate the param blob or realize cuBLAS here: the previous
// sub0_cuda_init()/ensure_cublas() allocated exactly the memory we are trying to measure free (it
// drove the budget negative -- "showed -2gb"). cuBLAS workspace is allocated lazily on the first GEMM,
// AFTER this probe, so the caller reserves headroom for it. Returns 0 on failure (caller falls back).
//
// Clears any PENDING sticky CUDA error first: this is also the probe GpuTrainer::enable() calls to
// compute a fallback batch right after an optimistic sub0_cuda_train_reserve() attempt failed (a real
// cudaMalloc OOM) -- an unhandled CUDA error is sticky and poisons every subsequent call on the same
// context, including THIS one's cudaMemGetInfo, until something consumes it. Without this, the
// fallback path silently measured free_mb=0 (not genuinely zero -- cudaMemGetInfo itself failing) and
// fell back to the less-accurate static-spec-minus-headroom budget instead of the real, more
// conservative live figure, right when accuracy matters most (immediately after proving the naive
// estimate was too optimistic).
SUB0_CUDA_API int sub0_cuda_free_vram_mb() {
    cudaGetLastError();                                    // clear any pending sticky error (see above)
    cudaFree(nullptr);                                     // ensure the context exists; allocates nothing
    std::size_t free_b = 0, total = 0;
    if (cudaMemGetInfo(&free_b, &total) != cudaSuccess) return 0;
    return static_cast<int>(free_b / (1024 * 1024));
}

// Self-validating footprint probe: allocate the FULL resident training set for `batch` on a clean
// device, measure the actual VRAM it consumed (cudaMemGetInfo delta), and return that alongside the
// pure model's prediction. The gap is how far the memplan.hpp mirror has drifted from the real
// allocations in this file -- the CUDA footprint test asserts it stays tiny, so adding/removing a
// device buffer without updating the mirror fails CI instead of silently mis-predicting in the field.
// Both outputs are MiB; either pointer may be null. Leaves the scratch allocated (callers re-tune or
// shut down as usual). Returns nonzero on a device/allocation failure.
SUB0_CUDA_API [[nodiscard]] int sub0_cuda_train_footprint(int batch, double* predicted_mb, double* actual_mb) {
    if (batch < 1) batch = 1;
    if (batch > MAX_FWD_BATCH) batch = MAX_FWD_BATCH;
    sub0_cuda_shutdown();                                   // clean slate so the delta is purely ours
    std::size_t free_before = 0, total = 0;
    if (cudaMemGetInfo(&free_before, &total) != cudaSuccess) return 1;
    if (sub0_cuda_init() || fwd_alloc(batch, false) || train_alloc(batch) || opt_alloc()) return 1;
    build_qkv_weights();    // BF16 builds: also resident-allocates g_w1_16/g_w2_16/g_wo16/g_wqkv16
                             // (persistent_bytes() counts these -- see memplan.hpp), so the measured
                             // delta below stays comparable to the prediction instead of undercounting it.
    cudaDeviceSynchronize();                                // ensure the allocations are physically resident
    std::size_t free_after = 0;
    if (cudaMemGetInfo(&free_after, &total) != cudaSuccess) return 1;
    const double actual = free_before > free_after ? static_cast<double>(free_before - free_after) : 0.0;
    const double predicted = static_cast<double>(sub0::memplan::train_resident_bytes(kFootprintDims, batch, sizeof(act_t)));
    constexpr double MiB = 1024.0 * 1024.0;
    if (predicted_mb) *predicted_mb = predicted / MiB;
    if (actual_mb)    *actual_mb    = actual / MiB;
    return 0;
}
