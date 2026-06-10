#include "backend.hpp"
#include <stdexcept>

#ifdef SUB0LLM_CUDA
#  include <cuda_runtime.h>
#  include "kernels.cuh"
#endif

namespace sub0llm::backend::cuda {

// ── Memory helpers ────────────────────────────────────────────────────────────

std::shared_ptr<Storage> alloc(std::size_t byte_size, int device_index) {
#ifdef SUB0LLM_CUDA
    cudaSetDevice(device_index);
    void* raw = nullptr;
    const cudaError_t err = cudaMalloc(&raw, byte_size);
    if (err != cudaSuccess) {
        throw std::runtime_error(
            std::format("cudaMalloc({} bytes) failed: {}", byte_size, cudaGetErrorString(err)));
    }
    auto storage = std::make_shared<Storage>();
    storage->data = std::shared_ptr<std::byte[]>(
        static_cast<std::byte*>(raw),
        [](std::byte* p) { cudaFree(p); });
    storage->byte_capacity = byte_size;
    storage->device        = Device::cuda(device_index);
    return storage;
#else
    (void)byte_size; (void)device_index;
    throw std::runtime_error("CUDA backend not compiled in (rebuild with -DSUB0LLM_ENABLE_CUDA=ON)");
#endif
}

void memcpy_h2d(void* dst, const void* src, std::size_t bytes, int device_index) {
#ifdef SUB0LLM_CUDA
    cudaSetDevice(device_index);
    if (const cudaError_t e = cudaMemcpy(dst, src, bytes, cudaMemcpyHostToDevice); e != cudaSuccess)
        throw std::runtime_error(std::format("cudaMemcpy H→D failed: {}", cudaGetErrorString(e)));
#else
    (void)dst; (void)src; (void)bytes; (void)device_index;
    throw std::runtime_error("CUDA backend not compiled in");
#endif
}

void memcpy_d2h(void* dst, const void* src, std::size_t bytes, int device_index) {
#ifdef SUB0LLM_CUDA
    cudaSetDevice(device_index);
    if (const cudaError_t e = cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToHost); e != cudaSuccess)
        throw std::runtime_error(std::format("cudaMemcpy D→H failed: {}", cudaGetErrorString(e)));
#else
    (void)dst; (void)src; (void)bytes; (void)device_index;
    throw std::runtime_error("CUDA backend not compiled in");
#endif
}

void memcpy_d2d(void* dst, const void* src, std::size_t bytes, int device_index) {
#ifdef SUB0LLM_CUDA
    cudaSetDevice(device_index);
    if (const cudaError_t e = cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToDevice); e != cudaSuccess)
        throw std::runtime_error(std::format("cudaMemcpy D→D failed: {}", cudaGetErrorString(e)));
#else
    (void)dst; (void)src; (void)bytes; (void)device_index;
    throw std::runtime_error("CUDA backend not compiled in");
#endif
}

void memset_zero(void* dst, std::size_t bytes, int device_index) {
#ifdef SUB0LLM_CUDA
    cudaSetDevice(device_index);
    if (const cudaError_t e = cudaMemset(dst, 0, bytes); e != cudaSuccess)
        throw std::runtime_error(std::format("cudaMemset failed: {}", cudaGetErrorString(e)));
#else
    (void)dst; (void)bytes; (void)device_index;
    throw std::runtime_error("CUDA backend not compiled in");
#endif
}

// ── Ops ───────────────────────────────────────────────────────────────────────

Tensor add(const Tensor& a, const Tensor& b) {
#ifdef SUB0LLM_CUDA
    Tensor out(a.shape(), a.dtype(), a.device());
    const std::size_t n = static_cast<std::size_t>(a.numel());
    kernels::launch_add_f32(
        reinterpret_cast<const float*>(a.raw_ptr()),
        reinterpret_cast<const float*>(b.raw_ptr()),
        reinterpret_cast<float*>(out.raw_ptr()), n);
    return out;
#else
    (void)a; (void)b;
    throw std::runtime_error("CUDA backend not compiled in");
#endif
}

Tensor mul(const Tensor& a, const Tensor& b) {
#ifdef SUB0LLM_CUDA
    Tensor out(a.shape(), a.dtype(), a.device());
    const std::size_t n = static_cast<std::size_t>(a.numel());
    kernels::launch_mul_f32(
        reinterpret_cast<const float*>(a.raw_ptr()),
        reinterpret_cast<const float*>(b.raw_ptr()),
        reinterpret_cast<float*>(out.raw_ptr()), n);
    return out;
#else
    (void)a; (void)b;
    throw std::runtime_error("CUDA backend not compiled in");
#endif
}

Tensor relu(const Tensor& a) {
#ifdef SUB0LLM_CUDA
    Tensor out(a.shape(), a.dtype(), a.device());
    const std::size_t n = static_cast<std::size_t>(a.numel());
    kernels::launch_relu_f32(
        reinterpret_cast<const float*>(a.raw_ptr()),
        reinterpret_cast<float*>(out.raw_ptr()), n);
    return out;
#else
    (void)a;
    throw std::runtime_error("CUDA backend not compiled in");
#endif
}

Tensor matmul(const Tensor& a, const Tensor& b) {
#ifdef SUB0LLM_CUDA
    const auto M = static_cast<std::size_t>(a.shape(0));
    const auto K = static_cast<std::size_t>(a.shape(1));
    const auto N = static_cast<std::size_t>(b.shape(1));
    Tensor out(Tensor::Shape{static_cast<std::int64_t>(M), static_cast<std::int64_t>(N)},
               DType::Float32, a.device());
    kernels::launch_matmul_f32(
        reinterpret_cast<const float*>(a.raw_ptr()),
        reinterpret_cast<const float*>(b.raw_ptr()),
        reinterpret_cast<float*>(out.raw_ptr()), M, N, K);
    return out;
#else
    (void)a; (void)b;
    throw std::runtime_error("CUDA backend not compiled in");
#endif
}

double matmul_q8_0_bench(const cpu::BlockQ8_0* Wq, const cpu::BlockQ8_0* Xq, float* Y,
                         int M, int K, int T, int reps, int variant) {
#ifdef SUB0LLM_CUDA
    const auto launch = (variant == 1) ? kernels::launch_matmul_q8_0_mma
                                       : kernels::launch_matmul_q8_0;
    const std::size_t nb = static_cast<std::size_t>(K) / cpu::QK8_0;
    const std::size_t wbytes = static_cast<std::size_t>(M) * nb * sizeof(cpu::BlockQ8_0);
    const std::size_t xbytes = static_cast<std::size_t>(T) * nb * sizeof(cpu::BlockQ8_0);
    const std::size_t ybytes = static_cast<std::size_t>(M) * static_cast<std::size_t>(T) * sizeof(float);

    cpu::BlockQ8_0 *dW = nullptr, *dX = nullptr;  float* dY = nullptr;
    auto ck = [](cudaError_t e, const char* what) {
        if (e != cudaSuccess) throw std::runtime_error(std::format("{}: {}", what, cudaGetErrorString(e)));
    };
    ck(cudaMalloc(&dW, wbytes), "cudaMalloc W");
    ck(cudaMalloc(&dX, xbytes), "cudaMalloc X");
    ck(cudaMalloc(&dY, ybytes), "cudaMalloc Y");
    ck(cudaMemcpy(dW, Wq, wbytes, cudaMemcpyHostToDevice), "H2D W");
    ck(cudaMemcpy(dX, Xq, xbytes, cudaMemcpyHostToDevice), "H2D X");

    for (int w = 0; w < 3; ++w) launch(dW, dX, dY, M, K, T);  // warm-up
    ck(cudaDeviceSynchronize(), "warmup sync");

    cudaEvent_t t0, t1;  cudaEventCreate(&t0);  cudaEventCreate(&t1);
    cudaEventRecord(t0);
    for (int r = 0; r < reps; ++r) launch(dW, dX, dY, M, K, T);
    cudaEventRecord(t1);
    ck(cudaEventSynchronize(t1), "kernel sync");
    float ms = 0.0f;  cudaEventElapsedTime(&ms, t0, t1);

    ck(cudaMemcpy(Y, dY, ybytes, cudaMemcpyDeviceToHost), "D2H Y");
    cudaEventDestroy(t0);  cudaEventDestroy(t1);
    cudaFree(dW);  cudaFree(dX);  cudaFree(dY);
    return static_cast<double>(ms) / 1000.0;
#else
    (void)Wq; (void)Xq; (void)Y; (void)M; (void)K; (void)T; (void)reps; (void)variant;
    throw std::runtime_error("CUDA backend not compiled in");
#endif
}

// ── Layer sub-kernel validation wrappers ────────────────────────────────────────────────
#ifdef SUB0LLM_CUDA
namespace {
inline void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) throw std::runtime_error(std::format("{}: {}", what, cudaGetErrorString(e)));
}
// Round-trip a host buffer to the device, run `fn(d...)`, copy the named output back.
struct DevBuf {
    void* p = nullptr;
    explicit DevBuf(std::size_t bytes) { ck(cudaMalloc(&p, bytes), "cudaMalloc"); }
    ~DevBuf() { if (p) cudaFree(p); }
    DevBuf(const DevBuf&) = delete; DevBuf& operator=(const DevBuf&) = delete;
};
} // namespace

void rmsnorm_dev(const float* x, const float* w, float* y, int n, float eps) {
    const std::size_t bytes = static_cast<std::size_t>(n) * sizeof(float);
    DevBuf dx(bytes), dy(bytes);
    DevBuf dw(w ? bytes : sizeof(float));
    ck(cudaMemcpy(dx.p, x, bytes, cudaMemcpyHostToDevice), "H2D x");
    if (w) ck(cudaMemcpy(dw.p, w, bytes, cudaMemcpyHostToDevice), "H2D w");
    kernels::launch_rmsnorm(static_cast<float*>(dx.p), w ? static_cast<float*>(dw.p) : nullptr,
                            static_cast<float*>(dy.p), n, eps);
    ck(cudaDeviceSynchronize(), "rmsnorm sync");
    ck(cudaMemcpy(y, dy.p, bytes, cudaMemcpyDeviceToHost), "D2H y");
}

void rope_neox_dev(const float* x_in, float* x_out, int dh, int pos, float base, const float* ff) {
    const std::size_t bytes  = static_cast<std::size_t>(dh) * sizeof(float);
    const std::size_t fbytes = static_cast<std::size_t>(dh / 2) * sizeof(float);
    DevBuf din(bytes), dout(bytes);
    DevBuf dff(ff ? fbytes : sizeof(float));
    ck(cudaMemcpy(din.p, x_in, bytes, cudaMemcpyHostToDevice), "H2D x_in");
    if (ff) ck(cudaMemcpy(dff.p, ff, fbytes, cudaMemcpyHostToDevice), "H2D ff");
    kernels::launch_rope_neox(static_cast<float*>(din.p), static_cast<float*>(dout.p), dh, pos,
                              base, ff ? static_cast<float*>(dff.p) : nullptr);
    ck(cudaDeviceSynchronize(), "rope sync");
    ck(cudaMemcpy(x_out, dout.p, bytes, cudaMemcpyDeviceToHost), "D2H x_out");
}

void geglu_dev(const float* gate, const float* up, float* out, int n) {
    const std::size_t bytes = static_cast<std::size_t>(n) * sizeof(float);
    DevBuf dg(bytes), du(bytes), dout(bytes);
    ck(cudaMemcpy(dg.p, gate, bytes, cudaMemcpyHostToDevice), "H2D gate");
    ck(cudaMemcpy(du.p, up,   bytes, cudaMemcpyHostToDevice), "H2D up");
    kernels::launch_geglu(static_cast<float*>(dg.p), static_cast<float*>(du.p),
                          static_cast<float*>(dout.p), n);
    ck(cudaDeviceSynchronize(), "geglu sync");
    ck(cudaMemcpy(out, dout.p, bytes, cudaMemcpyDeviceToHost), "D2H out");
}

void flash_attn_decode_dev(const float* q, const float* K, const float* V, float* o,
                           int dh, int kvlen) {
    const std::size_t qb  = static_cast<std::size_t>(dh) * sizeof(float);
    const std::size_t kvb = static_cast<std::size_t>(kvlen) * static_cast<std::size_t>(dh) * sizeof(float);
    DevBuf dq(qb), dk(kvb), dv(kvb), dout(qb);
    ck(cudaMemcpy(dq.p, q, qb,  cudaMemcpyHostToDevice), "H2D q");
    ck(cudaMemcpy(dk.p, K, kvb, cudaMemcpyHostToDevice), "H2D K");
    ck(cudaMemcpy(dv.p, V, kvb, cudaMemcpyHostToDevice), "H2D V");
    kernels::launch_flash_attn_decode(static_cast<float*>(dq.p), static_cast<float*>(dk.p),
                                      static_cast<float*>(dv.p), static_cast<float*>(dout.p),
                                      dh, kvlen);
    ck(cudaDeviceSynchronize(), "flash sync");
    ck(cudaMemcpy(o, dout.p, qb, cudaMemcpyDeviceToHost), "D2H o");
}
#else
void rmsnorm_dev(const float*, const float*, float*, int, float) {
    throw std::runtime_error("CUDA backend not compiled in");
}
void rope_neox_dev(const float*, float*, int, int, float, const float*) {
    throw std::runtime_error("CUDA backend not compiled in");
}
void geglu_dev(const float*, const float*, float*, int) {
    throw std::runtime_error("CUDA backend not compiled in");
}
void flash_attn_decode_dev(const float*, const float*, const float*, float*, int, int) {
    throw std::runtime_error("CUDA backend not compiled in");
}
#endif

} // namespace sub0llm::backend::cuda
