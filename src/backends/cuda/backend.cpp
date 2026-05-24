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
    cudaMemcpy(dst, src, bytes, cudaMemcpyHostToDevice);
#else
    (void)dst; (void)src; (void)bytes; (void)device_index;
    throw std::runtime_error("CUDA backend not compiled in");
#endif
}

void memcpy_d2h(void* dst, const void* src, std::size_t bytes, int device_index) {
#ifdef SUB0LLM_CUDA
    cudaSetDevice(device_index);
    cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToHost);
#else
    (void)dst; (void)src; (void)bytes; (void)device_index;
    throw std::runtime_error("CUDA backend not compiled in");
#endif
}

void memcpy_d2d(void* dst, const void* src, std::size_t bytes, int device_index) {
#ifdef SUB0LLM_CUDA
    cudaSetDevice(device_index);
    cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToDevice);
#else
    (void)dst; (void)src; (void)bytes; (void)device_index;
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

} // namespace sub0llm::backend::cuda
