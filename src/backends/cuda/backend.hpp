#pragma once
// CUDA backend interface — included by ops.cpp regardless of build config
// so the dispatch code compiles cleanly; the function bodies are no-ops /
// throw when CUDA is not compiled in.
#include "sub0llm/core/tensor.hpp"

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

} // namespace sub0llm::backend::cuda
