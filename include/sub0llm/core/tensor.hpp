#pragma once
#include <algorithm>
#include <cassert>
#include <cstring>
#include <format>
#include <functional>
#include <memory>
#include <numeric>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include "dtype.hpp"
#include "device.hpp"

namespace sub0llm {

// ── Storage ───────────────────────────────────────────────────────────────────
// Raw byte buffer shared across tensor views (slices, transposes, etc.).
// Device-specific allocation is handled by the backend (Ch02).  For now,
// CPU storage is a plain heap allocation.
struct Storage {
    std::shared_ptr<std::byte[]> data;
    std::size_t                  byte_capacity{0};
    Device                       device;
};

// ── Tensor ────────────────────────────────────────────────────────────────────
//
// A dynamically-shaped, dtype-tagged n-dimensional array.
//
// Design goals for Ch01:
//   • Correct — predictable strides, contiguous layout
//   • Readable — mirror NumPy/PyTorch naming so the book chapters translate
//   • C++23 — std::span data access, concepts, std::format
//
// Ch02 will add: SIMD kernels, CUDA/OpenVINO storage backends
// Ch05 will wrap this in a Node for autograd
class Tensor {
public:
    using Shape   = std::vector<std::int64_t>;
    using Strides = std::vector<std::int64_t>;

    // ── Constructors ─────────────────────────────────────────────────────────
    Tensor() = default;

    // Allocate an uninitialised tensor of the given shape.
    explicit Tensor(Shape shape,
                    DType  dtype  = DType::Float32,
                    Device device = Device::cpu());

    // Non-owning view into existing storage (used by slice/reshape).
    Tensor(Shape shape, Strides strides, std::size_t byte_offset,
           std::shared_ptr<Storage> storage, DType dtype);

    // ── Metadata ─────────────────────────────────────────────────────────────
    [[nodiscard]] const Shape&   shape()   const noexcept { return shape_;   }
    [[nodiscard]] const Strides& strides() const noexcept { return strides_; }
    [[nodiscard]] std::size_t    ndim()    const noexcept { return shape_.size(); }
    [[nodiscard]] std::int64_t   numel()   const noexcept { return numel_;   }
    [[nodiscard]] DType          dtype()   const noexcept { return dtype_;   }
    [[nodiscard]] Device         device()  const noexcept { return storage_ ? storage_->device : Device::cpu(); }
    [[nodiscard]] bool           defined() const noexcept { return storage_ != nullptr; }

    [[nodiscard]] std::int64_t shape(std::size_t dim) const {
        assert(dim < shape_.size());
        return shape_[dim];
    }

    // Is the data laid out contiguously in row-major (C) order?
    [[nodiscard]] bool is_contiguous() const noexcept;

    // ── Typed data access ─────────────────────────────────────────────────────
    // Returns a span over the element data.  Throws if T doesn't match dtype.
    template<ComputeScalar T>
    [[nodiscard]] std::span<T> data_as() {
        check_dtype<T>();
        return {reinterpret_cast<T*>(raw_ptr()), static_cast<std::size_t>(numel_)};
    }

    template<ComputeScalar T>
    [[nodiscard]] std::span<const T> data_as() const {
        check_dtype<T>();
        return {reinterpret_cast<const T*>(raw_ptr()), static_cast<std::size_t>(numel_)};
    }

    // Raw byte pointer (use sparingly).
    [[nodiscard]] std::byte*       raw_ptr()       { return storage_->data.get() + byte_offset_; }
    [[nodiscard]] const std::byte* raw_ptr() const { return storage_->data.get() + byte_offset_; }

    // ── Shape manipulation ────────────────────────────────────────────────────
    // Returns a view with a different shape (must have same numel).
    [[nodiscard]] Tensor reshape(Shape new_shape) const;

    // Returns a contiguous copy if not already contiguous, else *this.
    [[nodiscard]] Tensor contiguous() const;

    // Transpose two dimensions (returns a non-contiguous view).
    [[nodiscard]] Tensor transpose(std::size_t dim0, std::size_t dim1) const;

    // ── Device transfer ───────────────────────────────────────────────────────
    [[nodiscard]] Tensor to(Device target) const;

    // ── Scalar element access (slow — for debugging) ──────────────────────────
    template<ComputeScalar T>
    [[nodiscard]] T item() const {
        if (numel_ != 1) throw std::runtime_error("item() requires a scalar tensor");
        check_dtype<T>();
        T val;
        std::memcpy(&val, raw_ptr(), sizeof(T));
        return val;
    }

    // ── Printing ──────────────────────────────────────────────────────────────
    [[nodiscard]] std::string shape_str() const;
    [[nodiscard]] std::string to_string(std::size_t max_elements = 8) const;

    friend std::ostream& operator<<(std::ostream& os, const Tensor& t);

private:
    Shape                    shape_;
    Strides                  strides_;
    std::int64_t             numel_{0};
    DType                    dtype_{DType::Float32};
    std::shared_ptr<Storage> storage_;
    std::size_t              byte_offset_{0};

    // Compute C-contiguous (row-major) strides for a given shape.
    static Strides make_contiguous_strides(const Shape& shape, DType dtype) noexcept;

    // Compute total element count.
    static std::int64_t compute_numel(const Shape& shape) noexcept;

    template<ComputeScalar T>
    void check_dtype() const {
        if (dtype_of<T> != dtype_) {
            throw std::runtime_error(
                std::format("dtype mismatch: tensor is {} but {} was requested",
                    dtype_name(dtype_), dtype_name(dtype_of<T>)));
        }
    }
};

// ── Factory functions ─────────────────────────────────────────────────────────

[[nodiscard]] Tensor zeros(Tensor::Shape shape,
                           DType  dtype  = DType::Float32,
                           Device device = Device::cpu());

[[nodiscard]] Tensor ones(Tensor::Shape shape,
                          DType  dtype  = DType::Float32,
                          Device device = Device::cpu());

// Fills with values drawn from N(0, 1).  Uses a simple Box-Muller for now;
// Ch09 will replace with a proper PRNG infrastructure.
[[nodiscard]] Tensor randn(Tensor::Shape shape,
                           DType  dtype  = DType::Float32,
                           Device device = Device::cpu());

// [0, 1, 2, ..., n-1]
[[nodiscard]] Tensor arange(std::int64_t n,
                            DType  dtype  = DType::Float32,
                            Device device = Device::cpu());

// Deep copy.
[[nodiscard]] Tensor copy(const Tensor& src);

} // namespace sub0llm
