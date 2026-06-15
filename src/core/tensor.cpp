#include "sub0llm/core/tensor.hpp"
#include "sub0llm/core/block_pool.hpp"
#include "../backends/cpu/kernels.hpp"
#include "../backends/cuda/backend.hpp"
#include "pool.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <numeric>
#include <ostream>
#include <random>
#include <sstream>
#include <stdexcept>

namespace sub0llm {

// Returns `data` to the pool (or frees it). Buffer lifetime is the Storage refcount.
Storage::~Storage() noexcept {
    if (!data) return;
    if (pool_idx >= -1) TensorPool::get().reclaim_raw(data, pool_idx);
    else if (free_fn)   free_fn(data);
    // pool_idx == kExternal with null free_fn ⇒ externally owned: leave it alone.
}

// ── Internal helpers ──────────────────────────────────────────────────────────

Tensor::Strides Tensor::make_contiguous_strides(const Shape& shape, DType dtype) noexcept {
    const std::size_t rank = shape.size();
    Strides s(rank);
    if (rank == 0) return s;

    // Row-major: last dimension stride = element byte size, each outer dim
    // is the product of all inner dims × element size.
    std::int64_t stride = static_cast<std::int64_t>(dtype_size(dtype));
    for (std::size_t i = rank; i-- > 0;) {
        s[i] = stride;
        stride *= std::max<std::int64_t>(shape[i], 1);
    }
    return s;
}

std::int64_t Tensor::compute_numel(const Shape& shape) noexcept {
    if (shape.empty()) return 0;
    return std::accumulate(shape.begin(), shape.end(),
                           std::int64_t{1}, std::multiplies<>{});
}

// ── Constructors ──────────────────────────────────────────────────────────────

Tensor::Tensor(Shape shape, DType dtype, Device device)
    : shape_(std::move(shape))
    , strides_(make_contiguous_strides(shape_, dtype))
    , numel_(compute_numel(shape_))
    , dtype_(dtype)
    , storage_(std::allocate_shared<Storage>(PoolAllocator<Storage>{}))
    , byte_offset_(0)
{
    storage_->device        = device;
    storage_->byte_capacity = static_cast<std::size_t>(numel_) * dtype_size(dtype);

    if (storage_->byte_capacity > 0) {
        if (device.is_cuda()) {
            // Delegate to the CUDA backend allocator so we get the right deleter.
            auto cuda_storage = backend::cuda::alloc(storage_->byte_capacity, device.index);
            storage_ = cuda_storage;
            return; // storage_ already fully initialised
        }
        const auto buf = TensorPool::get().allocate_raw(storage_->byte_capacity);
        storage_->data     = buf.ptr;
        storage_->pool_idx = buf.idx;
    }
}

Tensor::Tensor(Shape shape, Strides strides, std::size_t byte_offset,
               std::shared_ptr<Storage> storage, DType dtype)
    : shape_(std::move(shape))
    , strides_(std::move(strides))
    , numel_(compute_numel(shape_))
    , dtype_(dtype)
    , storage_(std::move(storage))
    , byte_offset_(byte_offset)
{}

// ── is_contiguous ─────────────────────────────────────────────────────────────

bool Tensor::is_contiguous() const noexcept {
    return strides_ == make_contiguous_strides(shape_, dtype_);
}

// ── reshape ───────────────────────────────────────────────────────────────────

Tensor Tensor::reshape(Shape new_shape) const {
    const std::int64_t new_numel = compute_numel(new_shape);
    if (new_numel != numel_) {
        throw std::runtime_error(
            std::format("reshape: cannot reshape tensor of {} elements to shape {}",
                numel_, shape_str()));
    }
    if (!is_contiguous()) {
        return contiguous().reshape(std::move(new_shape));
    }
    Strides new_strides = make_contiguous_strides(new_shape, dtype_);
    return Tensor(std::move(new_shape), std::move(new_strides),
                  byte_offset_, storage_, dtype_);
}

// ── contiguous ────────────────────────────────────────────────────────────────

Tensor Tensor::contiguous() const {
    if (is_contiguous()) return *this;
    return copy(*this);
}

// ── transpose ─────────────────────────────────────────────────────────────────

Tensor Tensor::transpose(std::size_t dim0, std::size_t dim1) const {
    if (dim0 >= ndim() || dim1 >= ndim()) {
        throw std::out_of_range(
            std::format("transpose: dims {}/{} out of range for {}d tensor", dim0, dim1, ndim()));
    }
    Shape   new_shape   = shape_;
    Strides new_strides = strides_;
    std::swap(new_shape[dim0],   new_shape[dim1]);
    std::swap(new_strides[dim0], new_strides[dim1]);
    return Tensor(std::move(new_shape), std::move(new_strides),
                  byte_offset_, storage_, dtype_);
}

// ── to (device transfer) ──────────────────────────────────────────────────────

Tensor Tensor::to(Device target) const {
    if (device() == target) return *this;
    const std::size_t bytes = static_cast<std::size_t>(numel_) * dtype_size(dtype_);

    if (device().is_cpu() && target.is_cuda()) {
        Tensor dst(shape_, dtype_, target);
        backend::cuda::memcpy_h2d(dst.raw_ptr(), raw_ptr(), bytes, target.index);
        return dst;
    }
    if (device().is_cuda() && target.is_cpu()) {
        Tensor dst(shape_, dtype_, target);
        backend::cuda::memcpy_d2h(dst.raw_ptr(), raw_ptr(), bytes, device().index);
        return dst;
    }
    if (device().is_cuda() && target.is_cuda()) {
        Tensor dst(shape_, dtype_, target);
        backend::cuda::memcpy_d2d(dst.raw_ptr(), raw_ptr(), bytes, device().index);
        return dst;
    }
    if (target.is_cpu()) return copy(*this);

    throw std::runtime_error(
        std::format("Tensor::to({}→{}): unsupported transfer", device().str(), target.str()));
}

// ── Printing ──────────────────────────────────────────────────────────────────

std::string Tensor::shape_str() const {
    if (shape_.empty()) return "()";
    std::string s = "(";
    for (std::size_t i = 0; i < shape_.size(); ++i) {
        s += std::to_string(shape_[i]);
        if (i + 1 < shape_.size()) s += ", ";
    }
    s += ')';
    return s;
}

std::string Tensor::to_string(std::size_t max_elements) const {
    if (!defined()) return "Tensor(undefined)";

    std::ostringstream oss;
    oss << "Tensor(shape=" << shape_str()
        << ", dtype=" << dtype_name(dtype_)
        << ", device=" << device().str()
        << ", data=[";

    if (dtype_ == DType::Float32 && numel_ > 0) {
        auto span = data_as<float>();
        const std::size_t n = std::min(static_cast<std::size_t>(numel_), max_elements);
        for (std::size_t i = 0; i < n; ++i) {
            oss << span[i];
            if (i + 1 < n) oss << ", ";
        }
        if (static_cast<std::size_t>(numel_) > max_elements) oss << ", ...";
    } else {
        oss << "<" << dtype_name(dtype_) << ">";
    }

    oss << "])";
    return oss.str();
}

std::ostream& operator<<(std::ostream& os, const Tensor& t) {
    return os << t.to_string();
}

// ── Factory functions ─────────────────────────────────────────────────────────

Tensor zeros(Tensor::Shape shape, DType dtype, Device device) {
    Tensor t(std::move(shape), dtype, device);
    if (t.numel() > 0) {
        const std::size_t bytes = static_cast<std::size_t>(t.numel()) * dtype_size(dtype);
        if (t.device().is_cuda()) {
            backend::cuda::memset_zero(t.raw_ptr(), bytes, t.device().index);
        } else {
            std::memset(t.raw_ptr(), 0, bytes);
        }
    }
    return t;
}

Tensor ones(Tensor::Shape shape, DType dtype, Device device) {
    if (dtype == DType::Float16 || dtype == DType::BFloat16 || dtype == DType::Bool) {
        throw std::runtime_error(
            std::format("ones: dtype {} not yet supported", dtype_name(dtype)));
    }
    Tensor t = zeros(shape, dtype, device);
    if (dtype == DType::Float32) {
        auto sp = t.data_as<float>();
        std::fill(sp.begin(), sp.end(), 1.0f);
    } else if (dtype == DType::Float64) {
        auto sp = t.data_as<double>();
        std::fill(sp.begin(), sp.end(), 1.0);
    } else if (dtype == DType::Int8) {
        auto sp = t.data_as<std::int8_t>();
        std::fill(sp.begin(), sp.end(), std::int8_t{1});
    } else if (dtype == DType::Int16) {
        auto sp = t.data_as<std::int16_t>();
        std::fill(sp.begin(), sp.end(), std::int16_t{1});
    } else if (dtype == DType::Int32) {
        auto sp = t.data_as<std::int32_t>();
        std::fill(sp.begin(), sp.end(), std::int32_t{1});
    } else if (dtype == DType::Int64) {
        auto sp = t.data_as<std::int64_t>();
        std::fill(sp.begin(), sp.end(), std::int64_t{1});
    }
    return t;
}

Tensor randn(Tensor::Shape shape, DType dtype, Device device) {
    if (dtype != DType::Float32 && dtype != DType::Float64) {
        throw std::runtime_error("randn: only float32 and float64 are supported");
    }
    Tensor t(std::move(shape), dtype, device);
    std::mt19937_64 rng{std::random_device{}()};
    if (dtype == DType::Float32) {
        std::normal_distribution<float> dist{0.0f, 1.0f};
        for (auto& v : t.data_as<float>()) v = dist(rng);
    } else {
        std::normal_distribution<double> dist{0.0, 1.0};
        for (auto& v : t.data_as<double>()) v = dist(rng);
    }
    return t;
}

Tensor arange(std::int64_t n, DType dtype, Device device) {
    Tensor t({n}, dtype, device);
    if (dtype == DType::Float32) {
        auto sp = t.data_as<float>();
        for (std::int64_t i = 0; i < n; ++i) sp[static_cast<std::size_t>(i)] = static_cast<float>(i);
    } else if (dtype == DType::Int64) {
        auto sp = t.data_as<std::int64_t>();
        for (std::int64_t i = 0; i < n; ++i) sp[static_cast<std::size_t>(i)] = i;
    } else if (dtype == DType::Int32) {
        auto sp = t.data_as<std::int32_t>();
        for (std::int64_t i = 0; i < n; ++i) sp[static_cast<std::size_t>(i)] = static_cast<std::int32_t>(i);
    } else {
        throw std::runtime_error(
            std::format("arange: unsupported dtype {}", dtype_name(dtype)));
    }
    return t;
}

Tensor copy(const Tensor& src) {
    Tensor dst(src.shape(), src.dtype(), src.device());
    if (src.numel() <= 0) return dst;

    const std::size_t bytes = static_cast<std::size_t>(src.numel()) * dtype_size(src.dtype());

    if (src.is_contiguous()) {
        std::memcpy(dst.raw_ptr(), src.raw_ptr(), bytes);
        return dst;
    }

    // Stride-aware copy for non-contiguous float32 tensors.
    if (src.dtype() != DType::Float32) {
        throw std::runtime_error(std::format(
            "copy: non-contiguous copy of {} not yet implemented", dtype_name(src.dtype())));
    }

    // Fast path for 2D (covers transpose, the dominant non-contiguous case).
    // The blocked SIMD-friendly kernel is ~50× faster than the generic loop below
    // for the A.T.contiguous() pattern used by autograd matmul backward.
    if (src.ndim() == 2) {
        const std::size_t rows = static_cast<std::size_t>(src.shape()[0]);
        const std::size_t cols = static_cast<std::size_t>(src.shape()[1]);
        const float* sp = reinterpret_cast<const float*>(src.raw_ptr());
        auto dp = dst.data_as<float>();
        const std::size_t rs = static_cast<std::size_t>(src.strides()[0]) / sizeof(float);
        const std::size_t cs = static_cast<std::size_t>(src.strides()[1]) / sizeof(float);
        backend::cpu::copy_strided_2d_f32(sp, rs, cs, dp.data(), rows, cols);
        return dst;
    }

    // Generic N-D fallback: walk every element using its strided byte offset.
    const auto& shape   = src.shape();
    const auto& strides = src.strides();
    const std::size_t rank = src.ndim();
    const auto n = static_cast<std::size_t>(src.numel());

    auto dst_sp = dst.data_as<float>();
    std::vector<std::size_t> idx(rank, 0);

    for (std::size_t flat = 0; flat < n; ++flat) {
        std::size_t byte_off = 0;
        for (std::size_t d = 0; d < rank; ++d)
            byte_off += idx[d] * static_cast<std::size_t>(strides[d]);
        dst_sp[flat] = *reinterpret_cast<const float*>(src.raw_ptr() + byte_off);

        for (std::size_t d = rank; d-- > 0;) {
            if (++idx[d] < static_cast<std::size_t>(shape[d])) break;
            idx[d] = 0;
        }
    }

    return dst;
}

} // namespace sub0llm
