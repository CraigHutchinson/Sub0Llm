#pragma once

// workspace.hpp — preallocated, reusable scratch for the zero-alloc training path.
//
// The Ch29 measurements proved the per-window forward+backward is ~85% compute, and
// that the multi-core ceiling is the global heap-allocator lock convoy (the autograd
// graph allocates hundreds of Node/closure/Tensor objects per step). For a training
// loop the shapes are FIXED (arch × batch × seq_len), so every step's buffers are
// known ahead of time and can be allocated ONCE and reused — the std::span/constexpr
// direction. Workspace is that backing store: a single growable allocation with a
// bump pointer that rewinds each step, so a warmed-up step does ZERO heap allocation.
//
// It hands out std::span<float> views (and 2-D mdspan-style accessors); it never owns
// Tensors or shared_ptrs. A StaticGraph (built on top) records the denoiser's op
// sequence once and replays forward/backward over Workspace-owned buffers.
//
// Thread model: ONE Workspace per worker thread (the per-window data-parallel replica
// or the single batched graph thread). Never shared concurrently — no locks.

#include <algorithm>
#include <cstddef>
#include <span>
#include <vector>

namespace sub0llm {

class Workspace {
public:
    Workspace() = default;
    explicit Workspace(std::size_t capacity_floats) { arena_.resize(capacity_floats); }

    // Ensure the backing arena holds at least n floats. Call BETWEEN steps only (never
    // with live take() spans outstanding) — a StaticGraph reserves the exact per-step
    // total once at build time so take() never reallocates mid-step.
    void reserve(std::size_t n) { if (n > arena_.size()) arena_.resize(n); }

    // Bump-allocate n floats, 16-float aligned for SIMD. The span stays valid until
    // the next reset() — PROVIDED the arena was reserve()d large enough that take()
    // doesn't have to grow mid-step (growing reallocates and invalidates prior spans;
    // that only happens on the first warmup step before high_water() stabilizes).
    [[nodiscard]] std::span<float> take(std::size_t n) {
        const std::size_t off = (used_ + 15) & ~std::size_t{15};
        const std::size_t end = off + n;
        if (end > arena_.size()) arena_.resize(end);   // warmup growth only
        used_ = end;
        return std::span<float>(arena_.data() + off, n);
    }

    // Zeroed bump-allocation (for gradient accumulators that start at 0 each step).
    [[nodiscard]] std::span<float> take_zeroed(std::size_t n) {
        auto s = take(n);
        std::fill(s.begin(), s.end(), 0.0f);
        return s;
    }

    // Rewind to empty — call at the top of each step. Keeps the backing allocation,
    // so the next step reuses it with zero heap activity.
    void reset() noexcept { used_ = 0; }

    [[nodiscard]] std::size_t high_water() const noexcept { return arena_.size(); }
    [[nodiscard]] std::size_t used() const noexcept { return used_; }

private:
    std::vector<float> arena_;
    std::size_t        used_ = 0;
};

// Minimal 2-D row-major view over a span — used where std::mdspan isn't available or
// where an explicit (rows, cols) accessor reads more clearly in the kernels.
struct MatView {
    float*      data = nullptr;
    std::size_t rows = 0, cols = 0;
    [[nodiscard]] float&       at(std::size_t r, std::size_t c)       noexcept { return data[r * cols + c]; }
    [[nodiscard]] float        at(std::size_t r, std::size_t c) const noexcept { return data[r * cols + c]; }
    [[nodiscard]] std::span<float> row(std::size_t r) const noexcept { return {data + r * cols, cols}; }
    [[nodiscard]] std::size_t   size() const noexcept { return rows * cols; }
};

[[nodiscard]] inline MatView mat(std::span<float> s, std::size_t rows, std::size_t cols) noexcept {
    return MatView{s.data(), rows, cols};
}

} // namespace sub0llm
