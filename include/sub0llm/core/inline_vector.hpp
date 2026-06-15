#pragma once

// inline_vector.hpp — a small vector with inline storage for the first N elements that works
// for NON-TRIVIAL T (proper placement-new / destroy / move), unlike SmallVector which is
// memcpy-based and trivial-T only (Tensor shapes). Used for autograd Node::edges: an Edge
// holds a shared_ptr + an inline VJP closure, so storing up to N edges inline keeps a node's
// whole backward graph off the heap (the node itself is pooled). >N inputs spill to the heap.

#include <cstddef>
#include <new>
#include <utility>

namespace sub0llm {

template <class T, std::size_t N>
class InlineVector {
public:
    InlineVector() noexcept = default;
    InlineVector(const InlineVector& o) { for (std::size_t i = 0; i < o.size_; ++i) push_back(o[i]); }
    InlineVector(InlineVector&& o) noexcept { move_in(o); }
    InlineVector& operator=(const InlineVector& o) {
        if (this != &o) { clear_destroy(); for (std::size_t i = 0; i < o.size_; ++i) push_back(o[i]); }
        return *this;
    }
    InlineVector& operator=(InlineVector&& o) noexcept {
        if (this != &o) { destroy_free(); move_in(o); }
        return *this;
    }
    ~InlineVector() { destroy_free(); }

    void push_back(T v) {
        if (size_ == cap_) grow(cap_ ? cap_ * 2 : N + 1);
        ::new (data_ + size_) T(std::move(v));
        ++size_;
    }
    void pop_back() noexcept { data_[--size_].~T(); }
    void clear() noexcept { clear_destroy(); }

    [[nodiscard]] std::size_t size()  const noexcept { return size_; }
    [[nodiscard]] bool        empty() const noexcept { return size_ == 0; }

    [[nodiscard]] T&       operator[](std::size_t i)       noexcept { return data_[i]; }
    [[nodiscard]] const T& operator[](std::size_t i) const noexcept { return data_[i]; }

    [[nodiscard]] T*       begin()       noexcept { return data_; }
    [[nodiscard]] const T* begin() const noexcept { return data_; }
    [[nodiscard]] T*       end()         noexcept { return data_ + size_; }
    [[nodiscard]] const T* end()   const noexcept { return data_ + size_; }

private:
    [[nodiscard]] bool on_heap() const noexcept {
        return reinterpret_cast<const std::byte*>(data_) != inline_;
    }
    void clear_destroy() noexcept {
        for (std::size_t i = 0; i < size_; ++i) data_[i].~T();
        size_ = 0;
    }
    void destroy_free() noexcept {
        clear_destroy();
        if (on_heap()) ::operator delete(data_, std::align_val_t{alignof(T)});
    }
    void grow(std::size_t n) {
        T* nd = static_cast<T*>(::operator new(n * sizeof(T), std::align_val_t{alignof(T)}));
        for (std::size_t i = 0; i < size_; ++i) {
            ::new (nd + i) T(std::move(data_[i]));
            data_[i].~T();
        }
        if (on_heap()) ::operator delete(data_, std::align_val_t{alignof(T)});
        data_ = nd;
        cap_  = n;
    }
    void move_in(InlineVector& o) noexcept {
        if (o.on_heap()) {
            data_ = o.data_; size_ = o.size_; cap_ = o.cap_;
            o.data_ = reinterpret_cast<T*>(o.inline_); o.size_ = 0; o.cap_ = N;
        } else {
            data_ = reinterpret_cast<T*>(inline_); cap_ = N; size_ = 0;
            for (std::size_t i = 0; i < o.size_; ++i) { ::new (data_ + i) T(std::move(o[i])); ++size_; }
            o.clear_destroy();
        }
    }

    alignas(T) std::byte inline_[(N == 0 ? 1 : N) * sizeof(T)];
    T*          data_ = reinterpret_cast<T*>(inline_);
    std::size_t size_ = 0;
    std::size_t cap_  = N;
};

} // namespace sub0llm
