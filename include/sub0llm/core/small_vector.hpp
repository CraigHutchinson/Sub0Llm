#pragma once

// small_vector.hpp — a std::vector-compatible vector with inline storage for the
// first N elements (heap only when size exceeds N).
//
// Motivation (Ch29 zero-alloc): the per-window training step does ~15k heap
// allocations, 78% of them ≤32 bytes — dominated by every Tensor's shape and stride
// std::vectors (2 tiny heap allocs per Tensor). Tensor rank is ≤4 in practice, so
// inline storage of 4 int64s removes those allocations entirely while keeping a
// drop-in vector interface. Used as Tensor::Shape / Tensor::Strides.

#include <algorithm>
#include <cstddef>
#include <initializer_list>
#include <iterator>
#include <memory>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace sub0llm {

template <class T, std::size_t N>
class SmallVector {
    static_assert(std::is_trivially_copyable_v<T>, "SmallVector assumes trivial T (int64 shapes)");
public:
    using value_type      = T;
    using size_type       = std::size_t;
    using iterator        = T*;
    using const_iterator  = const T*;
    using reference       = T&;
    using const_reference = const T&;

    SmallVector() = default;
    SmallVector(std::initializer_list<T> il) { assign(il.begin(), il.end()); }
    explicit SmallVector(size_type n) { resize(n); }
    SmallVector(size_type n, const T& v) { resize(n, v); }
    template <class It> SmallVector(It first, It last) { assign(first, last); }

    SmallVector(const SmallVector& o) { assign(o.begin(), o.end()); }
    SmallVector(SmallVector&& o) noexcept { move_from(std::move(o)); }
    SmallVector& operator=(const SmallVector& o) {
        if (this != &o) assign(o.begin(), o.end());
        return *this;
    }
    SmallVector& operator=(SmallVector&& o) noexcept {
        if (this != &o) { free_heap(); move_from(std::move(o)); }
        return *this;
    }
    SmallVector& operator=(std::initializer_list<T> il) { assign(il.begin(), il.end()); return *this; }
    ~SmallVector() { free_heap(); }

    [[nodiscard]] size_type size()     const noexcept { return size_; }
    [[nodiscard]] bool      empty()    const noexcept { return size_ == 0; }
    [[nodiscard]] size_type capacity() const noexcept { return cap_; }

    [[nodiscard]] T*       data()       noexcept { return data_; }
    [[nodiscard]] const T* data() const noexcept { return data_; }

    [[nodiscard]] reference       operator[](size_type i)       noexcept { return data_[i]; }
    [[nodiscard]] const_reference operator[](size_type i) const noexcept { return data_[i]; }
    [[nodiscard]] reference       at(size_type i)       { check(i); return data_[i]; }
    [[nodiscard]] const_reference at(size_type i) const { check(i); return data_[i]; }
    [[nodiscard]] reference       front()       noexcept { return data_[0]; }
    [[nodiscard]] const_reference front() const noexcept { return data_[0]; }
    [[nodiscard]] reference       back()        noexcept { return data_[size_ - 1]; }
    [[nodiscard]] const_reference back()  const noexcept { return data_[size_ - 1]; }

    [[nodiscard]] iterator       begin()        noexcept { return data_; }
    [[nodiscard]] const_iterator begin()  const noexcept { return data_; }
    [[nodiscard]] const_iterator cbegin() const noexcept { return data_; }
    [[nodiscard]] iterator       end()          noexcept { return data_ + size_; }
    [[nodiscard]] const_iterator end()    const noexcept { return data_ + size_; }
    [[nodiscard]] const_iterator cend()   const noexcept { return data_ + size_; }

    void clear() noexcept { size_ = 0; }

    void reserve(size_type n) { if (n > cap_) grow_to(n); }

    void resize(size_type n) {
        if (n > cap_) grow_to(n);
        if (n > size_) std::fill(data_ + size_, data_ + n, T{});
        size_ = n;
    }
    void resize(size_type n, const T& v) {
        if (n > cap_) grow_to(n);
        if (n > size_) std::fill(data_ + size_, data_ + n, v);
        size_ = n;
    }

    void push_back(const T& v) {
        if (size_ == cap_) grow_to(cap_ ? cap_ * 2 : N + 1);
        data_[size_++] = v;
    }
    void pop_back() noexcept { --size_; }

    template <class It> void assign(It first, It last) {
        const auto n = static_cast<size_type>(std::distance(first, last));
        if (n > cap_) grow_to(n);
        std::copy(first, last, data_);
        size_ = n;
    }

    [[nodiscard]] friend bool operator==(const SmallVector& a, const SmallVector& b) {
        return a.size_ == b.size_ && std::equal(a.begin(), a.end(), b.begin());
    }
    [[nodiscard]] friend bool operator!=(const SmallVector& a, const SmallVector& b) { return !(a == b); }

private:
    void check(size_type i) const { if (i >= size_) throw std::out_of_range("SmallVector::at"); }
    [[nodiscard]] bool on_heap() const noexcept { return data_ != inline_; }

    void free_heap() noexcept {
        if (on_heap()) ::operator delete(data_, std::align_val_t{alignof(T)});
    }
    void grow_to(size_type n) {
        T* nd = static_cast<T*>(::operator new(n * sizeof(T), std::align_val_t{alignof(T)}));
        std::copy(data_, data_ + size_, nd);
        free_heap();
        data_ = nd;
        cap_  = n;
    }
    void move_from(SmallVector&& o) noexcept {
        if (o.on_heap()) {
            data_ = o.data_; cap_ = o.cap_; size_ = o.size_;
        } else {
            std::copy(o.inline_, o.inline_ + o.size_, inline_);
            data_ = inline_; cap_ = N; size_ = o.size_;
        }
        o.data_ = o.inline_; o.cap_ = N; o.size_ = 0;
    }

    T          inline_[N == 0 ? 1 : N];
    T*         data_ = inline_;
    size_type  size_ = 0;
    size_type  cap_  = N;
};

} // namespace sub0llm
