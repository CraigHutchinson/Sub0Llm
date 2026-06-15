#pragma once

// small_function.hpp — a std::function-like type-erased callable with an INLINE buffer
// (no heap allocation for closures that fit) and a heap fallback for larger ones.
//
// Motivation (Ch29 zero-alloc): every autograd op stores a VJP closure per backward edge.
// Captureless closures (add/sub) fit std::function's small-buffer optimisation, but any
// closure capturing a Tensor (~152 B: mul/matmul/...) overflows it and heap-allocates — the
// dominant ≤256/512 B chunk of the per-step alloc firehose. These closures capture one (or a
// few) Tensors, which fit a ~256-byte inline buffer; only the rare larger closure falls back
// to the heap. Copyable + movable (Edge is value-stored in std::vector).

#include <cstddef>
#include <new>
#include <type_traits>
#include <utility>

namespace sub0llm {

template <class Sig, std::size_t Buf = 256>
class SmallFunction;

template <class R, class... A, std::size_t Buf>
class SmallFunction<R(A...), Buf> {
    // Type-erased operations on the stored callable (one static table per concrete type).
    struct VTable {
        R    (*invoke)(const void*, A&&...);
        void (*copy)(void*, const void*);     // placement-copy-construct src→dst
        void (*move)(void*, void*) noexcept;   // placement-move-construct src→dst
        void (*destroy)(void*) noexcept;
        std::size_t size;
        std::size_t align;
    };
    template <class F>
    static const VTable& vtable_for() {
        static const VTable vt{
            [](const void* p, A&&... a) -> R {
                return (*static_cast<const F*>(p))(std::forward<A>(a)...);
            },
            [](void* d, const void* s) { ::new (d) F(*static_cast<const F*>(s)); },
            [](void* d, void* s) noexcept { ::new (d) F(std::move(*static_cast<F*>(s))); },
            [](void* p) noexcept { static_cast<F*>(p)->~F(); },
            sizeof(F), alignof(F),
        };
        return vt;
    }

public:
    SmallFunction() noexcept = default;

    template <class F, class DF = std::decay_t<F>,
              class = std::enable_if_t<!std::is_same_v<DF, SmallFunction> &&
                                       std::is_invocable_r_v<R, const DF&, A...>>>
    SmallFunction(F&& f) {
        vt_ = &vtable_for<DF>();
        ptr_ = storage_for(sizeof(DF), alignof(DF));
        ::new (ptr_) DF(std::forward<F>(f));
    }

    SmallFunction(const SmallFunction& o) { copy_from(o); }
    SmallFunction(SmallFunction&& o) noexcept { move_from(o); }
    SmallFunction& operator=(const SmallFunction& o) {
        if (this != &o) { reset(); copy_from(o); }
        return *this;
    }
    SmallFunction& operator=(SmallFunction&& o) noexcept {
        if (this != &o) { reset(); move_from(o); }
        return *this;
    }
    ~SmallFunction() { reset(); }

    [[nodiscard]] explicit operator bool() const noexcept { return vt_ != nullptr; }

    R operator()(A... a) const { return vt_->invoke(ptr_, std::forward<A>(a)...); }

private:
    // Returns inline buf_ if the callable fits, else a fresh aligned heap block.
    void* storage_for(std::size_t sz, std::size_t al) {
        if (sz <= Buf && al <= alignof(std::max_align_t)) { on_heap_ = false; return &buf_; }
        on_heap_ = true;
        return ::operator new(sz, std::align_val_t{al});
    }
    void reset() noexcept {
        if (!vt_) return;
        vt_->destroy(ptr_);
        if (on_heap_) ::operator delete(ptr_, std::align_val_t{vt_->align});
        vt_ = nullptr; ptr_ = nullptr; on_heap_ = false;
    }
    void copy_from(const SmallFunction& o) {
        if (!o.vt_) return;
        vt_  = o.vt_;
        ptr_ = storage_for(vt_->size, vt_->align);
        vt_->copy(ptr_, o.ptr_);
    }
    void move_from(SmallFunction& o) noexcept {
        if (!o.vt_) return;
        vt_ = o.vt_;
        if (o.on_heap_) {                       // steal the heap block
            ptr_ = o.ptr_; on_heap_ = true;
        } else {                                // move into our own inline buffer
            ptr_ = &buf_; on_heap_ = false;
            vt_->move(ptr_, o.ptr_);
            vt_->destroy(o.ptr_);
        }
        o.vt_ = nullptr; o.ptr_ = nullptr; o.on_heap_ = false;
    }

    alignas(std::max_align_t) std::byte buf_[Buf];
    const VTable* vt_      = nullptr;
    void*         ptr_     = nullptr;
    bool          on_heap_ = false;
};

} // namespace sub0llm
