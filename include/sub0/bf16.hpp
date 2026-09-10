// sub0/bf16.hpp -- bfloat16 storage: the type, the two conversions, and the read-only pointer proxy
// that lets an existing f32 kernel read bf16 storage without knowing it did.
//
// WHY A SEPARATE HEADER, and why it is deliberately config-free. gguf.hpp already had the READ half
// (`bf16_to_f32`) because the source model ships some tensors natively as bf16. B24 phase 1
// (docs/BACKBONE_PRECISION.md S1) needs the WRITE half in the offline transplant tool, and the SAME
// pair again in the engine's own parameter storage -- three consumers, one of which (the `*_math.hpp`
// row kernels) is deliberately engine-free and must not acquire a dependency on a generated config
// header. So the primitives live here, in the widest-reaching, lightest header of the three, and
// gguf.hpp delegates to them rather than keeping a second definition of the same shift (AGENTS.md S3's
// "two definitions of a fixed-size binary format is the wrong shape", applied to a numeric format).
//
// WHAT BF16 IS, stated precisely because it is routinely confused with f16 (gguf.hpp's own comment on
// `bf16_to_f32` makes the same point): bfloat16 is an IEEE binary32 with its low 16 mantissa bits
// removed. Same 8 exponent bits, same bias, same inf/NaN encodings, same subnormal boundary -- only 7
// explicit mantissa bits instead of 23. Widening is therefore a shift with no rebias and no
// renormalisation, and narrowing is a rounding of the mantissa alone. That shared exponent range is
// exactly why this is the right format for a backbone whose weights already trained at f32 magnitudes:
// nothing needs re-scaling or per-block calibration the way an integer format would.

#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <type_traits>

namespace sub0 {

// The storage element. A struct rather than a bare `std::uint16_t` alias so a bf16 array can never be
// silently passed where a raw-bits array is expected (or vice versa) -- the exact class of identity-swap
// bug memory `independent-reimplementation-catches-identity-swap-bugs` is about. Trivially copyable and
// exactly 2 bytes, so an array of these IS the on-disk/in-RAM byte layout with no packing directives.
struct bf16 {
    std::uint16_t bits;
};
static_assert(sizeof(bf16) == 2 && alignof(bf16) == 2, "bf16 must be exactly the two bytes it stores");
static_assert(std::is_trivially_copyable_v<bf16>);

// bfloat16 -> f32: widen by 16 bits. Exact and total -- every bf16 value, including every inf/NaN
// encoding, has an exact f32 representation, so this never rounds and never traps.
[[nodiscard]] inline float bf16_widen(std::uint16_t h) noexcept {
    const std::uint32_t bits = static_cast<std::uint32_t>(h) << 16;
    float f;
    std::memcpy(&f, &bits, sizeof f);
    return f;
}
[[nodiscard]] inline float to_f32(bf16 b) noexcept { return bf16_widen(b.bits); }

// f32 -> bfloat16, ROUND-TO-NEAREST, TIES-TO-EVEN -- the same rounding mode f32 arithmetic itself uses
// by default, and the one every mainstream bf16 conversion (PyTorch's `.bfloat16()`, oneDNN, ggml's own
// `ggml_fp32_to_bf16`) implements. Written out rather than recalled, because the naive alternative --
// truncating the low 16 bits -- is a real and different function: truncation biases every rounded value
// toward zero, which over 4.9e9 backbone weights is a systematic shrinkage of the whole model, not a
// wash. RNE has zero mean error instead.
//
// The mechanism, term by term:
//   `lsb`    = bit 16 of the input, i.e. the low bit of the bf16 that will SURVIVE. Adding it to the
//              rounding constant is what makes ties (exactly 0x8000 in the discarded bits) round to the
//              EVEN survivor rather than always up.
//   `0x7fff` = the largest discarded value that still rounds down. `bits + 0x7fff + lsb` therefore
//              carries into bit 16 exactly when the discarded part is above half, or is exactly half
//              with an odd survivor.
//   NaN      = handled before any of that: adding the rounding constant to a NaN's mantissa can carry
//              all the way out and turn it into an infinity, which would silently convert "this weight
//              is broken" into "this weight is enormous". A NaN in must stay a NaN out. (Infinities and
//              zeros need no special case -- their discarded mantissa bits are all zero, so the carry
//              term is inert and the result is exact.)
[[nodiscard]] inline bf16 f32_narrow(float f) noexcept {
    std::uint32_t bits;
    std::memcpy(&bits, &f, sizeof bits);
    // NaN: exponent all-ones with a non-zero mantissa. Quiet it and keep the sign; never let the
    // rounding carry below promote it to +/-inf.
    if ((bits & 0x7f800000u) == 0x7f800000u && (bits & 0x007fffffu) != 0u)
        return bf16{static_cast<std::uint16_t>((bits >> 16) | 0x0040u)};
    const std::uint32_t lsb   = (bits >> 16) & 1u;
    const std::uint32_t round = 0x7fffu + lsb;
    return bf16{static_cast<std::uint16_t>((bits + round) >> 16)};
}

// Round-trip through bf16 without changing the value's type -- what a "this build stores weights at
// bf16" claim means numerically at a single element, and what the correctness gate measures against.
[[nodiscard]] inline float bf16_round(float f) noexcept { return to_f32(f32_narrow(f)); }

// --- The read-only pointer proxy -------------------------------------------------------------------
//
// THE POINT OF THIS TYPE. Every parameter-reading kernel in this engine has the same two shapes:
// `const float* Wr = w + i * stride;` to pick a row, and `Wr[j]` to read an element. Bf16CPtr supports
// exactly those two operations and returns a `float` from the second, so a kernel templated on its
// weight-pointer type compiles unchanged against either `const float*` or this -- no `#ifdef`, no
// second copy of the loop body, and (crucially) no change at all to the F32 build, where the template
// parameter IS `const float*` and every one of these lines is the code that was there before.
//
// Deliberately NOT a full random-access iterator: it exposes the minimum the kernels actually use
// (AGENTS.md S8). In particular there is no `operator->`, no mutable access, and no implicit conversion
// to `const float*` -- the last of those would defeat the whole point by letting a missed call site
// compile into a reinterpretation of bf16 bytes as floats, which is precisely the silent misread this
// design exists to make impossible.
struct Bf16CPtr {
    const bf16* p = nullptr;

    Bf16CPtr() = default;
    explicit Bf16CPtr(const bf16* q) noexcept : p(q) {}

    template <class I>
    [[nodiscard]] float operator[](I i) const noexcept {
        return bf16_widen(p[i].bits);
    }
    [[nodiscard]] float operator*() const noexcept { return bf16_widen(p->bits); }

    // NOTE: `operator+` is the free template below, not a member -- a member taking `std::ptrdiff_t`
    // plus the template would make `ptr + some_size_t` pick between an exact-match template and a
    // converting non-template on every call site, and this type exists precisely to be invisible.
    [[nodiscard]] Bf16CPtr operator-(std::ptrdiff_t d) const noexcept { return Bf16CPtr{p - d}; }
    Bf16CPtr& operator+=(std::ptrdiff_t d) noexcept { p += d; return *this; }
    Bf16CPtr& operator++() noexcept { ++p; return *this; }

    [[nodiscard]] bool operator==(const Bf16CPtr& o) const noexcept = default;
    [[nodiscard]] bool operator==(std::nullptr_t) const noexcept { return p == nullptr; }
    [[nodiscard]] explicit operator bool() const noexcept { return p != nullptr; }
};

// `w + i` where i is a size_t/int: the free operator keeps call sites from having to cast an index that
// is naturally unsigned in half the kernels and `int` in the other half.
template <class I, class = std::enable_if_t<std::is_integral_v<I>>>
[[nodiscard]] inline Bf16CPtr operator+(Bf16CPtr a, I d) noexcept {
    return Bf16CPtr{a.p + static_cast<std::ptrdiff_t>(d)};
}

}  // namespace sub0
