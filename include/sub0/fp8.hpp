// sub0/fp8.hpp -- FP8 E4M3 storage: the type, the two conversions, and the read-only pointer proxy that
// lets an existing f32 kernel read fp8 storage without knowing it did. B33 (docs/BACKBONE_PRECISION.md
// S2), the same architecture as bf16.hpp's own B24-phase-1 pair, one dtype further.
//
// WHY THIS IS A SEPARATE HEADER, and why it is deliberately config-free. Same reasoning as bf16.hpp's
// own comment: this pair is consumed by three places (the offline transplant tool's WRITE side, the
// engine's own parameter storage, and the engine-free `*_math.hpp` row kernels), the last of which must
// not acquire a dependency on a generated config header. So the primitives live here.
//
// WHAT THIS IS, stated precisely because "FP8" alone is ambiguous (several real, different 8-bit float
// layouts share that name -- gguf.hpp's own block-quantized Q8_0 is NOT this, and NEITHER is E5M2, the
// other common FP8 variant). This header implements OCP/NVIDIA **E4M3**: 1 sign bit, 4 exponent bits
// (bias 7), 3 mantissa bits -- and specifically the **"e4m3fn"** convention (the one `torch.float8_e4m3fn`
// and most weight-only FP8 quantization use): NO infinities. The exponent-all-ones, mantissa-all-ones
// code point (0x7f / 0xff) is reserved for NaN ONLY; every other exponent-all-ones code point
// (mantissa 0b000-0b110) is an ordinary finite normal value, which is what pushes the maximum
// representable magnitude up to 1.75 * 2^8 = 448 rather than stopping at 2^8 the way a "with infinity"
// variant would. An input that would round to a magnitude >= 448's next tie point (see fp8_narrow's own
// comment) saturates to NaN rather than to an infinity that this format cannot encode -- the same choice
// `torch.float8_e4m3fn` makes for its own overflow/inf inputs.
//
// UNLIKE bf16 (an f32 with its low mantissa bits dropped, same exponent range, same bias -- a pure
// truncation), e4m3 is a genuinely different exponent range and bias, so widening/narrowing are real
// exponent remappings, not a bit-shift. ~2-3 significant decimal digits, dynamic range roughly +-448 --
// far coarser than bf16's ~7-8 exponent bits and 8-bit mantissa-equivalent precision. See
// docs/BACKBONE_PRECISION.md S2d for the measured correctness-gate consequence of that.

#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <type_traits>

namespace sub0 {

// The storage element. A struct, not a bare `std::uint8_t` alias, for the same identity-swap reason
// bf16.hpp's own comment gives (memory `independent-reimplementation-catches-identity-swap-bugs`).
struct fp8 {
    std::uint8_t bits;
};
static_assert(sizeof(fp8) == 1 && alignof(fp8) == 1, "fp8 must be exactly the one byte it stores");
static_assert(std::is_trivially_copyable_v<fp8>);

// e4m3 -> f32: a real exponent remap (not a shift), exact and total -- every one of the 256 e4m3 code
// points, including the reserved NaN pair, has an exact f32 representation, so this never rounds and
// never traps. `constexpr` deliberately: it is cheap enough, and small enough (a handful of branches, no
// loop) that a compiler can fold it at any call site where the input is itself a compile-time constant,
// and there is no reason to forbid that.
[[nodiscard]] constexpr float fp8_widen(std::uint8_t b) noexcept {
    const std::uint32_t sign  = static_cast<std::uint32_t>(b & 0x80u) << 24;  // fp8 bit7 -> f32 bit31
    const std::uint32_t exp4  = (b >> 3) & 0xfu;
    const std::uint32_t mant3 = b & 0x7u;
    std::uint32_t out;
    if (exp4 == 0u) {
        if (mant3 == 0u) {
            out = sign;  // signed zero
        } else {
            // Subnormal: true value = mant3/8 * 2^-6 = mant3 * 2^-9. Normalize mant3 (a 3-bit integer,
            // 1..7) to f32's own "1.frac * 2^E" form. Let p = the position of mant3's highest set bit
            // (0, 1, or 2): mant3 = 2^p * (1 + frac) with frac = (mant3 & (2^p - 1)) / 2^p, so the value
            // is (1+frac) * 2^(p-9) -- i.e. E = p-9, and the low p bits of mant3 become the TOP p bits of
            // f32's 23-bit mantissa field once left-shifted into place. Hand-verified for all 7 inputs
            // (fp8_narrow_tests.cpp): mant3=1 -> 2^-9; mant3=4 -> 2^-7; mant3=7 -> 1.75 * 2^-7.
            const std::uint32_t p     = (mant3 >= 4u) ? 2u : (mant3 >= 2u ? 1u : 0u);
            const std::uint32_t frac  = mant3 & ((1u << p) - 1u);
            const std::uint32_t exp32 = p + 118u;                 // (p - 9) + 127
            const std::uint32_t mant23 = frac << (23u - p);
            out = sign | (exp32 << 23) | mant23;
        }
    } else if (exp4 == 0xfu && mant3 == 0x7u) {
        out = sign | 0x7fc00000u;  // the reserved e4m3fn NaN code point -> a quiet f32 NaN
    } else {
        // Normal: exponent bias differs (7 here, 127 for f32), so the exponent field is rebiased, and
        // the 3-bit mantissa becomes the TOP 3 bits of f32's 23-bit field (the low 20 bits are exact 0 --
        // e4m3 carries no information there).
        const std::uint32_t exp32  = exp4 - 7u + 127u;
        const std::uint32_t mant23 = mant3 << 20u;
        out = sign | (exp32 << 23) | mant23;
    }
    float f;
    std::memcpy(&f, &out, sizeof f);
    return f;
}
[[nodiscard]] constexpr float to_f32(fp8 b) noexcept { return fp8_widen(b.bits); }

// f32 -> e4m3, ROUND-TO-NEAREST, TIES-TO-EVEN -- the same discipline bf16.hpp's own f32_narrow applies,
// for the same reason: truncation would bias every rounded value toward zero, a systematic shrinkage
// over billions of weights rather than a wash.
//
// One uniform construction handles both the normal and subnormal output ranges: extend the f32
// mantissa with its implicit leading 1 into a 24-bit "1.mantissa" significand (`sig24`), then right-shift
// it by however many bits are needed to land on either a 3-bit NORMAL mantissa (shift 20, unconditionally
// -- the exponent is rebiased separately and checked for overflow afterward) or a denormalized
// SUBNORMAL mantissa (shift 20 + however far the true exponent sits below the smallest normal exponent),
// using the identical round-half-to-even bit trick bf16.hpp's own f32_narrow uses (`round = (1 <<
// (shift-1)) - 1 + lsb`). A rounding carry that overflows the kept-bit budget is handled explicitly in
// both branches (it promotes a subnormal into the smallest normal, or a normal's mantissa into the next
// exponent) rather than assumed away, because e4m3's ranges are narrow enough that both happen routinely
// at real weight magnitudes, unlike bf16's much wider range where mantissa-only rounding never touches
// the exponent.
// NAMED `fp8_narrow`, NOT `f32_narrow` -- bf16.hpp already defines an `f32_narrow(float) -> bf16` in
// this same namespace, and C++ cannot overload on return type alone (both take a single `float`
// argument), so reusing that name here would be a hard redefinition error the moment both headers are
// included together (param_store.hpp does exactly that). Named to match `fp8_widen`'s own symmetry
// instead of bf16.hpp's `f32_narrow`/`bf16_widen` split naming.
[[nodiscard]] inline fp8 fp8_narrow(float f) noexcept {
    std::uint32_t bits;
    std::memcpy(&bits, &f, sizeof bits);
    const std::uint32_t sign8   = (bits >> 24) & 0x80u;   // f32 bit31 -> fp8 bit7
    const std::uint32_t absbits = bits & 0x7fffffffu;

    // NaN: f32 exponent all-ones, mantissa non-zero. Quiet it into the ONE reserved e4m3fn NaN code
    // point (mantissa all-ones); never let a rounding carry escape it into something that reads as a
    // finite value.
    if (absbits > 0x7f800000u) return fp8{static_cast<std::uint8_t>(sign8 | 0x7fu)};
    // +-Infinity: e4m3fn (this header's chosen variant -- see the file comment) has no infinity
    // encoding, so infinity saturates to the same NaN code point, matching torch.float8_e4m3fn's own
    // choice for this input.
    if (absbits == 0x7f800000u) return fp8{static_cast<std::uint8_t>(sign8 | 0x7fu)};
    if (absbits == 0u) return fp8{static_cast<std::uint8_t>(sign8)};  // signed zero, exact

    const std::int32_t  exp32 = static_cast<std::int32_t>((absbits >> 23) & 0xffu) - 127;  // unbiased
    const std::uint32_t mant32 = absbits & 0x7fffffu;
    const std::uint32_t sig24  = 0x800000u | mant32;  // 1.mantissa, hidden bit explicit at bit 23

    const std::int32_t te = exp32 + 7;  // the e4m3 biased exponent this value would land at if NORMAL

    // shift: normal path always drops the low 20 of sig24's 23 explicit mantissa bits, keeping the top
    // 3 (plus the hidden bit, handled via the carry check below). Subnormal path (te <= 0) shifts the
    // WHOLE 24-bit significand further right by (1 - te), aligning it to the fixed subnormal exponent -6.
    const std::int32_t shift = (te >= 1) ? 20 : (20 + (1 - te));
    // Anything shifted this far underflows to zero even accounting for the round-up carry (the largest
    // possible numerator, sig24 + round, cannot reach 1 at this shift) -- clamp here, before computing
    // `round`, to avoid an out-of-range shift (UB) for a tiny/subnormal f32 input.
    if (shift > 27) return fp8{static_cast<std::uint8_t>(sign8)};

    const std::uint32_t lsb    = (sig24 >> shift) & 1u;
    const std::uint32_t round  = (1u << (shift - 1)) - 1u + lsb;
    const std::uint32_t rounded = (sig24 + round) >> shift;

    std::int32_t oe;      // output biased exponent
    std::uint32_t mant3;  // output 3-bit mantissa
    if (te >= 1) {
        // rounded is in [8, 16]: bit3 is the (still-implicit) hidden one; 16 means the round carried
        // out of the mantissa entirely, promoting the exponent by one and resetting the mantissa to 0
        // (the standard "1.111...->10.000" carry, one exponent step up).
        if (rounded == 16u) { oe = te + 1; mant3 = 0u; }
        else                { oe = te;     mant3 = rounded & 0x7u; }
    } else {
        // rounded is in [0, 8]: 8 means the round carried a subnormal up into the smallest NORMAL value
        // (te=1, mantissa 0) -- a subnormal has no hidden bit to begin with, so this is not "the same"
        // carry as the normal branch's, but the destination is symmetric (next exponent step, mantissa
        // reset to 0).
        if (rounded == 8u) { oe = 1; mant3 = 0u; }
        else                { oe = 0; mant3 = rounded; }
    }

    // Overflow: this value (or its rounded-up neighbor) does not fit any representable e4m3fn code
    // point. oe > 15 covers any magnitude that was already too large before rounding; oe == 15 with
    // mant3 == 7 covers the one reserved slot AT the top of the normal range (the code point that would
    // otherwise read as NaN) -- e4m3fn's true maximum finite magnitude is oe=15, mant3=6 (448), so a
    // value that rounds to the "next" mantissa at that exponent has nowhere finite left to go.
    if (oe > 15 || (oe == 15 && mant3 == 7u)) return fp8{static_cast<std::uint8_t>(sign8 | 0x7fu)};

    return fp8{static_cast<std::uint8_t>(sign8 | (static_cast<std::uint32_t>(oe) << 3) | mant3)};
}

// Round-trip through fp8 without changing the value's type -- what an "fp8 backbone" claim means
// numerically at a single element, and what the correctness gate measures against.
[[nodiscard]] inline float fp8_round(float f) noexcept { return to_f32(fp8_narrow(f)); }

// --- The read-only pointer proxy -------------------------------------------------------------------
//
// Same interface, same reasoning, as bf16.hpp's own Bf16CPtr: the minimum a `WP`-templated kernel
// actually uses (AGENTS.md S8), deliberately NOT a full random-access iterator, and deliberately NOT
// implicitly convertible to `const float*` -- that conversion would let a missed call site compile into
// a reinterpretation of fp8 bytes as floats, exactly the silent misread this design exists to prevent.
struct Fp8CPtr {
    const fp8* p = nullptr;

    Fp8CPtr() = default;
    explicit Fp8CPtr(const fp8* q) noexcept : p(q) {}

    template <class I>
    [[nodiscard]] float operator[](I i) const noexcept {
        return fp8_widen(p[i].bits);
    }
    [[nodiscard]] float operator*() const noexcept { return fp8_widen(p->bits); }

    // NOTE: `operator+` is the free template below, not a member -- see Bf16CPtr's own comment for why
    // (avoids an exact-match-template-vs-converting-non-template ambiguity at call sites).
    [[nodiscard]] Fp8CPtr operator-(std::ptrdiff_t d) const noexcept { return Fp8CPtr{p - d}; }
    Fp8CPtr& operator+=(std::ptrdiff_t d) noexcept { p += d; return *this; }
    Fp8CPtr& operator++() noexcept { ++p; return *this; }

    [[nodiscard]] bool operator==(const Fp8CPtr& o) const noexcept = default;
    [[nodiscard]] bool operator==(std::nullptr_t) const noexcept { return p == nullptr; }
    [[nodiscard]] explicit operator bool() const noexcept { return p != nullptr; }
};

template <class I, class = std::enable_if_t<std::is_integral_v<I>>>
[[nodiscard]] inline Fp8CPtr operator+(Fp8CPtr a, I d) noexcept {
    return Fp8CPtr{a.p + static_cast<std::ptrdiff_t>(d)};
}

}  // namespace sub0
