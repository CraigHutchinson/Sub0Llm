// tests/fp8_tests.cpp -- correctness gate for sub0/fp8.hpp (B33, docs/BACKBONE_PRECISION.md S2d).
// Engine-free (sub0_frontend_tests), same rigor B24 applied to bf16.hpp's own RNE hand-verification:
// exhaustive round-trip over the whole 256-code-point domain, both RNE tie-breaking directions, exact
// small values, subnormal exactness, and overflow/NaN saturation.

#include "sub0/fp8.hpp"

#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <limits>

using namespace sub0;

TEST_CASE("fp8: every one of the 256 code points round-trips exactly through widen/narrow", "[fp8]") {
    // fp8_widen's output is, by construction, exactly representable in f32 for every input byte
    // (widening is exact and total -- see the header's own comment). Narrowing that exact value back
    // must therefore reproduce the SAME bit pattern for every non-NaN code, and a NaN for the one
    // reserved NaN pair (0x7f/0xff) -- a comprehensive self-consistency sweep of the entire domain, not a
    // sample.
    int checked = 0, nan_codes = 0;
    for (int i = 0; i < 256; ++i) {
        const fp8 orig{static_cast<std::uint8_t>(i)};
        const float f = to_f32(orig);
        const bool is_nan_code = (i & 0x7f) == 0x7f;
        if (is_nan_code) {
            CHECK(std::isnan(f));
            ++nan_codes;
            continue;
        }
        const fp8 back = fp8_narrow(f);
        CHECK(back.bits == orig.bits);
        ++checked;
    }
    CHECK(nan_codes == 2);
    CHECK(checked == 254);
}

TEST_CASE("fp8: exact small values narrow to their hand-derived e4m3 encoding", "[fp8]") {
    // Bit layout: sign(1) exp4(4, bias 7) mant3(3). Hand-derived, not read off the implementation.
    CHECK(fp8_narrow(0.0f).bits == 0x00);
    CHECK(fp8_narrow(-0.0f).bits == 0x80);
    CHECK(fp8_narrow(1.0f).bits == 0x38);    // exp=0111(7) mant=000
    CHECK(fp8_narrow(-1.0f).bits == 0xB8);
    CHECK(fp8_narrow(2.0f).bits == 0x40);    // exp=1000(8) mant=000
    CHECK(fp8_narrow(0.5f).bits == 0x30);    // exp=0110(6) mant=000
    CHECK(fp8_narrow(448.0f).bits == 0x7E);  // exp=1111(15) mant=110 -- e4m3fn's true max finite value
    CHECK(fp8_narrow(-448.0f).bits == 0xFE);

    // And each round-trips to the exact original value (no rounding loss for a value the format
    // represents exactly).
    CHECK(to_f32(fp8_narrow(1.0f)) == 1.0f);
    CHECK(to_f32(fp8_narrow(448.0f)) == 448.0f);
}

TEST_CASE("fp8: overflow and NaN saturate to the one reserved e4m3fn NaN code point", "[fp8]") {
    // e4m3fn (this header's chosen variant) has no infinity encoding -- both an overflowing finite
    // input and a genuine +-infinity input must saturate to NaN, matching torch.float8_e4m3fn's own
    // convention for this input class.
    CHECK(std::isnan(to_f32(fp8_narrow(1e30f))));
    CHECK(std::isnan(to_f32(fp8_narrow(std::numeric_limits<float>::infinity()))));
    CHECK(std::isnan(to_f32(fp8_narrow(-std::numeric_limits<float>::infinity()))));
    CHECK(std::isnan(to_f32(fp8_narrow(std::numeric_limits<float>::quiet_NaN()))));

    // A value just past the true max (448) must round to something plausible -- either the max itself
    // (if the rounding boundary keeps it finite) or NaN (if it overflows) -- never a silently wrong
    // finite value far from either.
    const float near = to_f32(fp8_narrow(460.0f));
    CHECK((std::isnan(near) || std::fabs(near - 448.0f) < 1e-3f));
}

TEST_CASE("fp8: round-to-nearest-even ties break toward the EVEN mantissa in both directions", "[fp8]") {
    // At exp=0111 (biased 7, true exponent 0), the representable values are 1 + k/8 for k=0..7.
    // 1.0625 is exactly halfway between k=0 (even, 1.0) and k=1 (odd, 1.125) -- RNE must round DOWN to
    // the even k=0, not simply "round half up."
    CHECK(fp8_narrow(1.0625f).bits == 0x38);
    // 1.1875 is exactly halfway between k=1 (odd, 1.125) and k=2 (even, 1.25) -- RNE must round UP to
    // the even k=2 this time, proving the tie-break is genuinely parity-based, not direction-based.
    CHECK(fp8_narrow(1.1875f).bits == 0x3A);
}

TEST_CASE("fp8: subnormal range is exact at its boundary, and RNE zeroes an exact half-ULP tie", "[fp8]") {
    // The smallest subnormal is mant3=1 at exp4=0: value = 1/8 * 2^-6 = 2^-9.
    const float smallest = 1.0f / 512.0f;
    const fp8 s = fp8_narrow(smallest);
    CHECK(s.bits == 0x01);
    CHECK(to_f32(s) == smallest);
    // Exactly half of the smallest subnormal is an exact tie between 0 (even) and the smallest
    // subnormal (odd, mantissa 001) -- RNE must round to zero.
    CHECK(fp8_narrow(smallest / 2.0f).bits == 0x00);
}

TEST_CASE("fp8: Fp8CPtr exposes the same minimal read-only interface Bf16CPtr does", "[fp8]") {
    const fp8 arr[4] = {fp8_narrow(1.0f), fp8_narrow(2.0f), fp8_narrow(3.0f), fp8_narrow(4.0f)};
    const Fp8CPtr p{arr};
    CHECK(p[0] == 1.0f);
    CHECK(p[1] == 2.0f);
    CHECK(*(p + 2) == 3.0f);

    Fp8CPtr p2 = p + 2;
    CHECK(*p2 == 3.0f);
    ++p2;
    CHECK(*p2 == 4.0f);

    const Fp8CPtr p3 = p2 - 3;
    CHECK(*p3 == 1.0f);
    CHECK(static_cast<bool>(p3));
    CHECK_FALSE(p3 == nullptr);

    const Fp8CPtr null_ptr;
    CHECK_FALSE(static_cast<bool>(null_ptr));
    CHECK(null_ptr == nullptr);
}
