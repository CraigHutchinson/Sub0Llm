// sub0/measure/ceilings.hpp -- turns measured roofs into token-rate bounds (issue #78, increment 3).
//
// A decode step reads the model's weights once per token, so tokens per second cannot exceed memory bandwidth
// divided by the bytes read per token. A prefill token costs a fixed number of floating-point operations, so
// tokens per second cannot exceed the compute roof divided by that count. These functions do only that arithmetic.
//
// Units throughout: bandwidth in GB/s with GB = 1e9 bytes, compute in GFLOP/s (1e9 operations per second),
// bytes and operations per token as plain counts. Every function is pure and constexpr. A zero or negative input
// (or NaN) returns 0, which means "no ceiling known"; callers must not read 0 as a real bound.

#pragma once

namespace sub0::measure {

/// Decode ceiling in tokens per second: bandwidthGBs x 1e9 / bytesPerToken. Returns 0 when either input is not
/// positive. Example: 16.8e9 bytes per token at 33 GB/s gives 1.9643 tokens per second.
[[nodiscard]] constexpr double decodeCeilingTokPerS(double bytesPerToken, double bandwidthGBs) {
    if (!(bytesPerToken > 0.0) || !(bandwidthGBs > 0.0)) return 0.0;
    return bandwidthGBs * 1e9 / bytesPerToken;
}

/// Prefill ceiling in tokens per second: gflops x 1e9 / flopsPerToken. Returns 0 when either input is not
/// positive. Example: 54 GFLOP per token at 2000 GFLOP/s gives 37.04 tokens per second.
[[nodiscard]] constexpr double prefillCeilingTokPerS(double flopsPerToken, double gflops) {
    if (!(flopsPerToken > 0.0) || !(gflops > 0.0)) return 0.0;
    return gflops * 1e9 / flopsPerToken;
}

/// The measured value as a fraction of its ceiling (1.0 means at the roof). Returns 0 when the ceiling is not
/// positive or the measured value is negative or NaN.
[[nodiscard]] constexpr double ratioToCeiling(double measured, double ceiling) {
    if (!(ceiling > 0.0) || !(measured >= 0.0)) return 0.0;
    return measured / ceiling;
}

}  // namespace sub0::measure
