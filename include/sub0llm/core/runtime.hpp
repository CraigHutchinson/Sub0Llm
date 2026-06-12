#pragma once

// runtime.hpp — process-level CPU prerequisites for compute binaries.
//
// Call init_cpu_compute() at the top of main() in every training binary.
//
// Why this exists: Adam's second-moment EMA (×0.999 decay) drifts into denormal
// float range as gradients shrink, and x86 handles denormals in microcode at a
// ~50-100× per-op penalty. Measured on Ch28 diffusion training: throughput decayed
// 254 → 65 steps/s as the loss fell; with FTZ+DAZ the same run held ~390 steps/s
// (1140s → 256s end-to-end, identical recovery results).
//
// FTZ/DAZ flushes denormals to zero — a (vanishingly small) accuracy trade. For
// inference binaries whose outputs are parity-checked against a reference
// (llama.cpp), confirm logit-exactness is preserved before adopting it there.

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#  include <immintrin.h>
#endif

namespace sub0llm {

// Set flush-to-zero (FTZ) + denormals-are-zero (DAZ) in MXCSR for the calling
// thread. Threads created afterwards inherit the creating thread's MXCSR on
// Windows/Linux, so calling this first thing in main() covers worker threads too.
inline void init_cpu_compute() noexcept {
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
    _mm_setcsr(_mm_getcsr() | 0x8040);  // 0x8000 = FTZ, 0x0040 = DAZ
#endif
}

} // namespace sub0llm
