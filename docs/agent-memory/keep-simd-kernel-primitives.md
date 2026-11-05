---
name: keep-simd-kernel-primitives
description: "Don't delete hard-won SIMD/kernel helpers when momentarily unused — mark [[maybe_unused]] and keep them"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: fb632281-974b-437a-89ff-47e56b5ae70c
---

In the sub0llm CPU backend (`backends/cpu/`), a correct, reusable vectorized
primitive (horizontal sum, fast dtype conversion, int8 dot, etc.) that isn't
currently on a hot path must be **kept**, marked `[[maybe_unused]]` with a comment
on its intended future use — NOT deleted to satisfy `-Wunused-function`.

**Why:** these building blocks cost real effort to get right (SIMD correctness is
fiddly); the user explicitly values keeping the toolbox ("we worked hard on those").
The next kernel (Q4 path, int-accumulate variant, batched matmul) will reuse them.

**How to apply:** prefer the standard `[[maybe_unused]]` attribute over a
compiler-specific pragma. Codified in CLAUDE.md "Code conventions". Example kept:
`hsum_epi32_avx` in `src/backends/cpu/quant.cpp` after the Q8 dot switched to
accumulate-as-float. Related: [[ch27-specialized-build-plan]].
