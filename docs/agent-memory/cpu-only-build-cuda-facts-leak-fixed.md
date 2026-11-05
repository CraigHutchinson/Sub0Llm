---
name: cpu-only-build-cuda-facts-leak-fixed
description: FIXED (2026-07-07, commit e9f056b) -- -DSUB0_COMPUTE=CPU on a GPU-equipped machine used to bake a self-inconsistent config (HAS_CUDA=false but CUDA_ARCH nonzero) and silently default to BF16 instead of F32
metadata:
  type: project
  originSessionId: 2a6b481b-6211-474b-818a-995223d7d5ac
---

**Root cause:** `cmake/Backends.cmake` derived `HAS_CUDA` from whether the CUDA backend was actually
BUILT (`SUB0_BUILD_CUDA`, respects `SUB0_COMPUTE`), but derived `CUDA_ARCH`/`GPU_VRAM_MB`/
`GPU_SHARED_MB` from raw hardware DETECTION, unconditionally -- so an explicit `-DSUB0_COMPUTE=CPU` on
a machine with a real CUDA device (this laptop) baked `HAS_CUDA=false, CUDA_ARCH=120` into the
generated config, which is self-inconsistent and trips `layout_tests.cpp`'s own
`STATIC_REQUIRE(HAS_CUDA || CUDA_ARCH == 0)` invariant -- blocking the whole test target from building.

**Not just a test-blocker:** the configurator's `--bf16 AUTO` heuristic ("on if GPU >= sm_80") reads
this same leaked `CUDA_ARCH` value, so a CPU-only build on this kind of machine was SILENTLY defaulting
to BF16 precision instead of F32 -- the wrong precision for a build that never touches the GPU. This
had been going unnoticed; found while running a fresh tinystories CPU-only smoke test.

**Fix:** in `cmake/Backends.cmake`, the `else()` branch that sets `SUB0_BUILD_CUDA_INT=0` now also
zeroes `SUB0_CUDA_ARCH`/`SUB0_GPU_VRAM_MB`/`SUB0_GPU_SHARED_MB` before they feed
`cmake/sub0_build_facts.hpp.in`'s `configure_file`. The informational `message(STATUS "CUDA: toolkit
found...")` log line still reports the real detected hardware (unaffected, runs earlier) -- only the
BAKED CONFIG DEFAULTS are corrected.

**This had been silently worked around at least twice before, not root-caused:** see [[qk-norm]]'s note
about `muon_ab`/`gated_test` build dirs hitting the identical `STATIC_REQUIRE` failure and being
patched per-invocation with explicit `--has-cuda 0 --cuda-arch 0` flags to `sub0llm-configure`, without
investigating why the CMake-derived defaults were wrong in the first place. If any OLDER build
directory still shows this symptom, it just means it was configured before commit `e9f056b` --
reconfigure (`cmake` will auto-detect the `.cmake` file changed) and rerun `sub0llm-configure` to pick
up the fix; no need to re-apply the manual workaround going forward.
