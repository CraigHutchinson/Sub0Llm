---
name: mock-device-backend-test-seam
description: "PATTERN — device_backend.hpp now has a SUB0_BUILD_MOCK_DEVICE branch backed by the CPU engine, so seam CONSUMERS are testable on any machine; use it for the next device consumer, not just eval"
metadata: 
  node_type: memory
  type: project
  originSessionId: 2a6b481b-6211-474b-818a-995223d7d5ac
  modified: 2026-07-29T17:24:59.433Z
---

Built 2026-07-29 alongside the GPU eval seam ([[gpu-eval-seam-needed-report-is-cpu-bound]]).

## The gap it closes

The device seam had exactly one implementation and it needed a GPU. So the CONSUMER-side logic around
it -- how work is batched, how inputs are paired, how partial groups are recombined -- had no test at
all: a CPU-only machine skipped the area entirely, and a GPU machine could only test it end-to-end
against real kernels, where a plumbing bug and a kernel bug are indistinguishable.

## The mechanism

`include/sub0/device_backend.hpp` gained a `SUB0_BUILD_MOCK_DEVICE` branch checked **before**
`SUB0_BUILD_CUDA`, so a CUDA-enabled tree still builds the mock target against the mock. The header
already anticipated this ("a future SUB0_DEVICE=ROCM branch defines SUB0_BUILD_DEVICE via its own
path") -- a mock is just a third backend, so no new mechanism was invented.

`tests/mock_device_backend.cpp` answers the seam by running the **real CPU engine** over the arguments
it was handed. It therefore computes the right answer if and only if it was handed the right question.
Its caps are honest (`eval` only, everything else 0), so a consumer that branches correctly on caps
exercises exactly the mocked path.

**It must be its own executable.** `SUB0_BUILD_MOCK_DEVICE` reroutes the inline `sub0_dev_*` functions,
and `sub0_tests` contains `cuda_tests.cpp` which needs the real ones -- two definitions of the same
inline functions in one binary is an ODR violation. Same reason it links `sub0_core` only, never
`sub0_train`.

## What it actually caught / guards

Silent-wrong-number bugs, none of which need a GPU to occur: a target shifted by one token, a window
start computed differently than the CPU path computes it, a ragged final group averaged as a plain
mean-of-means instead of weighted by count, a group reading past the rows it filled.

One test deliberately **guards the guard**: it scores deliberately mispaired targets and requires the
comparison to FAIL, so the parity assertion cannot be satisfied by any two numbers of roughly the right
magnitude.

## Reuse it

This is the pattern for the next device consumer, not a one-off for eval. When wiring anything new
through the seam, extend the mock's caps and add a consumer test there FIRST -- it runs in
milliseconds on any machine, and the real-hardware test then only has to prove the kernels agree.
Neither subsumes the other: the mock says nothing about kernel correctness, and a GPU-only test cannot
isolate plumbing from math.
