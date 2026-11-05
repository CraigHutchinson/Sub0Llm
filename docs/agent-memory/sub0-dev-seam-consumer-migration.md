---
name: sub0-dev-seam-consumer-migration
description: "DONE + committed (a5f09d8, 2026-07-18): decode.hpp + train_stage.cpp migrated from scattered sub0_cuda_* externs to the neutral sub0_dev_* seam (device_backend.hpp). Found + fixed a REAL pre-existing bug along the way: a -DSUB0_COMPUTE=CPU build of sub0_train failed to compile (undeclared sub0_cuda_upload_params in the hybrid_train branch) -- reproduced via git stash on the original committed code, confirmed fixed after. memplan.hpp/tools/ needed no changes."
metadata:
  type: project
  originSessionId: 2a6b481b-6211-474b-818a-995223d7d5ac
---

Follow-on to [[cuda-binding-compose-caps-flip-merged]] -- picked up the next queued item from
docs/BACKENDS.md's documented migration order (memplan.hpp -> decode.hpp -> train_stage.cpp -> tools),
skipping the riskier hybrid-router caps-flip-consumption item (that one needs real new engineering --
materializing GPU windows' token ids to build a per-step override table matching the GPU trainer's exact
flat-row layout -- not a mechanical migration, deliberately deferred as its own future task).

## What moved

`include/sub0/decode.hpp`: replaced its own `#if defined(SUB0_BUILD_CUDA)` block of 6 `extern "C"
sub0_cuda_*` declarations with one `#include "sub0/device_backend.hpp"`. `gpu_decode_try_enable()`,
`gpu_decode_shutdown()`, and `kv_decode_generate()`'s GPU branch all dropped their `#if defined
(SUB0_BUILD_CUDA)`/`#else` scaffolding -- the neutral seam's fail-fast stubs (init/kv_reset always return
-1 without a device backend) make it safe to compile the GPU branch unconditionally; `use_gpu` is only
ever true after a prior successful `gpu_decode_try_enable()`, which is false-by-construction on a no-device
build, so the branch is simply unreachable there rather than excluded at compile time.

`src/train_stage.cpp`: removed 8 redundant local `extern "C" sub0_cuda_*` declarations (device_backend.hpp
already declares them under SUB0_BUILD_CUDA); every call site in `GpuTrainer` (enable/step/backward_only/
sync_to_host/shutdown) and the `sub0_tune_stage` GPU sweep block now calls `sub0_dev_*`. Their own internal
`#if defined(SUB0_BUILD_CUDA)` guards were LEFT IN PLACE (not collapsed like decode.hpp's) -- these bodies
have genuinely divergent CPU-only fallback logic (`[[maybe_unused]]` params, real batch-fallback loops),
so removing the `#if` would be a bigger, riskier restructure than the seam migration itself warrants.

`include/sub0/memplan.hpp` and `tools/*.cpp`: needed ZERO changes. memplan.hpp only ever mentioned
`sub0_cuda_train_footprint` in prose comments, no actual declarations. Of the tools/, only
`tools/cuda_selftest.cpp` touches `sub0_cuda_*` directly -- correctly exempt by design (it exercises
per-backend DIAGNOSTIC symbols like `sub0_cuda_selftest`/`sub0_cuda_attn_check`, which docs/BACKENDS.md
explicitly excludes from the neutral seam, same reasoning as `tests/cuda_tests.cpp`).

## Real bug found (not just refactoring)

`train_stage.cpp`'s `hybrid_train` branch (unconditionally compiled -- NOT inside any `#if defined
(SUB0_BUILD_CUDA)`, since `hybrid_train` is an ordinary runtime `bool`) called `sub0_cuda_upload_params(...)`
directly after the merged optimizer step. That raw name was ONLY ever declared inside decode.hpp's (now-
removed) `#if defined(SUB0_BUILD_CUDA)` block -- so a `-DSUB0_COMPUTE=CPU` configure+build of `sub0_train`
would fail with "use of undeclared identifier 'sub0_cuda_upload_params'". Verified this was REAL, not a
misreading: `git stash`'d the migration, rebuilt `sub0_train` in a genuine CPU-only build dir (`tok_cpu`,
`cmake/Backends.cmake` confirms `SUB0_COMPUTE_RESOLVED=CPU` forces `SUB0_BUILD_CUDA OFF`) against the
ORIGINAL committed code -- reproduced the exact compile error. `git stash pop`'d, rebuilt clean.
`sub0_dev_upload_params` is always declared (a real forward under SUB0_BUILD_CUDA, a fail-fast stub
without it), so the migration fixes this as a side effect -- this branch was presumably never actually
built+tested in a genuine CPU-only configuration since the hybrid CPU/GPU commit landed
([[hybrid-cpu-gpu-execution-design]], verified only "on real RTX 5070 hardware" per that memory).

## Verification

Two from-scratch rebuilds, both green: `tok_cpu` (SUB0_COMPUTE=CPU) default suite 117 cases / 9,192,142
assertions; `gsm8k` (SUB0_COMPUTE=GPU) `[cuda]`-tagged suite 41 cases / 2,029,105 assertions (same numbers
as the binding-compose merge's own independent verification, confirming this migration didn't disturb it).

## What's still queued

- Hybrid router caps-flip consumption (`needs_cpu &= !sub0_dev_caps().supports_binding_compose`) --
  deliberately NOT done here; needs new host-side logic to materialize each GPU window's token ids and
  build a per-step override table matching `GpuTrainer::backward_only`'s exact flat-row (`m = i*T + t`)
  layout. A real design/implementation task, not a mechanical migration.
- Persistent-spike test loop conversion onto the MT `train_batch` path (mirror the committed pairspike
  pattern) -- unrelated backlog item, still open.
- Symbol flip (rename `sub0_cuda_*` exports in `backend_cuda.cu` to `sub0_dev_*` natively, delete the
  bridge forwards) -- docs/BACKENDS.md migration step 3, blocked on no production consumer naming
  `sub0_cuda_*` anymore. After this migration, only `backend_cuda.cu` itself and the two deliberately-
  exempt diagnostic files (`cuda_tests.cpp`, `cuda_selftest.cpp`) still use the raw names -- close, but the
  diagnostic files are permanent per-backend fixtures, not something the flip is meant to touch.
