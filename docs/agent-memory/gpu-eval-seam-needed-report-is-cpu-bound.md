---
name: gpu-eval-seam-needed-report-is-cpu-bound
description: "DONE 2026-07-29 — sub0_dev_forward_loss ships on CUDA, parity-gated on real hardware; report scores on the device. Also introduced the MOCK device backend that makes seam consumers testable without a GPU"
metadata: 
  node_type: memory
  type: project
  originSessionId: 2a6b481b-6211-474b-818a-995223d7d5ac
  modified: 2026-07-29T17:24:40.656Z
---

Scoped 2026-07-28, built 2026-07-29. `report`/`evaluate` used to run entirely on `sub0::forward` (CPU)
because the device seam had no batched forward-only entry.

## What shipped

**`sub0_dev_forward_loss(ids, targets, batch, T, out_loss, lengths)`** -- neutral seam entry, CUDA
implementation, no-device stub, plus a `supports_eval` caps bit (strictly weaker than `supports_train`:
a backend can evaluate without implementing a backward).

Two design calls worth remembering:

1. **Built on the INFERENCE forward (`forward_device`, graph-captured), not `forward_train`.** It is
   FP32 end to end -- matching the CPU path it is compared against, where `forward_train` stores its
   residual stream in bf16 -- and it is already gated against the CPU forward by
   `sub0_cuda_forward_check`, so eval inherits an existing parity claim instead of asserting a new one.
   It also allocates no per-execution activation buffers. The cost: it materializes the full
   `[batch*T, VOCAB]` logits buffer, so the CALLER picks a batch from a budget
   (`sub0::eval::device_batch`, 512 MiB).
2. **`ce_backward_kernel` gained a loss-only mode (`dlogits == nullptr`)** rather than a sibling
   kernel. The masking rules, the active-count denominator and the `1/(batch*denom)` weight then exist
   exactly once, so a forward-only eval structurally CANNOT drift from the loss the trainer reports.
   The loss-only branch also skips the third O(V) pass entirely (the target's probability is one
   element). The branch is block-uniform, so the block-wide reductions stay fully populated.

**`include/sub0/eval.hpp`** -- window planning + CPU/device dispatch, extracted from `train_stage.cpp`
where THREE call sites had open-coded the same window arithmetic. Fixed a latent `std::size_t`
underflow in it (`size - SEQ_LEN - 1` on a corpus shorter than one window).

## Verified, not assumed

- Real GPU: `sub0_cuda_forward_loss` vs `sub0_cuda_backward`'s loss (dense, ignore-index-masked, and
  length-masked), and `sub0::eval` device-vs-CPU end to end at two context widths.
- In production: arm A's device-scored `val nelbo 2.1332` reproduces the CPU-computed
  `best_val_nelbo 2.13319` recorded during training, to 5 decimals.

## The gate, and the one deliberate limitation

Enabled when `caps.supports_eval && !trainer_active()`. **A trainer's own periodic evals therefore stay
on the CPU** -- the trainer holds that same mutex. Deliberate: this path calls `fwd_alloc(full=true)`,
which would add the whole inference activation scratch on top of a training allocation already tuned to
fill VRAM. Giving the trainer a device eval needs a VRAM budget it can prove fits first.

`sub0::eval::Session::declined` carries WHY the device was not used, and `report` prints it -- a silent
CPU fallback is the failure to avoid, since it looks hung rather than degraded.

## Side finding worth keeping

`sub0_tests` linked `sub0_backend_cuda` while compiling `device_backend.hpp` in its NO-DEVICE
configuration (`SUB0_BUILD_CUDA` was only defined for `sub0_train`/`sub0_gen`). Anything driving a
device through the neutral seam in a test silently took the stub and "passed". Now defined for the test
target too.

See [[mock-device-backend-test-seam]] for the other half of the testing story.
