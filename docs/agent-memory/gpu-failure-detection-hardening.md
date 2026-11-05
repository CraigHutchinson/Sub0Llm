---
name: gpu-failure-detection-hardening
description: Fixed the "training loop can't detect a silently-failed GPU step" backlog item (real damage this session, see gpu-illegal-access-hardware-fault-not-code-bug) plus 2 related gaps found by auditing for the same bug class, then closed the class permanently with [[nodiscard]] -- 4 commits (84b5326, 2a1deaa, 4d6ffa2, 6a445db)
metadata:
  type: project
  originSessionId: 2a6b481b-6211-474b-818a-995223d7d5ac
---

**Trigger:** user asked to close the standing backlog item and "consider if there are further gaps
worth fixing -- use best judgment." Investigated the actual mechanism first rather than guessing:
`sub0_cuda_train_step` already synchronizes and checks `cudaGetLastError()` internally, so it DOES
return a proper nonzero code on a device fault -- the real gap was entirely on the CALLER side.
`GpuTrainer::step()` (`train_stage.cpp`) discarded that return code, and `gpu.sync_to_host()` (called
before every eval-tick/checkpoint-tick/final save) was `void` and discarded BOTH of its own
`sub0_cuda_download_params`/`sub0_cuda_download_opt` return codes -- exactly the pathway that produced
a real corrupted checkpoint earlier this session (see
[[gpu-illegal-access-hardware-fault-not-code-bug]]): a failed download leaves the host arenas holding
stale/garbage weights, which then get written to disk as if they were the real trained state.

**Fix 1 (`84b5326`): the training loop itself.** `GpuTrainer` gained an `ok` flag set by
`step()`/`sync_to_host()` from the real CUDA return codes; `sync_to_host()` changed `void` -> `bool`.
The training loop checks `gpu.ok` at all three device-touching points (per-step, the eval tick, the
CKPT_SECONDS crash-resistance tick) and stops immediately -- via `break`, not just setting a flag
checked at the top of the next iteration, so it can't run even one more step on a known-corrupted
context -- WITHOUT attempting any further device access afterward (no `sync_to_host()`, no
`save_model()`, no terminal `write_meta()`). This deliberately leaves the last GOOD periodic
checkpoint on disk with its `meta.txt` status still `"training"` -- a non-terminal status this same
session's earlier auto-resume work ([[model-identity-date-tag-and-comparison-table]], commit
`3927bbf`) already treats as "unfinished, offer to resume" -- so the recovery path needed zero new
mechanism, it just had to not corrupt the thing the existing mechanism already knows how to pick back
up. Also corrected a genuinely misleading comment in `backend_cuda.cu` claiming "the full backend
routes failures through the engine's fatal path" -- no such path existed before this fix.

**Live-verified, not just reasoned about:** temporarily forced `gpu.ok = false` at a fixed step
(`if (step == 50) gpu.ok = false;` right after the real assignment, reverted before committing) and
confirmed: exit code 1, the exact error message, `rs.step` correctly reflects the LAST GOOD step (49,
not the failed 50th), and -- critically -- zero files written to the model directory beyond the
initial `meta.txt` (no corrupted `model.bin`/checkpoint). Then reverted and ran a normal short training
session end-to-end (reaching `max_steps` via the real eval-tick path) to confirm the happy path --
checkpoint save, final `status=trained`, sample print -- is completely unaffected.

**Fix 2 (`2a1deaa`): the same bug class in `sub0/decode.hpp`.** `kv_decode_generate`'s GPU branch
(shared by `gen`, `preview_at`, `gen_self_stats`, and `models --refresh` since the KV-cache unification
earlier this session, see [[gen-decode-kv-cache-optimization]]) ignored every
`sub0_cuda_forward_one`/`sub0_cuda_kv_reset` return code -- a device fault mid-generation would leave
every subsequent call returning stale/garbage logits, which `sample_token` would sample from as if
real, producing plausible-looking but meaningless text with no error surfaced at all. Lower stakes than
Fix 1 (no persistent checkpoint corruption possible, just one bad sample or a stale cached metric that
a later good run overwrites) but the same underlying "trust corrupted device state" pattern, so worth
closing too. Now checks each call and stops generation early (keeps whatever's already sampled, same
shape as an EOS stop) with a clear stderr message. Verified live: `gen` still produces correct
GPU-backed output on the happy path after the change.

**Fix 3 (`4d6ffa2`): found by auditing for other unchecked call sites of the same functions.**
`GpuTrainer::enable()` checked `sub0_cuda_upload_params`'s return code (falls back to CPU training on
failure) but NOT the very next line, `sub0_cuda_upload_opt` (restores Adam m/v moments on a resumed
run) -- an inconsistency between two adjacent calls that fail the same way. A silent failure there
would resume training with wrong/stale optimizer state instead of falling back to CPU like every other
`enable()` failure already does. One-line fix, matching the adjacent line's exact pattern. Verified
live: GPU training still enables and completes normally.

**Considered and deliberately NOT fixed this pass:** retroactively validating checkpoints already on
disk from BEFORE this session's fixes (e.g. a NaN/Inf sanity check, or a checksum) for corruption --
the root cause that produced bad checkpoints is now closed, and the specific corrupted artifacts from
the actual incident were already found and cleaned up manually earlier this session. Would be a
reasonable defense-in-depth addition someday, but a separate, bigger feature (needs a real corruption
DETECTOR, not just a failure-propagation fix) -- not bundled in here.

**Fix 4 (`6a445db`), a follow-up ask ("add `[[nodiscard]]` where relevant to catch these issues"):
closes the whole bug class at compile time instead of relying on finding each instance by manual audit.**
Added `[[nodiscard]]` to the entire CUDA status-code API surface (`sub0_cuda_init`,
`sub0_cuda_upload_params`/`download_params`, `upload_opt`/`download_opt`, `kv_reset`, `forward_one`,
`train_step`, `train_reserve`, `time_train_step`, `train_footprint`) plus `load_model`/`save_model`/
`load_tokenizer` in `core.hpp`. Placement gotcha confirmed via test-build: the attribute goes AFTER the
linkage-spec/export-macro, i.e. `extern "C" [[nodiscard]] int foo(...)` and
`SUB0_CUDA_API [[nodiscard]] int foo(...)`, not before. No `-Werror`/`-WX` in this project's CMake, so
violations only warn, never break the build -- confirmed by checking CMakeLists.txt first.

Rebuilding surfaced 9 real discarded-value bugs the manual audits above had missed, none previously
suspected:
- 2 diagnostic-only tune-stage GPU warmup calls to `sub0_cuda_time_train_step` -- fixed to capture
  success and `log::warn` (not abort), since these only prime cuBLAS/context for the timing sweep that
  follows, not the live training path.
- 6 discarded `sub0::load_tokenizer()` calls, one each in: `load_corpus_tokens`'s on-demand-tokenize
  path, `sub0_train_stage`'s own preview load, `sub0_autotemp_stage`, `sub0_report_stage`'s sample
  section, `sub0_models_stage`'s `--refresh` path (the exact call site `4bf29fc` had already fixed
  once for a missing-load bug -- the return value STILL wasn't being checked even after that fix), and
  `sub0_ckpt2model_stage`. Each fixed with severity matched to actual blast radius: FAIL LOUDLY
  (`log::error` + `return 1`) where a silent failure would corrupt a measurement or a written artifact
  (autotemp's EOS-stop signal, models--refresh's same signal, ckpt2model's fingerprint stamp in
  model.bin); warn-and-continue where it only degrades a cosmetic preview/sample print (training's
  periodic preview, report's sample section).
- 1 already-known gap in `tests/engine_tests.cpp` (a discarded `save_model()` in a save/load
  round-trip test) -- promoted to `REQUIRE(...)`.

**Self-inflicted bug caught by the compiler, not by me:** the ckpt2model fix's first draft wrote
`sibling_tok.string().c_str()` inside a ternary -- `.c_str()` on a temporary `std::string` that gets
destroyed at the end of the full-expression, leaving a dangling pointer passed into
`load_tokenizer`/`log::error`. Caught immediately by `-Wdangling-gsl` on the very next rebuild, before
any test run or commit. Fixed by binding `sibling_tok.string()` to a named `std::string` first, taking
`.c_str()` from that. Direct, concrete evidence for why "rebuild after every edit" is worth the
friction, not just habit.

**Verification:** full project rebuild -> zero warnings anywhere (confirmed after the dangling-pointer
fix). Both test binaries rerun in full: `sub0_frontend_tests` 81/81 cases (13,019 assertions) clean;
`sub0_tests` 92/95 cases, the 3 failures deterministically reproduced as pre-existing float-tolerance
boundary cases in CUDA forward-pass parity checks (unrelated files/functions to anything touched here,
confirmed identical failure value across two reruns with different `--rng-seed`) -- not a regression.
