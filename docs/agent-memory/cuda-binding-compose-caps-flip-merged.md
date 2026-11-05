---
name: cuda-binding-compose-caps-flip-merged
description: "DONE + merged (2026-07-18): CUDA backend now composes scratch/persistent/sentinel-bound embed rows on-device (param-free MeanPool/Hash/HRR arms of encode_slot) via a host-computed override table; supports_binding_compose flipped to 1 in sub0_dev_caps(). Implemented by a delegated Fable agent, reviewed line-by-line, independently re-verified on real RTX 5070 hardware in TWO separate builds. Consuming flip (hybrid router's needs_cpu) ALSO DONE (2026-07-18): real A/B on hardware shows ~40-57% higher training throughput."
metadata:
  type: project
  originSessionId: 2a6b481b-6211-474b-818a-995223d7d5ac
---

Closes the design in [[hybrid-cpu-gpu-execution-design]]'s own aside (that memory's line ~41: "which is
effectively building the full CUDA content-embed port this project has always deferred, not a shortcut
around it") -- that deferred device-side content-embed port is now built, for the three param-free encoder
arms. Full design doc: docs/BACKENDS.md "Binding-compose on CUDA".

## What shipped

Delegated to a background Fable agent (worktree `agent-a1683d5085251cb7c`) after two session-limit
interruptions and resumes. Implementation: `src/backend_cuda.cu` (+511 lines), `include/sub0/device_backend.hpp`
(+45, the `SUB0_DEV_BIND_*` wire constants + `sub0_dev_set_window_bindings`), `tests/cuda_tests.cpp` (+199,
3 new parity test cases). Host precomputes an override table (`override_idx[m]` -> entry index, entries =
`{frag_offset, frag_len, encoding}`, flat `frags[]`) and uploads it via `sub0_cuda_set_window_bindings`;
device never parses binding SEMANTICS, only composes rows it's told to. Kernels touched:
`embed_kernel`/`embed_act_kernel`/`embed_backward_token_kernel`/`embed_one_body` (decode). Graph-capture-safe
by design: kernels dereference a FIXED device-resident `DevBindings` header pointer (`g_dev_bind`, allocated
once in `ensure_bind_hdr()`, called from `fwd_alloc()` so it exists before any graph capture) whose CONTENTS
change per install/clear -- same "fixed buffer, changing contents" pattern as `g_decode_state`. Unsupported
encodings (CharEncoder/ConvPool/Scalar -- the learned-`enc_w` arms) and Absolute-PE builds are REJECTED at
install (nonzero return), never silently mis-composed.

## Review (done by me, not rubber-stamped)

Read the full diff. Cross-checked the device MeanPool/Hash/HRR forward AND backward-adjoint math line-by-line
against the CPU reference (`sub0::encode_slot`/`encode_slot_bwd`, `scratch_slots.hpp`) -- exact match on
accumulation order, rotation-pair convention, and the HRR circular-convolution/-correlation adjoint pair.
Traced the CUDA-graph-capture lifetime through `fwd_alloc` -> `sub0_cuda_kv_reset` -> `capture_decode_graph`
call order to confirm `g_dev_bind` is always allocated before any capture. Verified host-side validation
(`sub0_cuda_set_window_bindings`) correctly rejects OOB `override_idx`/entry `off+len`/frag ids and unsupported
encodings before any upload. Confirmed `bound()` (ScratchBindings/PersistentBindings/SentinelBindings) never
produces an empty fragment list in production, so the device's `len<1 -> reject` never excludes a real case.

**Independently re-ran the parity suite myself on real hardware, in TWO separate builds** (not just trusting
the agent's report): the agent's own worktree build (`agentcuda`) -- 3 new cases green, 46 assertions, numbers
reproduced EXACTLY (MeanPool fwd bit-identical, Hash fwd maxrel 2.95e-7/bwd 2.95e-7, HRR fwd maxrel 2.23e-7/bwd
2.02e-7, end-to-end train-step cos >=0.9999 all three arms); then a from-scratch rebuild in the main checkout's
`gsm8k` CUDA build dir (hit an MSYS `cmd.exe`-argument-mangling quirk twice trying to source vcvars64 via the
Bash tool -- `/c` gets path-translated; fixed by using the PowerShell tool instead) -- clean build, full
`[cuda]`-tagged suite green (41 cases, 2,029,105 assertions), no regressions.

Two non-blocking findings filed (see this session's review): (1) the 4 host-side validation reject-branches in
`sub0_cuda_set_window_bindings` have only 1 of 4 directly tested (unsupported-encoding; OOB-index/off-len/frag-id
paths untested); (2) the `SUB0_DEV_BIND_ENC_*` wire constants in `device_backend.hpp` duplicate
`sub0::SlotEncoding`'s values with only `backend_cuda.cu`'s `static_assert` (CUDA-build-only) guarding drift --
deliberate, documented, matches the project's existing enum-mirror convention, not a defect.

## Merged

`worktree-agent-a1683d5085251cb7c` (commit `9c9e7be`) merged into `claude/llm-transformers-1bit-qf3t3i`
(merge commit `8016e64`, clean, no conflicts -- the only overlap risk, docs-only commit `b73c347`, touched
`docs/SCRATCH_TOKENS.md` exclusively). Worktree and branch cleaned up. `docs/BACKENDS.md`'s design section
updated from "in progress" to "DONE + merged."

## Consuming flip -- DONE (2026-07-18), same session

`train_stage.cpp`'s hybrid router now reads `needs_cpu = !src.doc_bindings.empty() && !dev_binding_compose`
(`dev_binding_compose` cached once from `sub0_dev_caps()`). That boolean flip alone isn't the hard part --
`GpuTrainer::backward_only` materializes its own `ids`/`targets` internally, so the caller has no host-visible
per-position binding state for the GPU sub-batch the way `materialize_cpu_window` already builds for the CPU
one. Added a new `install_gpu_bindings`-shaped block (inline in `sub0_train_stage`'s `hybrid_train` branch,
not a separate named function) that, for each compacted GPU-sub-batch row, resolves the owning document via
`doc_of(src.docs, starts[b])`, builds a `ScratchBindings` view over `doc_bindings[doc]`, and walks the
window's tokens to flatten bound positions into the `override_idx`/`entries`/`frags` wire format -- installed
via `sub0_dev_set_window_bindings` right before `gpu.backward_only`, cleared right after. No per-encoding gate
needed: production `content_embed` (`ContentEmbedKind`) only ever selects MeanPool/Hash/HRR, all three
device-supported, so `needs_cpu` is unconditionally false whenever `dev_binding_compose` is true -- the CPU
sub-batch side of `hybrid_train` goes idle in that case but is deliberately kept (not collapsed into
`gpu_train`) so a future *partial*-capability backend still degrades correctly per-window.

**Verification** (three-pillar: correctness + performance + no regressions):
- New differential test `cuda_tests.cpp` "doc_of-resolved bindings across multiple windows per document match
  CPU" -- 2 documents, 2 windows each, the SAME slot index bound to DIFFERENT content per document, closing
  the one gap the pre-existing end-to-end binding-compose test didn't cover (it never resolves bindings
  through `doc_of` at all, each of its windows carries its own directly-assigned table).
- Full `[cuda]` suite green on real hardware: 42 cases, 2,029,113 assertions (2,029,105 + the 8 new).
- CPU-only default suite green: 117 cases, 9,192,142 assertions, from a from-scratch rebuild of both configs.
- **Real hardware A/B**, not just unit tests: a live `--blend-config` training run (base + scratchspike,
  `content_embed: hrr`, gsm8k.txt corpus, d=128 L=8, same seed/schedule/tokenized corpus both times), code
  toggled via `git stash` (the same technique used to prove the `sub0_cuda_upload_params` pre-existing bug
  earlier this session). Steady-state throughput ~40-57% higher with the flip active (e.g. step 221: 517K vs
  365K tok/s; step 286: 441K vs 309K tok/s) across nearly every matching step; both runs' loss/val_nelbo
  curves equally clean, no divergence. Direct proof the GPU sub-batch is actually absorbing content-embed
  windows now, not a no-op flip.

## Unrelated finding noticed along the way (NOT investigated further, out of scope)

One full CPU-only suite run (`out/build/tok_cpu`) showed 7 failures in `persistent_slots_engine_tests.cpp`
(a size-mismatch assertion, "an unbound id (no table installed) is safe, not a crash"), in a subsystem this
session's changes never touch (`sub0_tests.exe` doesn't even link `train_stage.cpp`). Reran the identical
tag in isolation (`[persistent_slots]`) -- 9300 assertions, 5 cases, all green -- and reran the FULL suite
again -- 117 cases, 9,192,142 assertions, all green. Confirmed order-dependent/flaky (Catch2's randomized
test order), not a deterministic new bug and not caused by this session's work. Likely the same general bug
CLASS as [[cross-test-optimizer-state-contamination]] (thread_local/global state leaking between TEST_CASEs)
but a different specific manifestation (a logits-buffer SIZE mismatch, not stale optimizer moments) -- not
yet root-caused. Filed here rather than chased, since it doesn't block this session's actual task.
