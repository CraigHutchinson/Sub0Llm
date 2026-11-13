# Outstanding-work integration — 2026-11-13

Tracking: #77, #18, #66, #70, #74, #78, #63, #94.

The owner requested that all outstanding work be merged to main. The integration preserves the
head histories of PRs #67, #71, #91, #93, #99 and #102, including the locally prepared #67/main
integration commit. These are incremental deliveries; merging does not promote an experimental
kernel or close an unfinished qualification issue.

## Review and integration fixes

- Combined the IQ1_S dual-accumulator kernel with batched projection. The batch core now uses
  the same alternating accumulator chains and final reduction. Added exact serial/batch gates
  for all supported formats and five modes at widths 256/5120/17408, tiles 8/16/32 and 17 tokens,
  unaligned packed rows and padded outputs. Fused-IQ2 batch dispatch uses the qualified row core
  for IQ2 and matches the corresponding serial mode for other formats.
- Checked row span, null pointers and index multiplication before batch pointer arithmetic.
  Removed repeated worker-status heap allocation from the prefill prover.
- Reviewed the machine headers against cpp-review. Fixed allocation-size wraparound, failed-buffer
  access, partial thread-launch teardown, numeric-budget validation and Windows allocation/asm
  portability. Regression cases exercise failure paths rather than relying on successful timing.
- The unfiltered build exposed a pre-existing missing `USE_ATTENTION_GATE` constant in the
  checked-in Qwen4 real-axis config. It is explicitly false for that preview's QSA architecture;
  its existing shape gates verify this shared consumer without changing the dense-27B config.
- Tightened the proposed machine-lock design: age alone cannot evict a confirmed live owner.
- Preserved the untracked second-pass harness work item under #70 and narrowed its context to the
  native row core, so batch code does not consume or escape the experiment's edit budget.

## Qualification boundaries

AUTO remains baseline, the live chat artifacts are preserved, and the prefill prover remains an
explicit build target. No current end-to-end speedup is claimed from this integration. Historical
measurements in the constituent reports retain their original dates and candidate identities.

The full frozen 4,096-target quality gate is still open under #63/#70. The preserved #66 capture
manifest refers to its original fused-IQ2 artifact, not this newly combined IQ1/IQ2 candidate.
Do not rebuild that artifact in place and treat its old scores as combined-candidate evidence.
Stage a new candidate manifest for combined qualification; matching independent reference cache
records can be reused according to the existing provenance checks.

#74 still needs production-prefill/GDN integration and model-level qualification. #78 still needs
the remaining measurement features; native/machine-wide locking belongs to #94. Roof timing must
be coordinated with the existing shared performance workflow until that integration lands.

## Validation

On CrogBuntu1, 2026-11-13:

- `scripts/verify.sh out/build/pr5_check`: complete build passes; 523 registered tests, zero
  failures and two platform-dependent mapped-arena skips.
- Neutral engine: **44,292,784 assertions / 147 cases**, unchanged from the pre-integration binary.
  Forward/gradient/decode hashes remain `f7fdcb276307142f` / `29595b24c4355ecf` / `b89e320bb02b79eb`.
- Frontend: **220,980 assertions / 327 cases**, unchanged from the pre-integration binary.
- Shared performance/tooling CTest: **9/9** groups pass. Batch projection fixture passes generic,
  AVX2 and ASan/UBSan; no numerical tolerances were relaxed.
- Machine suites under AVX2 ASan/UBSan: **5,516 assertions**, 33 passed cases and four expected
  ISA-specific skips. Native variants pass in the unfiltered suite.
- Dense-token and prefill-prover explicit targets build. The actual downloaded GGUF is accepted by
  the prover parser; empty selection and zero-token calls correctly fail rather than report parity.
- AVX2 machine CLI builds with warnings as errors; facts JSON parses and `inf`, `nan`, `1e999`
  and excessive budget arguments return error 2. The second-pass harness item and its focused
  context validate against the existing harness contract.

Windows compilation remains unverified on this Linux machine; the portability branches are reviewed.
No new throughput measurements were taken during the correctness builds.
