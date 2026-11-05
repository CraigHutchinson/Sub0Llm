---
name: weight-tying-tied-embeddings
description: Tied input-embedding/output-head, CPU DONE + CUDA DONE (2026-07-04, CUDA via Fable); item 3 of the architecture backlog, fully complete on both backends
metadata:
  type: project
  originSessionId: 2a6b481b-6211-474b-818a-995223d7d5ac
---

Implemented `--tie-embeddings` (item 3 of [[architecture-ideas-backlog]]): the LM head reuses
`TokEmb` instead of a separate `[D_MODEL,VOCAB]` `LmHead` matrix + `LmBias`. `layout.hpp` drops both
slots when `USE_TIED_EMBEDDINGS` (`NUM_PARAMS` tail becomes 1 (`LnF` alone) vs 3
(`LnF`,`LmHead`,`LmBias`)). Checkpoint-format-safe with NO new header field, using the same reasoning
as gated-FFN: `param_floats`/`trainable_floats()` already differs whenever the tail param count
differs, so a tied-vs-untied mismatch is caught by the existing size check.

**Key architectural finding:** `TokEmb` is `[VOCAB,D_MODEL]` row-major -- efficient for embedding
LOOKUP (one contiguous row read per token). An axpy-efficient head GEMM (`op_linear`'s style) would
want `D_MODEL` as the outer/contiguous axis instead. These are INHERENTLY incompatible layouts for the
same buffer, so the tied head's FORWARD (`op_tied_head`) is necessarily a dot-product-per-output-column
pattern (slower per-output-element than `op_linear`'s axpy style, same total FLOPs) -- there is no way
to make forward as fast as an untied head without either transposing (defeats the point of tying) or
accepting the dot-product shape. BOTH backward passes (`dx = dY @ table`, `dtable[v,:] += dY[t,v] *
x[t,:]`) reduce back to `op_linear`'s own forward/dW shapes and stay axpy-efficient -- only forward pays
the layout mismatch cost.

**Verification performed (all real, not assumed):**
- Default (untied) `smoke_gpu` config: frontend suite 12,862/12,862 assertions; full non-CUDA suite
  124,694,440/124,694,440 (one MORE than the prior 124,694,438/124,694,439 baseline, exactly matching
  the new `STATIC_REQUIRE(!USE_TIED_EMBEDDINGS || COMPUTE_MODE==Cpu)` static check added to
  `layout_tests.cpp`), same single pre-existing unrelated gradient-check failure at d448 (see
  [[engine-tests-gradient-check-dims-dependent]]) -- i.e. tying added zero regressions to the default
  build.
- A real `--tie-embeddings 1` d32/l2/h4/sq32/v512 config against `data/english.txt`: full rebuild, full
  suite green at 264,991/264,991 assertions -- CRITICALLY including the finite-difference gradient
  check passing cleanly, which is the direct numerical proof `op_tied_head`'s forward+backward math is
  correct (both the embedding-lookup gradient and the head gradient correctly accumulate into the SAME
  shared buffer rather than one overwriting the other).
- Real end-to-end smoke test: trained 60 steps (val_nelbo ~6.42 -> 5.7677), ran `sub0llm-gen.exe`
  (confirms the KV-cache `tied_head_row` decode path), ran `sub0llm models` (confirms the `...v512rw...`
  registry tag and correct incompatible-vs-loadable gating against the real production model), resumed
  training 20 more steps (5.7677 -> 5.6150, smooth continuation, proving checkpoint round-trip
  correctness with no format break).

**CUDA support DONE (2026-07-04, via a phased Fable dispatch -- see [[fable-subagent-for-big-risky-tasks]]
for the delegation writeup).** The `static_assert(!USE_TIED_EMBEDDINGS, ...)` guard is lifted; GPU is
now a fully valid backend for tied embeddings, same as CPU.

Implementation: `launch_tied_head` (forward, `logits = a . tok_emb^T` via `CUBLAS_OP_T` -- the same
row-major-via-column-major trick `launch_linear_bwd`'s existing dX GEMM already used, just a new
operand-role combination) and `launch_tied_head_bwd` (two GEMMs: `dA = dY . tok_emb` reusing
`launch_linear`'s forward shape; `dTok += dY^T . a` reusing `launch_linear_bwd`'s dW shape with
`accumulate=true` UNCONDITIONALLY -- `tok_emb`'s grad slot receives two independent contributions per
step, the tied-head's own plus the ordinary embedding-lookup scatter-add, and unconditional accumulate
makes correctness independent of which runs first, matching the CPU reference's plain `+=` semantics
exactly on both sides). Both new functions take runtime `M/C/V` (not baked `D_MODEL`/`VOCAB`), matching
`launch_linear`'s own convention -- this is what let the self-test run BOTH a toy scale and the actual
production d448/vocab16517 scale in one binary.

Found+fixed 2 bugs beyond the original Phase 1 plan, both independent `memplan::Dims` construction
sites the initial audit had missed (a `kFootprintDims` in `backend_cuda.cu` itself, and 2 more in
`train_stage.cpp`'s GPU VRAM-fallback / tuning-stage code) -- a repo-wide grep for `memplan::Dims`
construction after the first one surfaced, closing the gap. Also self-caught a false-positive in its
own new parity test (per-element max-relative-diff blew up at production scale from an ordinary
near-zero-crossing denominator, not a real bug -- switched to whole-array relative-L2 norm, this
project's own established convention for CUDA-vs-CPU parity checks).

Verified: 3 real GPU builds (untied-regression d448, tied d64-small, tied d448-production/the ACTUAL
production config) all green -- `[cuda]` suite 100% at both scales (~41.46M assertions untied, ~34.04M
tied, matching between my own independent re-run and Fable's reported numbers exactly), including the
`TokEmb`-isolated gradient-parity test and the memplan footprint-prediction test. Throughput: tied is
not slower than untied (68.86ms vs 73.10ms per step at batch16/T64/d448, within measurement noise).
ncu profile (small build, avoiding the VBS/production-model blocker) showed nothing unusual for a
non-tensor-core FP32 GEMM. I independently re-derived and hand-traced the core GEMM math myself
(matched Fable's derivation exactly) before trusting the result, and separately re-ran the CUDA suite
myself at both tied and untied production scale after Fable's own report, rather than taking the report
at face value.
