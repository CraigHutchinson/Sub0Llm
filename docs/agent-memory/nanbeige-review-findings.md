---
name: nanbeige-review-findings
description: "What three review passes over the Nanbeige feature set (GQA/RoPE-scaling/LoopSplit) actually caught, 2026-07-27 -- two SILENT defects that full green test suites on both backends did not: a checkpoint that could not discriminate LoopSplit, and a GQA out-of-bounds read hidden by tag-filtered testing. Read before trusting 'suite green' as evidence."
metadata: 
  node_type: memory
  type: project
  originSessionId: 2a6b481b-6211-474b-818a-995223d7d5ac
  modified: 2026-07-27T15:00:36.670Z
---

Reviewed `e21e553^..3bbd3ac` (9 commits, ~973 insertions) in three passes: design/API/boundary
(`cpp-review`), aliasing + optimization (5 layers, helicopter to statement), and a sweep of RELATED
sources the diff never touched. Fixes landed as `af72772`; pre-fix state tagged
`nanbeige-pre-review-fixes` @ `3bbd3aca58d1`.

**The headline**: every suite was green on both backends, at the exact recorded assertion counts,
before this review. Both real defects were invisible to that. "Suite green at the recorded count" is
necessary and nowhere near sufficient.

## Defect 1 -- nothing in the checkpoint discriminated a LoopSplit model

LoopSplit re-executes EXISTING layer indices, so a looped and un-looped build at the same dims have
byte-identical parameter blobs. `PARAM_FLOATS` matches; so does every `Header` field. A model trained
at `--loop-repeats 3` loaded silently into a plain build and computed a different function.

`0ce9d88`'s commit message said "No checkpoint change (layout is keyed on layer index)" -- **true about
the layout, and it does not follow that the checkpoint can still tell the two architectures apart.**
That inference gap is the lesson: `AGENTS.md` 3 rule 1 says CHECK whether an existing field
discriminates. Working the arithmetic (shared weights => identical PARAM_FLOATS) takes a minute and was
skipped.

Fix uses each format's own established additive mechanism: a second trailing record on the model file
(the tokenizer-fingerprint pattern), `CKPT_VERSION` 4->5 on the `.ckpt`. A file with NO record is NOT
"unknown, skip the check" -- any such file predates LoopSplit and was therefore trained un-looped, so
the legacy path is a real guard. Scoped to the SCHEDULE only: RoPE scaling is excluded on purpose
(train short / run long with scaling on is the intended workflow), and GQA needs no entry because D_KV
narrows Wk/Wv so PARAM_FLOATS discriminates it monotonically.

## Defect 2 -- a GQA out-of-bounds read that tag-filtered testing hid

`tests/factspike_engine_tests.cpp` copied `D_MODEL` floats out of KV-cache rows that are now `D_KV`
wide -- an OOB read of `(D_MODEL - D_KV)` floats per row, running off the whole allocation at the last
`(layer, pos)`. 16 sites. In the default `sub0_tests` target.

**Why it survived**: I updated `core.hpp`'s doc comment to say "D_KV floats" and did not follow my own
doc change to its callers. Then the GQA test runs were CUDA-only and filtered to `[cuda]`, so this file
never executed under GQA at all; the CPU scales that DO run it were MHA-only.

**Standing rule this earns**: when a feature changes a WIDTH or a SEMANTIC of a shared accessor, run
the FULL suite with the feature ON at least once, unfiltered. Verified after the fix: `d196check` at
`--kv-heads 1` (MQA), full suite, 40388126 assertions / 127 cases, delta exactly 5.000x the Wk/Wv float
reduction.

## The related-sources sweep is what found it

Reviewing the diff found neither defect's caller. Grepping for CONSUMERS of every changed
surface -- `memplan::Dims`, `kv_krow_ptr`/`kv_vrow_ptr`/`kv_splice_row`, `PKind::Wk`/`PosEmb` -- found
both, plus:
- `configurator.cpp` omitted `exec_layers`, under-predicting activation memory 1.43x at 10 executions.
  That feeds `gpu_batch_estimate()`, i.e. the RECOMMENDED batch -- under-predicting pushes it UP, the
  dangerous direction given [[gpu-large-batch-access-violation]].
- `sub0_memplan_stage` built `Dims` from 6 of 12 fields, costing a gated+tied+qk-norm RoPE build as an
  untied non-gated absolute-position MHA one. `tied`/`qk_norm`/`gated` were ALREADY wrong pre-work.
  Third instance of the positional-aggregate-init hazard -> `current_build_dims()` in the roadmap.
- `arch_identity_tests`' recorded baselines were stale (aa64107 legitimately moved counts + grad hash)
  while telling the reader any drift is a leak -- actively misleading, re-recorded with the delta
  explained.

## Aliasing pass: verified clean, worth not re-litigating

- `q`/`k`/`v` as three `__restrict__` pointers into ONE allocation is LEGAL here -- accessed column
  ranges are disjoint and read-only. The `__restrict__` is load-bearing (enables `__ldg`). Post-GQA the
  sub-blocks are unequal widths, so the invariant is no longer obvious by symmetry; comment added.
- `dq`/`dk`/`dv` write-restrict into one buffer: disjoint, and the dv/dk KV-head regroup PRESERVED it.
- CPU GQA `+=` into shared KV rows is NOT a race: the `h` loop is `omp simd` only, and the outer
  parallel regions partition disjoint parameter tensors. Checked specifically; clean.
- The dv/dk regroup also CUT global K/V read traffic by `GQA_GROUP x` (k/v loaded once per block
  instead of once per query-head block). Record so it is not "optimised away" later.

## Hardening (`8ef7ab6`) -- the CLASSES, not more instances

Each of the three had a live case a point fix would have missed:

1. **`ARCH_SCHEDULE_ID` -> `ARCH_FINGERPRINT`.** `af72772` guarded LoopSplit's schedule; that class had
   a SECOND LIVE MEMBER -- **`ROPE_THETA`** is user-settable, changes every rotation angle, and is in
   neither the Header nor PARAM_FLOATS, so a model trained at 10000 loaded silently into a 500000 build.
   Verified broken before, rejected after. The fingerprint now carries a written CLASSIFICATION RULE so
   the next compile-time axis forces a decision: (1) shape-changing -> PARAM_FLOATS covers it;
   (2) computation-changing but shape-neutral -> ADD IT; (3) deliberately variable train-vs-inference ->
   exclude AND say why (RoPE scaling only). Packed not hashed, so a mismatch names actual values -- an
   undecodable fingerprint gives a "different" diagnostic, which is how the LoopSplit case hid.
2. **`current_build_dims()`** collapses five positional 12-field `memplan::Dims` initialisations into one
   designated-initialiser factory. That pattern produced THREE real bugs in one cycle. Configurator and
   config_tests keep their own -- they describe another config, not this build.
3. **`core.hpp` now includes `layout.hpp`.** The generated header gives RAW config at global scope;
   derived constants live in `namespace sub0`. That split cost three build breaks in one session.

Corrected a wrong claim of my own along the way: memplan is NOT "included by layout.hpp". It is
deliberately dependency-free so the tuner/configurator can call it before a device exists -- which is the
real reason it cannot use the constexpr constants, and why layout -> memplan is a safe one-way include.

## qknorm launch shape: MEASURED, and it refuted my own caution (`59afe68`)

I raised it with the caveat that 37.5% of blocks would NOT be 37.5% of runtime, because blocks that exit
immediately are cheap. **Wrong.** ncu, d384 L6 H8 kv=2, batch 32:

    before  grid (8125,8,2) = 130,000 blocks   267.8 us
    after   (8125,8,1) 135.6us + (8125,2,1) 35.8us = 171.4 us
    blocks -37.5%  ->  kernel time -36.0%

Almost exactly proportional: at 32 threads/block the per-block scheduling cost dominates the tiny
per-block work, so a no-op block costs nearly what a real one does. **The next time an early-out looks
free, measure it** -- and note the measurement moved me AGAINST my prior, which is the case where it pays.

End-to-end is much smaller and noisier: 3 INTERLEAVED pairs at 150 steps/batch 256 gave median +1.1%,
mean +2.6%, all three paired deltas positive (+6611/+3438/+14199) but per-trial spread (15,610) EXCEEDS
the gain. Defensible claim is "-36% on the kernel, small positive end-to-end" -- do NOT quote +2.6%.

**ncu on this machine**: needs VRAM headroom for its own collection, so profile a SMALL batch (32 worked
with 7.8 GB free); the binary is at
`C:/Program Files/NVIDIA Corporation/Nsight Compute 2026.2.1/target/windows-desktop-win7-x64/ncu.exe`
(not on PATH, and not under the CUDA toolkit dir). Useful invocation:
`ncu --target-processes all --kernel-name regex:<name> --launch-count N --metrics gpu__time_duration.sum,launch__grid_size <cmd>`.

## The wasted-block rule of thumb (both launch fixes, measured)

Two kernels had the same defect -- a fused `grid.z=2` launch sized for the WIDER sub-block, so the
narrower pass launched whole no-op blocks. Both fixed by making `which` a kernel ARGUMENT and giving
each pass its own grid (`59afe68` qknorm, `dac927a` rope). The measured costs differ a lot:

| kernel | threads/block | blocks | time |
|---|---|---|---|
| qknorm | 32  | -37.5% | **-36.0%** |
| rope   | 256 | -33.3% | **-9.1%**  |

**The cost of a wasted block scales with how SMALL the block is.** At 32 threads the per-block
scheduling cost dominates the tiny per-block work, so a no-op block costs nearly what a real one does.
At 256 threads that fixed cost is amortised over 8x more real work. Use this before assuming an
early-out is free -- and note I got qknorm wrong in the pessimistic direction and rope right for the
wrong reason ("threads, not blocks" -- they were entire blocks, 4 of 6 x-blocks on the K pass).

Neither is a throughput claim. qknorm's end-to-end was median +1.1% over 3 interleaved pairs with the
spread exceeding the gain; rope's is smaller still and no e2e was attempted. They are kept because they
are free and strictly less work.

## factspike96 is NOT a valid scale for numerical gradient work (2026-07-27)

Its 493-token toy vocabulary -- not its size -- makes the aggregate finite-difference check read ~3.8%
low. One-variable discriminator (factspike96 dims, d196check's real vocab+corpus):

| config | vocab | ratio |
|---|---|---|
| d96 L8 H2   | 493   | 0.96199 |
| d96 L8 H2   | 16535 | **0.999968** |
| d196 L11 H7 | 16535 | **0.999941** |

At a realistic vocabulary the EXISTING probe is exact to five decimals at both sizes. Consequences: the
"float32 floor" I recorded does not exist; RoPE scaling verifies cleanly (0.999979 / 0.999535 with
scaling x4, both PASS) so `19970f1`'s known-caveat is superseded; and the planned top-K probe redesign is
unnecessary -- I nearly rebuilt a test to fix a phantom.

`AGENTS.md` 7 asks for a tiny config AND a production-shaped one. "Tiny" must still mean a REALISTIC
vocabulary: use d96 L8 H2 at vocab 16535 (d196check's corpus with dims overridden) as the fast scale.
Green suites never surfaced this -- a user asking "could factspike be misleading, it's a spike config"
did.

## Coverage sweep (2026-07-27): blast radius verified CONTAINED

The changed KV-row API has exactly THREE consumers repo-wide -- `core.hpp` (decl), `backend_cpu.cpp`
(impl), `factspike_engine_tests.cpp` (the fixed one). `decode.hpp`, `gen_stage.cpp`, `factspike.hpp`,
`tune.hpp`, `gguf.hpp` have ZERO exposure: they go through `forward_one`/`kv_reset` and never index a KV
row. The only `D_MODEL` mentions in `scratch_slots.hpp`/`config_util.hpp` are comments. Worth recording
so the question is not re-opened.

The sweep procedure is now **`AGENTS.md` rule 10** (`edbf341`) rather than tribal knowledge.

## Open, deliberately unapplied (perf claims need measurement, not assertion)

- `gguf.hpp`'s `matches_plain_mha` framing is stale now the engine supports GQA (report-only, no
  production gate). Should carry the derived KV head count, which is what an importer needs.
