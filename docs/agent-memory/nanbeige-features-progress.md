---
name: nanbeige-features-progress
description: "Nanbeige-4.5-style architecture work for Sub0Llm. GQA: DONE both backends, now the DEFAULT (GQA-2), measured +8-13% tok/s and -50/-75% KV cache. LoopSplit: DONE but A/B-NEGATIVE at this scale. SEQ_LEN decoupled from the checkpoint. Remaining: depth attention, n-gram embeddings, mHC."
metadata: 
  node_type: memory
  type: project
  originSessionId: 2a6b481b-6211-474b-818a-995223d7d5ac
  modified: 2026-07-26T21:19:13.203Z
---

Started 2026-07-26 after a gap analysis of `Nanbeige/Nanbeige4.2-3B`'s real `modeling_nanbeige.py`
([[nanbeige-architecture-reference]]). Three of six differences selected; mHC, depth attention and
n-gram embeddings deliberately deferred (reasons in the plan/commit messages).

## Landed (branch `claude/llm-transformers-1bit-qf3t3i`)

- `e21e553` **arch identity harness** (`tests/arch_identity_tests.cpp`). The verification backbone for
  everything else: FNV-1a checksum + sum/absmax of forward logits, reduced gradients and a greedy
  decode trace. A checksum specifically because sums are permutation-insensitive and LoopSplit
  reorders the layer loop. Recorded neutral baselines live in the file header.
- `6824cbb` **GQA, CPU backend**. `--kv-heads` -> `constexpr N_KV_HEADS`, `GQA_GROUP`/`D_KV` +
  divisibility static_asserts in `layout.hpp`. Wk/Wv narrow to `[D_MODEL, D_KV]`; KV cache rows narrow
  to `D_KV`. Default `N_KV_HEADS == N_HEADS` (plain MHA) and verified BIT-IDENTICAL at that default.
- `050dda2` **CUDA build guard**. `static_assert(N_KV_HEADS == N_HEADS)` in `backend_cuda.cu` so a GQA
  config fails the BUILD instead of silently miscomputing. Verified it both compiles at MHA and fires
  at `--kv-heads 2`. Delete when CUDA GQA lands.
- `aa64107` **pos_emb dropped under RoPE -> SEQ_LEN decoupled from the checkpoint**. This was the
  enabling step for long context and it turned out much cheaper than "runtime-size every buffer": the
  table was the ONLY parameter whose shape depended on SEQ_LEN, and under RoPE it was already
  allocated-zeroed-never-read. Proven end-to-end: a model saved at SEQ_LEN=256 loads into a SEQ_LEN=512
  build, PARAM_FLOATS identical (934464) at both.

- `0ce9d88` **LoopSplit, CPU backend**. `--loop-middle-layers`/`--loop-repeats`; execution order is
  `consteval make_layer_execution_order()` in `layout.hpp` (NOT a per-call container -- `AGENTS.md` 1).
  No checkpoint change (the loop re-executes existing layer indices; layout is keyed on layer index);
  what grows is per-EXECUTION state -- `ACT_CAP`, `MAX_NODES`, KV slots now size on `LOOP_EXEC_COUNT`,
  and `forward_one` separates the execution index (cache slot) from the layer index (whose weights
  run). **The backward needed NO changes** -- CPU param grads are already `+=` -- and that prediction
  was confirmed empirically: the finite-difference gradient check WITH its per-parameter probes passes
  with 4-of-8 layers running 3x, and 5-of-11 running 2x at production scale. CUDA guarded by
  static_assert (its dW writes overwrite; failure mode would be a quietly wrong gradient, not a crash).

- `19970f1` **RoPE linear position scaling, CPU AND CUDA**. `--rope-scaling linear --rope-scale-factor N`
  divides the position before the angle is formed. Formula taken verbatim from the reference
  (`NanbeigeLinearScalingRotaryEmbedding`, whose whole body is `position_ids.float() / scaling_factor`),
  with the convention re-derivation recorded at `ROPE_POS_SCALE` in `layout.hpp` -- including that the
  reference rotates HALF-SPLIT pairs while this engine rotates INTERLEAVED pairs, which is real but
  orthogonal, since scaling touches only the angle's position term and never which components pair up.
  ONE constant applied at all SEVEN angle sites so forward/backward/decode cannot drift. Bit-identical
  when off (multiply by exactly 1.0f is exact in IEEE-754). This is the FIRST feature of the three to
  land on BOTH backends -- it needed no kernel restructuring because the CUDA kernels already took
  `theta` as a runtime argument.

- `890a0ec` **CUDA GQA -- the `050dda2` guard is GONE; `--kv-heads` is now real on BOTH backends.** The
  fused QKV buffer stays fused but goes ASYMMETRIC (`[M, D_MODEL + 2*D_KV]`); its column layout is named
  ONCE in `layout.hpp` (`QKV_STRIDE`/`QKV_K_OFF`/`QKV_V_OFF`/`QK_PRE_STRIDE`) because "sub-block s starts
  at s*D_MODEL" is simply false now. `memplan.hpp` CANNOT see those constants (layout.hpp includes
  memplan, not the reverse) so it carries runtime mirrors `d_kv()`/`qkv_stride()`/`qk_pre_stride()` --
  keep the two in lock-step. Both planned structural problems went as scoped: RoPE's
  single-thread-rotates-both-Q-and-K split into `blockIdx.z`-selected passes (following qknorm's existing
  idiom, not a new one), and dk/dv regrouped to KV-head-parallel with register summing so the atomic-free
  write survives. All three RoPE kernels also collapsed onto ONE templated body (backward = same rotation
  with `sign = -1`), verified against the ORIGINAL kernel rather than trusting my derivation.

- `3bbd3ac` **CUDA LoopSplit -- the last CPU-only guard is gone; all three features run on BOTH
  backends.** Every param-grad write accumulates under `LOOP_SPLIT_ON`; activation checkpoints and KV
  slots index by EXECUTION not layer; all four forward paths + the backward walk `LAYER_EXEC_ORDER`.
  Grads already using `atomicAdd` (dgamma, embeddings) needed NOTHING -- `backward_device` zeroes the
  grad blob at step start, which is also why accumulating is safe unconditionally. **The one bug was
  mine**: the accumulate DEFAULT applied to all seven `launch_linear_bwd_t` call sites, but one targets
  `g_tr.dwqkv`, a per-execution TEMP -- accumulating there AND in `split_dqkv` double-counted every
  repeat. Symptom was CPU/CUDA gradient cosine 0.36-0.52 while the LOSSES agreed to 4 decimals:
  forward right, gradient wrong, exactly the failure the guard existed to prevent.
- `af72772` **three review passes** (design/API, aliasing+optimization, related-sources sweep). Two
  silent defects, both detailed in [[nanbeige-review-findings]].

## Remaining

1. **CUDA LoopSplit** -- DONE, see above. Remaining: -- needs an `accumulate` flag threaded through `launch_linear_bwd_t`,
   `split_dqkv_kernel` and `bias_grad_act_kernel`, plus per-execution `TrainScratch` buffers
   (currently `[N_LAYERS]`) and the residual chain at `backend_cuda.cu:3155`.
2. **Nothing has been TRAINED with any of these enabled.** All three are verified CORRECT (gradients,
   parity, identity) but none has an accuracy or throughput number behind it -- no A/B, and the
   three-pillar policy ([[three-pillar-shootout-policy]]) is therefore unmet for all of them. GQA in
   particular is a memory/quality TRADE and its whole justification is the KV-cache drop, which has been
   computed from the layout but never measured on a real run. Do not describe any of these as
   "validated"; they are "implemented and verified correct".
3. **Perplexity-vs-context-length curve for RoPE scaling is unmeasured**, which is the only thing that
   says whether the scaling actually buys usable long context or just avoids a crash. Needs a model
   trained at short SEQ_LEN, then evaluated in a long-SEQ_LEN build with scaling off vs on.

## Verification method that has been working

Every stage: (a) fingerprints BIT-IDENTICAL at the neutral setting on two scales, (b) suite assertion
AND test-case counts compared against recorded baselines, with any delta explained arithmetically
rather than waved through, (c) the feature then actually enabled and exercised.

Six real findings were caught this way, all invisible to a "does it still pass" check:
- The K head-count was fixed in `forward_one` but not in `forward()` (`op_qknorm`/`op_rope` still passed
  `N_HEADS` for K). Caught by the forward_one-vs-forward parity test.
- `memplan::param_floats` independently re-derives the layout and still counted `pos_emb`. Caught by
  its own dedicated cuda test.
- LoopSplit's head/tail symmetry `static_assert` was written unconditionally, so it rejected every
  ODD-layer model even with the feature OFF (`N_LAYERS - 0` is odd) -- breaking the "zero effect when
  disabled" contract. **Only visible because of the two-scale rule** (`AGENTS.md` 7): factspike96 has
  8 layers and passed, d196check has 11 and did not. A single-scale check would have shipped it.
- `build_qkv_weights` sized its launch grid `n = C*C` -- correct while ONE thread wrote all three
  sub-blocks, but it leaves two thirds of the fused weight matrix GARBAGE under GQA's new
  one-thread-per-fused-element mapping. It surfaced as a gradient-cosine failure (0.53) in the MHA
  BASELINE -- the "no behaviour change when disabled" gate caught it; every GQA run would have looked
  fine. Strongest evidence yet for testing the NEUTRAL setting before the feature setting.
- Four kernels used `gridDim.y` AS the query-head count when indexing the softmax-stats arrays.
  Regrouping dv/dk to KV-head-parallel silently redefines `gridDim.y` to the KV count while
  `g_bwd_m/invZ/dot` stay laid out per QUERY head -- a wrong-index read with no crash. NOT in the plan;
  found only by re-verifying the roadmap's site map against real code, which the roadmap itself says to do.
- RoPE scaling appeared to trip `engine_tests`' aggregate finite-difference check (ratio 0.863). I
  diagnosed that as a float32 limitation of the CHECK and shipped `19970f1` with it as a known caveat.
  **That diagnosis was WRONG, and so was the caveat.** The 0.863 was an artefact of `factspike96`'s
  493-token TOY VOCABULARY. Re-measured at a realistic vocab with scaling x4: **0.999979** (d96 L8 H2)
  and **0.999535** (production d196 L11 H7) -- both pass. Held to the right instinct at the time
  (refused to widen a tolerance to make my own feature pass) but for the wrong reason; the real fault
  was gating a numerical test on a toy-vocab spike config. Full correction, including the discriminating
  experiment, in [[roadmap-2026-07-21]] section 3 and [[nanbeige-review-findings]].

Builds used: `out/build/factspike96` (d96/L8/H2, tiny/fast), `out/build/d196check` (d196/L11/H7,
production-shaped), `out/build/hd96_check` (d384/L2/H4, CUDA -- and H=4 makes `--kv-heads 2` and `1`
both valid GQA shapes). CUDA needs INCLUDE/LIB from the vcvars dump PLUS the MSVC HostX64 bin dir
PREPENDED to PATH so nvcc finds cl.exe -- do NOT replace PATH, that breaks the POSIX tools.

## Sequencing note

The plan ordered GQA -> LoopSplit -> SEQ_LEN -> RoPE. Actual order taken was GQA(CPU) -> SEQ_LEN,
deliberately: CUDA GQA is the largest and riskiest single chunk, and SEQ_LEN decoupling was small,
self-contained and independent of it, so it delivered standalone value first. CUDA GQA is safe to leave
pending because the build guard makes the unsupported config impossible rather than silently wrong.
