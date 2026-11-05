---
name: memplan-overhead-280-not-1650-and-host-is-bigger
description: "READ FIRST on Sub0Llm memory work — the \"~1650 MiB fixed overhead\" is RETRACTED (real ~280); kVramHeadroomMB=512 was always fine; and the HOST footprint is the bigger, unmodelled half"
metadata: 
  node_type: memory
  type: project
  originSessionId: 2a6b481b-6211-474b-818a-995223d7d5ac
  modified: 2026-07-30T11:12:22.723Z
---

Two corrections about Sub0Llm's memory model, established 2026-07-30. Both invalidate numbers I
previously stated confidently.

**1. The device overhead is ~280 MiB, NOT ~1650.** The 1650 figure came from hand-computed `memplan`
totals that evaluated the logits chunk count as `ceil(vocab/d_ff)` = 33 where `logits_n_chunks` clamps
to `[1, LOGITS_MAX_CHUNKS]` = 8 — under-counting the largest scratch term by ~1354 MiB. The same error
applied to two shapes inflated both deltas identically, which is *why* they agreed to 3 MiB and why
that agreement read as corroboration. **Two checks with one shared error are one check.** Corrected:
arm A 7219 predicted / 7502 measured / +283; arm D 8227 / 8507 / +280.

Consequence: `kVramHeadroomMB = 512` was **always adequate** (~230 MiB spare). Acting on the old advice to
raise it would have crippled the batch for nothing.

**The wrong figures were REMOVED from `docs/MEMORY_AUDIT.md`, not retracted in place**, so they cannot be
quoted out of context. This memory keeps the number only so a future session recalling "~1650" from an old
transcript or from git history lands on the correction. The doc keeps the *method* post-mortem — two checks
with one shared error are one check — without restating any of the wrong values.

The tell was recorded at the time and not chased: the section closed with *"the agreement to ~3 MiB is
better than that method deserves."* A suspiciously good result from a method you have just called
unreliable is evidence about the method, not the result.

**2. The HOST footprint is the bigger half, and was under-reported 27x.** `print_config()` printed one
`Worker`'s two activation arenas as the whole activation cost, omitting both the `DEFAULT_THREADS`
multiplier and the per-Worker `PARAM_FLOATS` gradient. At the d196 check config:

```
host (CPU) plan: shared 133 MiB + 24 x worker 336 MiB = 8188 MiB   (was reported as 302 MB)
  worker: gradient 33 MiB + activation arenas 302 MiB
device plan @ DEFAULT_GPU_BATCH=96:                     1287 MiB
```

So the half that was modelled, itemized and CI-validated (`memplan.hpp` + `sub0_cuda_train_footprint`)
was the **small** half. The 8.2 GiB host half is `calc_act_cap()` in `backend_cpu.cpp` — a second
footprint model with no shared vocabulary: build-fixed rather than `Dims`-parameterized (so it cannot
answer "what if"), a scalar rather than itemized, per-worker rather than total, slack baked in as
`* 3 / 2 + 8192` rather than named, and never validated against a measurement. Hybrid, which splits a
batch across both, is modelled nowhere. Design + staging in `docs/MEMORY_AUDIT.md` §5.

Note the per-worker gradient is pure duplication: 24 x 33 MiB = ~792 MiB of the same array. A
host-RAM clamp analogous to `max_batch_for_vram` was previously deferred as "no consumer"; at 8.2 GiB
for an 8.7M-param model that verdict should be revisited — the number is now large enough to bind.

Guards that landed with this: `memplan::ScratchTerms` (itemized, so a gap is attributable to a buffer
group), `Dims::logits_chunks` (ONE source of truth — the allocator and predictor had diverged by
1.35 GiB), and `[frontend][memplan]` tests that regenerate the census figures from the formula and WARN
with the `-DSUB0_LOGITS_MAX_CHUNKS` value a shape needs.

Related: [[memplan-vram-prediction-gap-at-scale]], [[gpu-batch-undertuned-vram-headroom]],
[[three-pillar-shootout-policy]], [[cpu-threading-model]], [[host-cpu-arrow-lake-hx]]
