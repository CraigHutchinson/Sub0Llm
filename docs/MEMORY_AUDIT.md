# Device memory: a full allocation census, and how to keep it honest

**Principle: no unexplained byte.** The device footprint should be predictable from the config alone, and
any gap between "what memplan says" and "what the process holds" is a defect until explained — the same
discipline `AGENTS.md` §4 applies to test counts, applied to VRAM. This document exists because a real
run silently lost 2.7× throughput to a spill that no guard noticed.

## 0. The incident that motivated this

Arm D (d192, 16 executions, depth stride 4, batch 448) on an 8151 MiB card:

```
  measured, dGPU, mid-run:   7793 MiB dedicated  +  718 MiB SHARED (system RAM over PCIe)
  per-process:               sub0llm-train holds 7789 of the 7793 -- NOTHING else is on the dGPU
  memplan train_resident:    6859 MiB
  unexplained:              ~1650 MiB
```

Two things were NOT the cause, both checked rather than assumed: other applications (the desktop is on
the iGPU; `Display Active: Disabled` on the dGPU, and per-process counters show only the trainer), and
power/thermal (60 °C; 35 W at 2370/3090 MHz is the signature of memory-stalled SMs, not a power cap).

The failure was silent because **`cudaMalloc` succeeds on WDDM by migrating to shared system memory
instead of failing**. `GpuTrainer::enable` only clamps the batch when `train_reserve` returns nonzero, so
the guard never fired. On Linux this would have OOM'd and self-corrected.

## 1. Allocation census

Every `cudaMalloc` in `backend_cuda.cu`, by owner, with whether `memplan` models it. Test/self-test
harness allocations are excluded (they free immediately and never coexist with training).

| owner | buffers | modelled by |
|---|---|---|
| `upload_params` | `g_dev_params`, `g_dev_grad`, `g_dev_m`, `g_dev_vel`, `g_dev_normsq`, `g_dev_gs` | `persistent_bytes` ✅ |
| `build_qkv_weights` | `g_w1_16`, `g_w2_16`, `g_wg16`, `g_wo16`, `g_wqkv16` (per layer) | `persistent_bytes` ✅ |
| `wqkv_alloc` / `ensure_wqkv_f32` | `g_fwd.wqkv[l]` | `persistent_bytes` ✅ (F32 builds) |
| `train_alloc` | the `g_tr.*` set, incl. depth cache + `v_own` | `train_scratch_bytes` ✅ |
| `fwd_alloc(full=false)` | `g_fwd.dids` | `fwd_dids_bytes` ✅ |
| **`fwd_alloc(full=true)`** | `g_fwd.h/a/qkv/att/proj/fbuf/ff1/gact/ff2/logits` **+ `depth_k/v`** | `fwd_scratch_bytes` — **and it does NOT include the depth cache** ❌ |
| **`ensure_bwd_stats`** | `g_bwd_m`, `g_bwd_invZ`, `g_bwd_dot` — 3 × `batch·H·T` f32 | **unmodelled** ❌ |
| **Muon** | `g_dev_muon_upd/bx/a/aa/ss` | **unmodelled** ❌ |
| **`eval_alloc`** | `g_ev.targets/lengths/active/loss` | **unmodelled** ❌ (small, ~64 KiB) |
| **`ensure_bind_hdr`** | `g_dev_bind`, `g_bind_roles` | **unmodelled** ❌ (small) |
| **decode KV** | `g_kv_k`, `g_kv_v`, `g_decode_state` | **unmodelled** ❌ |
| CUDA runtime | primary context, cuBLAS workspace, graph exec | out of scope for memplan, but **in scope for the budget** |

Sizes at arm D's shape (M = 229,376 training rows):

```
  ensure_bwd_stats : 3 * 448*6*512 * 4 B                       ~=   16 MiB
  decode KV        : 2 * 16*512*96 * 4 B                       ~=    6 MiB   (only once decode runs)
  eval fwd scratch : device_batch caps logits at 512 MiB, so
                     M_eval ~= 7758 rows -> ~578 MiB, of which
                     logits alone is ~488 MiB                  ~=  578 MiB   <-- the big one
```

**CORRECTION (the eval scratch is NOT resident during training).** An earlier revision of this document
claimed the eval-path forward scratch (~578 MiB) was the largest unmodelled item. It is not resident at
all: `sub0_cuda_train_step` and `sub0_cuda_train_reserve` both call `fwd_alloc(batch, /*full=*/false, T)`,
and the training loop's own periodic eval deliberately runs on the CPU
(`sub0::eval::Session cpu_only(/*allow=*/false)` in train_stage.cpp). So `full=true` never fires during a
training run. The remaining unmodelled owners are real but small (`ensure_bwd_stats` ~16 MiB — reserved up
front by `train_reserve` — plus Muon, bindings, eval and decode buffers).

That leaves the ~1650 MiB gap UNATTRIBUTED. It has now been mis-attributed twice (first to other
processes' graphics allocations, then to eval scratch), so the remaining candidates are stated as
candidates, not conclusions: the CUDA primary context, the loaded module/fatbin (this `.cu` is ~5.8K lines
of heavily templated kernels, so it is not small), the cuBLAS workspace, two captured CUDA graphs, and
allocator granularity across ~80 allocations. **Do not guess a third time — run 3b's checkpointed
`cudaMemGetInfo` sampling**, which attributes each of those exactly and costs one short run.

### Host offload: why it does not help here

Asked directly: can non-training allocations move to the CPU? Almost none, because **everything resident
is touched every step**, so offloading buys VRAM at the cost of a PCIe round-trip per step:

| candidate | size at arm D | verdict |
|---|---|---|
| `h_in`/`h_mid` per-execution checkpoints | 2716 MiB | read in every backward; streaming 2.7 GiB per step is seconds |
| optimizer moments `m`/`vel` | 55 MiB | AdamW runs on-device |
| bf16 weight mirrors | 8 MiB | GEMM operands |
| eval forward scratch | — | ALREADY CPU-side |

Offload pays for data touched RARELY, and the only thing that qualified is already there. The productive
direction is therefore to make the resident set smaller (§4), not to relocate it.

## 2. Why cuBLAS is not the explanation

Checked against the cuBLAS documentation rather than assumed
(<https://docs.nvidia.com/cuda/cublas/index.html>):

- Recommended workspace is **"NVIDIA Hopper Architecture (sm90): 32 MiB … Other: 4 MiB"**. This host is
  sm_120, i.e. the 4 MiB bucket — tens of MiB at most, not hundreds.
- `cublasStatus_t cublasSetWorkspace(cublasHandle_t handle, void *workspace, size_t workspaceSizeInBytes)`
  lets us own and account for it. The pointer must be **256-byte aligned**.
- **Gotcha that would silently undo it**: "Calling `cublasSetStream()` resets the workspace back to the
  default pool." Any `cublasSetWorkspace` must therefore run AFTER the stream is bound, and again after
  any later `cublasSetStream`.
- `CUBLAS_WORKSPACE_CONFIG` accepts `:16:8` (documented as possibly limiting performance) or `:4096:8`.
- `cublasLtMatmul` does not require a user workspace.

Conclusion: **do not write our own GEMM** — the user's instinct is right that it is too costly, and the
audit shows cuBLAS is not where the memory went. Owning the workspace via `cublasSetWorkspace` is still
worth doing, but for *accounting* (a known, fixed, ledgered number) rather than for savings.

## 3. Design: make the picture measurable, then keep it honest

### 3a. An allocation ledger

Route every device allocation through one accounted wrapper instead of raw `cudaMalloc`:

```cpp
  void* dev_alloc(const char* name, MemClass cls, size_t bytes);   // records (name, cls, bytes)
  void  dev_free (void* p);                                        // removes it
```

`MemClass` = `Persistent | TrainScratch | FwdScratch | EvalScratch | DecodeKV | Diagnostic`. That gives a
dump-on-demand ledger which can be **diffed against memplan term by term**, so a new buffer added without
a matching model term is caught mechanically rather than by someone remembering. This is the structural
fix: today the correspondence is maintained by comment discipline alone, and the census above shows six
places where it has already drifted.

### 3a-bis. MEASURED: the runtime overhead is a FIXED ~1650 MiB

Two independent shapes, measured on real runs, give the same delta between `memplan train_resident` and
what the process actually holds (dedicated + shared):

| arm | shape | memplan predicted | measured total | delta |
|---|---|---|---|---|
| A | 16 exec, no depth, 9.66M | 5851 MiB | 7502 MiB | **+1651** |
| D | 16 exec, depth stride 4, 7.23M | 6859 MiB | 8507 MiB | **+1648** |

`memplan + 1650 MiB` predicts the real footprint to within ~3 MiB on both. That the delta is CONSTANT
across a 1 GiB difference in modelled scratch is the important part: it is fixed overhead, not something
that scales with the model, which is consistent with its being CUDA context + cuBLAS workspace + the
eval-path forward scratch (itself fixed, since `DEVICE_LOGITS_BUDGET_BYTES` caps it at 512 MiB
regardless of model size) + allocator granularity.

So `kVramHeadroomMB = 512` is not merely "too small" — it is wrong by a factor of ~3.2, and the right
value is measurable rather than guessable. Until 3b lands, **1650 MiB is the empirical constant**, and
the fit-check should be `memplan(batch) + 1650 <= free_vram`.

Consequence for the arms, on this 8151 MiB card (~7891 MiB usable):

```
  C shallow10        4790 + 1650 = 6440   comfortable
  B loop10x6         5809 + 1650 = 7459   ~430 MiB spare
  A deep16           5851 + 1650 = 7501   ~390 MiB spare -- CONFIRMED FINE (full speed, 464 MiB free)
  D loop+depth4      6859 + 1650 = 8509   OVER by ~620 MiB -- REAL spill of ~644 MiB, 2.7x slowdown
  E flat+depth4      5625 + 1650 = 7275   comfortable
```

Only arm D exceeds the card, and the model's ~620 MiB over-commit matches the ~644 MiB actually migrated
(718 measured minus the ~74 MiB baseline of 3d). Arm A, at ~390 MiB spare, runs at full speed — so the
model is usable for planning once the constant is applied, and the marginal-looking arms are genuinely
fine rather than merely lucky.

Caveat on the constant's derivation: 1650 was taken from arm D and then checked against arm A, and both
"predicted" figures are hand-computed from memplan's terms rather than read from
`sub0_cuda_train_predicted_mb`. The agreement to ~3 MiB is better than that method deserves, so treat
1650 as a good working number pending 3b's checkpointed measurement — not as a validated constant.

### 3b. Measure the runtime overhead precisely, don't guess a constant

`cudaMemGetInfo` sampled at defined checkpoints attributes the non-ours bytes exactly:

```
  t0 before any CUDA call        -> baseline
  t1 after context creation      -> CUDA context cost
  t2 after cublasCreate          -> handle cost
  t3 after the first GEMM        -> workspace materialization
  t4 after each alloc phase      -> should equal the ledger delta EXACTLY
  t5 after graph capture         -> graph exec cost
```

Any t4 step where `free-memory delta != ledger delta` is allocator granularity/fragmentation, and it is
then a measured number rather than a mystery. `kVramHeadroomMB = 512` should become **this measurement**,
taken once at init, not a hardcoded guess — it is currently ~3× too small.

Note the existing `sub0_cuda_train_footprint` measures the free-memory delta around `train_alloc` ONLY,
which by construction excludes the context, cuBLAS and the eval scratch. That is why it can look accurate
while the process total is ~1.65 GiB higher: **a scope gap, not a modelling error**. It also explains why
`memplan-vram-prediction-gap-at-scale` recorded an OVER-estimate at d768 while this run shows an
under-estimate — different measurement boundaries.

### 3c. Budget against the right number, before reserving

```
  needed  = memplan train_resident + eval fwd scratch + measured runtime overhead
  if (needed > free_vram) -> warn loudly (and clamp when the batch is not pinned)
```

Check this BEFORE `train_reserve`, because on WDDM the allocation will succeed regardless. A pinned batch
(A/B arm comparability) must warn rather than clamp — silently changing the batch would invalidate the
comparison, which is a worse failure than a slow run.

### 3d. Detect the spill itself — and the baseline that makes "shared > 0" the WRONG test

`\GPU Adapter Memory(*)\Shared Usage` (and its per-process sibling `\GPU Process Memory(*)`) reports
shared-memory bytes, but **non-zero shared usage is NORMAL and is not a spill**. Measured: arm A runs with
74 MiB shared while holding 464 MiB of dedicated FREE, at 111,642-111,693 tok/s against its own clean
baseline of 111,948-111,976 — **within 0.3%, i.e. no spill at all**. That ~74 MiB is pinned/staging and
WDDM command buffers, present as soon as any CUDA process runs.

Using `shared > 0` as the alarm therefore fires on every healthy run. The signal is:

```
  spill  <=>  dedicated_free ~= 0  AND  shared >> ~74 MiB baseline
```

Arm D fits: 7793 MiB dedicated of ~7891 usable (free ~0) with 718 MiB shared, i.e. ~644 MiB of REAL
migration once the baseline is subtracted — and a 2.7x throughput collapse to confirm it. Arm A fits
neither clause.

**Throughput against a known-clean baseline is the ground truth**; the counters are supporting evidence.
An alarm should require the memory condition AND corroborate with a rate drop before naming it a spill.

Complementary driver-side control: **NVIDIA Control Panel → Manage 3D settings → Program Settings → CUDA – Sysmem Fallback
Policy → "Prefer No Sysmem Fallback"** makes the allocation FAIL instead of migrating, which the existing
clamp path already handles. Requires driver 536+; this host is on 596.36.

## 4. Ranked levers, once the picture is honest

### 4a. The chunking lever, and why a chunk COUNT is the wrong generalization

`logits_n_chunks` returns a COUNT (`ceil(vocab/d_ff)`, clamped), so `chunk_rows = total_rows / count`
scales with `total_rows` — i.e. **with batch**. Doubling the batch doubles the logits buffer at the same
count, which is precisely what a memory lever is supposed to prevent. The count-based form therefore does
not bound the buffer at all; it only bounds its *ratio* to the other per-row activations, and only at the
`d_ff` the clamp was calibrated for.

**The generalized target is a chunk SIZE, equivalently a byte budget:**

```
  chunks = ceil(total_rows * vocab * 4 / LOGITS_TARGET_BYTES)     // batch-invariant
```

That expresses the actual intent ("keep the logits buffer under X MiB") and needs no per-shape clamp
calibration. Not yet implemented — the immediate need was arm D, and changing the derivation changes every
build's memory profile — but `tests/cuda_tests.cpp` now pins the current form's coverage and bounds AND
records the batch-scaling gap as an assertion that should INVERT when the byte-budget form lands, so the
gap is visible rather than assumed away.

Smoke coverage added alongside (the chunk loop's "mathematically identical" claim had never been tested):
same backward at chunk counts 1/2/4/7/8/13/32 with gradient cosine > 0.9999 against the unchunked
reference, where 7 and 13 deliberately do NOT divide the row count so the RAGGED last chunk and its
`row_offset` arithmetic are exercised — the path where an off-by-one corrupts only the final partial chunk
and stays invisible to an evenly-dividing test.

| lever | saving at arm D's shape | note |
|---|---|---|
| **`logits_chunk_rows`: 33 → 132 chunks** | **~328 MiB** | **the clean one.** The chunk loop already exists and is documented as mathematically identical to the unchunked path, so results do not move — only launch count rises, and each chunk still does a large GEMM so it amortises. Needs the derived ratio to become a knob. |
| depth f32 gradient accumulators → bf16 | ~352 MiB | slot 0 takes contributions from up to 15 executions; parity is cos 0.99992 today, so measure before spending it |
| depth stride 4 → 8 | ~525 MiB | changes the mechanism under test — not a free lever |
| eval scratch | 0 | already not resident (see the correction above) |

`logits_chunk_rows` + bf16 accumulators together (~680 MiB) would take arm D from ~620 MiB over the card
to comfortably inside it — and the larger half of that costs no precision.

The per-execution checkpoints (2716 MiB, 40%) are inherent to a 16-execution arm and are not a lever
without changing the experiment.
