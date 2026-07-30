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
  memplan train_resident:    8227 MiB
  unexplained:               ~280 MiB
```

(The `memplan` figure read `6859` and the gap `~1650` in earlier revisions of this document. That was a
hand-arithmetic error — it evaluated the logits chunk count as 33 where `logits_n_chunks` clamps it to 8,
under-counting the largest single term by ~1354 MiB. See §3a-bis. The *spill* was real either way: 8227
MiB does not fit an 8151 MiB card.)

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

That leaves a **~280 MiB** gap (not the ~1650 MiB an earlier revision claimed — see §3a-bis for why that
figure was wrong and how it survived two checks). It has already been mis-attributed twice, first to other
processes' graphics allocations and then to eval scratch, so the remaining candidates are stated as
candidates, not conclusions: the CUDA primary context, the loaded module/fatbin (this `.cu` is ~5.8K lines
of heavily templated kernels, so it is not small), the cuBLAS workspace, two captured CUDA graphs, and
allocator granularity across ~80 allocations. ~280 MiB is the right order of magnitude for that set, which
is corroboration but not attribution. **Do not guess a third time — run 3b's checkpointed
`cudaMemGetInfo` sampling**, which attributes each of those exactly and costs one short run. The urgency is
lower than it looked, though: at 280 MiB the gap sits comfortably inside `kVramHeadroomMB = 512`, so this
is now an accounting task rather than a correctness one.

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

### 3a-bis. MEASURED: the runtime overhead is ~280 MiB, and `kVramHeadroomMB = 512` is ADEQUATE

> **This section previously claimed the overhead was a fixed ~1650 MiB and that `kVramHeadroomMB` was
> "wrong by a factor of ~3.2". Both claims were false, and acting on them would have crippled the batch
> for no reason.** The retraction is kept in place rather than deleted because the *way* it was wrong is
> the most transferable thing in this document — see the post-mortem at the end of this section.

Two independent shapes, measured on real runs, against `memplan train_resident` computed correctly:

| arm | shape | memplan predicted | measured total | delta |
|---|---|---|---|---|
| A | 16 exec, no depth, 9.66M | 7219 MiB | 7502 MiB | **+283** |
| D | 16 exec, depth stride 4, 7.23M | 8227 MiB | 8507 MiB | **+280** |

The delta is genuinely near-constant across a 1 GiB difference in modelled scratch — that finding
survives. What changes is its magnitude: **~280 MiB, not ~1650**. That is the right scale for CUDA
context + cuBLAS workspace + allocator granularity, and it no longer needs the eval-path forward scratch
to explain it (which is just as well, since that scratch is not resident during training at all — the
training step passes `full=false` and the periodic eval is CPU-side; see *Host offload* above).

So `kVramHeadroomMB = 512` comfortably covers the measured overhead with ~230 MiB of genuine slack. **It
needed no change.** The fit check `memplan(batch) <= free_vram - 512` is sound, and `free_vram` is
already measured post-context-creation (`sub0_cuda_free_vram_mb` forces the context with
`cudaFree(nullptr)` before sampling), so the context cost is subtracted twice-over conservatively rather
than being missed.

Consequence for the arms, on this 8151 MiB card (~7891 MiB usable):

```
  B loop10x6         7177 + 280 = 7457   ~434 MiB spare  [GENERATED by `sub0llm memplan`]
  A deep16           7219 + 283 = 7502   ~390 MiB spare -- CONFIRMED FINE (full speed, 464 MiB free)
  D loop+depth4      8227 + 280 = 8507   OVER by ~620 MiB -- REAL spill of ~644 MiB, 2.7x slowdown
  C shallow10        7021 + 280 = 7301   ~590 MiB spare  [GENERATED, but at its baked batch 512]
  E flat+depth4      7933 + 280 = 8213   ~322 MiB OVER   [GENERATED, at its baked batch 512]
```

**Arm E does not fit at its build dir's baked `DEFAULT_GPU_BATCH=512`** — 8213 MiB against ~7891 usable.
It must be run with an explicit `--batch 448` (which every other arm uses anyway, so this is required for
comparability regardless); at 448 it lands ~7236 MiB, comfortable. This is the first thing the new
no-clamp guard would have caught rather than spilled, found by prediction before any GPU time was spent.

Note the baked default is **per build dir** and is 512 for C and E but 448 for A/B/D, so trusting it
silently reintroduces the batch confound the sweep already hit once. Arm C additionally has two model
dirs one character apart: `ls_C_shallow10_s1` (trained at 512, confounded) and `ls_C_shallow10_s1b`
(trained at 448, the clean one). `load_checkpoint` restores the batch from the checkpoint and OVERRIDES
`--batch`, so resuming the wrong dir cannot be rescued from the command line.

Arm D's ~620 MiB over-commit matches the ~644 MiB actually migrated (718 measured minus the ~74 MiB
baseline of 3d), so the model is trustworthy for planning. With `-DSUB0_LOGITS_MAX_CHUNKS=32` arm D's
logits term drops 1806 → 452 MiB, putting it at ~7153 MiB — ~740 MiB inside the card.

The remaining rows are deliberately blank. Their old values came from the same hand-arithmetic that
produced the retracted table, and "correct them by adding the delta" would repeat the mistake in the other
direction. Regenerate each with `sub0llm memplan` in that arm's build dir, which reads the itemized terms
from `memplan::train_scratch_terms` directly.

Arm B's row above is the first one generated that way, and it **confirms the itemization against the
original audit term by term** — every term the hand census got right, it got right, and only the clamped
one was wrong:

```
  per-execution checkpoints    2716 MiB (38.5%)  x16 executions     audit said 2716  ✓
  final block + singles        1121 MiB (15.9%)                     audit said 1121  ✓
  logits [chunk_rows,V]        1806 MiB (25.6%)  8 chunks            audit said  438  ✗ (4.1x low)
  backward gradients           1289 MiB (18.3%)                     audit said 1289  ✓
  qk-norm pre-stash             126 MiB ( 1.8%)                     audit said  126  ✓
  NOTE: chunk count CLAMPED 4x (wants 33, capped at 8). -DSUB0_LOGITS_MAX_CHUNKS=33 saves 1368 MiB.
```

`5809 + 1368 = 7177` exactly — the whole discrepancy was the one clamped term, which is why the old
delta looked like a clean constant. The `NOTE` line is emitted by the tool, so the next shape whose cap
truncates says so before any GPU time is spent on it.

#### Post-mortem: how a 4x error survived two independent checks

The old table's "predicted" column was hand-computed from memplan's terms, and the hand computation
evaluated `ceil(vocab/d_ff)` as 33 while `logits_n_chunks` clamps it to `[1, LOGITS_MAX_CHUNKS]` = 8. That
under-counted the largest single scratch term by ~1354 MiB. Applying the *same* error to both shapes
inflated both deltas by the *same* amount — which is exactly why they agreed to 3 MiB and why that
agreement read as corroboration. Two checks with one shared error are one check.

The tell was recorded at the time and not chased: the original section closed with "the agreement to
~3 MiB is better than that method deserves." A suspiciously good result from a method you have just
described as unreliable is evidence about the method, not about the result.

Three things changed so this class of error cannot recur silently:

1. **`train_scratch_bytes` is itemized** (`memplan::ScratchTerms`). The total was previously all memplan
   exposed, which is *why* the census had to be recomputed by hand. Terms are now the reviewable unit.
2. **The census is generated, not transcribed.** Any figure in this document that a formula can produce
   should be produced by the formula. The clamp truncation, the logits MiB, and its share of scratch are
   now computed by `[frontend][memplan]` tests, which print `4.125x truncation, 1805.56 MiB, 22.27% of
   scratch` and name the `-DSUB0_LOGITS_MAX_CHUNKS` value the shape needs.
3. **The clamp is loud when it binds.** A shape whose derived chunk count is truncated more than 2x now
   raises a test WARN naming the remedy, rather than being discoverable only as a spilled run.

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

## 5. Architecture: memplan is a DEVICE census wearing a backend-agnostic name

`memplan.hpp` opens by describing itself as a "pure, backend-agnostic device-memory footprint model". The
*purity* claim is true and valuable — it includes no CUDA headers and takes explicit `Dims`, which is what
lets the configurator reason about memory before a device exists. The *backend-agnostic* claim is not:
every term in `persistent_bytes` / `train_scratch_terms` / `fwd_scratch_bytes` mirrors one specific
allocator, `backend_cuda.cu`'s. Under `ComputeBackend::Cpu` the numbers it produces describe memory that
is never allocated; under `Hybrid` they describe one half of the picture.

This is not a hypothetical gap, because **a second footprint model already exists** — `calc_act_cap()` at
`backend_cpu.cpp:84`. It does structurally the same job (enumerate per-execution activation terms from the
config) and the two share no vocabulary at all:

| | GPU — `memplan.hpp` | CPU — `backend_cpu.cpp:84-112` |
|---|---|---|
| form | `constexpr` of a **`Dims` parameter** | `consteval` of the **baked** constants |
| can answer "what if?" | yes — this is why the configurator can clamp a batch | **no** — it only ever describes this build |
| unit | bytes, itemized by buffer group | floats, one scalar (`ACT_CAP`) |
| scope | whole training step at a given batch | one window (`T`), one worker |
| unmodelled slack | `kVramHeadroomMB = 512`, external and named | `* 3 / 2 + 8192`, baked into the constant |
| validated against reality | yes — `sub0_cuda_train_footprint`, tolerance-gated in CI | **no** |
| concurrency | N/A (one device arena) | **absent** — `ACT_CAP` is *per worker thread* |

Three consequences worth acting on, in descending order of how wrong the current answer is:

1. **The CPU footprint report is thread-blind.** `print_config()` (`backend_cpu.cpp:1597`) prints
   `2 * ACT_CAP * sizeof(float)` as "acts", which is *one* worker's value and grad arenas. Real host
   activation memory is `threads × that` — an 8× under-report on this host's 8 P-cores. The params figure
   (`4 * PARAM_FLOATS`) is correct and shared, so the two halves of one line disagree about what they
   are counting.
2. **The CPU side has no host-RAM budget check.** `apply_autosize` takes a `vram_budget` and there is no
   host equivalent, so a CPU or Hybrid run has nothing analogous to `max_batch_for_vram` — no clamp, no
   configure-time hard failure. The GPU path's guard exists because that path OOMs loudly; the CPU path
   is protected only by the `act_used + n > ACT_CAP` abort at `backend_cpu.cpp:167`, which fires at
   *runtime*, after the shape is already baked.
3. **Hybrid is modelled nowhere.** Hybrid source-routes windows between backends, so host and device each
   hold a working set sized by their *share* of the batch. Neither model takes a split fraction, so
   neither total is right in Hybrid mode, and the fit question — two simultaneous constraints — is not
   asked at all.

### 5a. The proposed shape

The seam that already exists is correct and should be made explicit rather than replaced: `Dims`,
`param_floats`, `d_kv`, `qkv_stride` are genuinely backend-independent *shape* math. What needs separating
is shape math from a per-backend allocation census:

```
sub0::memplan
  Dims, param_floats, d_kv, qkv_stride, qk_pre_stride    // shape -- shared, unchanged
  struct Terms { ... total(); }                           // ONE itemized shape, every backend
  namespace device {  persistent_bytes, train_scratch_terms, fwd_scratch_bytes, max_batch_for_vram  }
  namespace host   {  param_bytes, worker_bytes, train_resident_bytes(dims, batch, threads)         }
```

Two properties make this worth doing rather than just tidier:

* **Consumers become backend-agnostic.** `sub0llm memplan` currently reports a GPU plan on a CPU-only
  build; with a common `Terms` shape it reports whichever census matches `COMPUTE_MODE`, and Hybrid
  reports both plus the split.
* **The CPU census inherits the GPU census's honesty mechanism.** `train_scratch_terms` is trustworthy
  because `sub0_cuda_train_footprint` measures the real delta and CI fails on drift. A `Dims`-parameterized
  host model can be checked the same way against the arena high-water mark, which would replace
  `ACT_CAP`'s `* 3 / 2` fudge with counted terms — the same move that made `kVramHeadroomMB` defensible.

### 5b. What is justified now, and what is not

Deliberately staged, because the rename touches `configurator.cpp`, `train_stage.cpp`, `config_util.hpp`,
`backend_cuda.cu` and four test files, and the LoopSplit sweep currently depends on those build directories
reproducing bit-for-bit:

* **Now, and independently landable:** make the host activation figure thread-aware. It is a wrong number
  being printed today, the fix needs no new abstraction, and it does not touch the GPU path.
* **Next, once the sweep's build dirs are free:** the `device::` / `host::` split. Pure namespacing plus a
  `Terms`-shaped host census; no formula changes on the GPU side, so the footprint parity test is the
  regression gate for the whole move.
* **Not yet — no consumer:** a host-RAM `max_batch_for_host` clamp, and the Hybrid split model. Both are
  real gaps, but adding either before something reads it would be a knob nothing consumes (`AGENTS.md`
  §8). The trigger for the host clamp is a CPU/Hybrid training run that actually hits the `ACT_CAP` abort
  or the d768-class host limit; the trigger for the Hybrid model is a Hybrid run at a batch large enough
  for the split to matter.

The honest summary: the GPU half of this is well modelled and now itemized and tested; the CPU half is a
build-fixed scalar with a fudge factor and no validation; the Hybrid half does not exist. That ordering
matches where the GPU-hours have gone, so it is a reasonable place to have arrived — but the naming should
stop implying otherwise, and it now does not.
