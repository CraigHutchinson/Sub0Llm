# Qwen4-preview memory/data orchestration — cross-tier placement design

Status: **v2 — LIVING DOCUMENT, RESEARCH + DESIGN ONLY.** No engine code lands this pass. Every number
below is tagged with how it was obtained (a real fetch/measurement this pass, a real fetch from a prior
pass, an arithmetic derivation from a verified quantity, or an estimate) and how confident it is,
following `docs/QWEN4_PREVIEW_REFERENCE.md`'s own facts-table discipline. Section 6 names exactly what to
re-measure once each mechanism lands, and this doc should be revised (not superseded by a new one) as
those measurements come in — bump the revision log below each time.

**Revision log**:
- v1 (2026-09-02): initial version. No Sub0Llm mechanism code exists yet for GDN/QSA/Gated-Residual at
  Qwen4 scale (GDN has CPU forward+backward+CUDA-forward at *this project's own* toy training scale,
  gated against small real-weight fixtures — see `docs/GATED_DELTANET.md` — but nothing has run at
  Qwen4's real 2560-wide, 512-expert scale). Every §2/§3 byte number for GDN/QSA/MoE is therefore a
  hand-derived estimate from the real, verified `config.json` fields, not a measured footprint.
- **v2 (2026-09-03): §6c item 5 discharged.** Gated Residual (`docs/GATED_RESIDUAL.md`), MoE
  (`docs/MOE.md`) and QSA (`docs/QSA.md`) have all landed real, fixture-gated CPU forward
  implementations on `main` since v1, each with a real `*_param_delta()` closed form in `layout.hpp` and
  real `*_math.hpp` scratch/state sizing functions. **Every §2 and §3d number for GR, MoE, QSA and GDN
  has been replaced with a value computed by evaluating THIS PROJECT'S OWN `make_param_layout()` /
  `PARAM_LAYOUT` / `gr_param_delta()` / `moe_param_delta()` / `qsa_param_delta()` / `gdn::state_floats()`
  / `qsa::scratch_floats()` machinery at Qwen4's real dims** — not by re-doing v1's hand arithmetic. See
  §6a for exactly what that measurement was and what it is (and is not) evidence of. Four v1 estimates
  moved materially: **Gated Residual is 0.64B params, not "single-digit millions per layer, negligible"**
  (§2d); **QSA is 46% larger per layer than v1 assumed** because v1 never knew about `q_proj`'s doubled
  query|gate width (§2c); the **backbone total is 4.62B, above v1's whole 3.9-4.1B estimated range**
  (§2f); and v1's claimed "two independent routes agree" reconciliation against the published 125B is
  **retracted** — it was an artifact of those two undercounts (§2f). A new §2h records the single most
  consequential finding for WP4: this engine's current `RunConfig` axes **cannot express** the real GDN
  and QSA head geometry, and the resulting shape gap is 1.68B params. §6c items 1-4 and 6 are
  **untouched and still open** — they need real hardware benchmarking this pass did not do.

**Read first, not re-derived here**: `docs/QWEN4_PREVIEW_REFERENCE.md` (architecture facts),
`docs/GATED_DELTANET.md` (GDN math + this project's own arena/checkpoint design for it),
`docs/QWEN4_DEPLOYMENT_FEASIBILITY.md` (real GGUF quant-tier byte totals + MoE-offload prior art —
cited, not re-derived, throughout §1-§3), `docs/SUB0FIRN_SPEC.md` + `docs/NGRAM_TABLE_TIERED_STORAGE.md`
(the n-gram table's own tiered-storage design — cited for the seam in §3, out of scope for budgeting per
explicit user instruction), `include/sub0/memplan.hpp` (the existing `Dims`/`persistent_bytes`/
`train_scratch_bytes`/`fwd_dids_bytes` accounting this doc extends rather than parallels),
`include/sub0/device_backend.hpp` + `docs/BACKENDS.md` (the existing device-neutral seam + capability-bit
model this doc's placement policy builds on).

---

## 1. The full tier ladder, named and budgeted

Five tiers, GPU VRAM being the fastest/smallest and network/HTTP-range the slowest/largest. Every
capacity number below was **measured on this machine this session** (2026-09-02) via `nvidia-smi`,
`Get-CimInstance Win32_OperatingSystem`, `Get-Volume`, and `Get-PhysicalDisk` — not assumed from the
task brief's rounded figures, per the project's own `memplan-overhead-280-not-1650-and-host-is-bigger`
lesson (host/OS overhead has been under-reported before; verify, don't assume).

### 1a. GPU VRAM

| Quantity | Value | How obtained | Confidence |
|---|---:|---|---|
| Total VRAM | 8,151 MiB (8.55 GB decimal / 7.96 GiB) | `nvidia-smi --query-gpu=memory.total`, this session | **High — measured** |
| Free at idle (no CUDA context open) | 7,891 MiB | `nvidia-smi --query-gpu=memory.free`, this session | **High — measured**, but see caveat below |
| Realistic usable headroom | **~7.5-7.7 GiB**, not 7.96 GiB | derived: idle free (7891 MiB) already shows ~260 MiB reserved by the driver/OS compositor before any workload opens a context; opening a CUDA context itself costs additional MiB (typically 150-400 MiB depending on driver/CUDA version, not independently measured this pass) | **Medium** — the 260 MiB idle gap is measured; the CUDA-context overhead on top of it is a class-level estimate, not measured on this exact driver this pass |
| PCIe link (host↔device) | **Gen5, x8 width** (not x16) | `nvidia-smi --query-gpu=pcie.link.gen.current,pcie.link.width.current`, this session — this is the laptop's actual electrical link, not a power-saving downclock (max reported width is also 8, i.e. the slot itself is x8) | **High — measured** |
| PCIe theoretical bandwidth | ≈31.5 GB/s per direction (≈63 GB/s full-duplex aggregate) | derived: PCIe 5.0 encodes at 32 GT/s per lane with ~1.5% overhead ⇒ ≈3.94 GB/s/lane/direction × 8 lanes | **Medium** — spec arithmetic from a measured link width, not a measured transfer rate |
| PCIe real sustained bandwidth | not measured this pass; expect 60-85% of theoretical for pinned-buffer transfers (≈19-27 GB/s/direction), lower for small/unpinned transfers | class-level expectation from general PCIe/CUDA transfer literature | **Low** — no `cudaMemcpy` benchmark run on this exact box this pass; flagged in §6 as a number to actually measure before it's load-bearing for any latency claim |

**Consequence stated plainly, used repeatedly below**: this project already measured (via
`sub0_cuda_train_footprint`, `memplan-overhead...`) that a *training* CUDA context on this same class of
hardware costs real, non-trivial device memory before a single model weight is uploaded (~280 MiB at a
small training shape). An *inference-only* context is unmeasured but not assumed free — treat ~7.5 GiB,
not 7.96 GiB, as this tier's honest usable ceiling for the rest of this document.

### 1b. CPU RAM

| Quantity | Value | How obtained | Confidence |
|---|---:|---|---|
| Total physical RAM | 68,112,736,256 bytes = 63.44 GiB (68.11 GB decimal) | `Get-CimInstance Win32_ComputerSystem`, this session | **High — measured** |
| Free RAM, measured live | 45,888,536 KB = 43.76 GiB | `Get-CimInstance Win32_OperatingSystem`, this session, **while this Claude Code session, its IDE/terminal host, and normal desktop background processes were running** | **High — measured, but NOT an idle baseline** — see caveat |
| Realistic usable headroom for a future model-loading process | **~40-45 GiB**, not 63.44 GiB | the live free-RAM figure above already reflects today's real, non-exotic background load (a dev desktop with an editor, a terminal, a browser-class background footprint) — closer to what an actual inference session would compete against than a theoretical "nothing else running" idle number would be, per this project's own `memplan-overhead-280-not-1650-and-host-is-bigger` lesson that host overhead was previously under-reported 27x by assuming a clean baseline | **Medium** — a real live measurement, but a single point-in-time snapshot, not a controlled idle-vs-loaded comparison; will vary run to run |

**This is the single most important corrective this section makes relative to the task brief's rounded
"63GB RAM" framing**: the realistic planning number for "how much RAM can a Qwen4-preview inference
process actually claim" is **~40-45 GiB, not 63GB** — an 18-23 GiB gap, i.e. this machine's own everyday
background load already claims what would otherwise look like free headroom. Every RAM-residency
comparison in §2/§3 below uses this corrected figure, not the raw total.

### 1c. Local disk (D: and C:)

| Quantity | Value | How obtained | Confidence |
|---|---:|---|---|
| D: free space | 627,928,170,496 bytes = 584.75 GiB (627.93 GB decimal) | `Get-Volume`, this session | **High — measured** (close to, but not identical to, the task brief's rounded "589GB" — normal disk-usage drift since that figure was recorded, not a discrepancy worth chasing) |
| C: free space | 169,766,486,016 bytes = 158.11 GiB (169.77 GB decimal) | `Get-Volume`, this session | **High — measured** (the brief's rounded "179GB" has drifted down ~9GB of real usage since — stated honestly, not silently reconciled) |
| Disk hardware | Two NVMe SSDs: `SAMSUNG MZAL81T0HFLB-00BL2` (1TB) and `Predator SSD GM7 M.2` (1TB), both `BusType: NVMe` | `Get-PhysicalDisk`, this session | **High — measured** (drive presence/bus type); **Low** for either drive's sequential/random-read numbers below (not independently re-verified against a spec sheet this pass) |
| Typical NVMe-class sequential read | ~3.5-7 GB/s (Gen4-class NVMe, which both drives' model families are commonly rated at) | general spec-sheet class knowledge, not fetched/re-verified against either drive's actual datasheet this pass | **Low** — class-level, not device-specific, and not measured on this machine |
| Typical NVMe-class random 4KB read latency | tens of µs at queue depth 1 | same class-level source | **Low**, but this figure is the same one `docs/NGRAM_TABLE_TIERED_STORAGE.md` §3 already relies on for its own disk-tier row-read estimate — reused here for consistency, not independently re-derived |

**No actual disk-throughput benchmark (e.g. a real sequential-read timing on this machine) was run this
pass** — named explicitly in §6 as a real gap, not glossed over. 589GB(D:)+158GB(C:) combined comfortably
covers even the largest GGUF quant tier from `docs/QWEN4_DEPLOYMENT_FEASIBILITY.md` §1b (354GB at BF16)
if anyone ever chose to download one — disk *capacity* is not a binding constraint anywhere in this
document; disk *bandwidth/latency* under real random-access load from a paging expert pool is the
open question, not disk space.

### 1d. Network / HTTP-range (tier below disk)

This project has **already validated** surgical HTTP-Range extraction against the real
`Qwen/Qwen3.8-Flash-Next` repository twice — `docs/QWEN4_PREVIEW_REFERENCE.md` Stage 1 (real GDN-layer
weights + n-gram rows, ~116MB total transferred, zero full-shard downloads) and
`docs/QWEN4_DEPLOYMENT_FEASIBILITY.md` §1 (real GGUF header probes across every quant tier, ~1.3MB total
transferred). Both are cited here as this tier's own working, already-proven access mechanism — not
re-implemented or re-verified in this pass.

| Quantity | Value | How obtained | Confidence |
|---|---:|---|---|
| Feasibility of exact-byte-range fetch against the real repo | **Proven working**, twice, on two different file formats (safetensors, GGUF) | direct reuse of prior verified work | **High** |
| Per-request latency | RTT-dominated (tens of ms per HTTP request), not payload-size-dominated for a row-sized (hundreds-of-bytes to low-KB) fetch | stated in `docs/NGRAM_TABLE_TIERED_STORAGE.md` §3's own ladder table, reused here | **Medium** — a reasoned characterization of HTTP-over-TLS request overhead, not a measured RTT number against Hugging Face's own edge from this machine |
| Sustained throughput (MB/s) for a *bulk* extraction | **Not measured, either prior pass or this one** | neither prior extraction recorded a wall-clock duration, only a total byte count and that it worked | **None — genuine gap**, named explicitly in §6. Both prior extractions were volume-light (116MB, 1.3MB) specifically because they were surgical single-tensor/single-header fetches, not sustained transfers; nothing about their success measures what a real page's worth of small requests (e.g. resolving a whole training window's n-gram rows, or one layer's worth of hot experts) would cost in wall-clock time |

**This tier exists in this document's ladder specifically as the seam behind the (out-of-scope) n-gram
table and, in principle, behind an on-demand expert-weight fetch if a local mirror is ever chosen not to
be downloaded** — it is not proposed as a placement target for this document's own budgeted components
(§2/§3), all of which assume a local — RAM, disk, or GPU — copy once acquired.

---

## 2. Per-component budget, computed from real config

All parameter counts below are **freshly derived this pass** by multiplying real fields fetched directly
from `https://huggingface.co/Qwen/Qwen3.8-Flash-Next/raw/main/config.json` (HTTP fetch, this session —
not re-summarized from a rendered page, per `AGENTS.md` §5/§9 and the exact lesson
`docs/QWEN4_DEPLOYMENT_FEASIBILITY.md` §1a already recorded about rendered-page summarizers losing
numeric fields) against the real module constructions already quoted verbatim in
`docs/QWEN4_PREVIEW_REFERENCE.md`/`docs/GATED_DELTANET.md`. Two config fields fetched fresh this pass,
not previously recorded in either doc:

```
moe_intermediate_size:            640
shared_expert_intermediate_size:  640   (same width as a routed expert — the shared expert is not narrower)
vocab_size:                       248320
tie_word_embeddings:              false  (confirms token embedding and lm_head are SEPARATE weight matrices)
```

No `first_k_dense_replace`/`mlp_only_layers`/`decoder_sparse_step`-style field exists in `config.json`
(checked explicitly this pass) — there is no per-layer dense-FFN exception the way some other MoE
configs carry; **every one of the 48 decoder layers (GDN and QSA alike) runs the MoE FFN block**. This is
an absence-of-a-field inference, not a directly-quoted "yes, every layer" statement — tagged **Medium-High**
confidence accordingly, not treated as certain as a directly-quoted field.

### 2a. MoE expert pool — the dominant term by a wide margin

Per-expert FFN (standard 3-matrix SwiGLU expert: `gate_proj`+`up_proj`+`down_proj`, no bias, matching
this family's convention elsewhere): `3 × hidden_size × moe_intermediate_size = 3 × 2560 × 640 =
4,915,200 params/expert`. The **shared** expert (always active, not one of the 512) has the identical
shape (`shared_expert_intermediate_size` == `moe_intermediate_size` == 640), so it costs the same
4,915,200 params.

| Quantity | Value | Confidence |
|---|---:|---|
| Params per expert | 4,915,200 (≈4.92M) | **High — direct multiplication of two real `config.json` fields** |
| Routed-pool params per layer (512 experts) | 2,516,582,400 (≈2.517B) | **High** |
| + shared expert per layer | 4,915,200 | **High** |
| Total MoE-FFN params per layer | ≈2.5215B | **High** |
| **Total MoE-FFN params, all 48 layers** | **≈121.03B** | **High** (arithmetic); consistent with, and independently corroborates, `docs/QWEN4_PREVIEW_REFERENCE.md`'s already-verified "125B total" figure — 125B − 121.03B ≈ 3.97B left for everything else (embeddings, lm_head, GDN/QSA mixer weights across 48 layers), which §2c below independently re-derives as ≈3.77-4.0B — the two routes agree to within the Gated-Residual estimate's own stated uncertainty |
| **Activated MoE-FFN params per token per layer** (top-10 + 1 shared = 11 of 513 slots) | 54,067,200 (≈54.07M) | **High** |
| **Activated MoE-FFN params per token, all 48 layers** | ≈2.595B | **High** |

**v2 correction (2026-09-03) — measured against `layout.hpp`'s own `moe_param_delta()` + the real
`PARAM_LAYOUT` `MoeRouter`/`MoeGate`/`MoeUp`/`MoeDown`/`MoeShared*` entries at `D_MODEL=2560, D_FF=640,
NUM_EXPERTS=512, EXPERTS_PER_TOK=10`.** Every routed/shared-expert number above **reproduces exactly** —
v1's arithmetic was right. Two terms v1 omitted entirely, both small in bytes but load-bearing for
placement policy:

| Quantity | v1 | v2 (measured, `PARAM_LAYOUT`) | What changed |
|---|---:|---:|---|
| MoE block params per layer | 2,521,497,600 (routed + shared only) | **2,522,810,880** | `+` `MoeRouter` [2560,512] = 1,310,720 and `MoeSharedGateProj` [2560,1] = 2,560 |
| **All 48 layers, absolute** | ≈121.03B | **121,094,922,240 (≈121.095B)** | same two terms × 48 |
| **Activated per token per layer** | 54,067,200 | **55,380,480** | the router is **not sparse** — see below |
| **Activated per token, 48 layers** | ≈2.595B | **2,658,263,040 (≈2.658B)** | |

**The qualitative finding v1 missed, and the reason this correction matters more than its 1.3M/layer
size suggests**: the router is a full `[hidden_size, num_experts]` matrix whose **every column is read
for every token**, because `Qwen4ExpTextTopKRouter` softmaxes over ALL 512 logits before taking the
top-10 (`docs/MOE.md` §1a, verified verbatim). So `MoeRouter` is **always-resident backbone**, not part
of the pageable expert pool — 1,310,720 params/layer × 48 = **62,914,560 params (240.0 MiB f32 /
120.0 MiB bf16)** that a placement policy must keep hot regardless of how aggressively the 512 experts
themselves are offloaded. §3b's static layer-range split must exempt it explicitly; v1's §3b did not
distinguish it from the expert weights at all.

**Per-expert byte size at plausible precisions** (bf16 is the real checkpoint's native dtype; the other
rows reuse `docs/QWEN4_DEPLOYMENT_FEASIBILITY.md` §1c/§1d's already-verified GGUF byte-per-element
formulas rather than re-deriving them):

| Precision | Bytes/element | Bytes/expert | MiB/expert | Full 512-expert layer pool | Full 48-layer pool (routed only) |
|---|---:|---:|---:|---:|---:|
| fp32 | 4 | 19,660,800 | 18.75 | 9.375 GiB | 450.0 GiB |
| bf16 (native) | 2 | 9,830,400 | 9.375 | 4.688 GiB | 225.0 GiB |
| Q8_0-class | 1.0625 | 5,222,400 | 4.98 | 2.490 GiB | 119.5 GiB |
| IQ4_NL-class | 0.5625 | 2,764,800 | 2.64 | 1.318 GiB | 63.3 GiB |

**This table is the single clearest illustration of the scale mismatch this whole document exists to
manage**: even the most aggressive quant tier this project has already verified real byte totals for
(`IQ4_NL`, §1c above) puts the **full** 48-layer routed-expert pool at ≈63.3 GiB — larger than this
document's own corrected §1b realistic RAM headroom (~40-45 GiB), before backbone, KV cache, or anything
else is added. No precision tier makes the full pool comfortably resident anywhere on this hardware; §3b
names the placement consequence.

### 2b. GDN mixer — per-layer parameter count, cross-checked against a real byte count

Using the real full-scale construction quoted in `docs/GATED_DELTANET.md` §1a (`hidden_size=2560,
num_v_heads=48, num_k_heads=16, head_k_dim=head_v_dim=128, conv_kernel=4`):

```
key_dim = 128 × 16 = 2048        value_dim = 128 × 48 = 6144        conv_dim = 2×2048+6144 = 10240

in_proj_qkv = 2560 × (2×2048+6144)  = 2560 × 10240 = 26,214,400
in_proj_z   = 2560 × 6144                          = 15,728,640
in_proj_b   = 2560 × 48                            =    122,880
in_proj_a   = 2560 × 48                            =    122,880
conv1d      = 10240 × 4 (depthwise, no bias)        =     40,960
dt_bias     = 48
A_log       = 48
norm        = 128                                  (RMSNormGated weight, shared across heads — §1c)
out_proj    = 6144 × 2560                          = 15,728,640
                                                     ------------
                                            total  = 57,958,624 params/GDN layer
```

**Cross-check, not a coincidence**: at bf16 (2 bytes/param), `57,958,624 × 2 = 115,917,248 bytes` — this
is the **exact same byte count** `docs/QWEN4_PREVIEW_REFERENCE.md` Stage 1 already reported as the real,
directly-fetched size of "the complete real weight set for one GDN layer" (`~110.6MB`). This document's
independent arithmetic from `config.json`'s fields reproduces a real, previously-fetched byte count
exactly — the strongest form of cross-validation available without a second live fetch.

| Quantity | Value | Confidence |
|---|---:|---:|
| Params per GDN layer | 57,958,624 | **High — matches a real fetched byte count exactly** |
| Bytes per GDN layer, bf16 | 115,917,248 (110.56 MiB) | **High — this IS the real fetched value, not a derivation** |
| Total, 36 GDN layers, bf16 | 4,173,020,928 B = **3.886 GiB** (v1 said "≈3.98 GiB" — a GiB/GB slip, corrected) | **High** |
| Total, 36 GDN layers, params | **2,086,510,464 (≈2.087B)** | **High** |

**v2 note**: this section needed no substantive correction — v1's GDN arithmetic was already anchored to
a real fetched byte count and stands. The only change is the GiB unit fix above. **But see §2h**: this
2.087B figure is what the REAL model costs, and is *not* what this engine currently builds at these
`RunConfig` axes, because GDN's real head geometry is not expressible today.

### 2c. QSA mixer — per-layer parameter count (attention projections; indexer weights not independently verified)

Using `config.json`'s real fields (`num_attention_heads=24, head_dim=256, num_key_value_heads=2`) and
this family's established no-bias-linear + per-head QK-norm convention:

```
q_proj = 2560 × (24×256) = 2560 × 6144 = 15,728,640
k_proj = 2560 × (2×256)  = 2560 × 512  =  1,310,720
v_proj = 2560 × (2×256)  = 2560 × 512  =  1,310,720
o_proj = 6144 × 2560                   = 15,728,640
q_norm + k_norm (per-head RMSNorm, width = head_dim = 256)  = 512
                                                               ----------
                                                       total = 34,079,232 params/QSA-layer attention proj
```

**The QSA indexer's own weight shapes were not independently fetched this pass** (`docs/
QWEN4_PREVIEW_REFERENCE.md` quotes the indexer's *forward*-path code, not its `__init__`) —
`indexer_n_heads=4, indexer_kv_heads=1, indexer_head_dim=128` implies small additional q/k projection
weights (order of 1-2M params, by the same shape of arithmetic as the main attention projections at
these narrower dims) plus a small `k_layernorm`. **Not claimed as a precise number here** — flagged
explicitly as an open item, tagged **Low** confidence, rather than fabricated to false precision.

| Quantity | Value | Confidence |
|---|---:|---|
| Attention-projection params per QSA layer | 34,079,232 | **High — direct arithmetic from real `config.json` fields** |
| + indexer weights (not independently derived) | +~1-2M, order-of-magnitude only | **Low** — named as a gap, see §6 |
| Bytes per QSA layer, bf16 (attn-proj only) | ≈68.16 MiB | **High** for the attn-proj part |
| Total, 12 QSA layers, bf16 (attn-proj only) | ≈0.80 GiB | **High** for the attn-proj part, an undercount by an unquantified small margin |

**v2 correction (2026-09-03) — the largest single arithmetic correction in this revision.** Everything
above is superseded. `docs/QSA.md` §1b fetched and quoted `Qwen4ExpTextAttention.__init__` in full (v1
had only ever seen the *indexer*'s code, never the attention side) and found a fact v1 could not have
known: **`q_proj` is DOUBLE width** — `Linear(hidden_size, num_attention_heads * head_dim * 2)`, chunked
per head into a `head_dim` query and a `head_dim` **output gate** (`attn_output * sigmoid(gate)` before
`o_proj`). v1's `q_proj = 2560 × 6144` is exactly half the real tensor.

Real per-layer shapes, now exact (each is a real `PARAM_LAYOUT` entry: `QsaQProj`, `QsaGateProj`,
`QsaKProj`, `QsaVProj`, `QsaOProj`, `QsaQNorm`, `QsaKNorm`, `QsaIdxQkProj`, `QsaIdxQNorm`, `QsaIdxKNorm`):

```
q_proj query half   = 2560 × (24×256)      = 15,728,640
q_proj GATE  half   = 2560 × (24×256)      = 15,728,640      <-- v1 missed this entirely
k_proj              = 2560 × (2×256)       =  1,310,720
v_proj              = 2560 × (2×256)       =  1,310,720
o_proj              = (24×256) × 2560      = 15,728,640
q_norm + k_norm     = 2 × 256              =        512
                                             ----------
        attention projections               = 49,807,872   (v1: 34,079,232 -- a 46% undercount)

index_qk_proj       = 2560 × ((4+1)×128)   =  1,638,400
q_layernorm + k_layernorm = 2 × 128        =        256
                                             ----------
        indexer                             =  1,638,656   (v1 estimated "~1-2M" -- CORRECT, vindicated)
                                             ==========
        TOTAL per QSA layer                 = 51,446,528
```

| Quantity | v1 | v2 (measured) |
|---|---:|---:|
| Params per QSA layer (attn + indexer) | 34,079,232 + "~1-2M" | **51,446,528** |
| Bytes per QSA layer, bf16 | ≈68.16 MiB | **98.13 MiB** |
| **Total, 12 QSA layers** | ≈0.409B / ≈0.80 GiB bf16 | **617,358,336 (≈0.617B) / 1.150 GiB bf16** |

**Confidence is now High for the whole layer, including the indexer** — the indexer's `__init__` was the
one "Low" item v1 flagged here, and `docs/QSA.md` §1a resolves it verbatim; its `Dims`-parameterised port
in `include/sub0/qsa_math.hpp` is fixture-gated against the real `Qwen4ExpTextAttention.forward()` to
`1.39e-09`, so the shapes are not merely read but exercised.

### 2d. Gated Residual (hyper-connections) — RESOLVED in v2; v1's estimate was materially wrong

**v1 said** (verbatim, kept here so the correction is auditable rather than silently overwritten): "a
generous order-of-magnitude estimate is **single-digit millions of params per layer** — negligible
(<0.1%) next to the 2.52B/layer MoE pool, and small (a few percent) next to the ~58M/34M GDN/QSA mixer
terms", tagged **Low** confidence pending someone fetching `Qwen4ExpTextGatedResidual.__init__`.

**v2**: `docs/GATED_RESIDUAL.md` §1a fetched and quoted that `__init__` verbatim, and `layout.hpp`'s
`gr_param_delta(n_layers, d_model, hc_count, hc_lowrank)` is the resulting closed form (hand-verified in
`tests/layout_tests.cpp` at two shapes, and realised as four real `PARAM_LAYOUT` entries — `GrHcNorm`,
`GrMixDown`, `GrMixUp`, `GrBlockInject` — whose per-tensor sum was confirmed this pass to equal the
closed form exactly at Qwen4's real axes). Evaluated at the real `hidden_size=2560, hc_count=4,
hc_lowrank=320` ⇒ `WIDE = 4 × 2560 = 10,240`:

```
per GatedResidual instance (with block_inject):
  hc_norm               = WIDE                  = 10,240      ([1, WIDE] -- one gain per (stream,channel))
  input_mix_weight_down = WIDE × hc_lowrank     =  3,276,800
  input_mix_weight_up   = hc_lowrank × WIDE     =  3,276,800
  block_inject_weight   = WIDE × hc_count       =     40,960
                                                  ---------
                                                  6,604,800

per DECODER LAYER: two such instances (attn_hyper_connection + mlp_hyper_connection) = 13,209,600
once, MODEL-LEVEL:  one instance WITHOUT block_inject (use_combine=False)            =  6,563,840
                                                                                      -----------
TOTAL, 48 layers + top:  48 × 13,209,600 + 6,563,840                                 = 640,624,640
```

| Quantity | v1 estimate | v2 (measured, `gr_param_delta`) | Confidence |
|---|---:|---:|---|
| Params per GatedResidual instance | — | **6,604,800** | **High** |
| Params per decoder layer (2 instances) | "single-digit millions" | **13,209,600** | **High** |
| Model-level exit instance (once) | not modelled at all | **6,563,840** | **High** |
| **Total, all 48 layers + exit** | not stated; implied ≈0.05-0.3B (§2f) | **640,624,640 (≈0.641B)** | **High** |
| Bytes, bf16 | — | 1,281,249,280 B = **1.193 GiB** | **High** |

**How wrong v1 was, and in which direction — stated plainly rather than buried**: 13.2M/layer is above
the top of v1's "single-digit millions" band, and the model-level exit instance was not modelled at all.
More importantly the *characterisation* was wrong: **Gated Residual is 13.9% of the backbone
(0.641B of 4.616B, §2f), the third-largest backbone term after GDN and the embedding pair** — not the
"negligible" rounding error v1 called it. It is, for instance, **larger than the entire 12-layer QSA
mixer set (0.617B)**. v1's "<0.1% next to the MoE pool" was true and remains true; it was simply the
wrong comparison, since nothing about the MoE pool's placement is decided by the backbone's size — §3a's
GPU-residency question is, and that is the comparison where 0.64B matters.

**Why it is this large, structurally** (worth stating so a future reader can sanity-check any variant):
the two low-rank projections are `WIDE × hc_lowrank` each, and `WIDE` is `hc_count × hidden_size`. The
`hc_count=4` multiplier applies to BOTH the width and (through `2 × WIDE × hc_lowrank`) the dominant
term, so GR's cost scales as `4 × hidden_size × hc_lowrank × 2 × 2 instances` — at `hc_lowrank=320` that
is `320/2560 = 1/8` of a full `[hidden, hidden]` projection per matrix, but there are 16 such
matrix-equivalents per layer. A "low-rank bottleneck ⇒ cheap" intuition is exactly what produced v1's
underestimate.

### 2e. Embeddings + LM head — untied, both real and separately resident

`vocab_size=248320`, `tie_word_embeddings=false` (both fetched fresh this pass) mean `tok_emb` and
`lm_head` are **two separate** `[248320, 2560]` matrices, not a shared/tied one:

| Quantity | Value | Confidence |
|---|---:|---|
| tok_emb params | 248,320 × 2560 = 635,699,200 | **High** |
| lm_head params (weight only, no bias assumed per family convention) | 635,699,200 | **High** (bias, if present, adds a negligible 248,320) |
| Combined, bf16 | ≈2.37 GiB | **High** |

### 2f. Backbone total (excludes MoE pool, n-gram table, MTP head) — reconciliation

```
36 × 57,958,624 (GDN)          ≈ 2.087 B
12 × 34,079,232 (QSA attn-proj) ≈ 0.409 B    [+ small indexer, not quantified — §2c]
tok_emb + lm_head               ≈ 1.271 B
48 × Gated Residual (unquantified, est. single-digit-M/layer) ≈ 0.05-0.3 B  [Low confidence]
ln_f, misc small norms                        negligible
                                              --------------
                                              ≈ 3.9-4.1 B backbone params
```

This reconciles with `docs/QWEN4_PREVIEW_REFERENCE.md`'s already-verified 125B total minus §2a's
independently-derived 121.03B MoE-FFN total (**125 − 121.03 ≈ 3.97B**), landing inside this section's own
3.9-4.1B range — **two independent derivation routes agreeing to within the least-confident term's own
stated uncertainty (Gated Residual)**, which is the strongest confidence statement this document can
honestly make about the backbone total without fetching the one remaining unquantified module.

**At bf16, the backbone alone is ≈7.6-8.2 GiB** — already at or past this document's own §1a realistic
GPU-VRAM ceiling (~7.5-7.7 GiB usable) **before any activations, KV cache, GDN state, or a single MoE
expert is added**. This is the load-bearing finding for §3's placement policy below.

#### v2 (2026-09-03): the real backbone total, and the retraction of v1's reconciliation

Every term above is now measured (§2b/§2c/§2d), so the range collapses to a number:

```
36 × 57,958,624  (GDN, §2b)                        = 2,086,510,464
12 × 51,446,528  (QSA attn + indexer, §2c)         =   617,358,336
     gr_param_delta(48, 2560, 4, 320)  (GR, §2d)   =   640,624,640
     tok_emb  [248320, 2560]                       =   635,699,200
     lm_head  [2560, 248320] + bias (UNTIED)       =   635,947,520
     ln_f                                          =         2,560
                                                     -------------
     BACKBONE TOTAL                                 = 4,616,142,720   (≈4.616B)
```

| Quantity | v1 estimate | v2 (measured) |
|---|---:|---:|
| Backbone params | 3.9-4.1B | **4,616,142,720 (≈4.616B)** |
| Backbone bytes, bf16 | ≈7.6-8.2 GiB | **8,804.59 MiB = 8.598 GiB** |
| Backbone bytes, Q8_0-class (1.0625 B/param) | ≈4.1-4.4 GiB | **4,904,651,640 B = 4.568 GiB** |

**The real total is 12.6% above the TOP of v1's estimated range**, driven almost entirely by the two
corrections above (GR +0.64B where v1 budgeted 0.05-0.3B; QSA +0.21B from the missed gate half).

**v1's "two independent routes agree" claim is RETRACTED.** The published/verified total is 125B; §2a's
measured MoE total is 121.095B; this section's measured backbone is 4.616B. **4.616 + 121.095 = 125.711B**
— the two routes now **disagree by ≈0.71B (0.57%)**, where v1 reported agreement. That apparent agreement
was an *artifact of the two undercuts this revision fixes*: v1's 3.97B "remainder" route and its 3.9-4.1B
"sum of parts" route agreed only because the sum-of-parts was ~0.7B too small in exactly the amount
needed. This is a worked example of this project's own `[[independent-reimplementation-catches-identity-
swap-bugs]]` lesson applied to arithmetic rather than code: two routes agreeing is not evidence when one
route's inputs are estimates chosen for plausibility.

**Candidate explanations for the residual 0.71B, named as hypotheses and NOT adopted**: (a) the tech
report's "125B" may exclude the untied `lm_head` from its parameter count — `125.711 − 0.636 = 125.075B`,
which lands on 125B suspiciously well, and excluding an output head from a "model size" figure is a
common convention; (b) "125B" may simply be rounded to three significant figures, in which case anything
in [124.5, 125.5] is consistent and 125.711 is marginally outside; (c) a real shape somewhere above is
still wrong. **Resolving this needs the tensor-by-tensor summation across all 131 shard headers that
`docs/QWEN4_PREVIEW_REFERENCE.md` explicitly declined to do** (~131 small header-only HTTP requests) —
recorded in §6c as a NEW open item rather than guessed at here.

**The load-bearing conclusion for §3 is unchanged in direction and stronger in magnitude**: at bf16 the
backbone alone is **8.598 GiB**, which does not fit this GPU's ~7.5-7.7 GiB realistic ceiling by a margin
of ~0.9-1.1 GiB — a clearer "no" than v1's 7.6-8.2 GiB straddle, which could still have been read as
"maybe, at the low end."

### 2g. N-gram/PLE table and MTP head — cited, not re-derived, and MTP flagged for the same exclusion as n-gram

- **N-gram/PLE table**: 51.2B params exactly, ~102.4GB bf16, per-tier byte totals already tabulated in
  `docs/QWEN4_DEPLOYMENT_FEASIBILITY.md` §1c. **Explicitly out of scope for this document's resident
  budget, per the task's own instruction** — see §3's seam description instead of a byte allocation here.
- **MTP head**: 4B params (README-verified, `docs/QWEN4_PREVIEW_REFERENCE.md`), ≈7.45 GiB at bf16. The
  task brief does not name this head explicitly, but the same reasoning that excludes the n-gram table
  applies structurally: it is an **additive, single-purpose auxiliary head** (speculative multi-token
  prediction) that a first "does the backbone forward pass compute the right numbers" test does not need
  to exercise — recommended, not mandated, to exclude it from the initial-run resident budget for the
  same reason, and named here explicitly rather than silently omitted.

### 2h. The engine-realized shape vs. the real model's shape — a 1.68B gap, and the WP4 blocker it names

**This is the finding with the largest consequence for WP4, and it exists only because a real
implementation now exists to compare against.** v1 could not have found it: it compared the real model
against itself.

This pass evaluated `include/sub0/layout.hpp`'s real `make_param_layout()` — the same `consteval`
function every build uses — against a `sub0_corpus.hpp` set to Qwen4's real `RunConfig` axes
(`D_MODEL=2560, N_LAYERS=48, N_HEADS=24, N_KV_HEADS=2, D_FF=640, VOCAB=248320, GDN_FULL_ATTN_STRIDE=4,
HC_COUNT=4, HC_LOWRANK=320, NUM_EXPERTS=512, EXPERTS_PER_TOK=10, QSA_INDEXER_*=4/1/128/2048/4,
USE_GATED_FFN=true, USE_QK_NORM=true, USE_TIED_EMBEDDINGS=false, POS_ENCODING=Rope`). Result:

| Quantity | Value |
|---|---:|
| `NUM_PARAMS` (tensor count `make_param_layout()` emits) | **74,899 tensors** |
| `PARAM_FLOATS` | **124,027,786,776 floats** |
| as f32 blob | **462.04 GiB** |
| as bf16 | **231.02 GiB** |
| Real model's own shape (§2f + §2a) | 125,711,064,960 |
| **Shape gap** | **−1,683,278,184 (−1.34%)** |

**The gap is entirely two mechanisms, and both have the same root cause** — `D_HEAD` is derived, not
configurable:

| Mechanism | Engine-realized | Real model | Gap | Why |
|---|---:|---:|---:|---|
| Gated Residual | 640,624,640 | 640,624,640 | **0 — exact** | `HC_COUNT`/`HC_LOWRANK` are real, independent axes |
| MoE | 121,094,922,240 | 121,094,922,240 | **0 — exact** | `NUM_EXPERTS`/`EXPERTS_PER_TOK`/`D_FF` are real, independent axes |
| QSA (12 layers) | 268,621,296 | 617,358,336 | **−348,737,040** | engine `head_dim = D_MODEL/N_HEADS = 106`, real is **256** (`docs/QSA.md` §2b.1) |
| GDN (36 layers) | 751,723,560 | 2,086,510,464 | **−1,334,786,904** | engine `GDN_DIMS{2560, N_KV_HEADS=2, N_HEADS=24, D_HEAD=106, D_HEAD=106, 4}`; real is `k_heads=16, v_heads=48, head_k=head_v=128` |

**Read this carefully, because it is easy to misread as "the mechanisms are wrong":** GR and MoE are
shape-exact at the real config today. QSA's and GDN's *math cores* are ALSO correct at the real shape —
both `qsa::Dims` and `gdn::Dims` take every geometry field explicitly, which is precisely why their
fixture tests can and do run at the real `head_dim=256` / `head_k_dim=128` values. What cannot express
the real shape is the **engine's `RunConfig`→`Layer` binding**: `sub0_config.hpp` bakes
`D_HEAD = D_MODEL / N_HEADS` (an invariant `docs/QSA.md` §2b.1 already flagged as "WP4's problem"), and
`layout.hpp` aliases GDN's key/value head counts and head dims onto `N_KV_HEADS`/`N_HEADS`/`D_HEAD`
(`GDN_DIMS`, `layout.hpp` line ~232) rather than giving them their own axes.

**Consequence for WP4, stated once**: a real weight transplant cannot succeed at these axes — the target
tensors are literally the wrong shape (e.g. `GdnInProjQkv` is `[2560, 2984]` here vs `[2560, 10240]` in
the checkpoint). **Breaking the `D_HEAD = D_MODEL / N_HEADS` invariant, and giving GDN its own head-count
axes, is a hard prerequisite for WP4's transplant stage, not a refinement of it** — see
`docs/WP4_SCOPE.md` §WP4b, which is scoped around exactly this.

---

## 3. Explicit placement policy per component, with justification

### 3a. Backbone: quantized, GPU-resident where it fits; CPU-resident remainder

> **v2 (2026-09-03) update to this section's inputs**: §2f's measured backbone is **4.616B params /
> 8.598 GiB bf16 / 4.568 GiB at Q8_0-class**, replacing the 3.9-4.1B / 7.6-8.2 GiB / 4.1-4.4 GiB
> estimates this section was written against. **Both of this section's conclusions survive and are
> strengthened**, so the prose below is left intact rather than rewritten: (a) "bf16 backbone does not fit
> the ~7.5-7.7 GiB ceiling" goes from a straddle to a clear 0.9-1.1 GiB overshoot; (b) the Q8_0-class
> recommendation still fits, at 4.568 GiB rather than 4.1-4.4 GiB, leaving **2.93-3.13 GiB** (not
> 3.1-3.6) for runtime state and any expert cache. One term to ADD when this becomes implementation:
> §2a(v2)'s **`MoeRouter`, 62.9M params / 120.0 MiB bf16 across 48 layers, is always-resident backbone**
> and belongs on the GPU-resident side of this split, even though every other MoE tensor does not.

§2f's finding is decisive: **backbone at bf16 (≈7.6-8.2 GiB) does not comfortably fit this GPU's own
realistic ≈7.5-7.7 GiB ceiling even alone**, let alone alongside runtime state. Extending this project's
existing hybrid design (`[[hybrid-cpu-gpu-execution-design]]`, `include/sub0/device_backend.hpp`) rather
than inventing a parallel one:

- **The existing hybrid design is a *source-routed*, structural, decided-once-per-step split** (which
  *training window* goes to which device, chosen by blend-source membership, never a per-token runtime
  decision) — built for a training-time regime where CPU-only mechanisms (content-embed) coexist with
  GPU-accelerated ones. The backbone-placement question here is the **inference-time, frozen-model**
  analogue: not "which windows run where" but "which **layers'** frozen weights are GPU-resident", decided
  once at model-load time, never per-token. Same shape of decision (static, structural, resolved ahead of
  the hot loop), different axis (layers, not batch windows).
- **Concrete recommendation**: quantize the backbone (a frozen, inference-only model — quantization for a
  read-only forward pass is exactly what `docs/QWEN4_DEPLOYMENT_FEASIBILITY.md`'s entire GGUF-tier survey
  already validates as this ecosystem's standard approach) to an **Q8_0-class precision** for the portion
  placed GPU-resident: at ≈1.0625 bytes/param, 3.9-4.1B backbone params ≈ **4.1-4.4 GiB** — fits this
  GPU's realistic ceiling with real headroom left for runtime state (§3d). A full-precision (bf16)
  backbone should be the CPU-resident fallback/default (this project's engine is fp32/bf16-native today;
  a quantize-for-GPU-only step is new surface area this document names, not implements).
  Extending `Sub0DeviceCaps`-style capability naming: a build would carry a compile-time count of
  **GPU-resident backbone layers** (§5), analogous to today's `GDN_FULL_ATTN_STRIDE`/`DEPTH_CACHE_MAX`
  baked constants, not a runtime-decided split — the backbone never changes at runtime (frozen weights),
  so there is no reason for this to be anything but a build-time choice once the target GPU's real budget
  is known.
- **This does not need the hybrid router's existing `needs_cpu`/caps-bit machinery** (`docs/BACKENDS.md`)
  as-is — that machinery routes *batch windows* by *mechanism support* (does this backend implement
  binding-compose); backbone layer placement routes *weight residency* by *byte budget*, a different axis
  entirely. The precedent this document draws from `device_backend.hpp` is narrower and more structural:
  **capability bits describe what a backend CAN run; a new, separate compile-time layer-range constant
  describes what a specific BUILD chooses to keep GPU-resident** — the two are orthogonal, and a future
  MoE/backbone work package should not conflate them.

### 3b. MoE expert pool: static layer-range CPU/GPU split (llama.cpp `-ot`/`--n-cpu-moe`-style), not an activation-aware cache — for v1

> **v2 (2026-09-03) update to this section's inputs**: with §2f's measured backbone (4.568 GiB at
> Q8_0-class, not 4.1-4.4), point 1's remaining GPU budget becomes `7.5-7.7 − 4.568 ≈ **2.93-3.13 GiB**`
> before runtime state — inside the 2.5-3.2 GiB band this section already used, so **its conclusion (a
> static layer-range split for v1, with plausibly zero GPU-resident expert slots) is unchanged**. Two
> refinements this revision does add: (a) the expert-slot ceiling recomputes to **~1,135-1,215 IQ4_NL-class
> slots across all 48 layers (2.93-3.13 GiB ÷ 2.64 MiB/expert), ~24-25 per layer of 513** — still far
> short of a meaningful cache, so the
> "budget, not algorithm, is the binding constraint" argument holds; (b) §2a(v2)'s `MoeRouter` must be
> **exempted from this split** — it is dense-read per token and belongs with the backbone (§3a), so
> "offload layer L's experts" must never be read as "offload layer L's MoE block".

`docs/QWEN4_DEPLOYMENT_FEASIBILITY.md` §2 already surveyed the real options (llama.cpp's static
`-ot`/`--n-cpu-moe`; MoE-Infinity/Fiddler's activation-aware caches; Eliseev & Mazur's router-lookahead
prefetch). **This document picks the static layer-range split as this project's own direction for v1**,
for reasons specific to this hardware's numbers, not a generic re-statement of that survey's conclusion:

1. **§2a's arithmetic leaves essentially zero spare GPU budget for expert caching in v1.** Even after
   quantizing the backbone to Q8_0-class (§3a, ≈4.1-4.4 GiB), the realistic remaining GPU budget is
   roughly `7.5-7.7 − 4.1-4.4 − (runtime state, §3d) ≈ 2.5-3.2 GiB` before subtracting runtime state at
   all. At the smallest per-expert byte size this project has a verified number for (`IQ4_NL`-class,
   ≈2.64 MiB/expert, §2a), that ceiling holds roughly **950-1,200 expert-slots total, across all 48
   layers combined** — an average of **~20-25 slots per layer** out of the 513 (512 routed + shared) a
   token could route to. This is not enough headroom for a meaningfully-sized activation-aware cache to
   pay for its own complexity in v1; it is enough, at best, for a token's *own* activated set (11/layer)
   plus a small margin. **The "which caching policy is smarter" question is close to moot when there is
   almost no cache to manage** — the decisive constraint is budget, not algorithm, at this hardware's
   actual numbers.
2. **A static split is compatible with this project's `AGENTS.md` §1 no-runtime-variable-latency rule in
   a way a reactive/adaptive cache is not for a first implementation.** `llama.cpp`'s own `-ot`/
   `--n-cpu-moe` (`docs/QWEN4_DEPLOYMENT_FEASIBILITY.md` §2a) is explicit that it "does not predict
   anything" — placement is fixed at load time, and every token pays the same, uniform, pre-known cost
   for an off-GPU expert. That uniformity is exactly the property an explicit resolve-pass-ahead-of-the-
   hot-loop design (§4) needs: a fixed policy means the resolve pass's *shape* (how many PCIe transfers,
   of what size) is knowable ahead of running it, even though *which* experts are being fetched is
   runtime. An activation-aware adaptive cache (MoE-Infinity-style) instead makes an eviction/promotion
   decision that is itself runtime-computed and workload-dependent — a legitimate v2 direction (see
   point 4), but one this project's own standing rules push toward building only once a real access trace
   exists to justify it (the same "measure, don't guess" discipline `docs/NGRAM_TABLE_TIERED_STORAGE.md`
   §1's Bandana citation and this project's `three-pillar-shootout-policy` both already apply elsewhere).
3. **Given point 1's near-zero GPU headroom, the honest v1 recommendation is that essentially the entire
   512-expert pool per layer is CPU/RAM-resident, and every activated expert (any of the 10+1 selected per
   token per layer) is a PCIe-fetch-on-demand or a CPU-side compute, decided by a per-build layer-range
   constant** — the direct analogue of `--n-cpu-moe`'s "N highest-numbered layers' experts on CPU," except
   at this hardware's numbers N is likely **all 48 layers** for a first working build, with 0-a-few layers'
   experts promoted to GPU only once real measurement (§6) shows spare budget exists after backbone/
   runtime-state are accounted for on real hardware, not the estimates here.
4. **v2 direction, named not scheduled**: once GDN/QSA/Gated-Residual are implemented and their *real*
   measured footprint (not this document's estimates) is known, and once a real access trace exists from
   an actual run, an activation-aware cache (MoE-Infinity's structure-aware prediction, or Eliseev &
   Mazur's one-layer router-lookahead prefetch — both already surveyed and cited in
   `docs/QWEN4_DEPLOYMENT_FEASIBILITY.md` §2b) becomes worth revisiting **if** spare GPU budget turns out
   to be large enough for caching more than a token's own activated set to matter. Not designed further
   here — named as the concrete follow-up question, per this section's own numbers.

**Resource cost of the chosen policy, stated plainly**: CPU RAM residency for the **full** expert pool.
Per §2a's table, even the most aggressive quant tier this project has verified byte counts for
(`IQ4_NL`-class) puts the full 48-layer routed pool at **≈63.3 GiB** — larger than this document's own
§1b corrected realistic RAM headroom (~40-45 GiB) **on its own**, before backbone or anything else.
Reusing `docs/QWEN4_DEPLOYMENT_FEASIBILITY.md` §1b's real, already-fetched total-file byte counts instead
of this document's own per-expert extrapolation: the smallest real quant tier that exists as an actual
downloadable file, `UD-IQ1_S` (72.55GB total, of which 28.80GB is the n-gram table this document excludes
per §2g), leaves **≈43.75GB for backbone+attention+MoE combined** — a number **that lands right at the
edge of, and plausibly just over, this document's own corrected ~40-45 GiB realistic RAM headroom**. This
is a **materially more informative comparison than `docs/QWEN4_DEPLOYMENT_FEASIBILITY.md`'s own §4a
hedge** (which compared 67.56 GiB — that tier's *whole* file, n-gram table included — against a flat,
uncorrected 63GB), because it (a) excludes the n-gram table consistently with this document's own scope
decision and (b) uses a measured, load-corrected RAM headroom instead of the raw total. **Restated
honestly**: full-expert-pool CPU residency at the most aggressive real quant tier is **plausible, still
tight, not comfortably clear of the realistic RAM ceiling** — a firmer conclusion than either document
could reach alone, but still not a "yes, this fits" verdict without an actual load attempt.

### 3c. N-gram/PLE table: out of scope for the initial budget; seam described, not implemented

Per explicit user instruction, this table is **omitted entirely from the initial inference-test resident
budget** — legitimate, not a correctness compromise, because it is one additive signal injected at a
single decoder layer (`ple_layer_ids=[2]`, i.e. 0-indexed layer 1), so a first test validating the OTHER
mechanisms (GDN, QSA, Gated Residual, MoE routing) does not need it present to be a meaningful test of
those mechanisms. **The seam for a future Sub0Firn-backed tier is exactly the one
`docs/NGRAM_EMBEDDING.md` §7's deferred section already names**: `ngram_tab[e]` would become "a thin
client issuing `resolve_into` calls instead of a raw parameter pointer" (`docs/SUB0FIRN_SPEC.md` §3's
`resolve_into(table_handle, row_indices[], dest_buffer)` contract) rather than the fully-resident
`PARAM_LAYOUT` leaf it is for this engine's own small, from-scratch-trained tables today. Nothing about
this document's own placement decisions for backbone/MoE/runtime-state depends on this table existing —
the seam is named here for completeness, and its resident-memory line item is deliberately **zero** in
every budget above.

### 3d. Runtime state: extending `memplan.hpp`'s `Dims`-based accounting to GDN/QSA/MoE terms

Following `memplan.hpp`'s existing style (one named term per buffer group, `Dims`-parameterized, a
`total()` aggregator) rather than inventing a parallel scheme. These are **new terms a future `Dims`
extension would add**, not implemented this pass — shown symbolically with a worked numeric illustration
at a chosen test context length (T=4096, an arbitrary but reasonable "first test" context — the terms
scale linearly with T where noted, so this is illustrative, not a fixed design point):

| Term (extends `memplan::Dims`/`ScratchTerms`) | Formula (batch=1, decode) | Value @ T=4096 illustrative | Persistent or scratch? |
|---|---|---:|---|
| GDN recurrent state, all 36 layers | `36 × num_v_heads(48) × head_k_dim(128) × head_v_dim(128) × 4B` | ≈108.0 MiB | **Persistent** (decode) / **train_scratch** (batched fwd) — exactly `docs/GATED_DELTANET.md` §2's existing `GdnCache`-vs-scratch distinction, now at real Qwen scale |
| GDN conv state, all 36 layers | `36 × conv_dim(10240) × (conv_kernel-1)(3) × 4B` | ≈4.2 MiB | **Persistent** (decode), alongside the recurrent state |
| QSA KV cache, all 12 layers, fp32 | `12 × 2(K,V) × n_kv_heads(2) × T × head_dim(256) × 4B` | ≈201.4 MiB @ T=4096; **scales linearly with T** — ≈12.9 GiB at the model's own native 262,144-token context | **Persistent** (grows with T, exactly `memplan.hpp`'s existing KV-cache-shaped term) — the QSA indexer's block-selection (§ below) does NOT shrink this: selection decides which cached K/V blocks a query ATTENDS to, not which are RETAINED, so long-context QSA KV residency is a real, T-scaling cost regardless of indexer sparsity |
| QSA indexer pooled-key cache, all 12 layers | `12 × indexer_kv_heads(1) × (T/compress_ratio(4)) × indexer_head_dim(128) × 4B` | ≈50.3 MiB @ T=4096 | **Persistent**, smaller than the full KV cache by `compress_ratio` |
| MoE routing scratch, per layer | router logits `[batch, 513]` + top-10 indices/weights — a handful of floats/ints per token | negligible (<1 KiB/token/layer) | **train_scratch**-shaped, negligible size |
| Backbone activations (batched forward, per window) | same shape as this engine's existing `fwd_scratch_bytes`/`train_scratch_terms` per-row buffers, at `D_MODEL=2560` instead of this project's own toy dims | not separately budgeted here — reuses the existing formula's shape, scaled to real `D_MODEL` | **train_scratch**, existing accounting applies directly once `Dims.d_model=2560` |

**All runtime-state terms above are small relative to §2's weight budgets** (hundreds of MiB, even at a
generous test context, vs. gigabytes for backbone and tens of gigabytes for the expert pool) — confirming
at real Qwen4 scale the same conclusion `docs/GATED_DELTANET.md` §2 already drew at this project's own toy
scale: GDN's O(1)-in-context state is a small, flat cost, and the dominant scaling risk in this whole
system is the **QSA KV cache at long context**, not GDN state and not MoE routing scratch.

#### v2 (2026-09-03): measured against each mechanism's OWN sizing functions

Every row above was a hand-written formula. All four mechanisms now ship real, `Dims`-parameterised
sizing functions (`gdn::state_floats`/`conv_hist_floats`/`scratch_floats`, `gr::normed_scratch_floats`/
`mix_scratch_floats`, `moe::scratch_floats`, `qsa::scratch_floats`/`block_key_cache_floats`), so this
pass called them directly at the real model's own `Dims` — `gdn::Dims{2560,16,48,128,128,4}` and
`qsa::Dims{2560,24,256,2,4,1,128,2048,4,64}`, i.e. the real geometry, which those math cores CAN express
even though the engine binding cannot (§2h):

| Term | v1 estimate | v2 measured (f32) | Verdict |
|---|---:|---:|---|
| GDN recurrent state, 36 layers | ≈108.0 MiB | `36 × gdn::state_floats` = 28,311,552 floats = **108.0 MiB** | **exact — v1 confirmed** |
| GDN conv state, 36 layers | ≈4.2 MiB | `36 × gdn::conv_hist_floats` = 1,105,920 floats = **4.219 MiB** | **exact — v1 confirmed** |
| GDN state total (persistent, decode) | ≈112.2 MiB | **112.219 MiB** | **exact** |
| QSA KV cache, 12 layers @T=4096 | "≈201.4 MiB" | 50,331,648 floats = **192.0 MiB** | **unit slip in v1** — 201.4 was the byte count in MB, labelled MiB. Formula was right. |
| QSA KV cache, 12 layers @T=262,144 (native context) | "≈12.9 GiB" | 3,221,225,472 floats = **12.0 GiB** | same unit slip; **still the dominant T-scaling term, as v1 concluded** |
| QSA indexer key caches, 12 layers @T=4096 | ≈50.3 MiB (modelled as ONE compressed `T/ratio` cache) | raw per-token keys 6,291,456 floats (**24.0 MiB**) + `qsa::block_key_cache_floats` 131,072 floats (**0.5 MiB**) = **24.5 MiB** | **v1's model of WHAT is cached was wrong, and the real answer is smaller.** The real implementation must keep the RAW per-token keys at full `T` (a block's pooled key is derived from them as blocks complete — `docs/QSA.md` §11) AND a small pooled block-key cache. v1 modelled only the compressed half. |
| QSA indexer key caches @T=262,144 | not stated | 402,653,184 + 8,388,608 floats = **1.531 GiB** | new |
| MoE routing scratch | "<1 KiB/token/layer" | `moe::scratch_floats(MOE_DIMS)` = 4,352 floats = **17.0 KiB, flat** (row-independent by construction — `docs/MOE.md` §2) | **v1's "negligible" confirmed, and it is even flatter than v1 assumed** |

**A term v1 did not budget at all, and the largest single number in this section**: the **batched-prefill
scratch** each math core needs. v1's last row waved at this ("reuses the existing formula's shape, scaled
to real `D_MODEL`"), but these are the mechanisms' own new buffers, not `memplan.hpp`'s existing ones:

| Term | Measured @ T=4096, f32 | Scaling |
|---|---:|---|
| `gdn::scratch_floats(REAL, T=4096)`, **per GDN layer** | 109,445,120 floats = **417.5 MiB** | linear in T |
| `gr::normed_scratch_floats(T=4096)`, per GR instance call | 41,943,040 floats = **160.0 MiB** | linear in T (`T × HC_COUNT × D_MODEL`) |
| `gr::mix_scratch_floats(T=4096)`, per GR instance call | 1,320,960 floats = **5.04 MiB** | linear in T (`T × HC_LOWRANK`) |
| `qsa::scratch_floats(REAL, T=4096)`, per QSA layer | 4,877,825 floats = **18.6 MiB** | linear in T |

These are **reused, not simultaneously live per layer** (one buffer serves whichever layer is executing),
so the resident cost is the max, not the sum — but **417.5 MiB for a single GDN layer at T=4096** is a
real number that nothing in v1 anticipated, and it is the term that sets the **maximum prefill chunk
length a real run can afford**: at T=16,384 the same buffer is ≈1.63 GiB, at the model's native
T=262,144 it is ≈26 GiB. **Prefill must be chunked**, and the chunk size is a first-class WP4 decision,
not a detail — recorded in `docs/WP4_SCOPE.md` §WP4c as an explicit scoping input rather than left to be
discovered at run time.

**The v1 conclusion this revision does NOT overturn**: GDN's recurrent state really is O(1) in context
and really is small (112 MiB flat), and the QSA KV cache really is the dominant *persistent* T-scaling
term (12.0 GiB at native context). What changed is that the *transient* prefill scratch, not the
persistent state, is what binds first at any realistic first-run context.

---

## 4. Handoff mechanics — resolve-pass-ahead-of-hot-loop vs. genuinely safe inline access

Per `AGENTS.md` §1 (no runtime-variable-latency branch in a hot path), classifying each §3 component:

| Component | Classification | Why |
|---|---|---|
| Backbone weights (GPU-resident slice + CPU-resident remainder) | **One-time load-time resolve** | Uploaded once at model-load (the direct analogue of today's `sub0_dev_upload_params`), never touched again mid-forward — trivially satisfies the rule since it happens entirely outside any per-token loop. |
| MoE expert weights (CPU-resident, fetched/computed on demand) | **Mandatory explicit resolve pass, never an inline fetch inside the compute loop** | Structurally identical to the n-gram table's own already-designed resolve-pass requirement (`docs/NGRAM_TABLE_TIERED_STORAGE.md` §2a) — but with a genuine, distinct opportunity `docs/QWEN4_DEPLOYMENT_FEASIBILITY.md` §2b already names (Eliseev & Mazur): the router for layer *L*'s experts is computable as soon as layer *L−1*'s hidden state exists, so a resolve pass for layer *L*'s selected experts can be issued while layer *L−1*'s remaining compute (or its own PCIe copies) is still in flight — a real latency-hiding opportunity the n-gram table's single-injection-point design does not have. Not implemented this pass; named as the concrete mechanism a future MoE work package should build the resolve pass around. |
| GDN recurrent + conv state | **Inline access, safe** | Not a cross-tier fetch at all — a local, arena-resident buffer read/written in place by the same compute (CPU or GPU, whichever this build placed that layer on) that owns it; no I/O, no latency variance, exactly `docs/GATED_DELTANET.md` §2's existing `GdnCache` design. |
| QSA KV cache + indexer pooled-key cache | **Inline access, safe** | Same reasoning as GDN state — compute-local, append-in-place per step, no external table lookup. |
| N-gram/PLE table | **Out of scope this pass**; when built, mandatory explicit resolve pass — already fully designed in `docs/NGRAM_TABLE_TIERED_STORAGE.md` §2a/§2c, cited not re-derived here. |
| A hypothetical fully-local, already-downloaded, page-cache-warm backbone/expert file | **Judgment call, not resolved by this document** | `docs/SUB0FIRN_SPEC.md` §3c's own hedge applies unchanged: an mmap page-fault against an already-local, already-warm file has bounded, NVMe-class (tens of µs) latency, which *might* be defensible to access inline without a separate resolve pass — but this is explicitly **not** the same case as a remote/cold fetch, which always requires the explicit resolve pass, and this document does not adjudicate the judgment call any more firmly than the source doc already declined to. |

---

## 5. Compile-time vs. runtime split — the dividing line for a future MoE work package

Per this project's standing compile-time-over-runtime preference (`AGENTS.md` §2, and the existing
`GDN_FULL_ATTN_STRIDE`/`DEPTH_CACHE_MAX`/`NGRAM_MAX_N`-style baked-constant precedent in
`tools/configurator.cpp`'s `RunConfig` X-macro):

### 5a. CAN be compile-time-fixed, for a Qwen4-specific build

- **Layer schedule** (which of the 48 layers are GDN vs QSA) — the direct extension of this project's
  existing `GDN_SCHEDULE`/`gdn_schedule_for<N_LAYERS>()` consteval mechanism (`docs/GATED_DELTANET.md`
  §3c) to Qwen4's real `3×GDN + 1×QSA, repeated ×12` pattern.
- **Which layers' backbone weights are GPU-resident vs. CPU-resident** (§3a) — the backbone never
  changes at runtime (frozen weights), so this is a build-time choice once the target GPU's real budget
  is known, exactly like `GDN_FULL_ATTN_STRIDE` is a build-time choice today.
- **The GPU-resident expert-slot CAPACITY per layer** (§3b) — a fixed count of slots reserved,
  compile-time-sized, even though which experts occupy them is runtime (§5b). Per §3b's own numbers,
  this capacity is likely **0, or a small single-digit number, for a v1 build on this hardware** — a
  legitimate, honestly-reported compile-time constant, not a placeholder for "figure it out later."
- **GDN recurrent-state and conv-state buffer sizes** — fully determined by `num_v_heads`/`head_k_dim`/
  `head_v_dim`/`conv_kernel`, all real, fixed `config.json` fields; no runtime dimension involved.
- **QSA KV-cache buffer sizing**, bound by a chosen maximum context length — exactly this project's
  existing `SEQ_LEN` compile-time cap, applied to Qwen4's own dims.
- **Precision/quantization tier chosen for the backbone and for the (CPU-resident) expert pool** — a
  per-build choice, the same shape of decision as this project's existing `USE_TERNARY`.
- **The n-gram seam's mere PRESENCE in a build** (whether the thin-client interface point exists at all)
  — but never its table's residency, which is entirely out of scope this pass (§3c).

### 5b. INHERENTLY runtime — cannot be known at compile time

- **Which specific experts are hot for a given token** — the router's output depends on the input; no
  amount of compile-time analysis of a frozen model's weights predicts a specific future token's routing
  decision.
- **The QSA indexer's block-selection** — which key-blocks a query attends to depends on content, per
  query, per token.
- **The actual bytes fetched from CPU RAM (or, later, disk/network) for a given request's activated
  experts** — runtime I/O, by construction.
- **Any future adaptive cache's eviction/promotion decisions** (§3b point 4, not built this pass) — if
  ever added, these are runtime-computed by definition; naming this here mainly to be explicit that
  adopting an activation-aware policy later does NOT change §5a's compile-time terms, only what runtime
  logic populates the compile-time-sized structure those terms describe.

### 5c. The dividing line, stated once, for the next work package

**Build a compile-time-SIZED, fixed-capacity GPU-resident expert-slot table per layer (capacity chosen
per build from the target GPU's real measured budget after backbone and runtime-state are subtracted —
per §3b's numbers, plausibly `0` for a first working build on this hardware), whose CONTENTS are
populated and evicted entirely at runtime by whichever placement mechanism is chosen (a static
layer-range assignment for v1, meaning "no GPU-resident expert cache at all, every expert access is a
runtime CPU-resident compute or PCIe fetch" is a legitimate, honestly-named v1 outcome — not a fallback
to apologize for).** This is the one sentence a future MoE work package needs: never a fully dynamic,
unbounded, heap-growing expert-storage structure (which would violate `AGENTS.md` §1 regardless of
placement policy), and never an attempt to make the *routing decision itself* compile-time (which is
architecturally impossible — it is a function of the input). Everything in §5a is a build-time
configuration surface; everything in §5b is what that fixed-size structure's runtime logic fills in.

---

## 6. Budget headroom and iteration plan — what's solid, what's estimated, what to re-measure

### 6a. Solid (real byte counts, directly fetched or exactly reproduced)

- Every `config.json`-derived parameter count in §2 (MoE per-expert size, GDN mixer, QSA attention
  projections, embeddings/lm_head) — fetched fresh this pass from the live repository, and the GDN number
  independently reproduces a previously-fetched real byte count exactly (§2b).
- Every GGUF per-tier byte total and n-gram-table byte total cited from `docs/
  QWEN4_DEPLOYMENT_FEASIBILITY.md` — that document's own §1 already did the direct-header-read work; this
  document reuses those numbers rather than re-deriving them.
- This machine's real total/free RAM, total/free disk, GPU total VRAM, and PCIe link width/generation
  (§1) — all measured this session via `nvidia-smi`/`Get-CimInstance`/`Get-Volume`/`Get-PhysicalDisk`.

**Promoted to solid in v2 (2026-09-03) — and exactly what that promotion is and is not evidence of.**
The measurement was: compile a throwaway translation unit against this repository's real
`include/sub0/layout.hpp` (and, through it, `gated_residual_math.hpp`/`moe_math.hpp`/`qsa_math.hpp`/
`gdn_math.hpp`) with a `sub0_corpus.hpp` set to Qwen4's real `RunConfig` axes, then read out the values
the project's OWN `consteval make_param_layout()` and `*_param_delta()`/`*_scratch_floats()` functions
produce. That is a **real evaluation of this engine's own shipped shape/sizing machinery at real dims** —
the same code every build compiles, gated in the default suite by `tests/layout_tests.cpp` (which pins
each closed form against hand-computed values at two shapes) and by each mechanism's real-weight fixture
test. What it is **NOT**: a running process, an allocator measurement, or a `sub0_cuda_train_footprint`-
style measured-vs-predicted check. **No model was loaded and no memory was actually allocated at these
dims** (a 462 GiB f32 blob is not allocatable here), so these are exact *shape* facts, not *residency*
facts. §6c items 1-4 and 6 are what would make them residency facts.

- Every §2a MoE parameter count, now including the `MoeRouter`/`MoeSharedGateProj` terms v1 omitted, and
  the corrected activated-per-token figures.
- The full §2c QSA per-layer count **including the indexer and the previously-unknown `q_proj` gate half**.
- The full §2d Gated Residual per-layer and total count.
- §2f's backbone total (4,616,142,720) and its bf16/Q8_0-class byte figures.
- §2h's engine-realized `NUM_PARAMS`/`PARAM_FLOATS` at the real axes, and the 1.68B shape gap.
- Every §3d runtime-state and prefill-scratch number, each read from the owning mechanism's own
  `Dims`-parameterised sizing function at the real geometry.

### 6b. Estimated (an arithmetic derivation from a real quantity, or a class-level figure — not a measurement)

- ~~The Gated Residual per-layer parameter count (§2d)~~ — **RESOLVED in v2**, see §2d. v1's estimate was
  wrong by enough to change a placement conclusion; kept in the list, struck through, so the correction
  stays visible.
- ~~The QSA indexer's own weight shapes (§2c)~~ — **RESOLVED in v2**, see §2c. v1's "~1-2M" estimate was
  correct (real: 1,638,656); the *attention* side it sat next to was not.
- PCIe **sustained** transfer bandwidth (§1a), NVMe sequential/random throughput (§1c), and HTTP-range
  **sustained** throughput (§1d) — all class-level or theoretical-spec numbers, none measured on this
  exact hardware, in v1 or v2. **Still estimated.**
- The realistic GPU-VRAM and RAM headroom figures (§1a/§1b) — real measurements of a specific point-in-time
  snapshot, but not a controlled idle-vs-loaded comparison, and will shift run to run. **Still estimated.**
- ~~Every §3d runtime-state number~~ — **the formulas are now the mechanisms' own shipped functions, not
  hand transcriptions** (§3d v2). What remains estimated is only the *choice* of T=4096 as an
  illustration point, and the assumption that a real run's per-layer scratch is reused rather than
  simultaneously live (true of the current CPU implementations; unverified for any future device path).
- **New in v2, and genuinely unresolved**: the 0.71B discrepancy between §2f's measured 125.711B total
  and the published 125B (§2f names three candidate explanations and adopts none). Estimated, not solid.

### 6c. What should get re-measured once real mechanism implementations exist, named explicitly per this project's own repeated lesson that estimates and reality diverge at scale (`[[memplan-vram-prediction-gap-at-scale]]`, `[[gpu-tune-benchmark-vs-production-discrepancy]]`)

> **v2 status board (2026-09-03) — read this before the list.** Item **5 is DONE** (§6a records what the
> measurement was and its limits). Items **1, 2, 3, 4 and 6 are UNTOUCHED and still open**: every one of
> them needs a real *running* artefact this pass did not have — a live inference CUDA context (1), a
> `cudaMemcpy` benchmark on this GPU/driver (2), a disk benchmark on these NVMe drives (3), a wall-clock
> HTTP-range transfer (4), and an actual MoE access trace from an actual run (6). **None of them is
> unlocked by having CPU forward implementations**, which is all that landed since v1. No number for any
> of them appears anywhere in this revision, and none should be invented from the shape facts §6a
> promoted — a shape is not a bandwidth. Items 1, 2 and 4 are on `docs/WP4_SCOPE.md`'s critical path;
> item 6 is explicitly downstream of a first working run; item 3 remains a Sub0Firn prerequisite,
> unscheduled. **Item 7 is NEW in v2.**

1. **A real `sub0_cuda_train_footprint`-style measured-vs-predicted check, extended to an inference-only
   CUDA context**, once any backbone-GPU-residency code exists — this document's §1a "~7.5-7.7 GiB usable"
   figure is itself an estimate layered on a measurement (idle free VRAM), not a measured inference-context
   footprint.
2. **A real `cudaMemcpy` (or equivalent) bandwidth benchmark on this exact GPU/driver**, before any
   claim about MoE-expert-fetch latency becomes load-bearing for a design decision — §1a's PCIe
   bandwidth numbers are spec arithmetic, not a measurement.
3. **A real sequential/random-read benchmark against this machine's actual NVMe drives**, before any
   disk-tier latency number in a future Sub0Firn integration is trusted — §1c's NVMe figures are
   class-level, not device-specific.
4. **A real wall-clock-timed HTTP-range extraction** (even a modest one — e.g. resolving a few hundred
   rows/tensors and timing it), to finally put a throughput number on this project's own already-proven
   extraction mechanism — §1d names this as a genuine, currently-unfilled gap across BOTH prior
   extraction passes.
5. **Real GDN/QSA/MoE-routing measured footprints**, the moment each mechanism has even a CPU forward
   implementation gated against a real fixture (following exactly the `docs/GATED_DELTANET.md` §6
   precedent) — every §2b/§2c/§2a-activated number in this document is a hand-derived estimate from
   `config.json` arithmetic, not a measured allocation, and should be replaced with real numbers the
   moment they exist, the same way `memplan.hpp`'s own header describes itself as "a PREDICTION, and
   predictions rot," kept honest by a measured-footprint test.
   **— DONE in v2 (2026-09-03).** §2a/§2c/§2d/§2f/§2h/§3d all now carry values produced by this project's
   own `make_param_layout()`/`*_param_delta()`/`*_scratch_floats()` at real dims. §6a states precisely
   what that measurement is evidence of (exact shapes) and what it is not (residency). **The "kept honest
   by a measured-footprint test" half of this item is NOT done**: there is no analogue of
   `sub0_cuda_train_footprint` for an inference-shaped load, which is item 1.
6. **A real access trace from an actual run**, before any MoE-expert placement policy more sophisticated
   than §3b's static split is attempted — per `docs/NGRAM_TABLE_TIERED_STORAGE.md` §1's own Bandana
   citation ("measure-then-size the cache, don't guess it"), no such trace exists yet for this model on
   this engine.
7. **NEW in v2 — a tensor-by-tensor summation across all 131 real shard headers**, to settle §2f's 0.71B
   discrepancy between this document's measured 125.711B and the published 125B. `docs/QWEN4_PREVIEW_
   REFERENCE.md` explicitly declined this ("~131 small header-only HTTP requests, not done this round, as
   the report's own table is a stronger source than a hand re-derivation would be") — that judgement was
   sound when the alternative was a hand re-derivation, and is no longer, now that the competing number
   comes from real, fixture-gated shapes. It is also the cheapest of the seven items (header-only Range
   requests, the technique this project has already validated three times) and it would independently
   confirm or refute §2h's shape gap from the checkpoint side. **Do this before, not after, any real
   weight transplant** — a transplant that silently mismatches on 0.7B of params is exactly the failure
   this settles cheaply in advance.

This document should be revised in place (a new revision-log entry, not a new file) as each of the above
lands — per its own header, it is explicitly a living document, not a final specification.
