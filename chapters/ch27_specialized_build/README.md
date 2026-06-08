# Chapter 27 — Specialized Model Compilation

**Thesis:** a general-purpose inference engine (llama.cpp, and our own
`sub0llm-cli`) treats every architectural axis — vocab size, embedding width,
head counts, head dim, layer count, FFN width, RoPE base, norm epsilon,
activation, attention windowing — as a *runtime* value. Strides are recomputed,
loop bounds are dynamic, buffers are heap tensors. If we instead **lift every
one of those axes to compile time** for a single target model, the optimizer can
size buffers as `std::array`, unroll bounded loops, constant-fold strides, inline
the whole forward into straight-line code, and — under `-march=native` —
vectorize against *known* shapes. This chapter builds the compilation stack that
does that and measures whether it actually beats the generic engine.

The motivating target is **Gemma 4 12B** (vs llama.cpp as the baseline server);
prototyping is done on **Qwen3-0.6B** (far smaller / less memory).

> Data is runtime, *shape* is compile time. Weights are still loaded from the
> GGUF at startup — only the dimensions are baked into the binary.

## Target model & baseline (local resources)

| Resource | Location |
|----------|----------|
| Prototype model | `models/Qwen3-0.6B-Q8_0.gguf` |
| Target model | `models/gemma-4-12b-it-Q8_0.gguf` (12.67 GB, Q8_0) |
| Baseline engine | `D:\tools\llamacpp` (prebuilt CPU bins: `vendor\llama.cpp-prebuilt\b9334\cpu\` — `llama-cli.exe`, `llama-bench.exe`, `llama-perplexity.exe`; put that dir on `PATH` for the DLLs) |
| Gemma 4 blog | <https://blog.google/innovation-and-ai/technology/developers-tools/introducing-gemma-4-12B/> |
| Model card | <https://huggingface.co/google/gemma-4-12B-it> |
| GGUF (unsloth) | <https://huggingface.co/unsloth/gemma-4-12b-it-GGUF> |

**Gemma 4 12B is a hard, heterogeneous, multimodal architecture** (ground truth from
`sub0llm-specialize --dump-meta` on the local Q8 GGUF). This is *not* a uniform
model and our current `ModernGPT` cannot represent it as-is:

| field | value | notes |
|-------|-------|-------|
| vocab_size | 262144 | |
| embed_dim | 3840 | embed × √3840 ≈ 61.97 |
| n_heads | 16 | |
| **n_kv_heads** | **array[48]** | **per-layer** (not a scalar) |
| **head_dim** | **512 global / 256 sliding** | `key_length` vs `key_length_swa` — varies by layer |
| n_layers | 48 | |
| d_ff | 15360 | GeGLU (`gelu_pytorch_tanh`) |
| **rope_base** | **1e6 global / 1e4 sliding** | `freq_base` vs `freq_base_swa` — dual RoPE |
| rope.dimension_count | 512 / 256 (swa) | partial-rotary related |
| **window** | **1024 + pattern[48]** | per-layer local/global mask, not every-6 |
| final_logit_softcapping | 30 | |
| norm | (1 + weight) RMSNorm, eps 1e-6 | qk_norm yes; tied emb yes |

> **Resolved by tensor shapes (authoritative — they determine what we load):**
> `attn_q_norm=[256]` and `attn_q` out=4096=16×256 ⇒ **head_dim=256** (the earlier
> `key_length=512` was vision-tower pollution). And kv-head count **varies per layer**:
> `blk.0/1.attn_k` out=2048 ⇒ 8 kv heads, but `blk.5.attn_k` out=512 ⇒ **2 kv heads**.
> So Gemma 4 is genuinely **per-layer heterogeneous** (kv count, plus per-layer window
> and RoPE base) — confirmed from the file itself, no llama.cpp source needed.
> Our uniform `ModernGPT` (single `n_kv_heads`, one window, one RoPE base) cannot
> represent it; a faithful Gemma 4 forward needs a per-layer model.

### Cross-referenced & verified architecture (official docs + GGUF tensors + HF config)

The official **dev guide** and **model card** are high-level — they confirm the
*structural* numbers and contradict nothing; the internals come from the GGUF tensor
shapes (authoritative for the file we load) and HF `config.json`, which agree:

| Param | Value | Source agreement |
|-------|-------|------------------|
| layers | 48 | model card = GGUF |
| context / window | 256K / 1024 | model card = HF = GGUF |
| vocab | 262144 | model card (262K) = GGUF |
| d_model | 3840 | GGUF = HF |
| n_heads | 16 (all layers) | GGUF tensors |
| **head_dim** | **256 local / 512 global** | **GGUF `attn_q_norm` dim per layer** (varies!) |
| **n_kv_heads** | **8 local / 1 global (MQA)** | **GGUF tensors per-layer** (q_out/head_dim) |
| global-attn layers | every 6th: 5,11,…,47 (8 of 48) | GGUF (the head_dim-512 / 1-kv layers) |
| per-layer extras | 4 norms (in/post-attn/pre-ffn/post-ffn) + `layer_output_scale` scalar | GGUF tensors |
| d_ff | 15360 | GGUF = HF |
| RoPE θ | 1e4 local / 1e6 global | HF |
| activation / norm | gelu_pytorch_tanh / (1+w) RMSNorm | HF + Gemma lineage |
| final logit soft-cap | 30 | HF |
| embed scale / tied | √3840 / yes | HF |

The standout (only the *tensors* reveal it): **global layers use 2 kv heads, local
layers 8** — a deliberate KV-cache optimization (global layers attend over the whole
256K context, so fewer KV heads slash their cache; windowed local layers keep 8).
The 5-local : 1-global pattern is uniform across all 48 layers (global = index%6==5).

### Gemma 4 tokenizer gate — PASSES (SentencePiece tokenizer built)

`tokenizer.ggml.model = gemma4` is a **SentencePiece** tokenizer; our `BPETokenizer`
is GPT-2 byte-level BPE and produced garbage (`▁` leaked as token 245237). Built
`SPTokenizer` (`tokenizer/sentencepiece.{hpp,cpp}`): score-based priority-queue merge
of adjacent symbols, `▁` (U+2581) word marker, byte fallback (`<0xNN>`), BOS prepend —
matching llama.cpp's `llm_tokenizer_spm`. The GGUF loader now also captures
`tokenizer.ggml.scores` / `token_type` / `bos`/`eos`/`add_bos`.

**Parity vs `llama-tokenize` — exact match**, ASCII and non-ASCII:
- "The capital of France is" → `2 818 5279 529 7001 563` ✓
- "the café costs $5" (UTF-8) → `2 1437 33443 5384 609 236810` ✓ (incl. " café"=33443)

(The earlier "café" divergence was the Windows console mangling UTF-8 in argv — clean
via `--file`.) `sub0llm-verify` auto-selects SPM for `gemma*`/`llama` vocabs, BPE
otherwise. Unit-tested (`test_sentencepiece.cpp`, 493 tests total). The tokenizer
prerequisite is cleared; the per-layer forward is next.

**Tokenizer performance note (staged win, off the generation hot path).** Tokenization
is a one-time prompt-ingestion cost — negligible vs per-token generation — so it does
*not* move tok/s. It *does* matter for long-prompt / document / high-throughput serving,
where two levers apply: (1) the **algorithm** — the current BPE merge is the documented
O(n²) bottleneck (CLAUDE.md), worth a priority-queue/linked-list rewrite; (2) **SIMD
string parsing** — vectorized whitespace/UTF-8 scanning for pre-tokenization (a
well-trodden domain). Both are real "staged" improvements that compound with the Q8 /
tiling wins for end-to-end latency on prompt-heavy workloads; sequence them after the
SentencePiece correctness work.

### Gemma 4 12B forward — DONE, at parity with llama.cpp

A dedicated, self-contained **`GemmaModel`** (`include/sub0llm/nn/gemma.{hpp,cpp}`)
implements the faithful per-layer-heterogeneous forward — the uniform `ModernGPT`
cannot represent it. Driven by `sub0llm-gemma` (`tokenize` / `logits` / `greedy`).
Q8 quantize-on-load (no f32 materialized): **11.91 B params, ~12.1 GiB RSS**.

**The architecture was resolved from the GGUF tensor shapes _and_ llama.cpp's
`src/models/gemma4.cpp`** (the authoritative interpreter of this exact file — the HF
config and even some GGUF metadata disagree). Several details would have been
impossible to guess and were each parity-critical:

| feature | resolution |
|---------|-----------|
| head_dim / kv-heads | **per layer**: local (idx%6≠5) 256-dim, 16q/8kv, window 1024, θ=1e4; global (idx%6==5) 512-dim, 16q/**1kv (MQA)**, full attn, θ=1e6 |
| attention scale | **1.0** — *not* 1/√head_dim (QK-norm controls magnitude; `f_attention_scale=1.0`) |
| V projection | global layers omit `attn_v` → **V = the raw K projection** (`Vcur = Kcur`) |
| V norm | plain `rms_norm` with **no learned weight** (easy to miss) |
| (1+w) RMSNorm | the +1 is **baked into the GGUF weights at conversion** → plain `x_normed·w` |
| RoPE | NEOX half-split; global layers apply `rope_freqs[256]` = `[1.0]×64, [1e30]×192` → only the first 64 freq-pairs rotate |
| layer_output_scale | real per-layer scalar (~0.05–0.36) multiplying the **whole** residual stream |
| FFN | GeGLU (`gelu_pytorch_tanh` gate) ; embed ×√3840 ; final logit soft-cap 30 |

**Correctness gate — PASSED** (vs llama.cpp `b9334`, raw completion `-no-cnv --temp 0`):

| check | result |
|-------|--------|
| Tokenize "The capital of France is" | `2 818 5279 529 7001 563` — **exact** vs `llama-tokenize` |
| Top next-token logit | ` Paris` @ 19.23 (margin 1.7 over ` a`) — **correct** |
| Greedy continuation (multiple clean prompts) | **byte-identical** to `llama-completion` across 24–48 tokens, e.g. " Paris.\n<\|channel>thought\n<channel\|>That is correct. Paris is the capital and most populous city of France." |

The greedy match runs through all 48 heterogeneous layers (both head dims, MQA,
dual RoPE + the `rope_freqs` partial rotation, the V=K fallback, per-layer scales,
GeGLU, soft-cap) — a byte-identical 48-token continuation is only possible if every
one of those is correct. (Adversarial/ambiguous prompts where the model is genuinely
uncertain can flip a near-tied token under Q8 activation quantization — llama itself
gives nonsensical answers there — but every clean prompt matches.)

**Speed / RAM (this CPU, gen tok/s, single-token decode).** Measured apples-to-apples:
each engine's *own* decode-loop timer (ours; llama's `common_perf_print` **eval time**,
which excludes load, prompt-eval, and sampling — "unaccounted time = 0.0%"), both
running greedy temp-0 on the same prompt and generating byte-identical tokens. No
`llama-bench` is used for the headline — it's two engines self-timing the identical
workload (one forward / token, batch 1, growing KV; the model's `<|channel>thought`
tokens are emitted identically by both, so the per-token work is the same).

| engine | 1 thread | best multi-thread | RSS | scaling |
|--------|---------:|------------------:|----:|--------:|
| llama.cpp (eval time) | 1.37 | 6.00 (24t) | 11.78 GiB | 4.4× |
| our `GemmaModel` (naive fork-join) | 1.60 | 1.96 | 12.1 GiB | 1.2× |
| our `GemmaModel` (CV pool) | 1.60 | 5.32 (16t) | 12.1 GiB | 3.3× |
| **our `GemmaModel` (spin + affinity)** | **1.60** | **5.59 (20t)** | 12.1 GiB | **3.5×** |

**Per core, our int8 GEMV is _faster_ than llama.cpp — 1.60 vs 1.37 tok/s (117%).** The
single-thread baseline is the cleanest kernel comparison; our Q8 quantize-on-load +
`dot_q8_0_q8_0` holds up against heavily-tuned ggml (which here has AVX-VNNI + weight
repack) on one core. The multi-thread gap was *scaling*, not kernels, and closing it took
three things — each matching ggml's threadpool design and each **zero logit change**
(every `y[m]` is the same row dot, just on a different worker — bitwise-deterministic,
greedy stays byte-identical to llama.cpp):

1. **Persistent pool** (workers spawned once, not ~340×/token): 1.96 → 5.32.
2. **Spin-wait barrier** instead of a condition variable — at ~340 sync points/token a
   CV's ~10 µs kernel wake dominates; workers busy-poll an atomic generation counter with
   `_mm_pause`, dropping dispatch latency to tens of ns.
3. **Thread affinity** (`SetThreadAffinityMask`, one thread per logical CPU) + an
   off-by-one fix (the caller counts as a thread): stops OS migration thrashing each
   core's L2, and on this hybrid part places the first 8 threads on the P-cores. → 5.59.

### Why we win at 1 core but llama wins at 20 (the bottleneck moves)

This *looks* contradictory — faster per core, yet slower with all cores — but the two
measurements are different races, because the bottleneck shifts:

- **At 1 core, the bottleneck is the core.** One core streams weights at ~20 GB/s, far
  below the chip's ~76 GB/s memory ceiling — the bus has spare capacity, so the limit is
  how fast *that core* issues loads and does the int8 math. Our kernel does that a touch
  tighter → **1.60 vs 1.37, we win.** A *compute-throughput* race.
- **At 20 cores, the bottleneck is the shared DRAM bus.** Every core now pulls from the
  same memory controller, which saturates at its physical peak. Per-core compute speed
  becomes **irrelevant** — all cores are just waiting on memory. The race becomes "who
  keeps the bus fullest," and llama wins it by ~7% (76 vs 71 GB/s) because AVX-VNNI +
  weight repacking keep more memory requests in flight. A *bandwidth-extraction* race.

| | bottleneck | what's measured | winner |
|---|---|---|---|
| 1 core | the core (bus idle) | compute throughput / core | **us** — 1.60 vs 1.37 |
| 20 cores | shared DRAM bus | aggregate bandwidth extraction | **llama** — 6.00 vs 5.59 |

So the winner flips because the *metric* flips. (Analogy: one person drawing from a
wide-pipe well — the faster runner wins; twenty people drawing from the *same* well —
the well's refill rate is the limit, not how fast anyone runs, and llama's crew just uses
slightly better buckets.) Our scaling looks lower (3.5× vs 4.4×) for the same reason: we
*start* higher per core, so there's less room to climb before hitting the wall — and our
wall is a touch lower (71 vs 76 GB/s).

### The physics of the wall

This CPU is an **Intel Core Ultra 9 275HX — Arrow Lake-HX: 8 P-cores + 16 E-cores, 24
threads, no SMT**. Single-token decode reads **every weight exactly once** (zero
arithmetic reuse), so throughput is **hard-capped by DRAM bandwidth ÷ model bytes**, not
compute. Measured (tok/s × 12.65 GB/token):

| threads | tok/s | speedup | eff. BW | note |
|--------:|------:|--------:|--------:|------|
| 1 | 1.60 | 1.0× | 20 GB/s | one core ≈ ¼ of DRAM BW |
| 2 | 2.25 | 1.41× | 28 GB/s | |
| 4 | 3.64 | 2.27× | 46 GB/s | |
| 8 | 4.42 | 2.76× | 56 GB/s | **P-cores only** (logical CPUs 0–7) |
| 16 | 5.52 | 3.45× | 70 GB/s | + E-cores; near the BW wall |
| 20 | **5.59** | 3.49× | **71 GB/s** | best — leaves 4 cores for the OS |
| 24 | 4.70 | — | 59 GB/s | **regresses**: spin workers on every core starve the OS + main thread |
| llama (24) | 6.00 | — | 76 GB/s | extracts ~7% more BW (VNNI/prefetch) |

Two regimes explain why *2 cores ≠ 3.2 tok/s*:
- **Low core count → Amdahl.** Only the GEMVs parallelize; the per-token serial work
  (RMSNorms, RoPE, per-head QK-norm, the attention softmax, GeGLU, residuals, activation
  quantization) runs on the caller. That serial fraction (~25–30%) caps low-end scaling,
  so 1→2 cores gives 1.41×, not 2×.
- **High core count → bandwidth wall.** By ~8–16 cores the memory controller saturates
  at **~70 GB/s**, so extra cores add almost nothing (8→20 cores: +1.17 tok/s total).
  The **P-cores are ~4× more bandwidth-efficient per core** than the E-cores (P: ~7.0
  GB/s/core to 56 GB/s at 8 cores; each added E-core: ~0.1 tok/s), so `-t 8` (P-cores
  only) is the efficiency sweet spot and `-t 20` the throughput sweet spot. (The dynamic
  32-row tile cursor is what lets the fast P-cores and slow E-cores share work without the
  barrier waiting on a straggler.) **Both engines land at ~5.5–6 tok/s — but that is *not*
  the bandwidth wall**, as the next section proves: a pure-read probe pulls the *full*
  ~104 GB/s out of this bus, which would be ~8 tok/s. The decode GEMV stops short at
  ~71–76 GB/s because the int8 math interleaved with the loads throttles per-core
  load-issue, not because the bus is full. The recoverable gap is a *kernel* one
  (AVX-VNNI + prefetch), not a threading one.

`sub0llm-gemma -t N` picks the thread count (`-t 1` = single-core baseline, `-t 8` =
P-cores only); the default leaves 4 cores free. The last ~7% vs llama is its extra
bandwidth extraction (AVX-VNNI lets each core issue more outstanding loads, plus weight
repacking for friendlier access) — a kernel/prefetch refinement, not a threading one.
Multimodal (vision/audio) and the 26B-A4B MoE variant are out of scope (text-only, dense).

### Is 71 GB/s actually the wall? Paper vs exercised vs real (`bench_membw`)

The decode table above *infers* a bandwidth ceiling from tok/s. To check it directly —
rather than trusting the spec sheet — `benchmarks/bench_membw.cpp` is a STREAM-style probe
that hammers raw DRAM with the same threading regime the model uses (pinned threads,
`std::barrier` sync) and reports three kernels: **Read** (pure loads — the metric that
matches single-token decode, which is read-only weight streaming), **Copy** (1 read + 1
write) and **Triad** (2 read + 1 write — the classic STREAM kernel).

```bash
cmake --build build-native --target bench_membw
./build-native/bin/bench_membw --peak 102
```

Measured on this machine (256 MiB/array, 30 reps, threads pinned), giving the three
numbers that bracket the LLM result:

| level | GB/s | as tok/s (÷12.65 GB) | source |
|-------|-----:|---------------------:|--------|
| **Theoretical (JEDEC paper)** | 102 | ~8.1 | RAM spec |
| **Exercised max — pure Read** | **104 @ 20t** | **~8.2** | `bench_membw` |
| Exercised max — Triad (2R+1W) | 61 | — | `bench_membw` |
| Exercised max — Copy (1R+1W) | 55 | — | `bench_membw` |
| LLM real — our decode | ~71 | 5.59 | `sub0llm-gemma -t 20` |
| LLM real — llama.cpp | ~76 | 6.00 | `llama` eval time |

**The probe overturns the "~6 tok/s is the hard wall" reading.** Pure reads reach the
*entire* 102 GB/s paper peak (the JEDEC number is real, not aspirational, for read-only
streaming) — so the true decode ceiling is **~8 tok/s, not ~6**. Our GEMV leaves ~30 GB/s
on the table because each core interleaves the int8 dot (`maddubs` + scale-decode) with
its loads; that instruction stream limits how many loads stay *in flight*, so the bus is
starved, not saturated. A pure-read loop has no such interleave and fills the bus.
→ The recoverable win is **per-core load-issue density**: **AVX-VNNI** (`vpdpbusd` —
one instruction for the int8 dot, vs our three-op chain → more load slots) plus
**software prefetch** of the next weight row ~1–2 iterations ahead. This is exactly the
lever behind llama's +7%, and the read ceiling says there is ~1.4× of headroom above 71,
not the few percent the inferred-wall reading implied.

#### What about writes — is write saturation / cross-thread sync a lever too?

Short answer for **decode: no.** A single-token forward *writes* almost nothing — the
activations are a few KB per layer (a `(1, D)` vector), dwarfed by the ~12.65 GB of weight
*reads*. So write bandwidth never binds decode, and there is no cross-thread write
coordination to tune: each pool worker writes its own disjoint `y[m]` rows, fully
independent — no "delaying / synchronising writes" buys anything when the writes are
negligible. The two write concerns that *do* exist are correctness-of-performance issues,
not saturation levers:

- **False sharing** — if two threads wrote `y[]` elements sharing a 64-byte cache line,
  the line would ping-pong between cores (coherence/RFO storms). Our **32-row tile cursor**
  already prevents this: each chunk is a contiguous 128-byte-aligned run, so worker writes
  almost never share a line. This is why the tiling is correct, not faster per se.
- **Write-allocate / RFO** — a store to a fresh line first *reads* it for ownership, so a
  "write" costs read+write bandwidth. That is exactly why Copy/Triad (~55–61 GB/s) come in
  at ~half the Read ceiling. Non-temporal stores (`_mm256_stream_ps`) bypass RFO by writing
  straight to memory.

Where writes genuinely *do* matter is the **write-heavy regimes** — training (gradient and
optimiser-state writes) and long-context prompt processing / KV-cache fills — whose ceiling
is the **Copy/Triad ~55–61 GB/s**, not the 104 read figure. There, NT stores for the large
write-once-read-later buffers (KV cache, activation checkpoints) are the relevant lever,
and avoiding false sharing on shared accumulators (e.g. gradient reductions) is the
cross-thread sync concern. For the decode hot path this chapter optimises, though, **reads
are the whole story** and the next kernel step is VNNI + prefetch.

#### AVX-VNNI kernel — measured (the bandwidth-wall thesis, confirmed)

The int8 dot `dot_q8_0_q8_0` now compiles to **AVX-VNNI `vpdpbusd`** (one unsigned×signed
4-byte dot accumulated to int32) where the host supports it — `_mm256_dpbusd_avx_epi32`
on this Arrow Lake part, `_mm256_dpbusd_epi32` on AVX-512-VNNI — replacing the
`maddubs`(int16) + `madd`(int16→int32) two-op chain, plus a software prefetch of the next
weight row. `-DSUB0LLM_DISABLE_VNNI` forces the old `maddubs` fallback, giving a
**controlled A/B from identical source** (the only valid test on a thermally-drifting
laptop — the two binaries are run *interleaved* so warmup/throttle cancels). Greedy output
is byte-identical between the two (VNNI changes only instruction selection, not the
arithmetic: both reduce the same int8 products to the same int32).

| build | **1 thread** (core-bound) | **20 threads** (bus-bound) |
|-------|--------------------------:|---------------------------:|
| `maddubs` (VNNI off) | 1.24 tok/s | 4.87 tok/s |
| **`vpdpbusd` (VNNI on)** | **1.32 tok/s** | 4.82 tok/s |
| delta | **+6.5%** | within noise (±1%) |

This is the bandwidth-wall thesis falsifiably confirmed, not merely asserted:
- **At 1 thread the core is the bottleneck**, so denser instructions (fewer uops per
  consumed weight byte → more loads in flight) show through directly: **+6.5%**,
  repeatable every round. This matches the microbench, where the kernel alone gained
  1.4–2.6× on the largest GEMVs (`sub0llm-qbench`).
- **At 20 threads the shared DRAM bus is the wall** — every core is already stalled on
  memory, so a faster *per-core* kernel buys nothing measurable (4.82 vs 4.87 is noise).
  The win is real but **invisible behind the bus**, exactly where the wall theory says it
  must be.

So VNNI is the right kernel (and is now the default), but on *this* memory-bound 12B-on-CPU
workload its decode payoff only appears below the bandwidth knee. The multi-thread gap to
llama is therefore **not** the int8 op — both engines now issue `vpdpbusd` — it is the
remaining **load-scheduling / weight-repacking** edge (ggml pre-shuffles weights into a
VNNI-friendly layout so each core sustains more outstanding loads against the bus). Same
session, `llama-bench tg32 -t 20` = **5.15 tok/s** vs our best **4.82** → **~94%**.
Closing the last ~6% is a memory-access-pattern problem (repack + deeper prefetch), the
one lever the read probe shows still has headroom.

#### The Q8 decode ladder — the gap is *orchestration*, not the kernel

The read-vs-decode framing above leaves a question the STREAM kernels can't answer: is our
*int8 GEMV kernel* itself leaving bandwidth on the table, or is it the *per-token plumbing
around it*? `bench_membw` now walks a **ladder** from pure memory to the real kernel — pure
f32 **read** → **q8_stream** (read the Q8 weight bytes, ~no math) → **q8_gemv** (the actual
`dot_q8_0_q8_0` over a 512 MiB weight buffer, activation reused across rows = the decode hot
path) — all thread-swept and core-pinned, best-of-2 per cell with cooldowns, and a
thermal-drift recheck. Representative run (this machine, exercised GB/s):

| threads | read | q8_stream | q8_gemv | gemv / read |
|--------:|-----:|----------:|--------:|------------:|
| 1  | 28.6 | 17.1 | 18.4 | 64% |
| 4  | 59.6 | 43.7 | 47.4 | 80% |
| 16 | 91.9 | 79.2 | 84.6 | 92% |
| 24 | 108.5 | 84.1 | 94.2 | 87% |

The diagnosis is decisive: **`q8_gemv` ≈ `q8_stream` ≈ `read`** at scale (94 vs 84 vs 108
GB/s) — adding the int8 dot to a raw byte stream costs almost nothing, so the **kernel is
memory-bound, not compute-bound**, and it reaches **~94 GB/s in isolation (87% of the pure-
read ceiling)**. But real Gemma decode only sustains **~71 GB/s** (5.59 tok/s × 12.65 GB) =
**75% of what the same kernel does over one big buffer.** So the missing ~25% is **not** in
the kernel — it is the **per-token orchestration**: ~25–30% serial main-thread work (the
RMSNorms, RoPE, QK-norm, attention softmax, GeGLU, residuals that don't parallelize), **340
barriers/token**, and **small per-GEMV M** (real projections are M = 2048…262144, far below
the ladder's single 131 586-row dispatch, so sync overhead and load-imbalance eat a larger
slice). That refines the earlier "repack/prefetch" reading: vs *llama* the residual ~6% is
indeed its VNNI-friendly repack, but vs the *silicon* our bigger recoverable win is the
orchestration layer — **fuse/batch GEMVs to cut barrier count, and overlap the serial
fraction** — which the ladder shows is worth ~20 GB/s, more than the kernel has left to give.
(The recheck also makes the **thermal drift** concrete: −18% between a cool and a hot `read`,
so every cross-engine number must be **interleaved A/B/A/B**, never two long back-to-back runs.)

## The stack

```
  model.gguf ──► sub0llm-specialize ──► <model>_spec.hpp   (constexpr struct)
                       │                 <model>_arch.json  (feature manifest)
                       │
                       ▼
        ch27 / gemma4-cli / gemma4-server  #include the generated header
                       │
                       ▼
        monomorphized forward  (StaticSpec S → std::array buffers, unrolled loops)
```

### 1. `sub0llm-specialize` (the front-end) — **done**

Reads a GGUF's *metadata only* and emits:

- `generated/<model>_spec.hpp` — a struct where every axis is `static constexpr`,
  with feature flags (`use_qk_norm`, `tied_embeddings`, `norm_plus_one`,
  `attn_qkv_bias`, `activation`, `sliding_window`, `local_global_stride`) and a
  `static_assert` sanity gate. Features are detected from the GGUF architecture
  string and tensor-name presence — so the numbers are the file's, not a guess.
- `generated/<model>_arch.json` — the same manifest as JSON, for tooling.

```bash
sub0llm-specialize --model models/Qwen3-0.6B-Q8_0.gguf \
                   --out-dir chapters/ch27_specialized_build/generated
```

The committed `generated/qwen3_0_6b_q8_0_spec.hpp` is the reference output, so the
chapter builds with no model file present.

### 2. `StaticSpec` concept + monomorphized forward — *in progress (P2)*

`include/sub0llm/nn/static_spec.hpp` defines the `StaticSpec` concept (the
contract the generated header satisfies) and `SpecDerived<S>` (constants a
forward pass indexes with: GQA fan-out, q/kv projection widths,
`1/sqrt(head_dim)`, KV-cache floats per token — all folded at compile time). The
`ch27_specialized_build` demo proves the generated spec satisfies the concept and
prints the baked-in dimensions. Next: a monomorphized RMSNorm + attention + FFN
forward microbenchmarked against the dynamic `Tensor` path.

### 3. Gemma-family architecture — *planned (P3)*

Gemma differs from our Qwen3/LLaMA path: GeGLU (GELU gate) not SwiGLU, embeddings
scaled by `sqrt(D)`, `(1 + weight)` RMSNorm, and interleaved local
(sliding-window) / global attention with a dual RoPE base. These are added to the
model **gated by the manifest flags** so Qwen3 is unaffected, and load the moment
a Gemma 4 GGUF is dropped in `models/`.

### 4. `gemma4-cli` / `gemma4-server` + llama.cpp baseline — *planned (P4)*

Dedicated binaries built against the Gemma spec, plus a harness that runs the same
prompts through llama.cpp and tabulates load time / tokens-per-second / RSS.

## Correctness is a gate on performance

**A faster forward that produces different logits is a regression, not a win.**
llama.cpp is therefore the *correctness oracle*, not only the perf baseline. No
speedup is reported for the specialized path until it matches llama.cpp on the same
model + prompt:

1. **Tokenizer parity** — our BPE token IDs equal llama's for the same text.
2. **Logit parity** — next-token logits vs llama's within tolerance (max-abs-diff /
   cosine) — the strictest check, catches weight-layout / RoPE / QK-norm / GeGLU /
   `(1+w)`-norm / embed-scale / window bugs.
3. **Greedy parity** — identical argmax continuation at temperature 0.
4. **Perplexity parity** — within tolerance on a fixed passage (`llama-perplexity`).

This runs as a `verify` mode in the runner (and unit tests where feasible), and
gates every performance claim.

## Qwen3-0.6B: correctness parity + perf baseline vs llama.cpp

Verified with `sub0llm-verify` against the llama.cpp `b9334` CPU prebuilt:

| Check | Ours | llama.cpp | Verdict |
|-------|------|-----------|---------|
| Tokenize "The capital of France is" | `785 6722 315 9625 374` | identical | **exact** |
| Greedy (temp 0) continuation | " Paris. The capital of Italy is Rome. The capital of Spain is Madrid…" | (chat-templated) | **all facts correct** |
| Top next-token logit | ' Paris' @ 16.56 (margin 2.2) | — | **correct** |
| Perplexity (repetitive 512 tok) | 1.71 | — | sane |
| **Generation throughput** | **11.6 tok/s** | **17.0 tok/s** | we're at **~68%** |
| Prompt throughput | — | 196 t/s | baseline |
| Model size (Q8 dequant→f32 vs Q8) | f32 in RAM | 604 MiB | (we dequantize) |

**Correctness is established** for the Qwen3 forward (tokenizer exact; greedy
produces correct world-facts over a long continuation; argmax correct). Exact
perplexity-vs-`llama-perplexity` magnitude matching is deferred — methodologies
differ (llama uses strided sliding windows) and our full-vocab LM head over all
positions makes a 512-token teacher-forced pass minute-scale.

**Honest headroom:** our *dynamic* engine already runs at ~68% of llama.cpp's
generation speed — so there's a real ~1.46× gap to close just to match, and
llama.cpp's CPU kernels are heavily tuned. That's the bar the specialized build is
measured against; note also we currently dequantize Q8→f32 (≈4× the RAM of llama's
native Q8), a memory cost to address.

## End-to-end: Q8 wired into generation — we beat llama.cpp, on speed and RAM

The int8 kernels are wired into the whole generation path:
`ModernGPT::quantize_for_inference(drop_f32)` quantizes **attention Q/K/V/O, FFN
projections, and the LM head** to Q8; `forward_one()` runs int8 throughout. With
`drop_f32` it frees the f32 weights afterward — including the tied embedding table
(the embedding lookup then dequantizes its row from the Q8 head). `sub0llm-verify
--q8` (keep f32) / `--q8-only` (drop f32) toggle it.

| Qwen3-0.6B generation (this CPU) | tok/s | vs llama.cpp | RSS (peak) |
|----------------------------------|------:|-------------:|----:|
| our f32                          | ~12   | 72%  | 2323 MiB |
| our Q8, f32 kept                 | ~23   | 143% | 2927 MiB |
| our Q8-only (drop after load)    | ~22   | 124% | 853 MiB |
| **our Q8-load (quantize-on-load)** | **~23** | **134%** | **651 MiB (3.57×)** |
| llama.cpp `tg64`                 | 17.0  | 100% | ~Q8 |

- **~2× over our own f32** and **byte-identical greedy** output throughout
  ("Paris. The capital of Italy is Rome…") — fast *and* correct, across all modes.
- **`--q8-load` is the winner**: `load_gguf_model_q8()` copies the GGUF's raw Q8
  blocks straight into the int8 buffers (our `BlockQ8_0` is byte-identical to the GGUF
  block; attention is sliced per head), so **f32 is never materialized** — peak RSS is
  **651 MiB (3.57× less than f32)**, *and* it runs at int8 speed (134% of llama).
- This is what makes **Gemma 4 12B loadable**: its 12.7 GB Q8 stays ~13 GB instead of
  spiking to ~50 GB f32. (`f32` weight elision threads a `alloc_weights=false` flag
  through the constructors; tied embedding served by dequantizing a row from the Q8 head.)

## Memory layout & tiling — where it helps (and where it can't)

The natural next lever is cache locality. The honest analysis, by regime:

- **Single-token generation (T=1, GEMV)** — what the table above measures — is
  **bandwidth-bound with zero weight reuse**: every weight is read exactly once per
  token, so it must stream from RAM no matter how it's laid out. Tiling/cache-blocking
  the *weights* therefore can't help here; the only lever that moved the needle was
  shrinking the bytes (Q8 → 3.76× less to stream), which we did. Our layout is already
  the cache-friendly one: row-major `(out, in)`, each output row read contiguously, the
  activation (tiny) hot in L1.
- **Prompt processing (T>1, GEMM)** — this is where tiling pays off, now **measured**
  (`sub0llm-qbench --batch T`). Same FLOPs and pre-quantized inputs; the only difference
  is loop order — per-column GEMVs re-stream W once *per* column, while the row-reuse
  matmul (`matmul_q8_0_q8_0`, rows outer / columns inner) streams W *once* and reuses
  each row across all T columns from cache:

  | GEMV (M×K), T=16 | per-column GEMVs | **row-reuse matmul** |
  |------------------|-----------------:|---------------------:|
  | ffn gate/up 3072×1024 | 38.7 GFLOP/s | **52.0 (1.34×)** |
  | ffn down 1024×3072 | 28.4 | **53.0 (1.86×)** |
  | lm head 151936×1024 | 28.9 | **55.1 (1.91×)** |

  The win is biggest on the memory-bound LM head (1.91×) — exactly where reducing W's
  memory traffic by ~T× matters most. This is the prompt-throughput lever; KV-cache
  layout (`K[layer][head]` as `(seq, head_dim)`) is the analogous one for attention.

So: for the generation hot path, **representation (Q8) was the locality win** (no reuse
to tile); for prompt/batched throughput, **row-reuse tiling is a measured 1.3–1.9×**.

## Defaults: the best options are now the defaults

`sub0llm-cli` and `sub0llm-server` **default to Q8 quantize-on-load** for GGUF models
(fastest generation, ~3.6× less RAM, no f32 peak). `sub0llm-cli --f32` opts back into
full precision. The episodic tool keeps the f32 path (it trains/writes LoRA deltas).

## Does *specialization* win? (honest verdict)

Yes — but it's worth being precise about *which* specialization paid off:
- **Representation/kernel specialization — decisive.** Choosing the best weight format
  (Q8) + int8 kernel + dropping the f32 copies for *this* model on *this* CPU is what
  delivered 2× speed, 2.7× RAM, and beating llama.cpp. This is the heart of "a
  dedicated build for one model."
- **Compile-time *dimension* monomorphization — marginal here.** Baking shapes into
  `constexpr` (the spec) gave ~1.02× on the FFN GEMV microbench: these ops are
  memory/compute-bound, not loop-overhead-bound, so the compiler's runtime-dim code is
  already near-optimal. It would matter more for control-flow-heavy or tiny-tensor ops.

So the specialized-build thesis holds strongly in its *useful* form (pick the optimal
representation + kernels + memory layout per model/CPU), while the narrow
"constexpr dims make the compiler faster" effect is small for transformer GEMVs — a
finding worth stating plainly rather than assuming.

## Weight representation: profile, don't assume (`sub0llm-qbench`)

We currently dequantize everything to f32 (4 B/weight). Four resident
representations, profiled per GEMV shape (AVX2/F16C native, vs an equally
vectorized f32 baseline; kernels in `backends/cpu/quant.{hpp,cpp}`, correctness
gated by `test_quant.cpp`):

- **f32** — baseline, 4 B/weight.
- **f16** — half precision, 2 B/weight; one F16C instruction converts 8→f32.
- **Q8 dequant-on-the-fly** — Q8 (1.06 B/weight), expand to f32 then f32 FMA.
- **Q8 quantized int8** — Q8, quantize the activation once, then `maddubs` int8
  block dots accumulated as float (one reduction per row; F16C scales).

  Includes the **dequant target** axis — Q8 expanded to an f16 vs f32 intermediate
  before the f32 FMA — so every dimension is covered.

**Qwen3-0.6B GEMVs** — speedup vs f32 (relRMS in text):

| GEMV (M×K) | f16 (2 B) | Q8 deq→f32 | Q8 deq→f16 | **Q8 quantized** |
|------------|----------:|-----------:|-----------:|-----------------:|
| attn q_proj 2048×1024 | 1.16× | 0.98× | 0.76× | **1.38×** |
| attn kv   1024×1024 | 1.15× | 1.03× | 0.75× | **1.55×** |
| ffn gate/up 3072×1024 | 1.39× | 1.20× | 0.90× | **1.68×** |
| ffn down  1024×3072 | 1.10× | 1.01× | 0.84× | **1.48×** |
| lm head 151936×1024 | 1.27× | 0.85× | 0.68× | **1.74×** |

relRMS: f16 ≈ 1.5e-4 · Q8 paths ≈ 3–5e-3. Gemma-4-12B shapes show the same ordering,
amplified (bigger = more memory-bound): Q8-quantized **1.65–2.50×**, f16 **1.0–1.71×**,
both Q8-dequant paths ≤ 1× on the large layers.

**Findings (the answer is shape- and goal-dependent — as predicted):**
- **Q8 quantized int8 is the overall winner**: fastest on *every* shape
  (**1.37–1.77× Qwen, up to 2.4× Gemma**) *and* smallest RAM (**3.76×**), at ~0.4%
  relRMS. The earlier "loses on the LM head" result was a kernel bug — fixed with
  **F16C hardware scale conversion** + **accumulate-as-float** (one reduction per
  row, not per block), which flipped the memory-bound LM head from 0.82× to 1.77×.
- **f16 is the near-lossless middle**: 1.0–1.36× faster, **2× less RAM**, relRMS
  **~1.5e-4** (≈10× more accurate than Q8). Best when accuracy is critical or the
  model ships as f16/bf16. Notably f16 **beats Q8-dequant** on both speed and (per
  byte) is the better f32-accuracy path.
- **Q8 dequant-on-the-fly is dominated**: same RAM as Q8-quantized but slower —
  keep only as the "exact-f32-activation" variant.
- **Dequant target — f16 vs f32 intermediate**: for single-token GEMV, expanding
  Q8 to an **f16 intermediate is the *worst* path (0.68–0.90×)** — the extra
  round-trip is pure overhead with no reuse and no accuracy gain (it rounds an
  already-Q8 value). The f16 intermediate only pays off in **batched/tiled** matmul
  (prompt processing, T>1) where a dequantized tile is reused across many columns
  and the half-width scratch eases cache pressure — a separate batched experiment.

→ Direction: store **Q8 resident + quantized int8 matmul** for max speed/min RAM
(the LLM default), with **f16** as the accuracy-preserving option. This also makes
**Gemma 4 12B loadable** — its 12.7 GB Q8 stays ~13 GB instead of ballooning to
~50 GB as f32.

## Microbench finding (honest)

A single SwiGLU FFN GEMV (D=1024, F=3072), compile-time vs runtime dims, both
`-march=native`, vectorized (8 accumulators): **~1.02× — essentially equal**
(7.95 vs 7.81 GFLOP/s). A large GEMV is throughput/bandwidth-bound, so the *value*
of the loop bound doesn't change the generated inner loop — matmul throughput is
shape-agnostic. **The monomorphization win is not raw matmul speed** (that's
BLAS-class either way); it must come end-to-end from eliminating per-op dispatch
and allocation, unrolling the many *small* fixed-dim loops (head_dim, RoPE,
softmax) with the remainder folded away, and cross-kernel fusion. That is what the
P4 end-to-end comparison vs llama.cpp measures.

## Build & run

```bash
cmake --build build-debug --target ch27_specialized_build
./build-debug/bin/ch27_specialized_build

# Re-specialize the default spec from the local GGUF:
cmake --build build-debug --target ch27-regen-spec

# Target a different generated spec:
cmake -B build-debug -DCH27_SPEC=<stem> -DCH27_SPEC_TYPE=<sub0llm::spec::Struct>
```

## Status

| Phase | Item | State |
|-------|------|-------|
| P1 | GGUF metadata: `norm_eps`, `sliding_window` parsing | ✅ |
| P1 | `sub0llm-specialize` codegen → spec.hpp + arch.json | ✅ proven on Qwen3-0.6B |
| P2 | `StaticSpec` concept + `SpecDerived` + demo binary | ✅ |
| P2 | Monomorphized forward + microbench vs dynamic path | ⏳ |
| P3 | Gemma-family arch features (manifest-gated) | ⏳ |
| P4 | `gemma4-cli`/`server` + llama.cpp baseline + writeup | ⏳ (needs Gemma 4 GGUF + llama.cpp) |
