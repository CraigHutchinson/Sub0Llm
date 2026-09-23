# Speculation, the n-gram table, and the MoE — how they interact, and in what order to build them

**Status:** design, 2026-09-23. Nothing here is implemented yet. It exists to set priority and order across
three threads that turn out to share one constraint: MTP / speculative decoding, the model's n-gram (PLE)
table, and routed-expert (MoE) efficiency. Research sources are cited inline. The research pass behind
this doc quoted its primary sources; claims it could not verify are marked as such here too.

## Summary

- **The n-gram table should move up the priority list — for fidelity first.** The engine omits it, so we
  are not running the real model. llama.cpp includes it by default, so our outputs and the oracle's
  differ for a reason unrelated to any kernel. Per token it is cheap: 16 row reads (~1.4 KB) plus ~35 MB
  of fusion projections. The work is building the *real* gated fusion, which the shipped n-gram stage
  does not implement, plus a tier for a 28.8 GB table.
- **Speculation is worth building, but its ceiling here is set by the routed experts, not the backbone.**
  Verifying K drafted tokens reads the backbone once. It does not reduce expert work: with 512 experts,
  each verified token brings its own ~10 experts per layer, and expert compute is ~22% of a token today.
  A time model (§4) puts the realistic ceiling on this CPU at roughly 1.2–1.5x, not the 2–3.6x GPU
  papers report.
- **So the next MoE lever raises both plain decode and speculation's ceiling:** the routed-expert kernels
  run at ~5 GB/s effective against a 79 GB/s roof. They are compute-bound on IQ unpacking, the same
  shape O5 just fixed for the backbone's K-quants. Named here as **O7**.
- **The n-gram → MoE link is mostly about memory, not prediction.** The table and the experts compete
  for the same 63 GB, and the table is the one that can tier down, because its rows are known exactly
  the moment a token exists. Predicting expert routing from n-gram context is plausible but unproven;
  the literature finds token-keyed predictors weaker than hidden-state ones. It is a measure-first item,
  and it only pays if experts start paging from disk.
- **One cheap measurement answers several of these questions at once** (§6, E1): log real expert routes
  over a mixed prompt set. It gives the expert-union growth that bounds speculation on *this* model, and
  the predictability of routes from n-gram context, before either is built.

**Recommended order:** finish O5 (in flight) → E1/E2 measurements → PLE fidelity (P1) → O7 expert
kernels (P2) → batched verify path (P3) → prompt-lookup speculation (P4a) → MTP drafter (P4b). Expert
prediction (P5) is built only if E1 and the post-PLE memory budget both say it would pay.

## 1. Verified facts this design rests on

**Model and files.**
- The local unsloth `UD-IQ1_S` GGUF has **no MTP head**: 1,224 tensors, `blk.0`–`blk.47`, and no
  `mtp`/`nextn` tensors or metadata (parsed from the three shard headers, 2026-09-23).
- The original HF checkpoint does have it: 31 `mtp.*` tensors across safetensors shards 37–124 of 131
  (parsed from the real `model.safetensors.index.json`).
- The n-gram table is in the GGUF: `per_layer_token_embd.weight`, IQ4_NL, `[160, 320001536]`, ~28.8 GB,
  injected at decoder layer 1 (`ple_layer_ids=[2]`, 1-based). The engine omits it and six small
  `blk.1.ple_*` tensors (`docs/WP4_SCOPE.md` §5).

**The MTP head** (reference: vLLM `vllm/models/qwen4_exp/nvidia/mtp.py`; `transformers` implements no MTP
forward at all and drops `mtp.*` weights on load):
- It is **one full QSA decoder layer, including the full 512-expert routed MoE and shared expert**, plus
  a Gated-Residual mixer. That is where the "4B params" comes from. It is not a lightweight head.
- Inputs: the next token's embedding and the backbone's **pre-final-mixer multi-stream hidden state**
  (`[T, hc_count*H]`), each through its own RMSNorm and linear (`fc_embedding`, and `fc_hidden` shared
  across HC streams). This is not DeepSeek-V3's concat-then-project MTP; don't port that design.
- It **reuses the main model's embedding and lm_head** (the checkpoint ships none of its own), and
  **runs with PLE forced off**.
- Only one trained layer exists (`mtp.num_hidden_layers = 1`). Deeper drafts reuse it recurrently,
  feeding its own multi-stream output back in.
- `mtp.hybrid = true` is consumed nowhere we could find. **Unverified** what it controls.
- Measured acceptance: SGLang reports an **accept length of 3.3 including the bonus token on this exact
  model** (B200, NVFP4, batch 1). vLLM reports per-position acceptance of 0.897 / 0.719 / 0.476 for
  Qwen3-Next's shipped MTP head: acceptance decays with depth because the head is trained on ground
  truth, not on its own drafts. Both are GPU, higher-precision numbers: treat them as upper bounds.
- The local llama.cpp fork has generic `nextn`/MTP speculative infrastructure (used by the DeepSeek and
  GLM families), but its `qwen4exp` model does not use it.

**The n-gram (PLE) table** (reference: llama.cpp fork `src/models/qwen4exp.cpp`, `build_ple` and
`llm_graph_input_ple::set_input`; agrees with `transformers` and `docs/QWEN4_PREVIEW_REFERENCE.md`):
- **Rows for token t are a pure function of tokens ≤ t**: t itself plus its two predecessors, hashed into
  16 heads (8 bigram, 8 trigram), each a prime-sized ~20M-row slice. So the row set is known the instant
  a token — sampled or drafted — exists.
- Per token: 16 rows × 160 elements × 18/32 bytes = **1,440 bytes at IQ4_NL**. (`NGRAM_TABLE_TIERED_STORAGE.md`
  quotes 5,120 bytes, which is the bf16 figure.)
- **The real fusion is a gated multi-branch injection, not the additive residual the engine's shipped
  n-gram stage implements.** The gathered 2,560-wide embedding goes through `ple_key`/`ple_value`, a
  sigmoid gate from the normalized hidden query against the normalized key, a short causal depthwise
  conv dilated by `ngram_size` (`ple_conv1d`, `ple_norm_conv`), and then into the Gated-Residual stream.
  `docs/NGRAM_EMBEDDING.md` §7 had deferred exactly this gated variant, and the conv step is not
  described there at all.
- llama.cpp serves the table **lazily by default**: a tensor marked lazy and larger than 4 GiB is
  memory-mapped and read by row on demand, always host-side. SGLang keeps it in pinned host memory and
  gathers rows into a small device buffer, measuring **−23.5 GiB GPU memory for a −0.07% throughput
  change** on this model.
- Engram (DeepSeek, arXiv:2601.07372) is the same family of idea. It states that deterministic
  addressing "enables runtime prefetching from host memory, incurring negligible overhead", and treats
  it as a sparsity axis complementary to MoE, not a variant of it.

**Speculation on MoE models** (measured, GPU, batch 1):
- *The Limits of Speculation* (arXiv:2609.22156; Qwen3-Coder-30B, EAGLE-3, A100) models cost as
  proportional to the unique experts touched across the verify pass. It reaches a ~2.34x ceiling, and
  average accepted length is ~2.1.
- EVICT (arXiv:2605.00342; Qwen3-30B-A3B, 128 experts) cuts verified tokens adaptively, for 26.6% lower
  verify latency and 1.21x over EAGLE-3.
- Both models have ~128 experts. With 512, drafted tokens overlap less, so the union grows closer to
  linearly. **This is unmeasured anywhere for a 512-expert model**, which is what E1 is for.

**Expert-routing prediction** (literature; mostly search-summary confidence, not independently fetched):
predictors keyed on the previous layer's hidden state dominate and report 80–97% accuracy (Fate,
HOBBIT, MoE-Infinity, ExpertFlow). Token-identity predictors are reported weaker, especially in early
layers. All are probabilistic, unlike the n-gram table's exact addressing.

## 2. Where time and bytes go today

At the recommended flags, decode is ~0.211 s/token (`docs/optimization/opportunities/O4_*.md`):

| phase | ms/token | nature |
|---|---:|---|
| GDN mixer | 73 | backbone bytes (projections) + serial recurrence |
| routed experts | 46 | **compute-bound**: ~0.25 GB/token read in 46 ms ≈ 5 GB/s |
| QSA mixer | 32 | backbone bytes + attention |
| Gated Residual | 31 | backbone bytes |
| lm_head | 21 | backbone bytes (1.27 GB bf16) |
| other | ~8 | router, shared expert, combine |

Backbone bytes are ~8.35 GB/token in bf16. O5's native path cuts them to roughly 3–4 GB once the GDN
roles are included (O5 phase 2b-2b, in flight). After that, **routed experts become the largest single
phase**, and they are nowhere near the bandwidth roof.

## 3. The n-gram table: priority and placement

### 3a. Priority — fidelity comes first

The model we run is not the model we loaded: one additive signal at layer 1 is missing, and the external
oracle includes it. That undermines every quality comparison against llama.cpp, unless the table is also
disabled on the llama.cpp side (`WP4_SCOPE.md` §5's option (a)). The cost per token is small:
- 16 random row reads (1.4 KB);
- the six `blk.1.ple_*` tensors (~35 MB of projections, +0.4% of backbone bytes);
- a short depthwise conv.

So this is a correctness debt worth paying early, not a performance feature.

### 3b. Placement — the table tiers down, the experts stay up

The resident sets don't fit together today:

| component | bytes | access per token |
|---|---:|---|
| bf16 backbone blob | 9.8 GB | all of it |
| native backbone sidecar (after O5) | ~3–4 GB | all of it |
| expert sidecar (`.moeq`, mapped) | 37 GB | ~0.25 GB, 480 expert planes |
| n-gram table | 28.8 GB | 1.4 KB, 16 rows |
| MTP head, if added | ~1.5–2.5 GB quantized (unmeasured) | one layer + its experts, per draft token |

RAM is 63 GB. **Once O5 lets the bf16 blob go, the expert sidecar and backbone fit (~41 GB), and the
n-gram table is the one component that cannot be resident too — and the one that doesn't need to be.**
Its rows are known the moment a token is sampled, and they are not consumed until layer 1. Layer 0's
compute (~4 ms today) is ample lead time for even a cold NVMe read of 16 small rows.

**Staging:**
1. Memory-mapped, read by row, OS page cache as the only tier (llama.cpp's shape). Resolve the 16 row
   addresses at sample time, touch them with a prefetch (e.g. `PrefetchVirtualMemory` on Windows), and
   consume at layer 1. This is AGENTS.md §1-clean: no allocation, a fixed 16-row staging buffer.
2. Only if measurement shows misses stalling layer 1: move to **Sub0TieredCache** (the spec already
   written for exactly this table, `docs/SUB0TIEREDCACHE_SPEC.md`), with a RAM hot-row tier. Frequent
   n-grams are highly skewed, so a modest cache should catch most rows.

## 4. Speculation on this model, on this CPU — a time model

Let a plain decode step cost `B + E + S`: backbone bytes `B`, routed-expert compute `E`, and serial or
other work `S`. Verifying `K` drafted tokens plus one bonus token is a batched pass of `K+1` positions:

- **Backbone:** read once. At `K+1 ≤ 5` the skinny GEMM stays close to bandwidth-bound, so roughly
  `1.2–1.5 × B` (estimate: native int8 dots on VNNI have compute headroom; bf16 less so).
- **Routed experts:** with 512 experts, each position brings ~10 mostly-distinct experts per layer, so
  roughly `(K+1) × E`. **Nothing amortizes.**
- **Serial work:** GDN's recurrence runs per position; attention grows with the batch. Roughly
  `1.5–2 × S` (estimate).

Post-O5 illustration (`B ≈ 60`, `E ≈ 46`, `S ≈ 25` ms; K = 3; accept length 3.3 per verify, taken from
SGLang's number for this model and therefore optimistic on IQ1_S weights):

| drafter | draft cost | verify cost | ms per accepted token | vs plain (~131) |
|---|---:|---:|---:|---:|
| none (plain decode) | — | — | ~131 | 1.00x |
| prompt-lookup | ~0 | 78 + 184 + 45 ≈ 307 | ~93 | ~1.4x (if it accepted 3.3; on prose it won't) |
| MTP, full-vocab lm_head per draft | 3 × ~13 ≈ 40 | ≈ 307 | ~105 | ~1.25x |
| MTP, reduced draft vocab | 3 × ~4 ≈ 12 | ≈ 307 | ~97 | ~1.35x |

(An MTP draft token costs about one layer's backbone bytes (~1.3 ms), that layer's ~10 experts (~1 ms),
the fusion (~1 ms) and the lm_head: ~10 ms with O5's native Q4_K head, 21 ms on the bf16 one.)

These are model estimates, not measurements. E1 and E2 replace them. Three conclusions survive the
uncertainty:

1. **The expert term dominates verification.** Halving `E` (O7) lifts plain decode by ~18% and
   speculation's ceiling by more, because `E` is multiplied by `K+1`.
2. **An MTP drafter's own cost is mostly its lm_head.** Every draft token projects onto the full
   248,320-token vocabulary, 8–16% of a step. Drafting over a frequency-ranked sub-vocabulary is a
   known remedy. It changes only which drafts get proposed, never what the verify pass accepts, so it
   cannot change output.
3. **Draft length should adapt.** Because the union of experts grows with K, longer drafts cost more on
   rejection. Cap K low (2–3), or adapt it from recent acceptance (EVICT's idea), rather than fixing a
   long draft.

**Correctness gate for every drafter:** under greedy decoding, speculative output must be
token-for-token identical to plain decode. This is exact and cheap to test. With sampling, the
acceptance rule must preserve the target distribution, and that needs its own statistical test before
sampling is ever enabled with speculation.

## 5. How the n-gram table can help the MoE — options evaluated

| option | mechanism | verdict |
|---|---|---|
| **Memory budget** | The table tiers to disk/page cache (§3b) so the expert sidecar stays resident. | **Do it**, as part of P1. This is the main link. |
| **Expert prefetch from n-gram context** | Key a small side table on the bigram/trigram hash, and record each layer's experts the last time that context occurred, to warm expert pages ahead of the layer. | **Measure first (E1), build only if experts page from disk.** In steady state experts are resident and compute-bound, so warming them buys little. The literature finds token-keyed predictors weaker than hidden-state ones. |
| **Prompt-lookup drafting** | Match the context's trailing n-gram against earlier text and propose what followed. It is the n-gram idea applied to drafting. | **Yes, P4a**: zero weights, zero draft cost, strong on code and repetitive text. |
| **Drafted-token row prefetch** | A drafted token's 16 table rows are known before verification, so fetch them during drafting. | **Yes, free once P1 and P4 exist.** A rejected draft wastes a prefetch, never correctness. The MTP draft itself runs with PLE off, so it needs no rows. |
| **Expert-aware drafting** | Prefer drafts, or cap draft depth, where the expert union stays small. | **Later**, after E1 measures union growth on this model. |

## 6. Measurements to take before building

- **E1 — expert-route log.** Record each token's per-layer top-10 expert ids over a mixed prompt set:
  prose, code, chat and repetitive text, at least a few thousand tokens. This needs a debug-gated route
  dump in decode; the router already computes the ids. From one log:
  - **union growth**: distinct experts per layer across K+1 consecutive tokens, for K = 1..6. This is
    `E`'s multiplier in §4, measured on 512 experts instead of extrapolated from 128;
  - **n-gram predictability**: how often the experts recorded for a bigram/trigram context recur the
    next time that context appears, per layer (decides P5);
  - **reuse distance** of experts across tokens (informs any expert cache).
- **E2 — prompt-lookup acceptance, offline.** Take greedy decode transcripts from the same prompt set
  and simulate prompt-lookup drafting on the token sequences alone: what fraction of positions would a
  trailing-n-gram match have predicted, and how many tokens deep. This needs no engine change.
- **E3 — memory budget.** Measure peak working set and page-fault rate with the native backbone and the
  bf16 blob dropped, then with the n-gram table memory-mapped under a real prompt. This decides whether
  §3b's step 2 (Sub0TieredCache) and P5 are needed at all.

## 7. Work packages, in order

| # | package | depends on | gate |
|---|---|---|---|
| P0 | **O5**: native backbone wired into decode, including the GDN roles (in flight as 2b-2b) | — | default-off exact; opt-in L2 and argmax reported; interleaved A/B |
| — | **E1, E2, E3** | P0 for E3 | numbers recorded here |
| P1 | **PLE fidelity**: the real gated fusion (`ple_key`/`ple_value`, gate, dilated conv) at layer 1; the table memory-mapped with sample-time row prefetch | — | parity against llama.cpp's `build_ple` on real rows (both sides with PLE on); decode timing unchanged within noise |
| P2 | **O7**: routed-expert kernels, IQ1_S / IQ2_XXS / IQ4_NL streaming unpack and VNNI, the same method as O5 phase 2a | — | bit-exact against the current kernels where the arithmetic is unchanged; expert phase ms/token |
| P3 | **Batched verify path**: `forward()` for small K on decode's kernels (native quant, threading) | P0 | `forward`/`forward_one` parity, as today |
| P4a | **Prompt-lookup speculation**, behind a configurator flag | P3, E2 | greedy output token-identical; accepted tokens per verify; interleaved A/B on the mixed set |
| P4b | **MTP drafter**: extract the 31 `mtp.*` tensors from the HF safetensors by HTTP range, quantize, implement vLLM's HC-aware fusion, reuse the layer recurrently; reduced draft vocabulary | P3, P4a | greedy token identity; per-position acceptance against vLLM's published decay; A/B against P4a |
| P5 | **Expert prefetch from n-gram context** | E1, E3 | only if experts page from disk; hit rate, then expert phase ms/token |

**Why P1 before P2 even though P2 is faster:** P1 is a correctness debt that silently biases every
quality comparison against the oracle, and its per-token cost is small. P2 and P1 touch different code
and can run in parallel, so the order matters only where one team does both.

**What would change this plan:**
- If E1 shows union growth well below linear (heavy expert overlap between neighbouring tokens), the
  speculation ceiling rises and P3/P4 move ahead of P2.
- If E2 shows prompt-lookup accepting well on the target workloads, P4b's case weakens.
- If E3 shows experts paging even with the table memory-mapped, P5 and Sub0TieredCache move up.

## 8. Open questions

- What `mtp.hybrid = true` controls. Nothing found consumes it.
- The MTP head's quantized size and its experts' formats: the checkpoint is bf16 and the GGUF lacks it,
  so it would need its own quantization (Q4_K/IQ4_NL are simplest; IQ1_S needs an importance matrix).
- Whether llama.cpp-side comparisons done so far had PLE enabled. Its default is on. Any past
  comparison that did not disable it on that side compared different models.
