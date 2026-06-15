# Ch29 — Text Diffusion Training: the model-quality axes (reference)

A coherent map of **every axis that gates recall** for the Ch29 BPE bidirectional-diffusion
denoiser, with the measured finding and status for each. Written so future work starts from
evidence, not from re-deriving the same dead ends. (Throughput/zero-alloc/threading notes live
in the memory + `bench_parallel.cpp`; this doc is about *model quality*.)

## TL;DR — current state

- **Baseline** (D=128, L8, 8/4 heads, d_ff=384, seq_len=64, vocab≈514, full Shakespeare 2.19M tokens):
  **held-out recall 15.4%** (33/27/17/8% at 10/25/50/75% noise), NELBO 4.19.
- **The model UNDERFITS** — *train* recall is only **19.0%** vs held-out 15.4% (a **3.6-pt gap**).
  So the ceiling is **fitting capacity, NOT coverage or generalization**.
- **The 15% is HONEST, and the gap is *content prediction*** (`--inspect`, axes H/I): word-level
  **14.1%** ≈ token 15.4% (not a metric artifact), but it splits into word-START **11.4%** (predict a
  word from context — hard) vs continuation **17.8%** (complete a partial word — easy). Per-sample
  dumps show the model recovers punctuation, completions, and function words, but substitutes a
  frequency prior (`·the`/`·of`) for content word-starts. It learned the easy structure, not prediction.
- **Context helps then saturates at the block size**: recall climbs steeply with `seq_len`
  (16→6.8%, 32→9.2%, 64→14.8%) then **flattens** (128→16.4%, +1.6 only). seq_len=64 ≈ optimal;
  longer windows hold more complete lines but cross-block context adds little. Not a big lever.
- **Architecture lever is exhausted**: wider/re-proportioned models are *worse* AND seed-fragile
  (often stuck in the unigram basin). D=128 is a genuine recall peak on a width sweep.
- **Mildly undertrained (not a hard plateau)**: resuming completeb *past* its early-stop with
  `--track-recall` showed recall SLOWLY CLIMB — 21.4%→22.8% on the {0.25,0.50} monitor (START
  16.1%→17.6%), NELBO 4.29→4.20, over ~110K extra steps (~32 epochs). The gains are real but
  sub-threshold (below the early-stop's `min_improvement=0.01`), so the stop was *slightly*
  premature with steeply diminishing returns. Implication: the model is **optimization-limited, not
  at a hard capacity wall** — which is exactly what a better optimizer (Muon, axis F) targets.
- **Ch28's 90% is NOT comparable**: it was char-level (≈40-vocab) memorization of ~38 hardcoded
  proverbs, *scored on the training set*. The comparable Ch29 number is our 19% *train* recall.

## The axes (systematic), each with finding + status

### A. Tokenization — **char vs BPE, and BPE VOCAB SIZE** · settled (smaller wins on this corpus)
char (Ch28, ≈40-vocab, low entropy, easy) vs BPE (Ch29, 514-vocab, high entropy, hard). The
single biggest reason Ch28 scored 90% and Ch29 scores ~15%. BPE exact-match is intrinsically hard.
**Vocab-size sweep (BUDGET-MATCHED, all 40k steps, early-stop off, proxy, word-level recall =
the comparable metric):** V=512 **13.6%** > V=1024 8.1% > V=2048 5.2%, NELBO 4.13/5.04/6.01.
Smaller vocab wins DECISIVELY even budget-matched — the mechanism is **data-sparsity** (bigger vocab
→ each token rarer → harder to fit on limited tokens), NOT over-fragmentation (the `--inspect`
single-char word-starts at V=512 are NOT a bottleneck). **Methodology lesson (important):** the NAIVE
sweep (self-terminating) was MISLEADING — bigger vocab early-stopped sooner (V=512/1024/2048 ran
24030/15515/**6222** steps), so V=2048 read **1.1%** purely from undertraining; budget-matched it's
**5.2%** (5×). Always budget-match capacity/vocab A/Bs. CAVEAT: the proxy (136K tokens) structurally
penalizes large vocab; full-corpus optimal could sit higher than 512 (the definitive full-corpus
budget-matched sweep is ~10h and low-priority — unlikely to break the content-prediction ceiling).

### B. Eval regime — **memorization vs generalization** · settled (via `--eval-train`)
Ch28 scored on its *training* corpus (memorization). Ch29 holds out 5% (generalization).
Measured train↔held-out gap = **3.6 pts** → the model isn't over/under on generalization; it
**fails to fit even training**. So generalization/coverage is NOT the limiter.

### C. Context length (`--seq-len`) — helps then **SATURATES at the block size** · settled
Recall in seq_len: **16→6.8%, 32→9.2%, 64→14.8%, 128→16.4%** (gains +2.4, +5.6, **+1.6**). Climbs
strongly to ~64 then **flattens** — the model captures the *within-coherent-unit* context by ~64
(lines ≈8 tokens, blocks ≈52), and cross-block context past that adds little at 3×+ the O(T²) compute.
**seq_len=64 is already near-optimal; longer context is NOT a meaningful lever.** Train tracks held-out
at every length (still underfitting). NOTE: longer seq_len is partly an *easier task* (more context per
masked token) — but it's the real denoising use case, and it saturates regardless.

### D. Window / data shape — **coherence, offsets, block structure** · partly explored
Corpus: 145,938 lines (≈8 tokens each), 42,392 blank-line-separated blocks (≈52 tokens each).
`read_paragraphs` splits on every newline, then `flatten` concatenates all lines, so a 64-token
window spans ~7 semi-unrelated lines. The "coherent windows" hypothesis (restrict windows within a
block) was *reframed* by axis C: the win comes from longer windows holding more complete lines, not
from restricting them. Still open: **word-aligned offsets**, **block-separator token** (keep all data
+ mark boundaries), **windowing within blocks** at a seq_len ≤ block size.

### E. Architecture — **width / depth / heads** · exhausted (negative result)
Width sweep (8h/8L, d_ff=3·D, 2 seeds): D=64→~11%, D=96→7–13% (fragile), **D=128→~15% (peak)**,
D=192→4–9% (fragile). Wider is *worse* and seed-fragile. The "founded" char-AR proportions
(head_dim≥32, d_ff=4·D, fewer layers) were **refuted** for this model — D=128/8-heads (head_dim 16)
beats 4-heads/head_dim-32 by 14.8% vs 3.9%. More heads + depth win here. `--founded` preset + the
head_dim warning are advisory only.

### F. Optimization — **optimizer / gradient variance** · the ROOT (AdamW wins; gradient is high-variance)
The deepest finding. A `--grad-probe` (cosine of the mean gradient per noise level + within-level
consistency) shows the **diffusion gradient is high-variance and noise-level-decoupled**, unlike AR's
stable per-token signal: at random init, within-level consistency (t=0.5) is only **0.31** and easy↔hard
gradients (t=0.10↔0.75) are nearly orthogonal (**0.09**); at convergence consistency crashes to **0.026**
(pure sampling noise at the minimum — which independently cross-confirms completeb converged). So one
step is a noisy, diluted average → the sticky unigram basin. **Optimizer A/B** (V=512, 20k steps,
budget-matched): **AdamW WINS** (word-level 11.1% vs Adam 9.7%, NELBO 4.39 vs 4.50) — decoupled weight
decay improves *conditioning*, not just regularisation. **Muon LOSES** (escapes the basin ~3-4× slower)
— it orthogonalises *all* singular directions to magnitude 1 *including noise*, amplifying the high
variance, whereas Adam/AdamW's 1/√v damps noisy coordinates. **The lever is variance reduction, not a
fancier optimizer** → `--shared-t` (one t/step so the W-window average sharpens the per-t gradient) is
the test. Flags: `--optimizer adam|adamw|muon`, `--grad-probe`, `--shared-t`. (See the Ch31
diffusion-optimization-sandbox plan.) The legacy LR/warmup/seed notes below still hold:

### F-legacy. LR / warmup / seed fragility
Bigger/different configs get *stuck in the unigram basin* (NELBO ~5.5–5.9, never learn) from a bad
init — seed-dependent (D=192: seed 42 stuck @3.9%, seed 7 trained @8.8%). **LR warmup+cosine did NOT
rescue them and HURT the baseline** (12.0% vs 14.8% — decayed it into an earlier early-stop), so
warmup is now **opt-in** (`--warmup-steps`, default constant LR). Why wider models can't be optimized
from scratch on CPU is the open research question if capacity scaling is ever wanted (init-scaling,
normalization, multi-seed). Flags added: `--lr`, `--warmup-steps`, `--seed`.

### G. Early-stopping / coverage / schedule — **fixed** (tested)
The held-out eval is meaningful only after ≥50% epoch coverage between evals (Ch28 rule), folded into
one tested place: `sub0diff/train/schedule.hpp` (`make_schedule`, 9 unit tests). Two bugs fixed:
(1) the eval cadence was *printed* pre-÷threads (said 17116, actual 4279); (2) on small corpora the
NELBO plateaus within the patience window after ~4 epochs → **unigram collapse** → an invalid A/B.
Fix: **`--min-epochs` floor** (default 20) — no early-stop before N full passes. Without it, two
different architectures gave *bit-identical* recall (both collapsed). One *sliding* pass over the full
corpus is ≈2.19M windows; we train ~1.2 sliding passes (~80 non-overlapping epochs) — coverage is fine
(confirmed by the small train↔held-out gap).

### H. Masking strategy — **per-token vs whole-word** · MEASURED (the 15% is honest)
Random per-BPE-token masking lets the model "complete" a partially-visible word (`Ham`→`let`).
Measured the split on the completeb baseline (`--inspect`, `--whole-word`): **word-START recall
11.4%** (predict a word's first subword from context — the honest, hard target) vs **continuation
17.8%** (complete a partial word). So per-token recall IS inflated by completion, but only modestly
(1.56×), and **word-level recall is 14.1% ≈ token 15.4%** — the headline is *not* a metric artifact.
Per-sample dumps (`--inspect`) show the mechanism: the model reliably recovers **punctuation**,
**subword completions**, and **high-frequency function words** (`·he`,`·and`,`·my`), but **fails on
content word-starts**, substituting a frequency prior (`·the`/`·of`/`·and`/`·that`) for the real word;
0%-recall windows are dense-masking or proper-noun/poetry passages where context is destroyed. So the
model genuinely underfits *content prediction* — it learned the easy structure, not the hard part.
**Whole-word *training* A/B — REFUTED as a lever (proxy, identical tokenization, 71094 word targets):**
on the identical honest whole-word task, the whole-word-TRAINED model scores **word-START 3.8%**
(word-level 1.8%) — statistically TIED with, and on word-level WORSE than, the per-token-trained
model evaluated under whole-word masking (**word-START 3.7%**, word-level 2.5%). The fully-converged
(24k-step) per-token model also only gets 3.7%, so ~3.8% is the genuine whole-word ceiling, not an
undertraining artifact. **Masking strategy is NOT a lever** — the hard ceiling is content prediction
(~4% whole words), unchanged by *how* we mask. The per-token 15% is ~⅔ completion (continuation
17.4%) over ~4% true word prediction. Reinforces the capacity/data limit (axes B/E).

### I. Recall metric — **BPE exact-match vs word-level** · MEASURED
Added a **word-level** credit (a word counts only if ALL its masked subwords are recovered) and a
**word-START/continuation** split, both tokenizer-agnostic (a bool `is_word_start` table). On the
baseline: word-level **14.1%**, word-START **11.4%**, continuation **17.8%**. Word-level ≈ token, so
the exact-match number is representative; the START/continuation split is the more *diagnostic* view.

### J. Objective — **formal NELBO vs curriculum** · partly explored
Formal objective = uniform t ∈ (0.02, 1]. Frontier-point `--curriculum` (Ch28's winner) HURT here
(8.9% vs 15.4%) because, once converged at ceiling=1.0, it trained *only* t=1.0 (fully masked, no
context) → degraded the easy regime → premature early-stop. **Fixed**: on convergence it switches to
the band [0.02, 1.0] = the formal objective (easy-first *warmup* then formal). Re-run at scale still
pending.

### K. Corpus / data regime — settled
1.65M params on 2.19M tokens (~1.3 tokens/param — data-light by Chinchilla, but **not** data-bound:
136K-token subset gives ~14.8% ≈ full-corpus 15.4%, so 16× more data buys nothing). The small corpus
is a **valid fast proxy** (~15-20 min/run) for architecture/seq_len experiments.

## Measured results (small-corpus proxy = 10k paragraphs unless noted)

| Experiment | held-out recall | notes |
|---|---|---|
| Baseline D=128/L8 (full corpus) | **15.4%** | train 19.0%; gap 3.6 pts |
| Baseline D=128/L8 (proxy) | 14.8% | proxy ≈ full corpus |
| seq_len 16 / 32 / 64 | 6.8 / 9.2 / 14.8% | climbs steeply (+2.4, +5.6) |
| seq_len 128 / 256 | 16.4% / *pending* | **saturates** (+1.6 only); 64 ≈ optimal |
| Width D=64/96/128/192 | ~11 / 7–13 / **~15** / 4–9% | D=128 peak; wider fragile |
| Vocab 512/1024/2048 (budget-matched, word-level) | **13.6** / 8.1 / 5.2% | smaller wins; data-sparsity (not fragmentation) |
| Undertrain (resume past early-stop) | 21.4→22.8% over 110K steps | MILDLY undertrained; slow sub-threshold climb |
| Whole-word train (word-START, honest) | 3.8% ≈ per-token 3.7% | masking not a lever |
| "founded" 4h/6L/head32 | 3.9% | refuted |
| `--curriculum` (point, full) | 8.9% | t=1.0 degeneracy; since fixed |
| LR warmup+cosine (baseline) | 12.0% | hurt → warmup opt-in |

## Tooling / flags added this round

- `--whole-word` — mask whole words (all BPE subwords of a word together) in BOTH training and eval;
  the honest task (no partial-word completion). Drives the `is_word_start` (Ġ-marker) table.
- `--inspect N` — dump the N best/worst-recovered held-out windows with per-masked-position
  truth→prediction detail + word-START/continuation tag (what the headline recall is made of).
- `--track-recall` — print a quick held-out recall (token/word/START) at every NELBO eval, to watch
  recall vs the early-stop signal during training (the "are we stopping too early" test → NO).
- `--pin auto|all|P|E|"lo-hi"|"a,b,c"` — worker core-affinity policy (shared `cpu_topology::
  resolve_pin_set`). Lets concurrent training jobs take DISJOINT cores (e.g. one run `--pin
  0,1,10,11`, another `--pin 12,13,22,23`) instead of all oversubscribing the same P-cores — the
  dominant cause of multi-job slowdown. Default `auto` = P-cores-first (unchanged behaviour).
- Every recall sweep now also prints a `breakdown:` line (word-level | word-START | continuation).
- `--eval-train` — also sweep recall on the train stream (memorization vs generalization).
- `--min-epochs N` — early-stop floor (default 20 full passes).
- `--seq-len`, `--embed-dim/--n-layers/--n-heads/--n-kv-heads/--d-ff` — no-recompile experiments.
- `--founded` — char-AR reference proportions (advisory; refuted here).
- `--curriculum [--curriculum-end --curriculum-min-tokens]` — frontier-point + token-gated ceiling.
- `--lr --warmup-steps --seed` — LR schedule (opt-in) + reproducibility.
- BPE caching (load saved tokenizer; skip ~110 s retrain on resume/eval).
- **Token-stream caching** (`ckpt_dir/tokens.bin`) — the encoded train/eval streams are cached
  (fingerprinted on corpus size + paragraph limit + vocab), eliminating the ~23 s full-corpus
  re-encode that ran on EVERY prior invocation (the dominant startup cost). Self-invalidating.
- `sub0diff/train/schedule.hpp` (tested), `curriculum.hpp` (`NoiseCurriculum`), `Adam::set_lr`.

## Open questions / future work (priority order)

1. **Optimizer (Muon / AdamW)** — THE remaining promising lever now that masking (H), context (C),
   architecture (E), vocab (A), coverage (B/K), and early-stop/undertraining (G) are all
   characterized. The codebase uses plain Adam; **Muon** (orthogonalized momentum) directly targets
   the optimization/fit ceiling (axis F) and could rescue wider models from the unigram basin. See
   the optimizer + methodology backlog memories. (Whole-word masking H/I: DONE — refuted as a lever.
   Undertraining: DONE — negative, converged. Vocab: DONE — smaller wins, budget-matched.)
3. **Optimization fragility** (axis F) — the only path to a *bigger, better-fitting* model: init-
   scaling with width, normalization, multi-seed. The recall peak at D=128 is an optimization peak,
   not necessarily a capacity peak. **Optimizer backlog**: the codebase uses plain **Adam** (no weight
   decay, constant LR). Add **AdamW** (standard default; but targets generalization, and we underfit),
   and try **Muon** (Newton-Schulz-orthogonalized momentum, hybrid w/ AdamW) — Muon directly targets
   the fit/optimization ceiling and famously improves small-transformer convergence, so it's a real
   candidate to break the ~11% word-START ceiling, not just a nicety. A/B on the proxy; test whether
   it rescues wider models (D=192) from the unigram basin.
4. **Coherent / block-aware windowing + block-separator token** (axis D) — keep all short blocks,
   mark boundaries; compose with longer seq_len.
5. **Curriculum-as-warmup at scale** (axis J, fix shipped) — easy-first warmup then formal objective.

## Reproduce

```bash
cmake --build --preset native --target ch29_diffusion_training
# baseline, full corpus
./build-native/bin/ch29_diffusion_training --ckpt-dir /tmp/ch29 \
  --corpus data/complete_shakespeare.txt --threads 4
# memorization vs generalization on a checkpoint (BPE-cached, fast)
./build-native/bin/ch29_diffusion_training --ckpt-dir /tmp/ch29 \
  --corpus data/complete_shakespeare.txt --eval-only --eval-train
# fast architecture/seq_len proxy (~15-20 min, self-terminating, min 20 epochs)
./build-native/bin/ch29_diffusion_training --ckpt-dir /tmp/ab --paragraphs 10000 \
  --corpus data/complete_shakespeare.txt --threads 4 --seq-len 128 --eval-train
```
