---
name: tinystories-sweep-saturation
description: "The TinyStories Chinchilla size-sweep (d128->d768, gated+tied+qk-norm+Muon) is NEAR ITS CORPUS CEILING — val_nelbo gain per size step collapsed (320->448 -0.070, but 448->512 only -0.025), and both d448 AND d512 began OVERFITTING at plateau (val rising while train flat; train/val gap widening 0.11->0.13). d448 is the efficiency knee; d512/d768 are diminishing returns. The real lever is a RICHER corpus, not more params/epochs on TinyStories."
metadata: 
  node_type: memory
  type: project
  originSessionId: 2a6b481b-6211-474b-818a-995223d7d5ac
---

Analysis 2026-07-10 (answering the user's queued question: "is the 448 or 512 oversized already
as we didn't capture reports" -> is continuing d768 worthwhile). Read from each model dir's meta.txt
+ train.log; all models same recipe (gated_ffn + tied_embeddings + qk_norm + Muon, seq 256, vocab
16517 for d128-d512, 16668 for d768 after the Stage-2 retokenize).

**Best val_nelbo by size (and the marginal gain per step):**
| d_model (L,H) | best val_nelbo | epochs | Δ vs prev |
|---|---|---|---|
| 128 (4,2) | 1.5719 | 2.00 | -- |
| 192 (5,3) | 1.4296 | 1.70 | -0.142 |
| 256 (6,4) | 1.3381 | 2.00 | -0.092 |
| 320 (8,5) | 1.2597 | 1.80 | -0.078 |
| 448 (11,7) | 1.1902 | 1.90 | -0.070 |
| 512 (12,8) | 1.1653 | 2.50 | **-0.025** (collapse) |
| 768 (11,12) | **1.1235 (final)** | 2.30 (plateaued step 25392) | -0.042 |

**Two overfitting signals -> the CORPUS, not capacity, is the ceiling:**
1. Both d448 and d512 had val_nelbo RISING at plateau while train stayed flat/low: d448 best 1.1902
   then 1.1978->1.2141; d512 best 1.1653 then 1.1874->1.1937. The trend-plateau detector correctly
   stopped them at the val floor (val degradation = overfitting onset). More EPOCHS won't help.
2. Train/val gap widens with size: d448 train ~1.085 vs val 1.190 (gap ~0.11); d512 train ~1.039 vs
   val 1.165 (gap ~0.13). Bigger models increasingly MEMORIZE the simple, finite TinyStories corpus.

Caveat: epochs aren't perfectly controlled (1.70-2.50), and d512 trained MORE epochs than d448, so
part of the 448->512 gain is extra training, not size -- making the true size benefit at 512 even
SMALLER than the -0.025 shown. Also d512=47.14M params; ~1.2-1.6B tokens_seen at ~2 epochs is ~25-34
tokens/param, near Chinchilla (20:1) and far below the autosizer's 100:1 target -- but MORE epochs
just overfits (signal 1), so the corpus is data-limited for these sizes, not compute-limited.

**Verdict:** d448 ~= the efficiency knee; d512 already into strongly diminishing returns; d768 is
marginal for TinyStories val_nelbo (FINAL: beat d512 by -0.042 -> 1.1235 at plateau step 25392/2.30ep,
a bit more than the -0.02-0.03 partial estimate but still deep in diminishing returns, at ~2x the cost
-- 67K vs 120K+ tok/s at batch 240). Continuing d768 (user chose "let it plate") was defensible ONLY for
curve-completeness + exercising the larger-model path, NOT for TinyStories quality. **The real lever
is corpus RICHNESS** (FineWeb-Edu / conversational data per docs/ROADMAP.md, see
[[capability-roadmap-discrete-experts]]) -- more params/epochs on TinyStories just buys memorization.
See [[model-identity-date-tag-and-comparison-table]] for the `models --metrics` table this could also
be read from, and [[d512-fault-resume-optimizer-switch-bug]] for the d512 run's own history.
