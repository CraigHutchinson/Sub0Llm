---
name: ch29-char-level-run-results
description: First ch29 char-level training run — Shakespeare FORM + real words on OUR engine (proves §13.6); greedy collapses to blanks, needs temperature
metadata:
  node_type: memory
  type: project
  originSessionId: f6f73404-ddcc-4bc1-b640-2b89a8b5c475
---

First real char-level ch29 run (2026-06-19, after [[char-level-tokenization-lever]] was implemented). Config: `--char-level --seq-len 128 --batch-size 32`, founded dims (D=256/L6/H8, 5.94M), complete_shakespeare.txt, ckpt `/d/tmp/ch29_char`, 8 P-cores. Throughput **56 windows/s** at seq128 (23 at seq256 — seq len is the CPU speed lever). Clean A/B vs prior BPE founded runs: ONLY tokenization changed.

**RESULT — §13.6 proven on OUR engine.** Held-out NELBO 3.4→1.91 by step ~5000 (still descending). Ch30 free generation (temperature 0.6) at step 4900:
```
To be, or not to be wished.

LONOLLT.
I
As thase place, I out tue nighthe daes,
And she that will he
```
Real English + Shakespeare FORM (speaker labels, verse lines, comma breaks, graceful char-degradation: errors are plausible letter strings `nighthe`/`daes`, NOT BPE subword salad `hTee`/`Cirj`). Structure visible as early as step 1400 (`KING HENRY:` / `[_Exerans` learning `[_Exeunt._]`). Roughly tiny-diffusion's ~step-1000–2000 quality and improving. This is the coherence BPE-512 NEVER produced on the same corpus/engine.

**GREEDY COLLAPSE (important Ch30 finding):** temperature 0 (our Ch30 default, "deterministic/educational") produces a BLANK canvas — with `conf≥0.90` commit the model commits the single safest high-frequency token (space ~18% of chars) everywhere. The weak-model iterative-refinement failure ([[ch30-iterative-refinement-precondition]]) made concrete: needs temperature 0.6–0.7 to get real output. FIXED: Ch30 default temperature → 0.9 (commit ea55d21).

**DECODE-BUG #2 — `min_commit_frac` (2026-06-20, MAJOR; commit 7882818).** The "loose generation / char-salad" across ALL our diffusion samples was largely a SAMPLER DEFAULT BUG, not model/data/architecture. TinyStories char model trained well (NELBO 1.28 @ step10k) but generated salad ("there whe a birty it you have") at the default. Raising conf_threshold 0.90→0.98 did NOTHING (byte-identical) ⇒ culprit = `min_commit_frac=0.10` (progress floor: commit ≥10% of still-masked canvas EVERY iter regardless of confidence). In the EARLY all-masked steps the model has no context, so it force-commits ~12 chars from near-unigram stats → garbage anchors garbage, bootstrap never recovers. Dropping to `--min-commit 0.01` on the SAME checkpoint → COHERENT prose ("Once upon a time, there was a big hole. She had a little girl named Sue… Anna said,"); Shakespeare char jumps same way ("come … you to your mother with her … not the leave"). Sweep: 0.10→29 iters(garbage), 0.05→46, 0.03→58(coherent, NEW DEFAULT), 0.01→87(cleanest). Matches tiny-diffusion's conservative "commit the confident few, or ≥1". ⇒ The model was ALWAYS this capable; the sampler destroyed it. Net of whole investigation: SALAD=tokenization (§13.6-7); LOOSE GRAMMAR=decode defaults (temperature + min_commit), NOT data/capacity/objective/flat-architecture. Strongest confirmation that coherence is RECIPE/DECODE-bound (§13.8c/d) → tree predictor parking vindicated.

**Speed reality (see [[char-level-tokenization-lever]]):** tiny-diffusion = ~0.32s/iter GPU, coherent by step ~2000 (~11 min), ~53 min full 10k (its "6.5hr" log was a laptop-sleep artifact). Our 8.7× gap (200 vs 23 win/s at seq256) is PURE CPU-vs-GPU — our model is SMALLER and runs the SAME 640k-window budget, so we're not over-doing it. The durable speed fix is the CUDA backend (Stage-4); the cheap CPU lever is seq_len (128 vs 256 = 2.4×).

**WORD-LEVEL CONFIRMED (2026-06-19, commit 8089ceb).** Added `--word-level` (BPETokenizer::word_level: one token per whole word, auto-derived vocab, no cap — rare words stay emittable but rarely generated, which is fine). complete_shakespeare → 32,818-word vocab, 14.3M params, seq64, /d/tmp/ch29_word. At step 1000 (Ch30 temp 0.6/0.9): EVERY emitted token is a real word, ZERO spelling salad (e.g. "thou my well … With make too I to matter / To me with his me as to him", + speaker label "ROSALIND."). Errors are now wrong word CHOICE/ORDER (grammatical salad), never broken spelling — closes the mechanism loop (remove fragments → non-words vanish ⇒ BPE-512 salad WAS fragment mis-coordination under parallel denoising). Error-mode by granularity: char=1-char blemish (near-word), BPE-512=mis-assembled fragment (non-word), word=wrong real word. Docs: TRAINING_DESIGN §13.7 + chapters/ch03_tokenization/README.md (granularity spectrum + AR-vs-diffusion consumer-matters). Char-level work committed b224e30; char-level best banked /d/tmp/ch29_char step6300 NELBO1.87.

**Pending:** word-level still training (sharper grammar); char-level resumable to ~10–15k; one faithful seq256 run; re-run recall probe / contiguous A/B on a char-level ckpt ([[diffusion-contiguous-masking-lever]] BPE-confounded); AR-GPT baseline on complete_shakespeare (now ALSO tests the AR-vs-diffusion salad prediction in §13.7); Ch30 weak-model temperature default fix.
