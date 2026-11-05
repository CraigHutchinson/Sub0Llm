---
name: autotemp-coherence-tuner
description: "`sub0llm autotemp` self-validating temperature tuner; key finding = statistic-matching OVERSHOOTS for underfit models, cap at 1.0 + diagnose"
metadata: 
  node_type: memory
  type: project
  originSessionId: 2a6b481b-6211-474b-818a-995223d7d5ac
---

Built `sub0llm autotemp <model>` — the coherence analogue of `tune`. Sweeps a temperature grid, generating from real held-out prefixes, and measures two self-validating statistics vs real text: generation self-perplexity (scored under the model's true T=1 dist; rises with temp) and 4-gram repeat rate (falls with temp). Interpolates the temperature where each crosses the real-text value. Pure math (`ngram_repeat`, `interp_cross`) lives in `include/sub0/coherence.hpp` (engine-free, like tune.hpp) with `tests/coherence_tests.cpp`.

**Key finding (validated on tinystories.bin/diag.bin, 2026-06-25):** matching generation statistics to real text OVERSHOOTS the subjectively-best temperature for an underfit model. Ground truth: temp 0.6–0.8 is most coherent; 1.1+ produces invented words ("fwasher") and topic drift. But both anchors recommend ~1.3–1.5, because (a) at temp 1.4 the gen perplexity (4.4) is STILL below real (5.17) — **the underfit model cannot tell its own gibberish is worse than real text**, the fundamental limit of any self-validating loop; (b) the model is MORE repetitive than real at coherent temps, so matching real diversity needs heat that breaks coherence. This is the capacity-bound repetition weakness (see [[ch29-model-quality-founded-dims]]) seen through temperature.

**Why:** a self-validating coherence metric is only as good a judge as the model. An underfit/high-entropy model both over-commits (low greedy perplexity) AND can't penalise its own incoherence.

**How to apply:** recommendation is **capped at the training-natural ceiling temp 1.0** (cross-entropy training fits the dist at T=1; sampling hotter only helps a calibrated model). When both matches exceed 1.0 → report the UNDERFIT DIAGNOSIS (real lever is capacity/training, not temperature), don't recommend the hot temp. For a future well-calibrated model the matches fall below 1.0 and the cap doesn't bind → returns the true crossing. So autotemp doubles as a calibration diagnostic. Repetition anchor is noisy near the floor on short windows (use GEN_LEN~96 so loops manifest); the 1.0 cap keeps the final answer robust to that noise.
