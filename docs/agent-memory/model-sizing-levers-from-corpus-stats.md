---
name: model-sizing-levers-from-corpus-stats
description: "Data-derived model-size levers for the tinystories corpus — biggest is space-token waste (40.7%) + seq starvation (0% stories fit), THEN capacity"
metadata: 
  node_type: memory
  type: project
  originSessionId: 2a6b481b-6211-474b-818a-995223d7d5ac
---

Built a corpus/tokenizer dimensioning analysis (local Python, scratchpad `dims_analysis.py`; reads corpus.tok + tokenizer.bin — reimplement in C++ as `sub0llm dims` if it earns its keep). Findings on the 22MB tinystories corpus (vocab 2048):

- **Space token = 40.7% of ALL tokens** (the single biggest lever). Spaces never merge (configurator keeps punctuation/space/markers standalone by design), so ~40% of the sequence budget AND model capacity is spent predicting the trivially-predictable space. tokens/word = 2.40, half of which is the space. Folding leading-space into the following word token (GPT-style "Ġword") would ~halve token count → ~2x effective context per seq AND ~2x throughput. This is the "token per word matters" axis the user flagged — it's a TOKENIZATION lever, bigger than raw width.
- **seq_len starvation: 0% of stories fit in a 64-token window.** Story length p50=341 / p90=515 tokens (~142 / ~215 words); seq=64 ≈ 27 words. The model never sees a whole story → can't learn story-level coherence. seq_len is a major lever (compounds with space-folding, which stretches the same seq ~2x further).
- **vocab 98% saturated** (2012/2048 used, only 43 rare) → room to grow vocab productively, which also cuts tokens/word.
- **Capacity headroom: model val_nelbo 1.4655 vs unigram-entropy floor H1=3.928 nats (ppl 50.8) → captures 63% of the gap, 37% remains; train≈val = underfit** → capacity helps. head_dim was 24 < founded 32.

**Lever ranking (data-derived):** 1) tokenization efficiency (space-fold / bigger vocab), 2) seq_len, 3) capacity (D/L/head_dim).

**Capacity result (RAN):** founded D=256/L6/H8 (head_dim 32), 5.8M params → val **1.3530** (vs 0.96M L5 @ 1.4655: ~8% lower NELBO), models/cap_d256_l6_h8.bin, plateaued step 6520 / 4ep. Sample notably more coherent. BUT train 1.329 < val 1.355 = a small gap appeared → at 5.8M params the **22MB corpus is becoming the data ceiling** (L5 had train≈val). So capacity helped AND we've now hit the point where the bigger corpus (FineWeb) is the next lever. JOIN tokenization (implicit-space + sentinel): measured **24.4% fewer tokens = 1.32x context/throughput** + composition operator (`sun⌒day`) for OOV/morphology. See [[ch29-model-quality-founded-dims]], [[tokenizer-normalization-and-factored-tokens]] (space-fold belongs there).
