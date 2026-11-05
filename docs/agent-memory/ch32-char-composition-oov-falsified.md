---
name: ch32-char-composition-oov-falsified
description: "ch32 P1 char-composition OOV fix is FALSIFIED at premise level; the robust OOV-cliff lever is the rare-weighted loss; don't rebuild the codec"
metadata: 
  node_type: memory
  type: project
  originSessionId: f6f73404-ddcc-4bc1-b640-2b89a8b5c475
---

Ch32 P1's char-composition codec (CharComposer: spelling→word vector, fed as the weight-tied
embedding) was the proposed fix for the M1 OOV cliff (rare types predicted ~6–8× worse). It is
**FALSIFIED across three integrations**, all losing to a plain rare-weighted baseline:

1. additive gate, unweighted loss — cliff 6.5×→8.1× (worse)
2. additive gate, rare-weighted loss — codec 5.5–5.7× vs baseline 4.85×
3. **convex blend (replace lookup for rare) + composer pretrained to recon-CE 0.026** — codec
   NLL_rare 23.1 vs baseline 12.1, cliff **7.68× vs 4.72×** (much worse)

**Mechanism (the durable insight):** a composed vector is a faithful *spelling* encoding, but the
embedding is *weight-tied to the LM head*, so spelling-similar words ("cat"/"cap"/"car") get
near-identical output rows and **collide** at prediction time. **Spelling geometry ≠ prediction
geometry.** Even a near-perfect spelling autoencoder hurts. Filling the OOV "hole" with a
spelling-addressed vector is worse than the hole.

**What works instead:** the **rare-weighted / frequency-balanced masked loss** alone (upweight rare
masked tokens, auto weight = common:rare token-mass ratio). It lowers the cliff on every model
(baseline 6.36→4.72–4.85×, the best arm of all) — cheap, robust, no new params. Implemented as
`tok_weight` in `batched_diffusion_loss` ([[diffusion-loss-weighting-and-level-cadence]]).

**Direction:** do NOT rebuild char-composition for OOV in P2/P3. If sub-word/OOV info is wanted, it
must enter as an *input-side feature the model re-embeds* (concatenated, NOT tied, NOT substituted
into the output projection), with its own kill-test. Full record: `diffusion/chapters/
ch32_tree_predictor/1C_RESULTS.md`. P1 premise (the cliff is real) stands; the *fix* does not.
Resume order after P1: P2 gist conditioning, P3 MERA. Relates to [[tree-predictor-parked-research-diffusion-training]].
