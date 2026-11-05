---
name: spellspike-poc-results
description: "Token-granularity spike RESOLVED SUCCESS -- combine/uncombine (model REQUESTS a deterministic tokenizer-layer op, harness provides the fragments) + LOSS-MASKING makes a d128 model do char-level reasoning (Nth-char, count, round-trip) that GENERALIZES to unseen words (held-out Nth-char 0.24 unmasked -> 0.75-0.84 masked; held-out round-trip 0.00 -> 0.48-0.65). Both tied and untied work; the ch32 tied-head penalty does NOT apply to intercept-and-provide. Committed e3a4ddc."
metadata: 
  node_type: memory
  type: project
  originSessionId: 2a6b481b-6211-474b-818a-995223d7d5ac
---

## RESOLUTION (masked combine/uncombine, 2026-07-10) -- the decisive result

The mechanism + loss-masking WORKS and generalizes. This section is the headline; the v1 arc below is
superseded background.

**Setup:** d128/L4/H4/seq64, vocab 774 (486 word pieces), CPU, `spike_untied`/`spike_tied`. The model
learns to EMIT `TOK_UNCOMBINE` after a word; the decode interceptor (`decode.hpp` kv_decode_ops)
deterministically injects that word's byte fragments (from `tokenizer.expansion`) + `TOK_UNCOMBINE_END`;
the model then reads them to answer. `TOK_COMBINE..TOK_COMBINE_END` re-tokenizes fragments back to a
token (inverse). Curriculum = Nth-char / count-X / round-trip (`spellspike.hpp`). **Loss mask** (the
LOSS_IGNORE_INDEX foundation) grades ONLY the emitted ops + answer + fragment copy, NEVER the injected
content -- so it can't memorize letters, only the protocol. Held-out words (never trained) = the clean
generalization probe. 1500 steps, AdamW lr 0.003, batch 16.

**Results (held-out = words NEVER trained -> a correct answer PROVES it uses injected chars):**
| | steps | drilled nth | drilled rt | held-out nth | held-out cnt | held-out rt |
|---|---|---|---|---|---|---|
| untied UNMASKED | 1500 | 0.62 | 0.98 | 0.24 (~floor) | 0.60* | 0.00 |
| untied MASKED   | 1500 | 0.92 | 0.98 | 0.75 | 0.73 | 0.48 |
| tied   MASKED   | 1500 | 1.00 | 0.95 | 0.84 | 0.67 | 0.65 |
| untied MASKED   | 3000 | 0.98 | 1.00 | 0.86 | 0.71 | 0.70 |
| **tied MASKED (CEILING)** | 3000 | 1.00 | 1.00 | **0.97** | **0.76** | **0.90** |

**CEILING (2026-07-10, kEvalRounds bumped 5->10 = 3000 steps):** the TIED production-geometry model
reaches held-out Nth-char **0.966**, round-trip **0.90**, count **0.76** on UNSEEN words -- indexing
any word's Nth char at 97% and uncombine->combine round-trip at 90% with zero memorization, all still
rising. Tied >= untied throughout (tied held-out nth 0.97 vs untied 0.86; rt 0.90 vs 0.70) -- so this
is not just safe for the tied production models, it's BETTER on them. The clean direct-count version at
3000 steps also matched the reverted scaffold's nth/rt highs WITHOUT the count regression, confirming
the scaffold's apparent gains were just extra training signal (see the scaffold note below).

**Conclusions:**
1. **Masking is the enabler.** Held-out nth 0.24->0.75 (untied), round-trip 0.00->0.48 -- went from
   memorizing drilled traces to the WORD-AGNOSTIC protocol. Exactly the fix the unmasked memorization
   signature predicted.
2. **Generalizes to unseen words -> scales to the whole fixed vocab for free** (nothing memorized; the
   harness provides every word's chars). The "count the r's in strawberry" class solved via a
   tokenizer-layer TOOL the model learns to invoke.
3. **The ch32 tied-head penalty does NOT apply here.** In the MEMORIZATION spike, tied hurt (spelling
   forced into the prediction geometry). With intercept-and-provide, tied is as good or BETTER than
   untied (held-out nth 0.84 vs 0.75) -- content is external, never in the embedding. So this is SAFE
   for the TIED production models (d768). Retroactively validates the reframe away from memorization.
4. Count (~0.67 held-out) lags nth/round-trip -- aggregation is harder than indexing/copying; still
   well above floor. A reasoning-scaffold (enumerate-then-count) curriculum is the lever if needed.
* the unmasked held-out count 0.60 is prior-inflated (guess 0/1), not real generalization.

**Enumerate-then-count scaffold TRIED and REVERTED (2026-07-10).** Added a running-tally reasoning span
to count_task (after the injected chars, the model emits one digit per position = count-so-far; answer
= last tally). Result: it HURT its own target -- held-out count 0.73 -> 0.56 -- because the tally block
sits AFTER the char block, so each tally digit must align to a REMOTE char position, which is harder
for a d128 model than a direct one-shot count, and one tally slip breaks the final answer. Held-out
nth (0.75->0.91) and round-trip (0.48->0.93) rose in the same run, but that is CONFOUNDED: changing
count_task lengthened ~1/3 of the docs, shifting the whole dataset's token/length distribution (more
training signal), not a genuine scaffold benefit. Reverted to the clean direct-count version (still
committed e3a4ddc). Takeaways: (a) a block-after running tally is the wrong scaffold shape here -- an
INTERLEAVED char/tally scaffold (each char immediately followed by its tally slot) would align
trivially but needs the interceptor to inject char-by-char with model turns between, a real protocol
change, untested; (b) the confounded nth/rt bump implies the CLEAN version likely reaches higher with
just MORE STEPS (the metrics were still rising at 1500) -- worth a longer ceiling run before adding any
scaffold machinery.

**Foundation shipped with this (committed e3a4ddc):** LOSS_IGNORE_INDEX ignore-index cross-entropy
(CPU op_cross_entropy + backward + train_batch loss_mask) -- reusable for instruction-tuning
assistant-span masking. GPU ce_backward_kernel parity is the documented next step, DEFERRED while the
GPU is busy with d768. See the plan smooth-noodling-kurzweil.md.

## v1 background (MEMORIZATION spike, SUPERSEDED by the above)

Ran the approved spelling spike (plan smooth-noodling-kurzweil.md; building blocks BB0-BB3, all
mainline + committed-worthy). Question: can a tiny model learn to reach the BYTE tokens that make up
one of its own word-piece tokens, framed as a `TOK_SPELL_Q <word> TOK_SPELL_A <bytes> TOK_EOS`
query/answer trace — and does it GENERALIZE to words never drilled?

**Setup:** d128 / L4 / H4 (d_head=32) / seq64, vocab 774 (486 word pieces), CPU-only, two build dirs
`out/build/spike_untied` + `spike_tied` differing ONLY in USE_TIED_EMBEDDINGS. Curriculum: each of
364 DRILLED words' exemplars x8 reps interleaved with english.txt corpus.tok as plain text; 122
words HELD OUT (never drilled). AdamW lr 0.003, batch 16, T48, 800 steps. Fair A/B: run A trains
with the spelling curriculum from a snapshot of the random init; run B restores that same init +
zeroed Adam moments and trains the SAME #steps on plain text ONLY. Metrics = free-generation
exact-match + teacher-forced answer accuracy, reported separately for drilled vs held-out, plus a
plain-text held-out next-token loss (with-curriculum vs plain-only baseline).

**Results (step 200/400/600/800):**
| | drilled tf | drilled exact | held-out tf | held-out exact | plain-loss delta (with - baseline) |
|---|---|---|---|---|---|
| untied | .47/.77/.91/.94 | .04/.48/.73/.83 | ~.21-.28 (flat) | 0.000 | -1.55 (spelling "helped" — CONFOUNDED) |
| tied | .43/.55/.75/.82 | .04/.10/.42/.58 | ~.25-.29 (flat) | 0.000 (one .021 blip) | +0.32 (spelling COST prediction) |

**Robust findings (both geometries, clear):**
1. **Drilled-word spelling IS learnable** and still climbing at 800 steps — a word-piece token is NOT
   hopelessly opaque to its own letters; a model can be taught to emit them on demand. Untied hit
   0.83 free-gen exact-match, tied 0.58, both trajectories still rising.
2. **NO zero-shot generalization to un-drilled words** — held-out exact-match stuck at 0.000, tf
   accuracy flat at the byte-prior floor (~0.25). The model MEMORIZES the drilled word->spelling map;
   it does not learn a general "read this token's constituent bytes" operation. Expected: a word
   piece's embedding doesn't inherently expose its bytes, so there's nothing to generalize from.
3. **Untied learns spelling faster and to a higher ceiling than tied** (0.83 vs 0.58 exact at equal
   budget) — the tied LM head measurably FIGHTS the spelling objective, directly consistent with
   [[ch32-char-composition-oov-falsified]] (which is why the A/B was requested).

**Caveated finding:** the prediction-cost delta is CONFOUNDED — english.txt is only ~1270 plain
tokens, so the plain-only baseline barely trains and the absolute losses are unreliable (untied
baseline 5.44 vs tied baseline 3.80 on the same corpus/steps is itself suspicious). The DELTA's SIGN
across the identical setup is still suggestive (tied +0.32 = spelling hurt prediction; untied -1.55)
and points the same way as ch32, but do NOT quote the magnitude. To trust it: rerun against a bigger
plain corpus (tinystories) with an equal-plain-token-budget baseline.

**Implications for the roadmap (the actual payoff of the spike):**
- For sub0llm's FIXED, known vocab, "spell ANY word in the token set" (the user's "please spell
  Amlardinr" goal) is achievable by DRILLING ALL word tokens exhaustively (486 here, ~16k in
  production) — memorization is FINE because we control the whole vocab; the held-out-generalization
  failure only matters for byte-sequences that AREN'T single vocab tokens, which are already byte
  tokens (trivially "spellable" as identity). A cheap unrun confirmation: set drilled_frac=1.0 + more
  steps and expect near-100% on the whole vocab.
- BUT it works best UNTIED, and production (d768) is TIED, where spelling is slower AND costs
  prediction. So a real spelling feature argues for either untying the head OR a SEPARATE readout/
  auxiliary path that doesn't fight the tied prediction geometry — exactly what
  [[spelling-via-injected-char-word-training]]'s caveat + the reasoning-harness framing already
  anticipated (spelling as an on-demand SIDE capability, not baked into base next-token prediction).
- The scratch-token direction ([[scratch-tokens-context-translation-layer]]) needs the model to spell
  a DYNAMICALLY-bound token whose bytes aren't a fixed vocab word — the no-generalization result says
  that won't come for free from this drill-the-vocab approach; it needs the content-derived-embedding
  engine change (deferred stage C), reinforcing the original staging.

**Building blocks left in mainline (reusable, tested, NOT yet committed):**
- `casing.hpp`: TOK_RESERVED_0/1 renamed -> TOK_SPELL_Q/TOK_SPELL_A (zero-cost, count still 32, inert
  in the text path); tok_interchange.cpp marker_literal + 3 tests updated for the rename.
- `include/sub0/spellspike.hpp` (engine-free curriculum generator) + `tests/spellspike_tests.cpp`
  (structure/byte-alignment/split/doc-index, in the engine-free suite).
- `tests/spellspike_engine_tests.cpp` (the train+eval harness, tag `[.spellspike]` hidden so it's
  opt-in via `sub0_tests "[spellspike]"`, not the default ctest sweep).
- Two CPU build dirs `out/build/spike_untied` + `spike_tied` (not committed) are the run environment.
Default ctest 174/174 green in the spike build (mainline changes sound). See
[[spelling-via-injected-char-word-training]] (the idea) and the tokenizer redesign's reserved-marker
headroom ([[tokenizer-redesign-implementation-complete]]) this consumed 2 slots of.
