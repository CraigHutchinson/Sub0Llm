# Degenerate repetition: what actually causes it, and which fixes are real

Observed on arm A (d192, 9.66M params, val_nelbo ~2.095) at the default sampler (`--temp 0.8 --topk 20`),
reproducible bit-for-bit at `--seed 42`:

```
... the "Dear Dear Dear Dear Dear Dear" - a book explaining how ...
... "What do you mean by '', '''? Is 'Bronx' like 'Dear'?" ...
... her own strengths and strengths ...  ... similarities and similarities even among people ...
```

Both artifacts are the **same defect**. The `''`/`'''` case is not a tokenizer bug: id 39 is a plain
apostrophe byte, the only markers in the special region are `<|cap|>`/`<|up|>`/`<|endoftext|>`, and none
of them render as a quote. The model learned the `' word '` shape from a corpus full of quoted terms and
emitted the delimiters with no content, then stuttered on them — a short instance of the same loop.

## The framing question

Is a repetition penalty a plaster over a data problem? **Mostly yes for the objective, no for the
sampler.** There are three separable mechanisms and they need different fixes; conflating them is how
people end up stacking decode-time hacks forever.

### 1. The sampler — `top-k` is genuinely the wrong mechanism (a REAL fix, not a plaster)

`sample_token` (engine_core.cpp) implements temperature + `top-k` only. There is no top-p, no min-p, and
no repetition penalty of any kind.

`top-k = 20` is a **fixed** cutoff that ignores the distribution's entropy:

* in a low-entropy context — inside a quotation, right after `Dear` — the true distribution may have two
  plausible continuations, but top-k admits twenty and **renormalizes**, inflating the self-repeat's mass
  to something far above what the model actually assigned it;
* in a high-entropy context it truncates legitimate diversity.

Replacing it with **min-p** (or nucleus/top-p) is not a band-aid — it swaps a crude fixed cutoff for one
that adapts to the model's own confidence. This is the mechanism Holtzman et al. 2019 identified as the
cause of neural text degeneration, and it is a correctness improvement independent of model quality.

### 2. The training objective — where the deep fix lives (the plaster is HERE)

Maximum likelihood with teacher forcing never shows the model its own output, so it never learns to
recover from a repeat it started, and is never penalized for assigning mass to a continuation that is
degenerate *when sampled autoregressively*. A decode-time penalty patches the symptom every time you
generate; the principled version moves it into training:

* **Unlikelihood training** (Welleck et al. 2019) adds a loss term penalizing tokens already present in
  the recent context, so the *model* stops wanting to repeat rather than being forbidden at sampling time.
* **Contrastive objectives** (SimCTG) keep token representations distinguishable, preventing the hidden
  state from collapsing into a fixed point that re-emits the same token.

This is the one that makes a decode-time penalty unnecessary rather than merely unused.

### 3. Data and capacity — real, but probably not the dominant term here

Duplicated or boilerplate text teaches that repetition is likely, and near-duplicate deduplication is
worth auditing. But the observed loops do not look like corpus boilerplate, and at 9.66M parameters and
~0.78 bits/byte the simpler explanation is capacity: the model cannot hold enough content-specific signal,
so the strongest signal available — *what did I just emit* — dominates. Arm A is also still improving at
3.8 epochs, so it is undertrained on top of being small.

## MEASURED: widening top-k does not reduce repetition

Before the distributional measurement below, a cheap behavioural probe — 10 seeds x 500 tokens per
condition, same prompt, counting immediate `X X` repeats and the `X and X` coordination pattern:

| `--topk` | words | immediate `X X` | `X and X` | rate |
|---|---|---|---|---|
| 20 | 2507 | 3 | 5 | 1 per 501 words |
| 100 | 2463 | 5 | 6 | 1 per 410 words |

**A 5x wider candidate set produced no reduction.** That is the opposite of what mechanism 1 predicts:
if top-20 renormalization were inflating a mid-probability repeat, admitting 100 candidates would spread
the mass and the rate would fall visibly. It did not move.

Statistical honesty: at counts of 5-6 the Poisson 95% interval is roughly [1.6, 11.7], so this does NOT
show top-k 100 is *worse* — the two conditions are statistically indistinguishable. What it does rule out
is the large reduction the sampler hypothesis requires. Absence of the predicted effect is the finding.

So the weight of evidence points at the **model's own distribution**, not the decoder: the repeat is
probably not an artifact of the cutoff, it is what the model actually wants. That moves mechanism 2 (the
objective) ahead of mechanism 1 and, notably, agrees with the original instinct that a decode-time
penalty would be a plaster.

The `X and X` pattern is the informative one, and it is more specific than a generic loop: the model has
learned the coordination frame `N and N` but cannot retrieve a *distinct* related noun for the second
slot, so it re-emits the most salient candidate in context — the one it just produced. That is a capacity
failure in content retrieval wearing the costume of a repetition bug, which is exactly the kind of thing
a decode-time penalty would paper over while leaving the cause untouched. It would suppress the visible
`science and science` and replace it with a *different* wrong noun, not a right one.

Immediate `X X` repeats are real but rarer (~1 per 700 words) than `X and X`, so the dramatic
`Dear Dear Dear` case is the tail of the distribution rather than the typical failure.

## Still worth doing: the distributional measurement

The three causes above predict **different, measurable things** at the moment the loop starts, and the
experiment is cheap. Take the exact generated prefix up to the second `Dear`, run it through batched
`forward()` (not the decode path — scoring uses `forward()` by standing policy), and record for that
position:

| measurement | if it shows | the binding cause is |
|---|---|---|
| `p(Dear)` **before** top-k, and its rank | low p (~0.05), mid rank, but top-k renormalization inflates it | **the sampler** — ship min-p, item 1 |
| `p(Dear)` before top-k | high p (~0.5+) | **the objective/data** — items 2 and 3; a sampler change only masks it |
| entropy of the full distribution | very low | low-entropy context; top-k's fixed width is actively harmful |

Do this before investing in either fix. It costs one scoring run and it converts "penalty vs deeper fix"
from a judgement call into a reading. Guessing here has a poor track record in this project.

## Ordering (revised after the top-k probe)

1. **Confirm with the distributional measurement** (above) once a build dir is free. The behavioural
   probe already points away from the sampler; this reads `p(repeat)` directly and settles it.
2. **Unlikelihood or contrastive term** in the training objective — promoted ahead of the sampler work,
   because the top-k probe found no sampler-width effect. This is the fix that addresses the cause.
3. **Corpus near-duplicate audit** — cheap, and it bounds how much of item 2 is really a data problem
   rather than a capacity one. Worth doing before investing in an objective change.
4. **min-p sampling** to replace/augment top-k — still worth doing on its own merits, since top-k's fixed
   cutoff is wrong in both entropy regimes, but it should **no longer be expected to fix this** and must
   not be reported as having done so.
5. **Repetition penalty** — last, and only as an explicitly documented stopgap. Given the `X and X`
   finding it would suppress a visible wrong noun in favour of a different wrong noun, which flatters the
   samples without improving the model.

Note the reordering is itself the point: the original list led with the sampler on a plausible-sounding
argument, and one cheap measurement inverted it.

## Do not land any of this mid-sweep

`report`'s sample battery runs at every eval, so changing the sampler changes generated samples inside a
running comparison. Wait for the LoopSplit arms to finish.
