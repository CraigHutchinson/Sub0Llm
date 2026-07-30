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

## Decide with a measurement, not an argument

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

## Ordering

1. **Measure** (above). One run, decides everything downstream.
2. **min-p sampling** to replace/augment top-k — justified on its own merits regardless of what the
   measurement says, since top-k's fixed cutoff is wrong in both entropy regimes.
3. **Repetition penalty** only if the measurement says the sampler is not the binding cause and the
   objective work is not yet affordable — explicitly as a stopgap, documented as one.
4. **Unlikelihood or contrastive term** in the training objective if the measurement shows the model
   genuinely assigns high probability to the repeat. This is the fix that removes the need for item 3.
5. **Corpus near-duplicate audit** — cheap to run, and it bounds how much of item 4 is really a data
   problem.

## Do not land any of this mid-sweep

`report`'s sample battery runs at every eval, so changing the sampler changes generated samples inside a
running comparison. Wait for the LoopSplit arms to finish.
