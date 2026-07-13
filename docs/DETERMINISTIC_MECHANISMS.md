# Deterministic mechanisms — the model as an orchestrator of exact computation

Scope note: this is a **strategic design review**, not committed engineering. It reframes *how we train*
around a single thesis that already runs through the codebase (combine/uncombine, scratch tokens, spelling)
and generalises it. It complements [ROADMAP.md](ROADMAP.md) (corpus/capability roadmap) and
[SCRATCH_TOKENS.md](SCRATCH_TOKENS.md) (the scratch-token feature, whose findings motivate this).

---

## The thesis

> A language model should **orchestrate exact computation, not approximate it.** For anything a classical
> scalar computer can compute deterministically — arithmetic, spelling, precise numbers, dates, unit
> conversions, symbol lookup, sorting — the model should learn *when and how to delegate*, and let a
> deterministic node produce the exact answer. Fuzzy, model-embedded knowledge is reserved for what is
> genuinely fuzzy: language, pattern, judgement, world-knowledge.

This is a **division of labour**, and it inverts the usual scaling reflex. When a capability is hard, the
first question is not "how much more model / data / steps do we need to approximate it?" but **"is this
deterministically computable, and can a mechanism make it trivial?"** If yes, we build the node — exact,
generalises perfectly, near-zero training — and the model only learns the cheap *routing* on top.

### Why this is not a detour — the evidence is already in this repo

- **combine/uncombine already IS this.** The model doesn't memorise how tokens decompose into bytes; it
  emits `TOK_UNCOMBINE` and a *deterministic* harness (`kv_decode_generate`'s `expand`/`combine` callbacks)
  produces the fragments. The model learned the *routing decision*, not the tokenizer.
- **The localization finding (see SCRATCH_TOKENS.md).** "Which slot contains letter X" looked like a
  capacity wall. Scale did nothing (d128→d256, more heads, more steps — all flat at random). A **mechanism**
  fixed it completely (local query grounding → held-out 1.000 at d128). *Structure beat scale, decisively.*
- **The spelling spike.** Char-level reasoning generalised only once the injected characters were **masked**
  so the *only* path to them was invoking resolution — the model couldn't cheat by reading them off the
  stream. (This is pillar 3 below, in miniature.)

The interceptor seam that powers all of this — the model emits a marker, the harness runs deterministic code
and injects the result, the injected span is loss-masked — is **already a general tool-invocation point.**
We have been building this architecture without naming it. This doc names it and asks what else it unlocks.

### PROVEN — big-number addition (arithspike, committed)

The thesis's crispest test now has a decisive answer. Same d128/4-layer model, same `A + B = ?` over 8–14
digit integers, same 3000-step budget — only the mechanism differs (`include/sub0/arithspike.hpp`, a
`compute` node added to `kv_decode_generate`):

| | held-out exact-match | per-digit |
|---|---|---|
| **DELEGATION** (emit `COMPUTE`, exact add node injects the sum) | **1.000** (from step 300) | 1.000 |
| **FUZZY** (model must produce the sum itself) | **0.000** (never) | ~0.10–0.15 |

Delegation generalises **perfectly** to never-seen numbers because it only ever learns the routing; fuzzy
internal arithmetic never generalises at all. **1.000 vs 0.000** — mechanism beats approximation by an
infinite margin on a deterministic task. This is the template for everything below.

**All three pillars are now proven with data** (spike tests, d128):

| Pillar | Test | Result |
|---|---|---|
| 1. PROVIDE (delegate vs approximate) | `arithspike` | delegation **1.000** vs fuzzy **0.000** held-out |
| 1b. Region frame — WORD op-names vs dedicated tokens | `nodespike` (4 ops) | **1.000 == 1.000** — words route as well as tokens, for free |
| 3. FILTER — mask scalar facts vs contamination | `arithspike` filter A/B | masked clean (sampled **1.000**) vs contaminated (sampled **~0.90**, ~10% fuzzy leak) |
| 4. COMPOSE — dynamic-turn chained reduction | `chainspike` (var. length) | trained k=2..6 **1.000**; held-out length k=7 **1.000** (k=8 only fails on context overflow) |

The region-frame result validates the natural-language-marker design (§1); the filter result is nuanced
(§3); chaining shows delegation **composes over a dynamic number of turns and length-generalises** (§4/staged
plan). The **production node registry** (`include/sub0/nodes.hpp`) is now a first-class, tested substrate.
Pillar 2 (numeric BIND) and the end-to-end production wiring are the remaining pieces.

---

## The three pillars

A deterministic mechanism is only a "win" when all three are in place. Give the model the node but leave the
fuzzy shortcut available, and it will take the shortcut; provide the shortcut but no node, and it stays
fuzzy. You have to **set the mechanism up to win** on all three sides.

### 1. PROVIDE — a deterministic node behind a region frame (named in ordinary words)

Generalise the `expand`/`combine` seam into a small registry of **computation nodes**: the model opens a
region, the harness runs the exact function, and injects the result span (loss-masked). The nodes are
**non-differentiable** (classical code between generation steps, exactly like the existing interceptor); the
model is trained on the *open-frame + op-name + operand* tokens (graded) and the injected result is masked —
so gradient only ever flows to the **routing and operand selection**, never to a fuzzy internal copy.

**Don't mint a token per op — use ONE region frame + a natural-language op name.** This is the same choice
the tokenizer already made for chat: `TOK_TURN_START`/`TOK_TURN_END` are *two* markers, not one-per-role, and
the role ("system"/"user"/"assistant") flows through as **ordinary text** the Unigram vocab already encodes
([casing.hpp](../include/sub0/casing.hpp) §turn-boundaries; reserved headroom there explicitly anticipates
"tool-call structure"). Apply the identical trick to compute: one delimiter pair, and the *op* is a word.

```
<|op|> add A B <|end|>          ▶ node("add", [A,B]) → exact result injected, masked
<|op|> solve "x^2 = A" <|end|>  ▶ node("solve", …)   → the CAS's answer injected
<|op|> uncombine <tok> <|end|>  ▶ expand (today's UNCOMBINE, re-expressed in the frame)
```

The op selector and operands are ordinary tokens the vocab already has, so **a new node costs ZERO tokenizer
budget** — no reserved-id per op, no format bump. (Contrast: the scratch pool maxed at K=6 precisely because
each slot is a reserved id; a region frame sidesteps that ceiling.) The registry keys on the op *word* after
the frame-open, not on a dedicated token. `expand`/`combine`/`compute` all collapse into this one shape;
today's per-op reserved markers (`TOK_UNCOMBINE`, …) and the arithspike's ASCII sentinels (`$`/`#`) are the
*spike* encoding — the region-frame is the production one, and it's the natural home for the whole registry.

### Solver nodes — what to embed (for "more complex maths")

The add node is a few lines; the power is delegating to a real, permissively-licensed, **deterministic**
library. Staged by capability (prefer MIT/BSD/Boost over GPL/LGPL for a permissive project; keep every node
**pure and sandboxed** — no I/O, bounded time/memory, since operands are model-generated):

| Capability | Candidate libraries | Notes |
|---|---|---|
| Exact arithmetic / rationals / arbitrary-precision decimals | **Boost.Multiprecision** (`cpp_int`, `cpp_dec_float`; Boost licence, header-mostly); GMP/MPFR (LGPL) | upgrade the self-contained add node to exact rationals + decimals (the user's `.32232` case) |
| General numeric expression eval | **exprtk** (MIT, header-only), **tinyexpr++** (zlib), muParser (BSD) | one "evaluate this expression exactly" node covers `+ - * / ^ %`, functions |
| Symbolic algebra (solve / simplify / differentiate) | **SymEngine** (MIT, C++, CPM/CMake-friendly — SymPy's C++ backend) | the real "complex maths" node: `solve`, `simplify`, `diff`, `expand` |
| Number theory / polynomials | FLINT, Pari/GP | heavier; likely an external-process node, not embedded |

Recommended order: (1) already have big-int add → extend to Boost.Multiprecision for exact decimals/rationals;
(2) an **exprtk** numeric-expression node; (3) a **SymEngine** symbolic node. All fetch cleanly via the
existing CPM setup. Avoid GiNaC (GPL) and be deliberate about GMP/MPFR (LGPL) given the project's licensing.

### 2. BIND — scratch tokens as symbolic (algebraic) variables

A scratch slot is **algebraic notation.** Binding a precise value to a slot lets the model reason over the
*symbol* and expand to the *content* only on demand. Numbers are the killer case — precision is exactly what
embeddings are worst at:

```
"12345678.32232 + 999836475.123 = X"
        │                │
     bind A            bind B          (each collapses to ONE scratch token)
"A + B = X"  ──▶  the model reasons algebraically; it never has to hold 13 digits in its activations
"COMPUTE add A B"  ──▶  node returns the exact value, bound to a fresh slot C
"X = C"
```

The model does **not** learn digit-arithmetic. Algebraically, `A + B = X` is trivial; the hard part
(carrying 13 digits) is delegated. A small model with near-zero arithmetic training should resolve this,
because it isn't doing arithmetic — it's doing *substitution + a node call.* This also slashes context and
KV: a 13-digit number is one token unless the model *asks* (uncombine) for the digits. Scratch tokens are a
**super-power for scalars** precisely because they are variable-binding.

This unifies threads that looked separate: OOV compression, spelling, and numeric/symbolic computation are
all "bind a precise thing to a symbol; expand or compute on demand." One mechanism, one set of markers.

### 3. FILTER — mask the fuzzy path out of the training data

*(This is the pillar the raw corpus fights hardest.)* If the training text contains `"12345 + 67890 =
80235"` as plain tokens, next-token loss will **reward the model for fuzzily memorising the answer** — and
it will, badly, instead of learning to delegate. Providing the node is not enough; you must **remove the
incentive to be fuzzy.** So a corpus pre-pass must find scalar-solvable spans and **mask them in the loss**
(or rewrite them into scratch/compute form), so the *only* rewarded path is the mechanism:

- **Detect** scalar-solvable spans: arithmetic and equalities, precise numerics, dates, unit conversions,
  spellings/char-enumerations, sorted lists, table lookups. Cheap classical regex/grammar catches most;
  a **small corpus-scanning model** ("embellish masking onto the dataset") handles the fuzzy boundary cases
  a regex misses (is this number a *quantity to compute with* or *rhetorical* — "one of the reasons"?).
- **Mask / rewrite**: mask the *result* tokens (so predicting `80235` earns no reward — the model must route
  to the node), and optionally rewrite the operands into bound scratch tokens so the training example *is*
  the mechanism. This reuses the existing `LOSS_IGNORE_INDEX` masking end-to-end — the same machinery the
  spelling spike used to force resolution.

The principle mirrors the spike exactly: *the only path to the answer must be the mechanism.* We proved this
matters — masking the injected spelling was what made char-reasoning generalise instead of memorise.

**What the filter A/B actually showed (nuanced — measure, don't assume).** A corpus of 50% delegation + 50%
plain arithmetic facts, masked vs unmasked, evaluated with the node available:

- **Greedy decoding self-corrects.** Even contaminated (facts graded), greedy delegates 100% and scores
  1.000 — because `COMPUTE` is the **modal** next-token after `=` (one frequent, learnable token beats the
  dispersed, *unpredictable* fuzzy digits). Contamination is *invisible* to greedy.
- **Sampling exposes the residue.** The contaminated model still carries **~10% fuzzy probability mass**
  (only ~10%, not 50% — a deterministic target the model can't compute earns high loss, so its learned
  probability stays low), which leaks under sampling → sampled score ~0.90.
- **Masking removes it entirely** → a clean distribution, 1.000 even sampled.

So the filter's job is precise: it isn't needed for greedy *correctness* (the mechanism self-selects, since
delegating is the low-loss option), but it is needed for a **clean, sampling/temperature-robust distribution**
— and it becomes *essential* for any scalar pattern the corpus covers with facts but **no** delegation
alternative (there, fuzzy is the model's only rewarded path). Mask, or ensure delegation coverage; never
leave an unmasked fuzzy-only scalar span.

---

## Why this is the "root of effective training" question

The localization saga is the argument in miniature. We burned real effort on scale (a d256 build, LR
re-tuning, head counts, a CoT scaffold) and moved the needle **zero**. Then a *mechanism* (local grounding)
solved it at the smallest model. The lesson isn't "local grounding" specifically — it's the **triage**:

> **Mechanism-first.** Before scaling a capability, ask whether structure can make it trivial. Scale is the
> lever of last resort, for irreducibly fuzzy tasks; it is the *wrong* lever for anything a scalar computer
> can do exactly.

Homogenising spelling + scratch + numerics + tool-use under one architecture (markers + a node registry +
symbolic binding + loss-masking the deterministic parts) shrinks the training surface, removes whole classes
of "the model is bad at X" (arithmetic, exact recall, long-number handling), and concentrates the model's
capacity on what only a fuzzy model can do. It also makes the training objective *honest*: we stop rewarding
the model for pretending to compute.

---

## Sketch of a unified architecture

```
                    ┌───────────────── the model (fuzzy) ─────────────────┐
   corpus ──[filter pass]──▶ masked/rewritten stream ──▶ learns: ROUTING + OPERAND SELECTION
                                                              │ emits marker + operands
                                                              ▼
   ┌──────────────── interceptor seam (kv_decode_generate) ───────────────┐
   │  registry of DETERMINISTIC NODES:  expand · combine · add · cmp · …   │
   │  runs exact classical code, injects result span (LOSS-MASKED)         │
   └──────────────────────────────────────────────────────────────────────┘
                                                              │ bound to a scratch slot
   scratch/binding table ◀── BIND precise values to symbols (A, B, C … ), expand on demand
```

Three surfaces, one idea. The seam and the binding table and the masking **already exist** — this is mostly
about (a) generalising the marker/registry, (b) adding a numeric-binding path, and (c) building the corpus
filter pass.

---

## Architecture directions (open considerations)

Design directions raised while building this out — recorded so they inform (and don't get lost from) later
work.

### A. Resolution is a discrete FORWARD-STEP event, not an iterative reasoning loop

The op-resolve is *one discrete step*: the moment an end-op marker is processed, the deterministic node runs
and its result is injected as the next tokens (each fed through `forward_one`). Crucially this means the model
**does not need an iterative thinking/reasoning mode** to compute — a single op is answered by one delegation
step, not a chain-of-thought.

**Frame it as a PLUGGABLE PREDICTOR — the classical computer is swapped in as the "brain" for the op span.**
Generation is a sequence of predictors: the neural model predicts the routing (the op region), and for the
result span a classical computer *is* the predictor. Today the classical computer already does the predicting
(the result is never sampled from the model), but we still run the model's `forward_one` on each result token
— purely to update the KV-cache so the model can *attend* to the result later. The refinement: **skip the
model forward too — the brain is genuinely off for that span.** Whether it can stay off is decided entirely by
the KV-cache: a neural prediction can only condition on what the KV holds.
- **Terminal op** (the result IS the final answer): nothing later depends on it → the brain stays fully off,
  the classical computer writes the answer for **zero model forwards** (a long CAS answer = O(len) forwards
  saved). A cheap, obviously-correct win.
- **Intermediate op** (the model must reason *with* the result): it must re-enter the KV — and the efficient
  re-entry is **one scalar embedding** (the value as a single vector), not N digit forwards. So a big-number
  op goes from O(result-length) model forwards to **0** (terminal) or **1** (scalar re-entry); the classical
  compute is ~free. This is the same build as the scalar-embedding result below (and the same backward choice:
  fixed encoding → stop-grad, learned number→vector encoder → grad to the encoder). **Done:** the interception is now factored (`decode.hpp resolve()`) and fires
both per generated token *and after prefill*, so a prompt that merely *poses* a computation (ends in an op
region, `12+34=[op add]`) is resolved in the **forward pass** — no generation loop.

**Training-time resolution — the forward/backward analysis (decided: keep bake-train/live-infer).** A node is
*non-differentiable* (classical code), so backward can only **stop-gradient** at its output: the result is a
constant for autodiff. There is nothing to learn about an exact computation — only *routing*, which the loss
on the op-header tokens already fully supervises (through-node gradient is both unnecessary and ill-defined:
the task fixes the operands, so there's no "better operand" for a gradient to find). And for teacher-forced
training, "bake the result offline" and "resolve live in `forward()` + stop-gradient" produce **identical
gradients** (given the operands the result is the same constant either way) — so baking is strictly better:
same signal, computed once instead of every epoch (matters for an expensive CAS node), no engine change. The
current **bake-train / live-infer** split *is* the right design (chainspike already does it: states baked in
training, node-injected at eval, matched 1.000). Live training-time resolution only earns its keep for
**non-teacher-forced operands** (the model generating operands itself) — a **free-running / RL** decision
(reward on the routing, no grad through the node), to take *only if exposure bias proves real*. **Backlog:**
(i) **mid-prompt** ops need a KV-cache-aware splice; (ii) **scalar-embedding** result — inject the value as
*one vector* (à la the content-derived slot embedding) instead of digit tokens: this is where a *backward*
choice is genuinely live — a **fixed** numeric encoding → stop-gradient, a **learned** number→vector encoder →
gradient flows to the encoder; (iii) free-running/RL, only on demonstrated exposure bias.

### B. Fold combine/uncombine into the op registry

`expand`/`combine` are already interceptor ops; they could become registry entries (`[op uncombine <tok>]`)
so there is **one** marker family and dispatch. **Trade-off (why it's a consideration, not a given):** they
are *token-group* operations — an expression *over/producing tokens* — whereas compute nodes are *value*
operations (operands → a value). The registry's `NodeFn` (strings→string) doesn't fit token-group I/O cleanly.
Unifying wants a second node signature (tokens→tokens) or an adapter. **Backlog** — low urgency; the win is
homogenisation, not capability.

### C. Scratch tokens: named regions vs reserved ids

Scratch slots are reserved ids today (`TOK_RESERVED_4..9`) — which is exactly why the pool **caps at K=6**,
and why a binding is a *special* id that can't be trained toward, stored in a corpus, or survive a
re-tokenisation (it's short-term, context-specific memory). The alternative the region-frame work suggests:
express a slot as a **named region** — `<|scratch> a` — with an ordinary in-vocab name, exactly like an op
(`[op add]`). Then:

- **Unlimited slots, zero reserved-id cost** — names never run out (the very ceiling the scratch content work
  hit). nodespike already evidences this: WORD names route as well as dedicated tokens, no penalty.
- **Survives encode/decode** as ordinary text, and the binding (name → fragments/value) can be **persisted to
  an external store** → the substrate for *long-term* memory, not just in-context.
- vs a "vocab+1 offset" per slot (the user's alternative): same *dispatch* effect, but it re-introduces a
  bounded, special-id pool that doesn't persist as text — so named regions strictly dominate for *unbounded +
  persistable* slots; the offset only wins if a slot must be a single token for KV/length reasons.

Recommendation: migrate scratch slots to named regions (unifying with the op frame), keeping a single-token
form only where a binding must occupy one position. This uncaps the pool and opens the long-term-memory path.
**Backlog** (a real refactor of the binding table + curricula).

---

## Open questions / risks

- **Routing accuracy is the new bottleneck.** The model must reliably decide *when* to invoke a node and
  *which operands* — a small, learnable pattern (much lower entropy than the computation), but the
  localization work shows even "small" routing can need the right *structure* (local grounding). Expect to
  design the trace shape, not just add a token.
- **Operand binding under precision.** Binding `999836475.123` correctly (all digits, right slot) is itself
  a resolve/copy task — the same one the spike is already probing. It must be exact or the node gets garbage.
- **Non-differentiability.** Nodes don't backprop; training is masking + (later) preference/RL on whether the
  routing *decision* was correct. The current masked-CE path covers the supervised case.
- **The filter's false-positive/negative cost.** Masking a *rhetorical* number as if computable, or missing
  a real one, both hurt. This is exactly where a small scanning model earns its keep over pure regex.
- **Scope/safety of nodes.** A `latex`/CAS/tool node is powerful and must be sandboxed and deterministic;
  keep the registry small, pure, and auditable.
- **Does it compose?** The real prize is *chained* delegation (compute A, feed to compare, feed to sort).
  The scratch table (bind intermediate results to fresh slots) is the composition substrate.

---

## Staged plan (proposal)

0. ~~Prove the thesis.~~ **DONE** — arithspike: delegation 1.000 vs fuzzy 0.000 held-out.
1. ~~Region frame + registry (spike).~~ **DONE** — nodespike: one `{ … }` frame + a word-keyed `ComputeNode`
   registry over 4 ops; WORD op-names route as well as dedicated tokens (1.000 == 1.000), so new nodes cost
   zero tokenizer budget. *Production form still to land:* adopt the real `<|op|> … <|end|>` delimiter
   (reuse/mirror `TOK_TURN_START/END`; one reserved slot at most) and move expand/combine/compute onto it.
2. **Exact + symbolic nodes.** Extend the add node to exact decimals/rationals (Boost.Multiprecision), then
   add an **exprtk** numeric-expression node and a **SymEngine** symbolic node (solve/simplify/diff) — the
   "more complex maths". Each pure + sandboxed.
3. ~~Filter pass (spike A/B).~~ **TESTED** — masking scalar facts yields a clean, sampling-robust delegation
   distribution; contamination leaks ~10% under sampling (greedy self-corrects). *Still to build:* the
   classical scanner (regex/grammar) + the small model for ambiguous spans, on the real corpus path.
4. ~~Chaining (dynamic turns).~~ **DONE (spike)** — chainspike: reduce a variable-length sum via a dynamic
   number of node calls, the model deciding termination locally. Trained k=2..6 = 1.000; held-out length
   k=7 = 1.000 (length-generalises; k=8 fails *only* on context overflow — the O(k²) re-emit, fixable with an
   O(k) running-accumulator/scratch-bound intermediates or a wider context). Delegation composes.
   *Remaining:* **numeric BIND** — bind literals to scratch slots (13 digits → 1 token unless asked) as the
   O(k) composition substrate.
5. ~~Production op frame LIVE.~~ **DONE** — `nodes.hpp` (registry) + `node_frame.hpp` dispatch the registry
   through `kv_decode_generate`'s compute seam on the EXISTING `TOK_TURN_START/END` markers (zero new ids; an
   OP region is `TOK_TURN_START op <name> <operands> TOK_TURN_END`, a chat tool-call). End-to-end: a d128 model
   emits `[op add]` for `A + B =` → the node injects the exact sum → held-out **1.000**. Non-op regions inert.
   *Remaining:* fold a compute-delegation **curriculum** into `--scratch-mix`; a decode-to-text step for a full
   Unigram deployment (the byte parse suffices for this project's byte-heavy tokenizer); real library nodes.

Each stage is a cheap, falsifiable experiment in the spike style — set the mechanism up to win, then check
whether a *small* model wins with near-zero training. That is the homogenised bet: **teach the model to use
the scalar, not to be one.**
