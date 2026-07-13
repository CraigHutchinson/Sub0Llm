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

0. ~~Prove the thesis.~~ **DONE** — the arithspike (`compute` node + `arithspike.hpp`): big-number addition
   delegation 1.000 vs fuzzy 0.000 on held-out. The seam and the masked-CE training already carry it.
1. **The region frame.** Adopt one `<|op|> … <|end|>` delimiter pair (reuse/mirror `TOK_TURN_START/END`;
   spend the reserved-headroom tool-call slot at most once) and a `ComputeNode` registry keyed on the op
   *word*. Re-express expand/combine/compute in the frame so new nodes cost zero tokenizer budget. (Rename +
   indirection; low-risk; makes everything below additive and un-caps the reserved-id ceiling.)
2. **Exact + symbolic nodes.** Extend the add node to exact decimals/rationals (Boost.Multiprecision), then
   add an **exprtk** numeric-expression node and a **SymEngine** symbolic node (solve/simplify/diff) — the
   "more complex maths". Each pure + sandboxed.
3. **The filter pass.** A classical scanner (regex/grammar) that masks arithmetic results and precise numerics
   in the corpus; measure whether it *stops* the model memorising answers. Add the small scanning model only
   where the classical pass is ambiguous.
4. **Numeric BIND + chaining.** Bind literals to scratch slots (algebraic compression: 13 digits → 1 token
   unless asked), and test **chained** delegation (compute A, feed to compare/sort) via scratch-bound
   intermediates — the composition substrate.
5. **Fold into `--scratch-mix`** alongside the resolution/associative/content capabilities already there.

Each stage is a cheap, falsifiable experiment in the spike style — set the mechanism up to win, then check
whether a *small* model wins with near-zero training. That is the homogenised bet: **teach the model to use
the scalar, not to be one.**
