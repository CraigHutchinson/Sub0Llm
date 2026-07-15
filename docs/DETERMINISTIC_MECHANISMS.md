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
| 2. BIND — op over bound scratch slots (algebra) | `bindspike` A/B | delegation **1.000** vs fuzzy **0.000** — model reasons over the *symbol* (digits absent from its stream), node derefs the *value* |
| 3. FILTER — mask scalar facts vs contamination | `arithspike` filter A/B | masked clean (sampled **1.000**) vs contaminated (sampled **~0.90**, ~10% fuzzy leak) |
| 4. COMPOSE — dynamic-turn chained reduction | `chainspike` (var. length) | trained k=2..6 **1.000**; held-out length k=7 **1.000** (k=8 only fails on context overflow) |

The region-frame result validates the natural-language-marker design (§1); the filter result is nuanced
(§3); chaining shows delegation **composes over a dynamic number of turns and length-generalises** (§4/staged
plan). The **production node registry** (`include/sub0/nodes.hpp`) is now a first-class, tested substrate.
Pillar 2 (numeric BIND) is now proven too (`bindspike` — op over bound slots, node dereferences the
bindings); the **end-to-end production wiring** (the op/compute callback + `set_scratch_bindings` in
`gen_stage.cpp` — today it wires only spell/scratch `expand`/`combine`) is the remaining piece.

**GSM8K capstone (`[.gsm8kcapstone]`) — end-to-end delegation IN PROSE, and a reusable lesson.** Arithmetic
consolidated onto ONE general node: `[op math <expr>]` evaluates a whole formula (precedence, parens, exact
big-int) — the per-op add/sub/mul/div FRAMES are removed (they stay as `math`'s internal primitives). A tiny
d128 model trained on synthetic GSM8K-format problems (via `gsm8k::build_stream`) and decoded through the
production callback reaches **delegation 1.000, exact-match 0.935**, and — because the node is exact —
**generalises to 3-digit operands it never trained on (OOD exact 0.675)**. The lesson: **prefer *contextual*
delegation over *self-contained* copy.** Making the model copy the expression into the frame
(`[op math 12+34]`) hits the localization/copy wall (exact ~0.2–0.4, unstable); a **bare `[op math]`** that
lets the node read the expression already posed in the prose (`… 12+34 = [op math]`, via
`node_frame::preceding_expr`) sidesteps it entirely. Errors then localise to *routing* — arithmetic error is
0 by construction whenever delegation fires — which is exactly the decomposed, auto-verifiable metric a
well-formed-math task affords. (Overtraining degrades this toy task past the early peak; a recipe-tuning item,
not a mechanism one.)

**Worked example — from raw GSM8K text to a training frame.** Concretely, for a corpus fragment
`get_gsm8k.py` writes verbatim from the dataset (GSM8K's own calculator-annotation format; see that script's
docstring for the ELI5 of `<<expr=result>>` itself):

```
There are 80/100 * 10 = <<80/100*10=8>>8 more purple flowers than yellow flowers.
```

`gsm8k::segment()` (`gsm8k.hpp`) splits this into alternating literal-text and annotation segments by a
plain `<<...>>` substring scan, turning the annotation `80/100*10=8` into one `Op{expr="80/100*10",
result="8"}`. `verify()` re-runs the SAME exact `math` node on `expr` and keeps the annotation only if it
reproduces `result` bit-for-bit — this IS the FILTER pillar's guard: a malformed or non-integer annotation
is left as plain (graded) prose instead of being turned into a delegation.

`build_stream()` then emits, in order:

1. the literal prose up to the annotation, tokenized and **GRADED** (mask=1) — `"There are 80/100 * 10 = "`;
2. the op-frame `TOK_TURN_START op math TOK_TURN_END`, **GRADED** (mask=1) — the ROUTING decision the model
   must learn: at this point, delegate. The expression is NOT copied into the frame — it's already sitting
   in the prose the model just emitted, and the node re-reads it from there (`node_frame::preceding_expr`,
   the same "prefer contextual over self-contained" lesson above);
3. the result, **MASKED** (mask=0) — either the collapsed scratch-slot token `SCRATCH_SLOT_BASE` (production
   / `--gsm8k` training) or the literal digits `8` (test-only, `collapse=false`). Predicting this token earns
   no loss signal either way — the only way to "know" it is to have delegated;
4. the redundant `8` GSM8K repeats right after `>>` is stripped from the following prose segment (else it
   would duplicate the masked result as graded, un-masked text — reintroducing the exact fuzzy-copy
   incentive FILTER exists to remove) before the remaining prose (`" more purple flowers..."`) resumes graded.

So the model's actual training signal for this fragment is: predict the prose, predict `[op math]` right
after `= `, predict nothing for the answer (the node supplies it). A whole GSM8K solution typically carries
several such annotations, so the training example is a natural mix of graded reasoning prose and masked
delegated arithmetic — with zero hand-authored curriculum needed, because GSM8K's own worked solutions
already have the shape the FILTER pillar wants.

**Multi-step chaining — COLLAPSE decisively beats digit-copy (`[.chaincapstone]`, 1.000 vs 0.095).** Chaining
`R2 = 7 + (A+B)` forces step 2 to *reference* the intermediate. If the intermediate stays as **digits** the
model must copy into `7+R1`, a tiny model fails (best exact **0.095** — the copy wall again). If instead the
result is **collapsed to a scratch-slot symbol** (`make_collapse_callback`: bind the exact value to a slot,
inject that one token; the next op derefs it), the model references `7+S0` — one stable token — and chains
**1.000, rock-stable every round, no overtraining**. This is the reusable principle, now proven twice:
**never make the model copy a value — let it route over *symbols/context* and let the node/binding hold the
value.** It is also the mechanism the "collapse a result to a scratch token" note (§1b) anticipated.

**Repeated mentions — HARNESS-driven collapse (no model request) beats fuzzy repeat-copy, and a MID-STREAM
binding works as well as a pre-arranged one (`[.repeatspike]`, held-out: 0.95 vs 0.13 by step 3000).** A third
proof of the same principle, this time for *entity tracking* rather than arithmetic: an OOV word appears three
times in a passage; a query asks for its Nth character. **FUZZY** (all three mentions spelled out in full,
answer via plain in-context lookup — no scratch mechanism at all) stays low and noisy on held-out OOVs (0.02 →
peaks near 0.32 → ends **0.13**) — the copy/localization wall bites even though the answer is plainly visible
in the fed context, the same wall arithmetic copying hits. **COLLAPSE** (mention 1 spelled out normally — no
request, no marker, exactly as a live harness would observe it and bind a slot — mentions 2 and 3 replaced by
the bound slot token, masked; the query references the slot, resolved via `UNCOMBINE`) climbs steadily to
**0.95**. Two things this specifically adds beyond the chaining result above: (1) it's the harness — not the
model — that decides to bind, mirroring `nodes::make_collapse_callback`'s pattern (the model never asks,
op-collapse already does this for computed results) but applied to a *recognised repeat* instead of a *computed
value*; (2) the binding forms **mid-stream**, from the passage's own first mention, rather than being
pre-arranged before the episode starts the way `scratchspike::nth_char_task` already tests — closing the gap
toward "the model just writes the word normally, the system silently compacts recurrence #2, #3." This proves
the *mechanism* (spike-style, over baked training traces — the full-forward path every other spike here uses)
independently of the harder, unbuilt engineering question: doing this live, mid-*generation*, inside a running
KV-cache needs a splice capability that does not exist today (see `include/sub0/repeatspike.hpp`'s header
comment and project memory `persistent-slot-range-engine-substrate` for the concrete gap). Also: this
directly confirms scratchspike's own prior finding from the *other* direction —
`define_task` (#1, model-driven: the model must ask via `combine`) is documented there as "copy-bottlenecked",
excluded from the production curriculum — harness-driven binding isn't just *an* option, it's the one that
actually works at this scale, now proven for a second trigger condition (repeat-recognition, not just
pre-arrangement).

**The unbounded (persistent) slot-range ENGINE SUBSTRATE is now real (2026-07-16), tested, not just
designed.** The scratch-slot mechanism above lives entirely inside the fixed, bounded reserved-marker
range (`SCRATCH_SLOT_BASE..+COUNT`, 6 ids). A structurally different, *unbounded* range — ids `>= VOCAB`,
for a future persistent compound-word cache (a recurring multi-token OOV gets ONE stable id forever,
composed live from its fragments, no per-mention token cost) — is now wired into the engine and proven
correct: `include/sub0/scratch_slots.hpp`'s `PersistentBindings`/`is_persistent_slot`/`persistent_fragments`,
`backend_cpu.cpp`'s `op_embed`/`Op::Embed` backward/`forward_one` all route any id `>= VOCAB` through the
SAME `encode_slot`/`encode_slot_bwd` composition the bounded pool uses (id-agnostic already, needed zero
changes) — never the raw embedding-table lookup, closing a real latent OOB-read risk (`VOCAB` sizes the
table; nothing previously stopped an id past it from indexing past the table's bounds if one ever
arrived). `tests/persistent_slots_engine_tests.cpp` (5 cases) proves: compose correctness against a
manual reference, real gradient flow through a full forward/backward cycle, safety when unbound/no table
is installed (finite output, no crash), and byte-identical invariance when the feature is untouched. This
is the SUBSTRATE only — no compound-word DB, no encode-time substitution, no live decode-time collapse
yet (repeatspike proved that mechanism using the *bounded* pool, not this range) — see project memory
`persistent-slot-range-engine-substrate` for the full design reasoning (why unbounded ids need no
tokenizer-format bump, unlike growing the bounded pool) and what's still open.

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
TOK_TURN_START op add A B TOK_TURN_END          ▶ node("add", [A,B]) → exact result injected, masked
TOK_TURN_START op solve "x^2 = A" TOK_TURN_END  ▶ node("solve", …)   → the CAS's answer injected
TOK_TURN_START op uncombine <tok> TOK_TURN_END  ▶ expand (today's UNCOMBINE, re-expressed in the frame)
```

**The word `op` is load-bearing, not decorative — it's the role discriminator.** `TOK_TURN_START`/`TOK_TURN_END`
are reused from ordinary chat turns, so the dispatcher needs a way to tell "this is a tool-call region" apart
from a normal system/user/assistant turn sharing the same two markers; `op` is that tell, exactly the way a
chat turn's first word is its role. The parser checks it literally
(`node_frame.hpp`: `if (toks.size() < 2 || toks[0] != "op") return {};`) — a region that opens with anything
else is inert (no dispatch, no injection). Drop the word and every plain chat turn beginning with a word that
happens to match a registered op name (e.g. a turn starting "add …") would misfire as a tool-call.

The op selector and operands are ordinary tokens the vocab already has, so **a new NODE costs ZERO tokenizer
budget** — no reserved-id per op, no format bump. (Contrast: the scratch pool maxed at K=6 precisely because
each slot is a reserved id; a region frame sidesteps that ceiling.) The registry keys on the op *word* after
the frame-open, not on a dedicated token — no new marker was minted; the op frame is `TOK_TURN_START`/`TOK_TURN_END`
itself, reused (this superseded an earlier sketch of a dedicated `<|op|>`/`<|end|>` pair, which was never built).
`expand`/`combine`/`compute` all collapse into this one shape; today's per-op reserved markers (`TOK_UNCOMBINE`, …)
and the arithspike's ASCII sentinels (`$`/`#`, its literal `COMPUTE`/`COMPUTE_END` constants) were that *spike's*
own encoding, superseded by the region-frame — the production one, and the natural home for the whole registry.

**Prior-art check (2026-07) — how Qwen and Gemma actually do it, and a revision to the claim above.** Checked
against real production tool-calling schemes:

- **Qwen (Hermes-style, Qwen2.5/Qwen3)** reuses `<|im_start|>`/`<|im_end|>` for the outer turn — the same move
  as our `TOK_TURN_START`/`TOK_TURN_END` reuse — but the CALL ITSELF is wrapped in its own dedicated marker
  pair added to the tokenizer, `<tool_call>...</tool_call>`, with results fed back as `<tool_response>
  ...</tool_response>` on a `tool`-role turn.
- **Gemma 4 / FunctionGemma** goes further: `<|tool_call>...` / `...<tool_call|>` and `<|tool_response>...` /
  `...<tool_response|>` are dedicated special tokens **explicitly distinct from** `<start_of_turn>`/
  `<end_of_turn>`, not nested inside the generic turn markers at all.

Neither reuses the generic turn pair *plus a magic first word* the way our `TOK_TURN_START op <name> …
TOK_TURN_END` does today. This **revises the "zero tokenizer budget" framing above**: true in the narrow
sense (no new reserved id), but the trade was a RECURRING per-call cost — the literal `op ` text, ~2-3 tokens
on this byte-level tokenizer since "op" is very unlikely to be a learned Unigram piece — to save a ONE-TIME
cost (0 reserved ids). Prior art makes the opposite trade: pay a small FIXED reserved-id cost once, save the
recurring per-call bytes forever. At any real op-call volume (arithmetic, dates, lookups — potentially many
per generation) that is the better trade, and matches the load-bearing-word finding just above: `op` is
already functioning as a de facto marker, just an expensive text-encoded one instead of a cheap reserved id.

**Backlog — mint dedicated `TOK_OP_START`/`TOK_OP_END`** (or `TOK_TOOL_START`/`TOK_TOOL_END`, generalising
straight to the unified tool-calling §1c anticipates), retiring the literal `"op"` discriminator. Concrete
blocker: there is currently **zero free reserved-id headroom** — all 6 `TOK_RESERVED_4..9` ids are already
claimed by the scratch-slot pool (`scratch_slots.hpp`: `SCRATCH_SLOT_COUNT = TOK_MARKER_COUNT -
TOK_RESERVED_4` = 6, no slack). Minting 2 new ids means either (a) shrinking the scratch pool 6→4, or (b)
bumping `TOK_MARKER_COUNT` 32→34. Either is a tokenizer-FORMAT bump (§C below) — `n_base`/VOCAB shift, every
learned piece id shifts, and it invalidates every currently-trained model (fineweb d448, the `--gsm8k`
model, …), the same cost §C already flags for growing the scratch pool past 6. Worth doing, but as one
deliberate, batched format bump — ideally the SAME bump that ever grows the scratch pool, so the retrain
cost is paid once for both, not twice.

**Notation used throughout the rest of this doc:** `[op <name> <args>]` is prose shorthand for one
`TOK_TURN_START op <name> <args> TOK_TURN_END` region (what actually gets rendered/decoded); `COMPUTE` /
`<|op|>...<|end|>` appearing anywhere below are spike-era names for encodings that were tried and superseded,
kept only where the text is describing that specific historical spike.

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

**DONE — a native exact `math` expression node (commit 60cb5e9).** Rather than the model decomposing an
expression into ordered binary op-frames and getting precedence right itself, ONE general node evaluates the
whole thing: `[op math 6+8*10-5]` → **81** in a single frame. *Parsing + precedence is itself deterministic,
so it belongs in the node, not the fuzzy model* — the model just copies the expression it is already reasoning
about. `nodes.hpp`'s `eval_expr` is an **exact signed big-integer** recursive-descent evaluator (`^` right-assoc
> `* /` > `+ -`, parens, unary ±) built on the add/sub/mul/div primitives — **no `double`**, and it declines
(returns "") on non-exact division / div-by-zero / malformed input rather than ever emitting a wrong answer.
The callback now passes the raw expression through (lexes the region with operators preserved). This is the
in-house version of the exprtk row; exprtk/SymEngine remain the escalation for decimals/rationals and symbolic
work. The per-op nodes stay as `math`'s primitives (and for the bound-slot BIND path). **Operators are single
ANSI bytes**, so the lexer is trivial — a nice property of the byte-level tokenizer.

**Follow-on — collapse a `math` result to a scratch token.** A computed result can be *bound to a scratch slot*
(one token) instead of spilled as digits, so the model reasons over the symbol and `uncombine`s to the inner
digits only if it actually needs them — the numeric-BIND substrate (§2) applied to *results*, not just inputs.
Keeps the context/KV small for long chained calculations. Must be A/B-tested (does collapsing the result help
or hurt downstream reasoning vs. leaving the digits inline). Backlog.

**Complex results — a result is not always a plain decimal (exponents, powers, rationals, symbols).** `2^100`
is 31 digits or the symbol `2^100`; `1.23e45` needs scientific form; division needs a rational; `sqrt(2)` is
irrational. The node result type is therefore a **canonical numeric grammar** — `int | decimal | scientific
(e) | rational (/) | symbolic (^, sqrt, …)` — emitted as **ordinary tokens** (digits, `.`, `e`, `-`, `^`, `/`
are already in-vocab; **zero new tokenizer ids**, same ethos as the region frame). Each node declares which
form it returns; the harness just injects those tokens. This is also where the **scalar re-entry** meets its
match: `SlotEncoding::Scalar` already encodes any of these to `(sign, exp, mantissa)` for a bounded
intermediate re-entry (§A), while the exact grammar string stays in the fragment binding for the terminal path.

**Verified / symbolic backends — Lean + mathlib.** A step beyond a CAS: Lean's `mathlib` is a vast, *machine-
checked* math library, so a Lean node can return not just an answer but a **proof-carrying** one (exact
symbolic algebra, number theory, `decide`/`norm_num`-style verified equalities). Two honest roles, matched to
its cost (JIT-elaborated tactics → up to seconds/query, a heavy toolchain — never an inline per-token decode
node): (1) an **offline oracle for the FILTER pillar** — generate *and verify* the exact answers we mask
scalar-solvable spans down to, so the training data's delegated results are provably correct (and to *mine*
new scalar-solvable templates); (2) a **high-value external-process node** for the rare hard symbolic/proof
query, sandboxed like the other heavy nodes (FLINT/Pari row). For fast inline symbolic work, SymEngine
remains the embed; Lean/mathlib is the *correctness oracle* and the proof-grade escalation, not the hot path.

### 1c. Ops vs. tools — the safe, engine-embedded subset of tool-calling

`math` is not the whole story: the registry is an **extensible table of operations**, and op-calling is the
**safe, in-engine sibling of tool-calling**. Both share the *exact same model-facing mechanism* — emit a
region-frame call, a handler runs, the result is injected (loss-masked) — so training a model to route to ops
teaches precisely the skill that transfers to tool-use. The difference lives entirely in the **handler**, and
it splits three ways along *purity × safety*:

| Category | Examples | Purity | Training treatment | Safety |
|---|---|---|---|---|
| **Pure op** | math, unit convert, gcd, sort, string ops, base convert | deterministic `f(inputs)` | **bake** the result (teacher-forced, masked) — current design | always safe, even speculatively |
| **Impure-but-safe op** | `date`, `now`, `elapsed`, RNG, **read-only** db / lookup | depends on runtime state, *no side effects* | **can't bake** → train only the *routing*, mask a placeholder, live-resolve the value | always safe |
| **Tool** (effectful) | bash, http, db-**write**, file-write | side effects | never baked, live-only | **guarded** — authorization / sandbox / confirm |

**Strategic consequence:** ops are pure-or-read-only and never touch bash/network/writes, so we can train the
tool-use *routing* behaviour **massively and safely, entirely offline via generate-and-verify** — and it
generalises to real, guarded tool-use. The op table is the safe training ground; a tool is the same frame with
a safety gate bolted on. The bake/live split is already the forward/backward analysis (§A): pure → bakeable,
impure-safe → the masking pillar handles it (mask the result, train the routing, inject live), effectful →
out of the always-safe path.

**Scope check — the substrate already supports this.** A new op is one `register_node(name, fn)` call, ZERO
tokenizer budget (word op-name; nodespike). The `NodeFn` (`operands → result`) already covers **nullary**
(`[op date]` → empty operands) and **variadic** ops, and the region frame carries an arbitrary op-name, so the
heterogeneous catalog needs no mechanism change. What a fuller catalog *will* want (backlog, build when a
consumer exists — [[only-add-arguments-we-need]]): a per-op **purity/safety tag** so the train-time baker
knows bake vs. mask-only vs. gate, and a **tool tier** with the permission/sandbox layer for effectful calls.
Useful near-term pure/impure-safe ops to seed the table: `date`/`now`/`elapsed`, unit + base conversion,
`sort`/`min`/`max`/`gcd`, string ops, read-only key/db lookups.

### 2. BIND — scratch tokens as symbolic (algebraic) variables

A scratch slot is **algebraic notation.** Binding a precise value to a slot lets the model reason over the
*symbol* and expand to the *content* only on demand. Numbers are the killer case — precision is exactly what
embeddings are worst at:

```
"12345678.32232 + 999836475.123 = X"
        │                │
     bind A            bind B          (each collapses to ONE scratch token)
"A + B = X"  ──▶  the model reasons algebraically; it never has to hold 13 digits in its activations
"A + B = [op add]"  ──▶  node dereferences A, B from the binding table, returns the exact value, bound to a fresh slot C
"X = C"
```

The model does **not** learn digit-arithmetic. Algebraically, `A + B = X` is trivial; the hard part
(carrying 13 digits) is delegated. A small model with near-zero arithmetic training should resolve this,
because it isn't doing arithmetic — it's doing *substitution + a node call.* This also slashes context and
KV: a 13-digit number is one token unless the model *asks* (uncombine) for the digits. Scratch tokens are a
**super-power for scalars** precisely because they are variable-binding.

**TESTED (`bindspike` A/B, committed).** The whole flow above now runs end-to-end. Operands are bound scratch
slots — the model sees `S0 + S1 =` where each `S`ᵢ is a single `SlotEncoding::Scalar` token (magnitude only;
the exact digits appear *nowhere* in the token stream) — emits the routing `[op add]`, and the bind-aware
node (`bindspike.hpp`, `make_bind_compute_callback`) resolves the operands by **dereferencing the slots from
the binding table**, returning the exact sum. d128, held-out fresh number pairs:

| arm | held-out exact |
|---|---|
| DELEGATION (route → node derefs the bindings) | **1.000** (from step 300) |
| FUZZY (model must produce the sum from the Scalar slots) | **0.000** (never) |

The **0.000 is the proof, not a disappointment**: the model *cannot* produce the sum because the digits are
genuinely not in its reasoning stream — so the only path to the exact answer is the node dereferencing the
binding. Two 13-digit numbers occupy two tokens; the model reasons over the *symbol* and delegates the
arithmetic; the value never touches its activations. This is arithspike's 1.000-vs-0.000 lifted from inline
digits onto **algebra** — pillar 2, closed. (`bindspike` is the spike; folding the bound-slot dereference
into `node_frame.hpp` is the productionisation step, alongside the gen wiring.)

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

**TESTED — the scalar re-entry holds for magnitude/leading-digit reasoning, breaks below the encoding
(`scalar_reentry` A/B, committed).** `SlotEncoding::Scalar` (`scratch_slots.hpp`) *is* the intermediate
re-entry: the classical result re-enters as **one** bounded `(sign, base-10 exponent, leading mantissa
digits)` vector — a fixed, stop-gradient encoding that is bounded *regardless of magnitude*, so a power's full
expansion (`2^100`) and a scientific literal (`1.23e45`) map into the same vector as a small integer (the
complex-result case, below). Single-number readout A/B (read a queried digit of a re-entered value; d128,
mantissa = 4 leading digits), held-out:

| re-entry channel | leading digit | trailing digit |
|---|---|---|
| DIGIT (N tokens) | 1.000 | 1.000 |
| SCALAR (one vector) | **1.000** | 0.125 (≈ chance) |

A one-vector re-entry is **as good as N digit tokens for the reasoning the encoding carries** (leading digits
/ order-of-magnitude — comparisons, routing, sign/size decisions) and **provably chance where it is lossy**
(the low-order digits it drops). That is the honest answer to *"will the brain-swap hold at scale?"* — **yes**
for magnitude/leading-digit reasoning, **no** for low-digit precision, which is exactly why the *exact* value
stays in the slot's fragment binding for the terminal/exact path. Widening the mantissa trades vector width
for how deep the scalar stays exact; a **learned** number→vector encoder (the live backward choice below)
could pack more discriminable value per dim — the natural next probe if the fixed encoding's ceiling bites.

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
(i) **mid-prompt** ops need a KV-cache-aware splice; (ii) ~~**scalar-embedding** result — inject the value as
*one vector* instead of digit tokens.~~ **DONE (fixed encoding):** `SlotEncoding::Scalar` + the A/B above —
the fixed `(sign,exp,mantissa)` encoder is stop-gradient and holds for magnitude/leading-digit reasoning; the
**learned** number→vector encoder (gradient flows to it) is the still-open half, worth it only if the fixed
encoding's low-digit ceiling actually bites a target task; (iii) free-running/RL, only on demonstrated
exposure bias.

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

**Superseded in practice by a different uncapping strategy (2026-07-16) — reconciling the two.** When the
pool actually needed uncapping (for a persistent compound-word cache, not a format-bump concern), the path
taken was NOT named regions — it was single-token ids `>= VOCAB` (`PersistentBindings`, see the PROVEN
section above and project memory `persistent-slot-range-engine-substrate`). Named regions (multi-token,
persists as plain text) and the unbounded id range (single-token, compact, but ids aren't human-readable
text) solve different problems: named regions win when a binding must **survive as text** (round-trip
through re-tokenization, a real corpus, an external store) — genuinely the long-term-memory case this
section was written for. The unbounded id range wins when the binding is **process-local and compactness
matters** (a compound-cache entry doesn't need to survive as text; it's rebuilt from a DB keyed by the word
itself, not by the id). Both remain valid, for different jobs — this section's "long-term memory" framing
is the named-region case specifically, not a general replacement for the ephemeral pool.

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
   zero tokenizer budget. *Production form landed differently than first sketched here* — see stage 5: no new
   delimiter was minted, `TOK_TURN_START/END` were reused directly (zero new ids, not "one reserved slot").
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
   *Remaining (updated 2026-07-16):* the compute-delegation curriculum landed as its OWN `--op-mix` flag,
   not folded into `--scratch-mix` as this line originally proposed — see `op_curriculum.hpp`/`gsm8k.hpp`
   and the GSM8K capstone above. `--op-mix` now records its collapsed results into a `doc_bindings`-shaped
   table too (mirroring `scratchspike::Dataset`; `gsm8k.hpp`'s collapse slots increment per annotation
   within a document — S0, S1, ... — instead of always reusing S0, so each slot holds exactly one binding
   per document, matching `--content-embed`'s invariant), so `--content-embed` now works from `--op-mix`
   alone, `--scratch-mix` alone, or both blended together (`train_stage.cpp`'s `content_embed_active`,
   `gen_stage.cpp`'s matching `content_embed_on` gate). Still open: a decode-to-text step for a full Unigram
   deployment (the byte parse suffices for this project's byte-heavy tokenizer); real library nodes.

Each stage is a cheap, falsifiable experiment in the spike style — set the mechanism up to win, then check
whether a *small* model wins with near-zero training. That is the homogenised bet: **teach the model to use
the scalar, not to be one.**
