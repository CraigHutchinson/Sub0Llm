---
name: deterministic-mechanisms-thesis
description: Strategic thesis -- model orchestrates exact computation (delegates to deterministic nodes) instead of approximating it; unify spelling/scratch/math/tool-use; mechanism-first over scale
metadata: 
  node_type: memory
  type: project
  originSessionId: 2a6b481b-6211-474b-818a-995223d7d5ac
---

Strategic direction the user set (2026-07-13), committed as docs/DETERMINISTIC_MECHANISMS.md (84c5ef9).
Reframes HOW we train, generalising the combine/uncombine + scratch-token pattern.

**Thesis:** a model should ORCHESTRATE exact computation, not APPROXIMATE it. For anything a classical
scalar computer computes deterministically (arithmetic, spelling, precise numbers, dates, unit conversions,
lookup, sort), the model learns *when/how to delegate* to a deterministic node and lets it produce the exact
answer. Fuzzy model-embedded knowledge is reserved for genuinely fuzzy tasks (language, judgement, world
knowledge). Inverts the scaling reflex: when a capability is hard, FIRST ask "is it deterministically
computable, can a mechanism make it trivial?" -- scale is the last resort, wrong lever for exact tasks.

**Three pillars (all needed to "set the mechanism up to win"):**
1. PROVIDE a deterministic node behind a marker token -- generalise kv_decode_generate's expand/combine
   callbacks (already a tool-invocation seam!) into a ComputeNode registry (add/cmp/sort/date/CAS/latex...).
   Nodes are non-differentiable classical code between gen steps; injected result is LOSS-MASKED so gradient
   only flows to ROUTING + operand selection, never a fuzzy internal copy.
2. BIND precise values to scratch tokens = ALGEBRAIC notation. `3.141` -> one scratch token A; big-number
   add `12345678.32232 + 999836475.123 = X` becomes bind A,B -> "A+B=X" (trivial substitution) -> COMPUTE
   add A B -> exact C. Model never does digit-arithmetic; also slashes context/KV (a 13-digit number is 1
   token unless it asks/uncombines). Scratch tokens are variable-binding -> a super-power for scalars.
   Unifies OOV-compression + spelling + numeric/symbolic computation under one "bind a precise thing to a
   symbol; expand or compute on demand".
3. FILTER the training data so the FUZZY path is masked out (user's 2nd note). Raw corpus text like
   "12345+67890=80235" REWARDS fuzzy memorisation of the answer -- providing the node isn't enough, you must
   REMOVE the incentive. A corpus PRE-PASS detects scalar-solvable spans (arithmetic/equalities, precise
   numerics, dates, unit conversions, spellings, sorted lists, lookups) and MASKS the result tokens in the
   loss (or rewrites operands into bound scratch tokens) so the ONLY rewarded path is the mechanism. Classical
   regex/grammar catches most; a SMALL corpus-scanning MODEL "embellishes masking onto the dataset" for the
   fuzzy boundary cases (is this number a quantity-to-compute or rhetorical?). Reuses existing
   LOSS_IGNORE_INDEX machinery -- exactly the spell-spike trick (mask injected chars so the only path is
   resolution) applied to the corpus.

**Why now:** the [[scratch-tokens-context-translation-layer]] localization saga -- scale (d256, LR-retune,
heads, CoT) moved the needle ZERO; a MECHANISM (local grounding) solved it at the smallest model. Evidence
for mechanism-first triage. The interceptor seam + binding table + masking ALREADY EXIST -- this is mostly
(a) generalise the marker/registry, (b) add a numeric-binding path, (c) build the corpus filter pass.

**Staged plan (falsifiable, spike-style):** (1) rename expand/combine into a ComputeNode registry; (2)
numeric binding + ONE arithmetic node -- crispest test: big-number addition near-free at d128 with trivial
training (if NOT, thesis is wrong, learned cheaply); (3) the filter pass (classical first, small model for
ambiguous), measure it STOPS answer-memorisation; (4) widen nodes + CHAINED delegation via scratch-bound
intermediates; (5) fold into --scratch-mix. Related: [[compile-time-decisions-preferred]] (bake known/
deterministic decisions), [[verify-correctness-against-reference-before-perf]], [[capability-roadmap-discrete-experts]].

PROVEN for arithmetic (committed 1efcb0f code, bdfd9a2 docs). Added a `compute` node to kv_decode_generate
(callback + caller-supplied marker, independent of expand/combine, NO tokenizer change). arithspike.hpp:
exact big-int add on decimal strings; delegation vs fuzzy A/B on held-out random 8-14 digit sums. RESULT
(same d128 model, same 3000-step budget): DELEGATION held-out exact-match 1.000 (from step 300, per-digit
1.000 -- learns only routing, generalises perfectly bc node is exact); FUZZY 0.000 (per-digit ~0.10-0.15,
never). 1.000 vs 0.000 -- the thesis proven as starkly as possible. Hidden test [.arithspike].

TWO REFINEMENTS (user notes 2026-07-13, folded into the doc):
(A) MARKERS = NATURAL-LANGUAGE REGION FRAME, not a token per op. The project ALREADY does this for chat:
casing.hpp TOK_TURN_START/TOK_TURN_END are TWO markers (not one-per-role), role name flows as ORDINARY text
(ChatML <|im_start|>system...); reserved headroom explicitly anticipates "tool-call structure". Apply to
compute: ONE `<|op|>...<|end|>` delimiter pair + op named in ordinary word tokens (`<|op|> add A B <|end|>`).
=> a new node costs ZERO tokenizer budget, sidesteps the reserved-id ceiling (which capped SCRATCH_POOL at
K=6). Registry keys on the op WORD. expand/combine/compute all collapse into this shape; per-op reserved
markers + the arithspike's $/# ASCII sentinels are the SPIKE encoding, the frame is PRODUCTION. Next-impl.
(B) SOLVER NODES for complex maths (permissive, deterministic, sandboxed): Boost.Multiprecision (exact
decimals/rationals -- the .32232 case), exprtk/tinyexpr++ (numeric expr eval, header-only MIT/zlib),
SymEngine (MIT C++ CAS -- solve/simplify/diff, SymPy's backend, CPM-friendly). Prefer MIT/BSD/Boost; avoid
GiNaC (GPL); deliberate re GMP/MPFR (LGPL). Fetch via existing CPM.

ALL THREE PILLARS PROVEN with data (committed 1efcb0f arith, 7f6f454 nodespike, f84487e filter, d0ebdd1
docs). (1b) REGION FRAME validated -- nodespike.hpp: one `{ ... }` frame + a WORD-keyed ComputeNode registry
over 4 ops (add/sub/max/min -- arithmetic AND comparison). A/B: op-name as WORD tokens (zero tokenizer cost)
vs DEDICATED per-op marker -> WORD 1.000 == TOKEN 1.000 held-out. So natural-language op-names route with NO
penalty -> the frame design is strictly better (free unbounded extensibility; dedicated tokens exhaust the
~6-10 reserved-id budget, words never do). Dispatch is on the MODEL's emitted header, so 1.000 = correct
routing. Extensible: a new op is one registry row.
(3) FILTER pillar TESTED (nuanced -- measured, didn't assume): corpus = 50% delegation + 50% plain arithmetic
facts, masked (filter) vs unmasked (contaminated), eval WITH the node. RESULT: GREEDY self-corrects -- both
1.000, because COMPUTE is the MODAL next-token after `=` (one frequent learnable token beats dispersed
UNPREDICTABLE fuzzy digits), so greedy always delegates even contaminated -> contamination INVISIBLE to
greedy. SAMPLING (topk=12) exposes it: contaminated keeps ~10% fuzzy probability mass (only ~10% not 50% --
unpredictable target earns high loss so low learned prob) -> sampled ~0.90; MASKED is clean -> sampled 1.000.
So the filter isn't needed for greedy correctness (mechanism self-selects, delegating is low-loss) but IS
needed for a CLEAN sampling/temperature-robust distribution, and is ESSENTIAL where a scalar pattern has
facts but NO delegation coverage (fuzzy is then the only rewarded path).

(4) CHAINING / DYNAMIC TURNS DONE (committed 6477de6 chainspike). User flagged "chaining may need a dynamic
number of turns." Task: reduce a variable-length sum n1+..+nk to one value via a DYNAMIC number of reduce-node
calls; model emits {} to reduce the leftmost +, node injects the shorter expression as a STATE (...), model
decides DONE when a single value remains (LOCAL termination decision). RESULT (d128/L4/H4 seq128, train
k=2..6): trained k=2..6 = 1.000; held-out LENGTH k=7 = 1.000 (LENGTH-GENERALISES -- the per-step decision is
local); k=8 = 0.00 but ONLY context overflow (trace k7=110 fits, k8=138 > SEQ_LEN 128; O(k^2) re-emit). So
delegation COMPOSES over a dynamic turn count, the model terminates itself, and it generalises to unseen
lengths -- ceiling is context, not reasoning. Fix: O(k) running-accumulator / scratch-bound intermediates.

PRODUCTION MOVE STARTED (committed e41c35e nodes.hpp + 2f3b48e docs): include/sub0/nodes.hpp = the
engine-facing extensible ComputeNode registry (name->NodeFn, N-ary) + exact decimal big-number primitives
(add/sub/cmp arbitrary length) + built-in add/sub/max/min; a new node is one register_node() call, zero
tokenizer cost. Engine-free frontend unit test nodes_tests.cpp (21 assertions: exactness, dispatch, graceful
miss, extensibility). Production MARKER plan documented: reuse casing.hpp TOK_TURN_START/END (op-name +
operands as text, like a chat tool-call) -> zero new reserved ids. REMAINING production wiring: decode
interceptor dispatch onto nodes.hpp, the real TOK_TURN region frame, a compute-delegation curriculum folded
into --scratch-mix; plus numeric BIND (literals->scratch slots, the O(k) composition substrate) and
exact/symbolic library nodes (Boost.MP/exprtk/SymEngine).

PRODUCTION OP FRAME LIVE (committed 8874245): node_frame.hpp bridges nodes::Registry to kv_decode_generate's
compute seam using the EXISTING TOK_TURN_START/END markers (zero new ids). OP region = `TOK_TURN_START op
<name> <operands> TOK_TURN_END` (a chat tool-call); on TOK_TURN_END the interceptor parses + dispatches on
<name> over operands (in-region else preceding inline expr) + injects `TOK_TURN_START <result> TOK_TURN_END`;
non-op regions (chat turns, the injected result) inert. Proof: mechanistic frontend test (28 assertions:
dispatch, inline/in-region operands, big-num exact, inert cases) + END-TO-END engine test (model emits [op
add] for A+B=, decoded through the production callback -> held-out exact 1.000 from step 300).

FORWARD-PASS (PREFILL) RESOLUTION + 3 ARCHITECTURE DIRECTIONS (committed 492232a code, 0e77c67 docs; user
architecture musings). (A) Resolution is a DISCRETE FORWARD-STEP event, not iterative reasoning -- one op =
one delegation step, no CoT needed. Factored decode.hpp interception into resolve(next); now fires per
generated token AND after prefill, so a prompt that POSES a computation (ends in an op region, `12+34=[op
add]`) resolves in the FORWARD PASS -- no generation loop (tested: -> [46] injected in prefill; behaviour-
preserving, inert unless prompt ends in a marker + callback given; full suite 78 green).
FORWARD/BACKWARD ANALYSIS (committed def69a9; user: "consider backward/forward op resolve before committing").
A node is NON-DIFFERENTIABLE -> backward can only STOP-GRADIENT at its output (nothing to learn about an exact
op; only ROUTING, already supervised by the op-header loss; through-node grad is ill-defined -- the task fixes
the operands). DECISION: for teacher-forced training, BAKE (resolve offline) and FORWARD-RESOLVE+STOP-GRADIENT
give IDENTICAL gradients, so baking is strictly better (once vs per-epoch, cheaper for a CAS node, no engine
change). KEEP the current BAKE-TRAIN / LIVE-INFER split (chainspike proves it: states baked in training,
node-injected at eval, matched 1.000). Live training-time resolution ONLY pays off for NON-teacher-forced
operands = a FREE-RUNNING/RL decision (reward on routing, no grad through node) -> defer until exposure bias
demonstrated. Backlog: mid-prompt ops (KV-cache splice); SCALAR-EMBEDDING result (inject value as ONE vector
not digit tokens) -- the ONE genuinely-live backward choice: FIXED numeric encoding -> stop-grad, LEARNED
number->vector encoder -> grad flows (like the CharEncoder); free-running/RL only on demonstrated exposure
bias. Next-build candidates: scalar-embedding prototype OR named-scratch migration (both > re-plumbing
supervised resolution).
PLUGGABLE-PREDICTOR framing (committed e95b5f8; user: "swap out the AI brain for a classical computer for a
few tokens"). Generation = a sequence of predictors; neural model predicts the routing (op region), a
CLASSICAL computer predicts the RESULT span. Today the classical computer already predicts the result (never
sampled from the model) but we still forward_one each result token to update KV; the refinement = skip the
model FORWARD too (brain genuinely off). The KV-cache decides if it can stay off: TERMINAL op (result IS the
answer) -> brain fully off, ZERO model forwards (needs a terminality protocol signal -- to predict EOS the
model would otherwise have to forward the result; backlog nuance). INTERMEDIATE op (model reasons with the
result) -> must re-enter KV; efficient re-entry = ONE scalar embedding not N digit forwards. So big-number op:
O(result-len) forwards -> 0 (terminal) or 1 (scalar re-entry); classical compute ~free. SAME build + backward
choice as the scalar-embedding result. => scalar RE-ENTRY is the real "brain swap" build; terminal-skip needs
a terminality signal (not trivial). (B) Fold
combine/uncombine into the op registry -- tractable but token-group(tokens->tokens) vs value(strings->string)
I/O mismatch needs a 2nd NodeFn signature; backlog, homogenisation not capability. (C) Scratch tokens as
NAMED REGIONS `<|scratch> a` (in-vocab name) vs reserved ids: reserved ids are exactly why the pool CAPS AT
K=6 + can't be trained/persisted; named regions -> UNLIMITED slots (zero reserved cost; nodespike: word-names
route free), survive encode/decode, PERSISTABLE to external long-term store (the long-term-memory substrate);
vocab-offset slot only wins if a binding must be 1 token. Recommend migrating scratch to named regions;
backlog (binding-table + curricula refactor). See [[scratch-tokens-context-translation-layer]] (the K=6 cap).

SCALAR RE-ENTRY — BUILT + TESTED (the "brain swap", committed a40927e Scalar encoder + 02b3ffe A/B + 9b2e217
docs). This is the pluggable-predictor payoff, now data-backed. `SlotEncoding::Scalar` (scratch_slots.hpp,
rides the EXISTING content-derived-slot scaffold — set_scratch_bindings + per-window train_batch bindings, NO
engine change): a value re-enters the model as ONE bounded vector `[sign, base-10 exp/64, 4 leading mantissa
digits/9]`, parsed from the slot's digit fragments; FIXED encoding -> stop-gradient (no params, no tok_emb
read). Being float-shaped is the point: BOUNDED regardless of magnitude, so a power's full expansion (2^100 ->
E=30) and a scientific literal (1.23e45) map in like a small int — the answer to the user's exponents/powers
concern. Un-evaluated symbolic forms (2^100) reject to a zero row. A/B (single-number READOUT to avoid the
alignment wall — first attempt used P-vs-c COMPARISON and the DIGIT baseline itself only hit 0.63 because
cross-number alignment is the localization wall, NOT the channel; readout gives a strong baseline that
isolates the channel): read a queried digit of a re-entered P. d128, held-out:
  DIGIT (N tokens)  : leading 1.000, trailing 1.000
  SCALAR (1 vector) : leading 1.000, trailing 0.125 (~chance, 1/10)
=> the brain-swap HOLDS PERFECTLY for the reasoning the encoding carries (leading digits / order-of-magnitude
= comparisons, routing, sign/size) and is PROVABLY CHANCE where it's lossy (low digits it drops). Honest
answer to user's "not sure it holds at scale": yes for magnitude/leading-digit, no for low-digit precision —
which is WHY the exact value stays in the fragment binding for the terminal/exact path. Widening mantissa
trades vector width for exact depth; a LEARNED number->vector encoder (the live backward choice) is the open
next probe, worth it only if the fixed encoding's low-digit ceiling bites a target task.
COMPLEX-RESULT REPRESENTATION (docs, user raised): a result isn't always a plain decimal — powers/exponents/
rationals/symbols. Node result type = a CANONICAL NUMERIC GRAMMAR (int|decimal|scientific-e|rational-/|
symbolic-^,sqrt) emitted as ORDINARY tokens (digits . e - ^ / already in-vocab -> ZERO new tokenizer ids,
same ethos as the region frame); each node declares its return form. LEAN+mathlib (user raised): proof-
CARRYING/verified backend; too heavy for inline decode (JIT tactics ~seconds) -> best as (1) the FILTER-pillar
CORRECTNESS ORACLE (generate+verify the exact masked answers; mine scalar-solvable templates) and (2) a
proof-grade external-process escalation node. SymEngine stays the inline symbolic embed. See solver survey in
docs/DETERMINISTIC_MECHANISMS.md (Boost.Multiprecision / exprtk / SymEngine, all CPM-friendly).

PILLAR 2 (numeric BIND) — PROVEN (committed b8bcf39 spike + 05bfa94 docs; user chose this as next build after
the scalar re-entry A/B). bindspike.hpp lifts arithspike's delegation template from inline digits onto
ALGEBRAIC SYMBOLS: operands are BOUND SCRATCH SLOTS. Model sees `S0 + S1 =` where S0,S1 are single
SlotEncoding::Scalar tokens (magnitude only — exact digits appear NOWHERE in the token stream), emits routing
`[op add]`, and make_bind_compute_callback resolves operands by DEREFERENCING the slots from the binding table
(slot_value = decode fragments to decimal string) -> injects the exact sum. d128 held-out fresh pairs:
DELEGATION 1.000 (from step 300) vs FUZZY 0.000 (never). The 0.000 IS the proof (not a failure): the model
CANNOT produce the sum because the digits aren't in its stream — only path to exact answer is the node deref.
=> two 13-digit numbers = two tokens; model reasons over the SYMBOL, delegates arithmetic, value never touches
activations. Context/KV compression + algebra + exact, end-to-end. Mechanistic [nodebind] test (123+456->579,
max->456, no training) + hidden [.numericbind] A/B. bindspike reuses nodes.hpp registry + node_frame.hpp frame
+ scratch_slots.hpp unchanged; deref falls back to inline digits so it degrades to node_frame for plain A+B=.
ALL 4 pillars now proven (PROVIDE arithspike, BIND bindspike, FILTER mask A/B, COMPOSE chainspike). ONLY
remaining gap = END-TO-END PRODUCTION WIRING: gen_stage.cpp wires ONLY spell/scratch expand/combine, NOT the
compute(op) callback NOR set_scratch_bindings -> everything is spike/test-only, not live in `sub0llm gen`.
Productionisation = fold bound-slot deref into node_frame.hpp + wire compute callback + set_scratch_bindings
into gen_stage. Also open: learned number->vector encoder (past Scalar's low-digit ceiling); GSM8K integrated
test (calc-annotations <<48*2=96>> = ready-made op-frame -> node calls + masked results = all-pillars-together
on REAL math data; see ROADMAP.md §D + 135b862). Math corpora: synthetic-we-verify (incl Lean oracle) cleanest;
GSM8K best real start. See docs/DETERMINISTIC_MECHANISMS.md PROVEN table + §2 BIND.

PRODUCTION GEN WIRING — DONE (committed 871028e; user said "production"). The op frame is now LIVE in
`sub0llm gen`, not spike-only. (A) node_frame::make_compute_callback UNIFIED: gained an optional
std::function<std::string(int)> slot_deref -> the ONE production callback covers inline-digit ops (A+B=) AND
numeric BIND ([op add S0 S1] derefs bound slots); bindspike::make_bind_compute_callback is now a thin wrapper
over it (DRY, single impl). (B) ScratchTable::value(slot) = the slot's bound value string (for the deref).
(C) gen_stage.cpp wires compute = make_compute_callback(nodes::builtin(), scratch.value-deref) + compute_marker
= TOK_TURN_END into kv_decode_generate, alongside the existing expand/combine interceptor. KEY: INERT for any
non-`[op ...]` turn (a normal chat/role turn scans->returns {}), so NO new recipe flag needed and existing
models are byte-identical (decode guards do_compute=bool(compute)&&marker>=0); like the interceptor it forces
CPU decode. No .cu includes any touched header (zero CUDA exposure). Verified: node_frame/numeric_bind/embed
engine tests + full frontend suite (97329 assertions) green through the unified callback. STILL OPEN for a
live op DEMO: a training recipe that actually EMITS the op frame (the GSM8K/op-curriculum chapter) -- no
production model emits ops yet, so the wiring is ready+tested but unexercised live; content-derived slot
embeddings at gen (set_scratch_bindings, for Scalar magnitude reasoning) = separate follow-on gated on a
content-embedding training recipe. Gating refinement (op-only vs scratch models) also deferred to when the op
curriculum lands. NOTE: user running a medium fineweb_edu d448 train (workflow build, plain LM, no op/scratch
curriculum) -- independent, needs nothing; it modified data/tokenizer_calibration.txt (per-corpus ledger
upsert) which is theirs, not mine to revert.

GSM8K CHAPTER — STARTED (user: "continue to gsm8k"). The integrated real-data testbed for all pillars
together (docs/ROADMAP.md §D). GSM8K worked solutions carry inline calc-annotations `<<48/2=24>>` = a
ready-made op-frame marking exactly the arithmetic to delegate+mask. Committed increments:
- a2eb932: nodes.hpp gained exact big-int mul (schoolbook) + div (long division, EXACT-only -> "" on
  non-zero remainder / div-by-zero, so a non-integer annotation is DECLINED not faked); registered in
  builtin() (now 6 nodes: add/sub/mul/div/max/min). gsm8k.hpp (FRONTEND, engine+tokenizer free):
  parse_annotation (a<sym>b=result -> registry op, non-neg INTEGER operands +-*/, filters decimals/negatives/
  spaces/multi-op to literal text; decimal/rational nodes = documented follow-on), segment() (text|op spans),
  verify() (rerun node == stated result = FILTER guard, bad label dropped). Tested on canonical Natalia
  example.
- cc6ce5d: build_stream() -> tokenized (tokens,mask) example: prose+op-frames GRADED, each verified result
  MASKED (node fills it). GSM8K repeats the answer after `<<..=R>> R` -> the redundant R is stripped from
  prose (delegated not written). Tokenizer-AGNOSTIC (encode injected -> pure frontend, unit-testable;
  train_stage passes real sub0::encode). op_frame(op) = TOK_TURN_START op <name> <a> <b> TOK_TURN_END
  (operands IN-frame so node reads them unambiguously). TRAIN<->GEN LOOP CLOSED at unit level: the exact
  frame build_stream emits dispatches through the PRODUCTION node_frame callback ([op mul 6 7]->[42]).
- d2692a2: fixed nodes_tests registry-size asserts (4->6, 5->7) for the new nodes (boy-scout).
REMAINING (not done): (3) GSM8K data fetch -- a get_gsm8k.py (NOTE python-not-committed rule vs the existing
committed scripts/get_fineweb.py/get_tinystories.py precedent -- unresolved which applies); (4) train_stage
--op-mix recipe blending build_stream examples (like --spell-mix/--scratch-mix via blend.hpp) + an actual
training run; (5) end-to-end demo: train an op-emitting model -> `sub0llm gen` dispatches nodes live (the
gen wiring 871028e is READY+tested but UNEXERCISED until such a model exists). A self-contained alternative
to (3)+(4): a SYNTHETIC generate-and-verify GSM8K-format end-to-end engine test (templated multi-op prose ->
build_stream -> train tiny model -> confirm it emits verifiable ops through the production callback) proves
the whole chapter with NO download/train_stage change -- the cleanest next capstone.

GENERAL `math` EXPRESSION NODE — DONE (committed 60cb5e9 + docs 943f4d4; user's design call: "op math" over
per-op frames). Instead of the MODEL decomposing an expression into ordered binary op-frames + getting
precedence right, ONE general node evaluates the whole expression: `[op math 6+8*10-5]` -> 81 in one frame.
Rationale (thesis-pure): parsing+precedence is ITSELF deterministic -> belongs in the node, not the fuzzy
model; the model just COPIES the expression it's reasoning about. nodes.hpp: lex_str (numbers + single-byte
operators + parens; operators are single ANSI bytes -> trivial lexer, user's point) + eval_expr = EXACT SIGNED
big-int recursive-descent (^ right-assoc > * / > + -, parens, unary +/-) on the add/sub/mul/div primitives,
NO double; declines ("") on non-exact div / div0 / malformed / runaway power -- never a wrong answer.
Registered "math" (builtin now 7). node_frame make_compute_callback now passes the RAW EXPRESSION THROUGH:
lexes the op region with OPERATORS PRESERVED (was scan_region which dropped them); operand resolution =
in-frame args first, then bound-slot deref (BIND), then preceding inline expr -- all prior paths unchanged.
Per-op nodes STAY as math's primitives + for the BIND path. FOLLOW-ON (user raised, backlog, A/B first):
collapse a math RESULT to a scratch token (combine/uncombine to reach inner digits on demand) = numeric BIND
applied to RESULTS not just inputs -> small KV for long chained calc. OPEN CONSIDERATION: GSM8K build_stream
currently emits per-op `[op div 48 2]`; could switch to `[op math 48/2]` (copy EXPR verbatim = simplest
routing, one op-name, handles compound exprs) -- aligns GSM8K with the math node; not yet done (offered).

CONSOLIDATED ON `math` — per-op arithmetic frames REMOVED (committed a23ca1b; user: "Remove the per op frames
... the new approach supersedes with all expectations"). builtin() = { math, max, min } -- add/sub/mul/div NO
LONGER registered dispatch nodes (kept as nodes.hpp INTERNAL primitives eval_expr runs on); max/min stay
(math can't express selection). node_frame callback: region_expr() renders the op region as an EXPRESSION
STRING with operators preserved AND bound scratch slots SUBSTITUTED with their values -> numeric BIND UNIFIES
into math (`[op math S0+S1]` and `S0 + S1 = [op math]` both eval by substituting slots); the old slot-deref
operand path + nums-only preceding fallback are GONE (one expression path). gsm8k emits `[op math <EXPR>]`
(copy annotation expr verbatim, verify via nodes::eval which IS the integer filter) not per-op `[op div 48 2]`
-- simpler + handles compound exprs free. ALL PROOFS PRESERVED via math: node_frame end-to-end 1.000;
numeric BIND A/B 1.000 vs 0.000 (re-ran, still holds); gsm8k converter; full frontend 97392 green. Self-
contained spike proofs (arithspike/nodespike/chainspike) LEFT FROZEN (own encodings, don't touch
builtin/node_frame -- untouched; offered to retire if wanted). OPEN: user wants a formal MATH EVAL harness for
the trained model (math is well-formed -> rich AUTO-verifiable metrics): GSM8K `#### answer` exact-match
(field standard) + decomposed metrics unique to delegation (routing/expression correctness, delegation
coverage, arithmetic-error=0-by-construction-when-delegated so errors localize to ROUTING not arithmetic) +
magnitude/complexity generalization probes (cheap generate-and-verify). To build alongside the synthetic
end-to-end capstone (train tiny model to emit [op math] in prose -> gen dispatches -> score).

GSM8K CAPSTONE — END-TO-END DELEGATION IN PROSE (committed ccc9a55). Synthetic GSM8K-format word problems ->
gsm8k::build_stream -> train tiny d128 model -> decode through the PRODUCTION node_frame callback -> score.
KEY FINDING (from data): making the model COPY the expression into the frame (`[op math 12+34]`) hits the
LOCALIZATION/COPY WALL -- delegation ~1.0 but exact noisy+low (0.2-0.4), unstable, LR-insensitive. FIX: the
expression is ALREADY in the prose (`... 12+34 = <<...>>`), so emit a BARE `[op math]` (route only) and the
node reads the posed expression from PRECEDING CONTEXT (new node_frame::preceding_expr = scan back over the
trailing arithmetic expr -- digits/operators/parens/bound-slots/spaces/'=' -- stopping at prose/letters).
Result: BEST step300 in-dist delegated=1.000 exact=0.935, OOD 3-digit exact=0.675. MAGNITUDE GENERALIZATION
CONFIRMED: bare frame => NO copy => node exact at any magnitude => generalizes to unseen 3-digit (arithmetic
not learned). OVERTRAINING: more steps collapse it (0.935->0.14) -- early-stop is the readout; a recipe-tuning
finding, NOT a mechanism failure. EVAL HARNESS (the "math well-formed" advantage): delegation-rate +
exact-match (a miss = ROUTING error; arithmetic error 0-by-construction when delegated) + magnitude-gen probe.
gsm8k::op_frame() now BARE; build_stream uses it. node_frame/numeric_bind unaffected (their whole prompt is
the expression, preceding_expr handles it). GENERAL LESSON: prefer CONTEXTUAL delegation (bare marker + node
reads context) over SELF-CONTAINED (copy expr into frame) whenever the expression is already present -- avoids
the copy wall. OPEN/next: fix overtraining (LR schedule/regularization/more data); multi-step chaining in
prose (2+ ops, intermediate result reused -- the chaining-copy is the next wall, candidate for scratch-collapse
of results); real GSM8K data fetch; --op-mix train_stage recipe; native sub0llm eval stage.

OPS vs TOOLS taxonomy (user design clarification, committed doc c72738e). Op-calling = the SAFE, engine-
EMBEDDED subset of tool-calling; SAME model-facing mechanism (emit region-frame call -> handler runs ->
result injected loss-masked), so routing-to-ops IS the transferable tool-use skill -> train it MASSIVELY +
SAFELY offline via generate-and-verify (no bash/net), generalizes to guarded tool-use. 3 tiers by purity x
safety: PURE op (math/unit/gcd/sort/string -> BAKE result, always safe) | IMPURE-BUT-SAFE op (date/now/
elapsed/RNG/READ-ONLY db -> can't bake, train ROUTING only + mask placeholder + live-resolve; masking pillar
handles it) | TOOL effectful (bash/write/http -> never baked, GUARDED permission/sandbox). Registry ALREADY
the extensible op TABLE: new op = one register_node call, ZERO tokenizer budget, NodeFn(operands->result)
handles NULLARY (`[op date]`) + variadic. Backlog when a consumer exists ([[only-add-arguments-we-need]]):
per-op purity/safety TAG for the baker + a TOOL tier w/ permission layer. Seed catalog: date/now/elapsed,
unit/base convert, sort/gcd, string ops, read-only lookups. See docs DETERMINISTIC_MECHANISMS.md §1c.

COLLAPSE mechanism BUILT (committed 5642edb) -- the result-side of BIND + the fix for the multi-step chaining
COPY WALL. make_collapse_callback (bindspike.hpp): when an op resolves, bind the exact result to the next
scratch slot and inject THAT ONE TOKEN (not the digits); a chained op then refers to the previous result as a
SYMBOL (`S0 - C`) = one-token copy not multi-digit copy; value stays exact in the binding for the next op to
deref (region_expr slot substitution). Mechanistic [nodecollapse] (no training): 12+34=[op math]->collapse 46
to S0; S0-20=[op math]->deref S0->46, eval 46-20, collapse 26 to S1 -- chained through slots. Falls back to
digit injection when pool full. SPIKE uses std::vector<std::string> value table; PRODUCTION form binds into
gen's stateful ScratchTable + set_scratch_bindings for Scalar embedding. NEXT (multi-step chapter): the
TRAINING A/B -- does a tiny model chain 2+ steps better with COLLAPSED SLOTS than with digit results (expect
collapse > digit-copy, mirroring single-step contextual>copy). Then productionize collapse into gen_stage.

MULTI-STEP CHAINING A/B — COLLAPSE DECISIVELY WINS (committed 22d4061). The chaining chapter's decisive
experiment. Problem R1=A+B then R2=7+R1, intermediate REFERENCED in step 2 after a `. 7+` separator (forces
the model to reproduce the reference = the chaining-copy). Tiny d128, two arms:
  DIGIT    (reproduce multi-digit R1 into `7+R1`)        : best final-answer exact = 0.095  (FAILS)
  COLLAPSE (emit single slot token S0 in `7+S0`, value in binding) : 1.000, STABLE every round, NO overtraining
=> COLLAPSE 1.000 vs DIGIT 0.095. Collapsing an op result to a scratch-slot SYMBOL is what makes multi-step
chaining work: the model references a stable symbol (trivial) instead of copying a multi-digit intermediate
(the localization/copy wall a tiny model can't clear). Same lesson as single-step (contextual>copy) now for
the chained case. Notably COLLAPSE is STABLE (no oscillation/overtrain) because it isn't fighting a hard copy.
Uses make_collapse_callback. GENERAL PRINCIPLE now proven twice: avoid making the model COPY values -- let it
route over SYMBOLS/CONTEXT and let the node/binding hold the values. STATUS: multi-step + collapse chapter
DONE (mechanism + A/B). REMAINING productionization: wire collapse into gen_stage's stateful ScratchTable +
set_scratch_bindings (Scalar embedding of collapsed slots); real GSM8K data fetch; --op-mix train_stage recipe
blending build_stream; native sub0llm eval stage (exact-match + delegation + routing decomposition). Also the
ops-vs-tools catalog (date/now/elapsed/unit/read-only-lookup) + purity/safety tag when a consumer exists.

PRODUCTIONIZED + END-TO-END VALIDATED (user: "tackle it all to production"; committed 6a59404..66dec69).
All 6 production pieces shipped + a real CLI-trained model validates the whole pipeline:
1. op_curriculum.hpp (6a59404): the --op-mix blend source. UNIFORM-SLOT (every op result collapses to a
   scratch slot; single-step + 40% multi-step chains). Uses tok::encode (digits/operators stay byte tokens --
   Unigram Viterbi runs only over word-byte runs). BONUS FIX: node_frame preceding_expr now skips TOK_JOIN +
   glue-bracket markers (the Unigram tokenizer sprinkles JOIN glue between adjacent digit/op bytes) and
   region_expr restores glue-brackets -> the node reads REAL tokenized prose, not just byte spikes.
2. --op-mix recipe (840aaa5): cli_stages --op-mix flag -> sub0_train_stage -> RunConfig op_mix (X-macro,
   persisted in config.json, re-enforced on resume, read by gen) -> train_stage blends op_curriculum as a
   masked BlendSource (mirrors --scratch-mix).
3. collapse in gen (e86b728): gen_stage, gated on cfg.op_mix -- a ScratchTable-backed COLLAPSE callback binds
   each op result to the next slot + injects that token (chained refs resolve via region_expr substitution);
   expand_op_slots() rewrites collapsed slots to their value digits for display.
4. op catalog + purity tag (1cebb4c): builtin() = {math,max,min,mod,gcd,abs,date,now}; Registry gains a per-op
   `pure` tag + is_pure() (ops-vs-tools taxonomy: pure=bakeable, impure date/now=live-only). Portable clock.
5. native eval (5550c4b): `sub0llm eval <model>` -- delegation-rate + exact-match + routing-accuracy (arith
   exact by construction -> a miss is a ROUTING error). sub0_eval_stage in gen_stage; gen_eval problems.
6. plateau fix (66dec69): explicit --steps N now runs to completion (plateau-stop gated on steps==0) -- was
   cutting --op-mix runs short because the base val plateaus before op-delegation is learned.
VALIDATION (real CLI, d128 spike_tied, --steps 2500 --op-mix 0.6): config.json op_mix=0.6; `sub0llm eval`
delegation 1.000 / exact 1.000 / routing 1.000; `sub0llm gen` delegates LIVE ("total 27+58="->[op math]->node
->85; 9*8->72; 100-37->63) -- exact arithmetic from the NODE, not the model. THE THESIS IS PRODUCTION.
NOTES: eval needs the model.bin path (not the dir -- minor CLI inconsistency, gen resolves dir); a toy
validation model sits in models/ (sub0llm_english_d128...20260714-120346, prunable). GPU d448 short runs are
LR-warmup-throttled (warmup 44661 steps) -- use spike_tied CPU or many steps for op validation. Real GSM8K
data fetch + a GPU-scale op model remain as future scale-up (mechanism + pipeline proven).

GSM8K WORKFLOW PRODUCTIONIZED (committed 5b0f6f1 scripts + 6e548cc file-delegation). Real GSM8K now trains a
DELEGATING model end-to-end:
- scripts/get_gsm8k.py: fetch GSM8K (pip install datasets) -> data/gsm8k.txt, <|endoftext|>-separated, calc
  annotations `<<expr=result>>` KEPT (committable data-fetch script like get_fineweb/get_tinystories).
- scripts/workflow.ps1: new -OpMix (->--op-mix) + -Gsm8k <path> (->--gsm8k) params.
- `--gsm8k <file>` (with --op-mix>0): switches the op-curriculum SOURCE from synthetic to REAL GSM8K --
  op_curriculum::gsm8k_file_dataset reads the file, runs gsm8k::build_stream(collapse=true) per problem,
  converting each `<<expr=result>>` to a graded [op math] + result COLLAPSED to a masked scratch slot. So
  GSM8K's OWN arithmetic delegates (never learned fuzzily). build_stream gained a `collapse` bool (default
  off=digit for existing tests/capstone; on=slot, uniform with op_curriculum+gen). Chaining stays via the
  prose's LITERAL numbers (node reads the posed expr via preceding_expr), so ONE slot id suffices (the slot
  is only the masked injected-result placeholder, the model never emits it). gen UNCHANGED (collapse gated on
  op_mix>0; GSM8K is collapse-based -> consistent).
FULL GSM8K MATH WORKFLOW:
  python scripts/get_gsm8k.py
  scripts/workflow.ps1 -BuildDir gsm8k -Corpus data/gsm8k.txt -OpMix 0.5 -Gsm8k data/gsm8k.txt -Steps 4000
  out/build/gsm8k/sub0llm.exe eval <model>/model.bin
(base corpus=GSM8K reasoning; -Gsm8k=delegated GSM8K arithmetic; proven arch stack now default-ON).
CAVEAT: GSM8K held-out CHAINING (reproducing an intermediate as a literal in the next expression) is still
the copy wall for a tiny model -- collapse solves it only for SYNTHETIC data (explicit slot refs); GSM8K's
natural-language literals aren't auto-rewritten to slots (would need number-tracking). So each STEP delegates
(no fuzzy arith), but multi-step held-out accuracy depends on model scale / small numbers. Backlog: rewrite
GSM8K intermediate-literals to slot refs for full-chain collapse. See [[arch-features-off-by-default-in-production]].

GSM8K WORKFLOW RUN + VALIDATED (2026-07-14; committed 0828c00 fetch-fix + 6ff238d workflow-fix). Ran the
whole thing: python scripts/get_gsm8k.py (fixed repo id "gsm8k"->"openai/gsm8k") -> data/gsm8k.txt (4MB,
7473 problems). workflow.ps1 GPU: configure (autosize d128 L8 seq256 vocab2257) + build (CUDA, arch stack
DEFAULT-ON: gated_ffn/tied/qk_norm=on, optimizer=muon -- config.json shows the NAMED values working) + train
--op-mix 0.5 --gsm8k (blend "GSM8K op-delegation 50% 7104 problems 1758707 tokens" -- the file conversion
WORKS). VALIDATED: gen on GSM8K-format prompts DELEGATES: "Natalia sold 48/2 = "->[op math]->node->24;
"60/3 = "->20 (node's exact answer, not the model's). So --gsm8k trains a real-GSM8K delegating model.
FINDINGS FROM THE RUN (baked into workflow.ps1 -Batch + example): (1) GPU AUTO-BATCH OVER-SHOOTS VRAM on
small autosized models (d128 -> batch 2144 -> 0xC0000005 crash); pin --batch 256 works fine (461K-1.9M
tok/s). Root-cause = gpu_batch_estimate for small models, BACKLOG (see [[gpu-batch-undertuned-vram-headroom]]
[[memplan-vram-prediction-gap-at-scale]]). (2) `sub0llm eval` scores the SYNTHETIC "total A+B=" format ->
delegation 0.005 on a GSM8K-format-trained model (format mismatch, NOT a failure); eval-on-GSM8K-format is a
gap. (3) BASE-CONFLICT: -Corpus data/gsm8k.txt (plain, <<..>> as text) FIGHTS the delegated --gsm8k (raw
arithmetic-as-text vs [op math]); use -OpMix HIGH (0.8) so delegated dominates, or a NON-math base. (4)
sub0llm eval needs model.bin path not dir; workflow builds stage tools not the sub0llm driver (need
--target sub0llm for eval). NOTE: the run only trained ~165 steps (killed early -- 3000 steps too slow due
to frequent eval on the small corpus) yet still delegated. Model at models/gsm8k_gsm8k (prunable).
