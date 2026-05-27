# Chapter 21 — Research Concept: Specialised Mathematical Neurons

> **Status**: Research / concept stage. No implementation yet.

---

## Motivation

Language models learn arithmetic through statistical pattern matching: "2 + 2"
→ "4" because that co-occurrence is overwhelmingly common in training text.
This forces the model to memorise multiplication tables, carry rules, and
floating-point identities as weight patterns — a wasteful and lossy encoding.

Classical computation handles arithmetic exactly, instantly, and with zero
approximation error. The research question is: **can we give the model a direct
path to exact arithmetic that bypasses statistical memorisation entirely, without
routing to an external API or a separate model?**

---

## Prior Art Survey

### Neural Arithmetic Logic Units (NALU / iNALU, 2018–2020)

Graves et al. (2018) introduced the Neural Accumulator (NAC) and Neural
Arithmetic Logic Unit (NALU): small, constrained MLP variants with weight
matrices forced into `{−1, 0, +1}` during training, capable of learning to
add and multiply. iNALU (2020) improves stability.

**Limitations**: Approximate (not exact). No transformer integration. No learned
router. Performance degrades on numbers outside the training distribution.

### Toolformer (Schick et al., NeurIPS 2023 / arXiv:2302.04761)

The model is fine-tuned to emit special "call" and "response" tokens that
bracket an API call to an external calculator. Arithmetic is exact because an
external Python evaluator handles it.

**Limitations**: Latency from API round-trip. Not in-stream (control leaves the
model). Gradient does not flow through the calculator. Special tokens are
manually placed, not learned.

### Clock / Helical Arithmetic (Charton, arXiv:2402.02619; Hägele et al., 2502.00873)

Probing studies reveal that transformers internally represent numbers via
**trigonometric helix patterns** in the residual stream — one sinusoidal
component per digit position. Arithmetic is processed in three overlapping
stages:

```
Layer phase          Operation
──────────────────────────────────────────────────
Early layers         Encode numeric tokens into helix representation
Attention ~9–14      Transport operand representations to output position
Late MLPs ~14–18     Write arithmetic result to residual stream
```

(Layer indices for a 24-layer GPT-2-scale model. Scales to ≈ 0.6–0.75 × N
for deeper models.)

**Implication**: Arithmetic is concentrated in the **last third** of the model.
A single math layer placed at depth ≈ 0.7 × N should intercept this phase.

### Abacus Embeddings (McLeish et al., arXiv:2405.17399, 2024)

Per-digit positional embeddings assign each digit of a multi-digit number its
own embedding slot, making the numeric structure explicit. Achieves >99%
accuracy on 100-digit addition.

**Limitation**: Requires multi-token numeric representation; does not compact a
number into one token; does not provide exact execution.

### Fourier Number Embeddings — FoNE (arXiv:2502.09741, Feb 2025)

Encodes a number as a **single token** whose embedding is a concatenation of
`cos(2π f_i × n)` / `sin(2π f_i × n)` terms, one pair per frequency. A single
linear layer then maps the pair of frequency-domain embeddings to the result
token's embedding, implementing addition exactly.

Results: 99%+ accuracy with **64× less training data** than standard
tokenisation; 100% on 6-digit addition. Closest prior work to this chapter's
numeric token range idea.

**Limitation**: No learned router — all tokens go through the numeric path.
FoNE re-defines the entire embedding, so it cannot sit inside a general LM.

### In-stream Gated Calculator — IGC (arXiv:2501.00684, Jan 2025)

Embeds a **GPU-emulated exact calculator** as a layer inside a Llama-3 model.
A gate learned during fine-tuning decides whether to pass the hidden state
through the calculator or the standard FFN. Achieves 98–99% on BigBench
Arithmetic.

**Limitation**: Non-differentiable calculator — gradients do not flow through
the compute path. Gate is trained post-hoc (not end-to-end from scratch).
Calculator is GPU kernel code, not a tensor op, making it opaque to autograd.

### PiERN (arXiv:2509.18169, Sep 2025)

Token-level router between a "reasoning LM" and a "compute expert".
The router is trained with reinforcement, rewarded when the compute expert
returns the correct answer.

**Limitation**: Compute expert is still a neural approximation — not exact.
Router is post-hoc RL, not a differentiable gate trained end-to-end.

### LASER (Sharma et al., arXiv:2312.13558, ICLR 2024)

Applies SVD rank reduction to late MLP layers of LLaMA/GPT-J; finds that
**discarding small singular values in late-layer MLP weights improves
multi-step reasoning by 20–30 percentage points**. Confirms that late MLPs
disproportionately encode arithmetic/reasoning computations.

**Implication**: Late MLPs are critical for arithmetic. A targeted replacement
of the late MLP (at depth ≈ 0.7N) with an exact math node should surgically
address the weakest part of the architecture.

---

## Gap Analysis

| Work | Numeric token detection | In-stream exact arithmetic | End-to-end differentiable router |
|------|------------------------|---------------------------|----------------------------------|
| NALU / iNALU | ✗ (approximate) | ✗ | ✓ |
| Toolformer | ✓ (via special tokens) | ✓ (external) | ✗ (no gradient through tool) |
| FoNE | ✓ (Fourier embedding) | ✓ (linear layer = exact add) | ✗ (no router, whole model) |
| IGC | ✓ | ✓ | ✗ (post-hoc gate) |
| PiERN | ✓ (RL router) | ✗ (neural approximation) | ✗ (RL, not gradient) |
| Abacus Embeddings | ✓ (per-digit PE) | ✗ | ✗ |
| **Ch21 proposal** | **✓** | **✓** | **✓** |

No prior work simultaneously achieves all three properties from within a
standard transformer block trained end-to-end with gradient descent.

---

## Core Concept

Replace the feedforward path at **one designated transformer layer** with a
**specialised execution node** — a deterministic function wired into the residual
stream that executes exact arithmetic rather than a learned matrix product.

```
standard path:
  token_ids → embedding → transformer → softmax → output token

proposed path (single math layer at depth L_math):
  layers 0 … L_math-1 : standard attention + FFN
                                 │
                      ┌─── is numeric position? ───┐
                      │ yes                         │ no
                      ▼                             ▼
            Math Execution Node              standard SwiGLU FFN
            (exact IEEE-754 arithmetic)           │
                      │                            │
                      └───────────────┬────────────┘
                                      ▼
              residual stream updated (math result or FFN output)
                                      │
  layers L_math+1 … N-1 : standard attention + FFN
                                      ▼
                               output token logits
```

---

## Single Tunable Math Layer

Mechanistic interpretability (Charton, LASER, helix probing) consistently shows
arithmetic computation is concentrated in the **last 25–40% of layers**.
The math layer is therefore treated as a **hyperparameter** with a principled
starting point:

```
L_math_default = round(0.7 × N_layers)   # last-third heuristic
```

### Ablation schedule

| L_math | Fraction of model | Rationale |
|--------|------------------|-----------|
| round(0.25N) | early | Can it intercept before helix transport? |
| round(0.50N) | mid | Transport phase |
| round(0.70N) | late (default) | Late-MLP compute phase |
| N−1 | last layer | Maximum context before logit projection |

The model is frozen except for:
1. The numeric token embeddings (new range)
2. The router at depth L_math
3. The fallback FFN at depth L_math (replaced by the gated router)

All other layers retain their pre-trained weights, so fine-tuning is fast.

### Router architecture

A lightweight `Linear(D, n_types + 1) + softmax` at position L_math produces
per-position routing weights:

```
route ∈ ℝ^{T × (n_types+1)}

n_types = 6 : add, sub, mul, div, compare, fallback-FFN
```

For training, the router uses a **straight-through estimator** (STE): the
argmax is used in the forward pass, but gradients flow through the softmax
probabilities. This makes the routing end-to-end differentiable while producing
hard routing decisions at inference time.

---

## Tokenizer: Numeric Token Range

The BPE tokenizer reserves a contiguous range of token IDs for exact numeric
representations. These IDs are deterministic mappings, not learned subword merges:

```
Reserved range: IDs [V_lang, V_lang + V_num)

Encoding examples:
  "42"     → numeric token for integer 42
  "3.14"   → numeric token for float 3.14 (nearest float16)
  "-7"     → numeric token for integer −7
  "1e-6"   → numeric token for scientific notation 1×10⁻⁶

V_num encoding:
  integers  : signed int16 range [−32768, 32767]  → 65536 IDs
  floats    : IEEE 754 float16 quantised           → 65536 IDs (separate range)
  fractions : rational p/q pairs (optional)
```

The tokenizer heuristic: numeric strings where the entire token is a number
go into the reserved range. Mixed tokens ("42nd", "IPv4", "#42") use the
standard BPE vocabulary.

**Connection to FoNE**: Rather than Fourier-encoding the number into the
embedding dimensions (FoNE), this design maps the numeric value into the
embedding via a dedicated learned embedding matrix for the numeric range —
simpler to implement and compatible with the existing `EmbeddingTable` op.

---

## Specialised Execution Nodes

A math execution node receives one or more token hidden states (each labelled
with its numeric value), applies an exact function, and returns the result
hidden state (labelled with the result value).

```cpp
// Conceptual interface
class MathNode {
public:
    virtual ~MathNode() = default;
    // inputs: hidden states of operand positions (carry numeric value in payload dims)
    // returns: hidden state of result position (numeric payload updated)
    virtual Variable forward(std::span<const Variable> operands) const = 0;
};

class AddNode    : public MathNode { ... };   // a + b  (exact integer / float)
class SubNode    : public MathNode { ... };   // a − b
class MulNode    : public MathNode { ... };   // a × b
class DivNode    : public MathNode { ... };   // a / b  (detect /0)
class CompareNode: public MathNode { ... };   // a < b, a == b → boolean embedding
```

### Numeric value propagation: side-channel register

A separate numeric register tensor `R ∈ ℝ^T` is maintained alongside the
residual stream `x ∈ ℝ^{T×D}`:

```
x  : residual stream (T, D) — standard learned features
R  : numeric register (T,)  — raw float values for numeric token positions
```

Math nodes read from and write to R. Attention and FFN treat R as opaque;
they only operate on x. This avoids polluting the semantic embedding dimensions.

The router at L_math reads both x and R to make its routing decision.

---

## Performance Analysis

### Overhead at a single layer

| Operation | Cost | vs attention |
|-----------|------|-------------|
| Attention (QKV + softmax + proj) | O(T²D + TD²) | 1× |
| Standard SwiGLU FFN | O(T × D × d_ff) ≈ O(8TD²/3) | ~2× attention flops |
| Router (Linear(D, n_types+1)) | O(T × D × n_types) | < 0.1% |
| Math node execution | O(1) per op | negligible |
| Net math layer (well-routed) | O(T × D × n_types) | < 0.1% |

When the router is accurate, the math layer is **faster** than a standard FFN
because math nodes have O(1) per-operation cost vs O(D × d_ff) for matrix
products.

### Memory

The numeric register R is T float32 values. At T=2048 that is 8 KB — negligible
vs the KV cache and attention tensors.

### Inference latency (estimated, D=512, T=128, n_types=6)

```
Standard FFN at L_math:  T × D × 8D/3 = 128 × 512 × 1365 = 89 M flops
Router gate:              T × D × 7    = 128 × 512 × 7    = 458 K flops  (0.5%)
Math node execution:      T × 1        = 128 flops         (< 0.001%)

Net saving when math nodes fire: ~99.5% of L_math FFN cost
(benefit only applies to numeric-heavy inputs; text-heavy inputs pay only the 0.5% router overhead)
```

---

## Learning/Test Dataset

A programmatically generated benchmark with five tiers totalling ≈ 225,000
examples. All examples are formatted as token sequences matching the model's
vocabulary; ground-truth answers are verified by Python before insertion.

### Tier 1 — Router discrimination (50,000 examples)

Purpose: teach the router to distinguish numeric token positions from text.

```
Format:  "The answer is <NUM>."      → label: route to math node
         "The meeting is <WORD>."    → label: route to FFN
         "<NUM> copies sold"         → mixed: NUM position → math, rest → FFN

Split: 60% numeric-heavy, 40% text-heavy
Range: integers −9999 to 9999, floats with 1–3 decimal places
```

### Tier 2 — Single-operation unit tests (100,000 examples)

Purpose: verify math node correctness for each operation type.

```
Op       Examples  Range              Format
───────────────────────────────────────────────────────
add      25,000    ints 0–9999        "A + B = ?"
sub      25,000    ints 0–9999        "A - B = ?"  (no negative results initially)
mul      25,000    ints 0–99          "A * B = ?"
div      15,000    divisors 1–99      "A / B = ?"  (exact integer, no remainder)
compare  10,000    ints 0–9999        "Is A > B?", "Is A == B?"
```

All answers are single numeric tokens. Accuracy metric: exact match (not
approximate — the whole point is exact arithmetic).

### Tier 3 — Arithmetic word problems (50,000 examples)

Purpose: test whether the model routes correctly when numbers are embedded
in natural language context.

```
Template examples:
  "Alice has {A} apples. Bob gives her {B} more. How many does she have?"
  "A train travels {A} km/h for {B} hours. What is the distance?"
  "Split {A} items equally among {B} groups. How many per group?"

Generated from 200 templates × 250 numeric instantiations each.
Difficulty distribution: 60% one-step, 30% two-step, 10% three-step.
```

### Tier 4 — Multi-step chained calculations (20,000 examples)

Purpose: test whether the model can correctly route and execute multiple
math nodes in sequence within a single forward pass context.

```
Format: "Compute: ({A} + {B}) × {C} = ?"   (two operations, one context)
Range:  ints 0–999, result fits in int16
Steps:  2–4 operations per example
```

### Tier 5 — Adversarial / boundary cases (5,000 examples)

Purpose: test robustness of the numeric tokenizer and router.

```
Case type                   Examples  Purpose
────────────────────────────────────────────────────────────────────────────
Division by zero            500       Math node must return NaN token + signal
Overflow (result > int16)   500       Test overflow detection
Near-integer floats          500       "3.0000001 ≈ 3?" — router boundary
Ordinal numbers              500       "42nd" should NOT route to math node
Embedded dates               500       "2025-01-01" should NOT route to math
IP addresses                 500       "192.168.1.1" should NOT route to math
Scientific notation           500       "1e3" should route as 1000
Negative results              500       Subtraction yielding negatives
Repeated equal operands      500       "5 × 5", "7 + 7" — basic identity
```

### Metrics

| Tier | Primary metric | Secondary metric |
|------|---------------|-----------------|
| 1 | Router precision / recall (numeric vs text) | F1 |
| 2 | Exact-match accuracy per op | Confidence calibration |
| 3 | End-to-end accuracy | Partial-credit (correct op, wrong number) |
| 4 | Exact-match accuracy | Step-level accuracy |
| 5 | False-positive rate (non-numeric routed to math) | NaN/overflow detection rate |

### Baseline comparisons

For each tier, compare:
1. **Math node (Ch21)** — proposed system
2. **Standard ModernGPT (Ch10)** fine-tuned on same data — statistical baseline
3. **GPT + Toolformer-style external calculator** — external API baseline

---

## Extensions Beyond Arithmetic

The same architecture generalises to any domain where classical computation
is exact and faster than learned approximation:

| Domain | Specialised Node | Classical operation |
|--------|-----------------|---------------------|
| Arithmetic | `AddNode`, `MulNode` | IEEE 754 float ops |
| Logic | `AndNode`, `OrNode` | Boolean algebra |
| String ops | `ConcatNode`, `LenNode` | std::string operations |
| Date/time | `DateDiffNode` | Calendar arithmetic |
| Unit conversion | `ConvertNode` | SI prefix tables |
| Regular expressions | `RegexNode` | NFA/DFA matching |
| Lookup tables | `LookupNode` | Read-only key→value map |

The unifying principle: **anything where the answer is deterministic and
well-defined given the inputs** is a candidate for a specialised node.
Language models should learn to *use* these nodes, not to *approximate* their outputs.

---

## Training Implications

### What is trained vs frozen

```
Frozen:         All pre-trained weights except the math layer and numeric embeddings
Learned:        Numeric embedding matrix (V_num × D)
                Router at depth L_math  (Linear(D, n_types+1))
                Fallback FFN at L_math  (only fine-tuned, not replaced)
```

### Gradient flow

- Arithmetic operations (add, mul) are differentiable — gradients flow back
  through math nodes to the router and embeddings.
- Comparison / Boolean outputs are discrete — straight-through estimator or
  Gumbel-softmax for gradient.

### Curriculum

1. Pre-train a `ModernGPT` (Ch10/Ch11) on standard language data.
2. Freeze all weights; insert numeric token range into tokenizer vocabulary;
   initialise numeric embeddings to be close to nearby language tokens.
3. Fine-tune on Tier 1 (router discrimination) with STE-gated router.
4. Fine-tune on Tier 2 (unit operations) to train math node dispatch.
5. Fine-tune on Tier 3–4 (word problems, multi-step) for integration.
6. Evaluate on Tier 5 (adversarial) to measure robustness.

This mirrors ToolFormer fine-tuning but with the tool executed inside the
residual stream rather than via an external API call.

---

## Relationship to Existing Work

| Work | Approach | Vs Ch21 |
|------|----------|---------|
| ToolFormer (2023) | External calculator API | API latency; gradient does not flow through tool |
| NALU / iNALU (2018/2020) | Constrained MLP approximates add/mul | Approximate, not exact; no routing, no transformer integration |
| FoNE (2025) | Fourier embedding; linear layer = exact add | Single token, but whole-model change; no selective router |
| IGC (2025) | GPU calculator gated into Llama | Nearest prior work: exact, in-stream; but non-differentiable gate |
| PiERN (2025) | RL router → neural compute expert | Router nearest; compute is still approximate |
| Helix / Clock (2024) | Probing shows late-MLP arithmetic | Explains WHERE; does not replace with exact computation |
| LASER (ICLR 2024) | SVD on late MLPs improves reasoning | Confirms late-MLP importance; does not replace with exact ops |
| **Ch21 proposal** | Exact arithmetic at single tunable layer, end-to-end differentiable router | All three properties together: detection + exact + differentiable |

---

## Open Research Questions

1. **Routing accuracy**: How precisely does the router learn to identify numeric
   token positions? What error rate is acceptable for downstream task accuracy?

2. **Layer depth sensitivity**: How sharply does performance peak around L_math ≈ 0.7N?
   Is the optimal depth task-dependent (arithmetic vs logic vs lookup)?

3. **Value extraction from context**: Multi-digit numbers like "12,345" are
   often tokenised as multiple BPE tokens. How does the model assemble the full
   value across tokens before routing?

4. **Generalisation beyond training range**: Does a model trained on int16
   arithmetic generalise to larger numbers via composition?

5. **Numerical stability**: What should the model output when arithmetic
   produces NaN (0/0) or overflow? A special error token? A probability
   distribution over "undefined"?

6. **Interaction with attention**: Operands at positions t₁ and t₂ must both
   be attended to by the output position t₃ before the math node fires.
   Does standard causal attention handle this, or is a dedicated operand-
   gathering attention head needed?

---

## Proposed Implementation Plan (Future Chapters)

```
Ch21a — Numeric Tokenizer Extension
  • Extend BPETokenizer with reserved numeric range [V_lang, V_lang + V_num)
  • Implement float16 quantisation mapping (int16 + float16 → 131072 IDs)
  • Tokeniser routing heuristic: numeric string detection
  • Generate and save Tier 1 router discrimination dataset

Ch21b — Math Execution Nodes
  • MathNode interface + AddNode, SubNode, MulNode, DivNode, CompareNode
  • Side-channel numeric register (T,) alongside residual stream
  • Exact IEEE-754 arithmetic with overflow / NaN detection
  • Unit tests: Tier 2 exact-match accuracy = 100%

Ch21c — Router Integration
  • NumericRouter: Linear(D, n_types+1) + softmax + straight-through estimator
  • Integration into ModernTransformerBlock at configurable depth L_math
  • L_math defaults to round(0.7 × N_layers); ablation at 0.25/0.5/0.75/N-1
  • L_math exposed as constructor parameter in ModernGPT

Ch21d — Training and Evaluation
  • Fine-tuning script for Tiers 1–4
  • Full evaluation on Tier 5 adversarial set
  • Benchmark: Ch21 vs ModernGPT baseline vs Toolformer-style external calc
  • Router activation analysis: which positions activate math nodes?
```

---

## Files (planned)

| Path | Description |
|------|-------------|
| `include/sub0llm/nn/math_nodes.hpp` | `MathNode` interface and implementations |
| `include/sub0llm/nn/numeric_router.hpp` | `NumericRouter` — per-position routing with STE |
| `src/nn/math_nodes.cpp` | Exact arithmetic with overflow/NaN detection |
| `src/nn/numeric_router.cpp` | Router forward + straight-through backward |
| `chapters/ch21_math_neurons/main.cpp` | Demo: §21.1 tokenizer, §21.2 math nodes, §21.3 router, §21.4 training, §21.5 evaluation |
| `tools/gen_arithmetic_dataset.py` | Generate Tiers 1–5 (~225k examples) |
| `tests/test_math_nodes.cpp` | Math node correctness (exact results, overflow/NaN) |
| `tests/test_numeric_router.cpp` | Router precision/recall on Tier 1 samples |
