# Chapter 21 — Research Concept: Specialised Mathematical Neurons

> **Status**: Research / concept stage. No implementation yet.

## Motivation

Language models learn arithmetic through statistical pattern matching: "2 + 2"
→ "4" because that co-occurrence is overwhelmingly common in training text.
This forces the model to memorise multiplication tables, carry rules, and
floating-point identities as weight patterns — a wasteful and lossy encoding.

Classical computation handles arithmetic exactly, instantly, and with zero
approximation error. The research question is: **can we give the model a direct
path to exact arithmetic that bypasses statistical memorisation entirely?**

---

## Core Concept

Replace the feedforward path for numeric tokens with **specialised execution
nodes** — deterministic functions that are wired into the model like neurons
but execute classical algorithms rather than learned matrix products.

```
standard path:
  token_ids → embedding → transformer → softmax → output token

proposed path (numeric tokens):
  token_ids → embedding → transformer
                                │
                     ┌──── is numeric? ───┐
                     │ yes                │ no
                     ▼                    ▼
           Math Execution Node      standard FFN
           (exact arithmetic)           │
                     │                  │
                     └────────┬─────────┘
                              ▼
                        output embedding
```

---

## Tokenizer: Numeric Token Range

The BPE tokenizer would reserve a contiguous range of token IDs for exact
numeric representations. These IDs are not learned subword merges — they are
deterministic mappings:

```
Reserved range: IDs [V_lang, V_lang + V_num)

Encoding examples:
  "42"     → numeric token for integer 42
  "3.14"   → numeric token for float 3.14
  "-7"     → numeric token for integer -7
  "1e-6"   → numeric token for scientific notation 1×10⁻⁶

V_num could encode:
  integers  : signed 32-bit or 64-bit  (e.g. 2^32 IDs for full int32 range)
  floats    : IEEE 754 float16 quantised (65536 IDs covers most practical values)
  fractions : rational p/q pairs
```

The tokenizer learns when to route numeric strings into the reserved range vs
treating them as regular subword tokens (for embedded numbers like "42nd", "IPv4
address", dates in context).

---

## Specialised Execution Nodes

A **math execution node** receives one or more numeric token embeddings, applies
an exact function, and returns an embedding for the result token.

```cpp
// Conceptual interface
class MathNode {
public:
    virtual ~MathNode() = default;
    // inputs: embeddings of operand tokens (each carries its numeric value)
    // returns: embedding of result token (also carries exact numeric value)
    virtual Variable forward(std::span<const Variable> operands) const = 0;
};

class AddNode    : public MathNode { ... };   // a + b
class MulNode    : public MathNode { ... };   // a * b
class DivNode    : public MathNode { ... };   // a / b (detect /0)
class PowNode    : public MathNode { ... };   // a ^ b
class SqrtNode   : public MathNode { ... };   // √a
class LogNode    : public MathNode { ... };   // ln(a)
class CompareNode: public MathNode { ... };   // a < b, a == b → boolean embedding
```

The numeric value is stored in the embedding alongside the standard semantic
vector. A thin router (learnable) decides whether the FFN path or a math node
handles each token position.

---

## Integration with the Transformer

### Option A: Post-attention gate

```
x (T, D)
     │
     ▼
GQA attention
     │ residual
     ▼
Router: per-position classifier
  ├─ numeric position → Math Execution Node(s)
  └─ text position    → SwiGLU FFN
     │
     ▼
merge results back into residual stream
```

The router is a lightweight MLP `Linear(D, n_node_types + 1)` + softmax. It
learns from context whether a position needs exact arithmetic or standard
language processing.

### Option B: Parallel path

Run both the FFN and the math node, then blend with a learned gate:

```
out = gate * math_node(x) + (1 - gate) * ffn(x)

gate ∈ [0,1] — learned per-position, conditioned on x
```

This allows soft transitions and avoids hard routing failures.

---

## Numeric Value Propagation

For math nodes to work, the numeric value must survive the embedding lookup and
survive attention. Two design options:

### Literal encoding
Reserve specific dimensions of the embedding vector for the numeric payload.
The math node reads these dimensions directly; the rest of the model treats them
as opaque features.

```
embedding[token_id = N]:
  dims [0..D/2-1] : standard learned features (context, syntax)
  dims [D/2..D-1] : numeric payload (IEEE 754 bits, fixed mapping)
```

### Side-channel register
Maintain a separate numeric register tensor `R ∈ ℝ^T` in parallel with the
residual stream `x ∈ ℝ^{T×D}`. The register stores raw float values for
numeric token positions; math nodes read from and write to the register
while leaving the residual stream untouched for attention.

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
| Sorting | `SortNode` | Comparison sort |
| Hashing | `HashNode` | CRC32, MD5 |
| Unit conversion | `ConvertNode` | SI prefix tables |
| Regular expressions | `RegexNode` | NFA/DFA matching |
| Lookup tables | `LookupNode` | Read-only key→value map |

The unifying principle: **anything where the answer is deterministic and
well-defined given the inputs** is a candidate for a specialised node. Language
models should learn to *use* these nodes, not to *approximate* their outputs.

---

## Training Implications

### Supervision
- Math nodes themselves have no learnable parameters — their output is exact.
- The **router** and the **numeric embedding mapping** are learned.
- Supervision comes from the standard next-token loss: the model is rewarded
  when it routes to the math node and produces the correct numeric output token.

### Gradient flow
- Arithmetic operations (add, mul) are differentiable — gradients flow back
  through math nodes to the router and embeddings.
- Comparison / Boolean outputs are discrete — require straight-through
  estimator or Gumbel-softmax for gradient.

### Curriculum
- Pre-train base LM normally.
- Fine-tune with arithmetic-heavy data (calculator traces, algebra problems)
  to teach the router when to activate math nodes.
- This mirrors tool-use fine-tuning (e.g. ToolFormer) but with the tool
  executed inside the residual stream rather than via an external API call.

---

## Relationship to Existing Work

| Work | Approach | Difference |
|------|----------|-----------|
| ToolFormer (Schick et al., 2023) | External calculator API via special tokens | Math is outside the model; API call latency |
| Neural Arithmetic Units (NAU, 2019) | Learned add/mul via constrained weights | Approximate, not exact; limited to simple ops |
| Program synthesis (Codex, AlphaCode) | Generate code, execute externally | Full program rather than inline ops |
| Scratchpad / CoT (Ch16) | Think through arithmetic step by step | Still statistical approximation |
| **This proposal** | Exact arithmetic baked into residual stream | Zero approximation, no API latency, differentiable router |

---

## Open Research Questions

1. **Routing accuracy**: How precisely does the router learn to identify numeric
   token positions? What error rate is acceptable?

2. **Numeric token range size**: int32 covers 4B values — far too many IDs.
   A quantised float16 range (65536 IDs) covers most practical values. How does
   precision affect downstream task performance?

3. **Value extraction from context**: Multi-digit numbers like "12,345" are
   often tokenised as multiple BPE tokens. How does the model assemble the full
   value across tokens before routing?

4. **Generalisation**: Does a model trained on single-step arithmetic
   generalise to multi-step calculations (e.g. 3×4+7) without explicit training?

5. **Numerical stability**: IEEE 754 arithmetic on numbers extracted from
   context can produce wrong results for ill-conditioned problems. How should
   the model signal uncertainty?

6. **Other specialisations**: Which non-arithmetic domains benefit most?
   Candidates: date arithmetic, unit conversion, regex matching, table lookup.

---

## Proposed Implementation Plan (Future Chapters)

```
Ch21a — Numeric Tokenizer Extension
  • Extend BPETokenizer with reserved numeric range
  • Implement float16 quantisation mapping
  • Tokeniser routing heuristics (numeric string detection)

Ch21b — Math Execution Nodes
  • MathNode interface + AddNode, MulNode, DivNode, CompareNode
  • Numeric value propagation (side-channel register)
  • Exact arithmetic with overflow/NaN detection

Ch21c — Router Integration
  • Lightweight router (Linear(D, n_types+1) + softmax)
  • Integration into ModernTransformerBlock
  • Straight-through estimator for discrete routing

Ch21d — Training and Evaluation
  • Fine-tuning on arithmetic-heavy corpus
  • Benchmark vs baseline LM on arithmetic tasks
  • Router activation analysis (which positions activate math nodes?)
```

---

## Files (planned)

| Path | Description |
|------|-------------|
| `include/sub0llm/nn/math_nodes.hpp` | `MathNode` interface and implementations |
| `include/sub0llm/nn/numeric_router.hpp` | `NumericRouter` — per-position routing |
| `src/nn/math_nodes.cpp` | Exact arithmetic implementations |
| `src/nn/numeric_router.cpp` | Router forward + straight-through backward |
| `chapters/ch21_math_neurons/main.cpp` | Demo: router accuracy, arithmetic benchmark |
| `tests/test_math_nodes.cpp` | Math node correctness (exact results) |
