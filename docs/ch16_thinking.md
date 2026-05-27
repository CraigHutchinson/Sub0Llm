# Chapter 16 — Thinking Tokens: Chain-of-Thought Generation

## Overview

Modern reasoning models (DeepSeek-R1, o1, Qwen-QwQ) generate an internal
chain-of-thought before producing their answer. This chapter implements the
mechanism at the generation level: the model emits tokens inside a
`<think>…</think>` delimited section before producing the answer.

The key insight is that thinking is just generation under a different sampling
config, terminated by a special `</think>` token rather than EOS. The same
weights support both thinking and non-thinking generation — the only difference
is which token IDs are treated as delimiters.

## How It Works

```
prompt tokens
     │
     ▼
generate_with_thinking()
     │
     ├─ Phase 1: thinking generation
     │    emit tokens until </think> or max_think_tokens exceeded
     │    uses inner_cfg (usually higher temperature for exploration)
     │
     └─ Phase 2: answer generation
          emit tokens until EOS or max_answer_tokens
          uses answer_cfg (usually lower temperature / top-p)
```

## Special Tokens

```cpp
// Add to tokenizer beyond regular vocabulary
int32_t think_bos = tokenizer.add_special_token("<think>");
int32_t think_eos = tokenizer.add_special_token("</think>");

// Retrieve later
int32_t id = tokenizer.token_id("<think>");
```

Special token IDs are assigned beyond the regular vocabulary range so they never
conflict with BPE merge tokens.

## API

```cpp
#include "sub0llm/nn/thinking.hpp"
using namespace sub0llm::nn;

// Configure thinking generation
ThinkingConfig cfg;
cfg.think_bos_id      = tokenizer.token_id("<think>");
cfg.think_eos_id      = tokenizer.token_id("</think>");
cfg.max_think_tokens  = 64;    // budget for internal reasoning
cfg.max_answer_tokens = 16;    // max answer length
cfg.inner_cfg.mode        = SamplingMode::Temperature;
cfg.inner_cfg.temperature = 1.2f;    // explore during thinking
cfg.answer_cfg.mode       = SamplingMode::TopP;
cfg.answer_cfg.top_p      = 0.9f;
cfg.answer_cfg.temperature = 0.8f;  // more focused for answer

std::mt19937 rng(42);
ThinkingResult result = generate_with_thinking(model, prompt, cfg, rng);

result.thinking_tokens;    // tokens inside <think>...</think>
result.answer_tokens;      // tokens after </think>
result.thinking_complete;  // true if </think> was found before budget exhausted

// Decode
std::string thinking = tokenizer.decode(result.thinking_tokens);
std::string answer   = tokenizer.decode(result.answer_tokens);

// Self-consistency voting: run N times, pick most common first-answer token
int32_t voted = think_self_consistency(model, prompt, cfg, /*n_samples=*/7, rng);
```

## Thinking Budget Comparison

```
budget= 4  think_len=4  complete=false  answer="was beg"
budget=16  think_len=16 complete=false  answer="was beg"
budget=48  think_len=23 complete=true   answer="inning to"
```

Longer budget allows more thinking before the answer, but with untrained weights
the thinking content is not yet meaningful — the mechanism is demonstrated.

## Self-Consistency Voting

Run K independent thinking+answer samples with temperature > 0, then vote on
the most frequent first answer token. Reduces variance when the model has
multiple plausible answers.

```
7 independent runs → first answer tokens: [3, 7, 3, 3, 12, 3, 7]
voted result: token 3  (majority)
```

## Comparison: Thinking Tokens (Ch16) vs LoopedGPT (Ch17)

| Dimension | Ch16 Thinking Tokens | Ch17 LoopedGPT |
|-----------|---------------------|----------------|
| Budget unit | Token count | Loop count K |
| Thinking visibility | Visible in sequence | Internal (hidden state) |
| KV cache overhead | Grows with thinking length | None |
| Interpretability | Readable chain-of-thought | Opaque refinement |
| RLHF compatibility | Natural (reward on tokens) | Implicit |

## Files

| Path | Description |
|------|-------------|
| `include/sub0llm/nn/thinking.hpp` | `ThinkingConfig`, `ThinkingResult`, `generate_with_thinking`, `think_self_consistency` |
| `src/nn/thinking.cpp` | Two-phase generation loop |
| `chapters/ch16_thinking/main.cpp` | Demo: §16.1 special tokens, §16.2 training+demo, §16.3 budget comparison, §16.4 self-consistency, §16.5 architecture preview |
| `tests/test_thinking.cpp` | Thinking generation tests |
