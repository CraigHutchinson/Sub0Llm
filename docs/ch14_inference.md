# Chapter 14 — Inference and Sampling

## Overview

A trained model produces a probability distribution over the vocabulary at each
step. **Sampling** determines which token is selected. The choice dramatically
affects output diversity, coherence, and quality.

This chapter implements four strategies and a full autoregressive generation
loop that uses any of them.

## Sampling Strategies

### Greedy
Always pick the argmax token. Deterministic, zero diversity.

### Temperature Sampling
Scale logits before softmax: `logits_scaled = logits / temperature`

| Temperature | Effect |
|-------------|--------|
| → 0 | Approaches greedy (deterministic) |
| 1.0 | Unmodified distribution |
| → ∞ | Uniform random |

### Top-k Sampling
Keep only the k highest-probability tokens; zero out the rest. Hard cutoff
regardless of the probability gap between rank k and k+1.

### Top-p (Nucleus) Sampling
Sort tokens by probability; keep the smallest set whose cumulative probability
≥ p. Adapts to the distribution shape — fewer tokens when the distribution is
peaked, more when it's flat.

```
Logits: [0.2, 0.4, 0.6, 3.0, 7.0]  → probs ≈ [0.1%, 0.2%, 0.2%, 4.9%, 94.6%]

top-k=3: tokens {2,3,4} always eligible
top-p=0.9: when peaked on tok4, only {4} eligible; when flatter, {3,4} etc.
```

## Strategy Comparison

With logits `[0.2, 0.4, 0.6, 3.0, 7.0]` over 5000 samples:

```
              strategy: tok0    tok1    tok2    tok3    tok4
greedy (always):         0.0%    0.0%    0.0%    0.0%  100.0%
temp=1.0:                0.1%    0.2%    0.2%    4.9%   94.6%
temp=0.5:                0.0%    0.0%    0.0%    1.2%   98.8%
top-k=3, t=1.0:          0.0%    0.0%    0.2%    4.9%   94.9%
top-p=0.9, t=1.0:        0.0%    0.0%    0.0%    4.9%   95.1%
```

## API

```cpp
#include "sub0llm/nn/sampler.hpp"
using namespace sub0llm::nn;

std::mt19937 rng(42);
Tensor logits = ...;   // (V,) float

// Individual samplers
int32_t tok_g  = greedy_sample(logits);
int32_t tok_t  = temperature_sample(logits, /*temp=*/0.7f, rng);
int32_t tok_k  = top_k_sample(logits, /*k=*/40, /*temp=*/1.0f, rng);
int32_t tok_p  = top_p_sample(logits, /*p=*/0.9f, /*temp=*/0.8f, rng);

// Full autoregressive generation loop
SamplingConfig cfg;
cfg.mode        = SamplingMode::TopP;
cfg.top_p       = 0.9f;
cfg.temperature = 0.8f;

std::vector<int32_t> prompt = {1, 2, 3};
auto output = generate(model, prompt, /*max_new_tokens=*/50, cfg, rng);
// output = prompt + generated_tokens
```

## SamplingConfig

```cpp
enum class SamplingMode { Greedy, Temperature, TopK, TopP };

struct SamplingConfig {
    SamplingMode mode        = SamplingMode::Greedy;
    float        temperature = 1.0f;
    int64_t      top_k       = 50;
    float        top_p       = 0.9f;
};
```

## Files

| Path | Description |
|------|-------------|
| `include/sub0llm/nn/sampler.hpp` | `SamplingConfig`, all samplers, `generate` |
| `src/nn/sampler.cpp` | Sampling implementations, generation loop |
| `chapters/ch14_inference/main.cpp` | Demo: §14.1 temperature, §14.2 top-k, §14.3 top-p, §14.4 generation, §14.5 comparison |
| `tests/test_sampler.cpp` | Sampler distribution tests |
