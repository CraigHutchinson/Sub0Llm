# Chapter 04 — Dataset & DataLoader

## Overview

Before a neural network can train, its input must be shaped into fixed-length
integer sequences and delivered in randomised mini-batches. This chapter builds
that pipeline on top of the BPE tokenizer from Ch03.

```
raw text  →  BPETokenizer  →  token id stream
          →  TextDataset (sliding window)  →  (input_ids, target_ids)
          →  DataLoader (shuffle + batch)  →  Batch
```

## Fixed-Length Sequences

Transformer attention is O(T²) in sequence length T, so we train on fixed-length
chunks of L tokens. Each chunk of L+1 tokens generates L (input, target) pairs
via a **1-token shift**:

```
tokens:  [A, B, C, D, E]   (L=4, with one lookahead)
input:   [A, B, C, D]
target:  [B, C, D, E]      target[t] = next token after input[t]
```

GPT-2 trains with L=1024. The library uses smaller L for demonstration.

## API

```cpp
#include "sub0llm/data/dataset.hpp"
#include "sub0llm/data/dataloader.hpp"
using namespace sub0llm;

// Build token stream from tokenizer
std::vector<int32_t> all_tokens = ...;

// Sliding-window dataset
constexpr std::size_t ctx_len = 64;   // context length L
constexpr std::size_t stride  = 32;   // window stride
TextDataset ds(all_tokens, ctx_len, stride);
ds.size();                             // number of samples

// Access individual samples
auto sample = ds[0];
sample.input_ids;   // vector<int32_t>, length ctx_len
sample.target_ids;  // vector<int32_t>, length ctx_len (shifted by 1)

// DataLoader — shuffles sample indices each epoch
DataLoader loader(ds, /*batch_size=*/32, /*shuffle=*/true, /*seed=*/42);
loader.batch_size();   // 32
loader.num_batches();  // ds.size() / batch_size

loader.reset();        // shuffle and reset to epoch start
while (auto batch = loader.next()) {
    batch->input_ids;        // flat span, length batch_size * ctx_len
    batch->target_ids;       // flat span, same shape
    batch->batch_size;       // number of sequences in this batch
    batch->context_length;   // ctx_len
}
```

## Key Points

- **Stride < ctx_len** means windows overlap — the same tokens appear in
  multiple samples. This is how GPT-2-style training works.
- **Epoch reset**: `loader.reset()` re-shuffles sample indices so each epoch
  sees a different ordering.
- **Reproducibility**: seed passed to `DataLoader` controls the shuffle RNG.
- **Memory**: `TextDataset` does not copy the token vector — it stores a
  reference and computes offsets on access.

## Files

| Path | Description |
|------|-------------|
| `include/sub0llm/data/dataset.hpp` | `TextDataset`, `Sample` |
| `include/sub0llm/data/dataloader.hpp` | `DataLoader`, `Batch` |
| `chapters/ch04_dataset/main.cpp` | Demo (§1–§7) |
| `tests/test_dataset.cpp` | Dataset and DataLoader tests |
