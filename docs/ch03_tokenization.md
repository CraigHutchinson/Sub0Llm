# Chapter 03 — Tokenization

## Overview

Neural networks require fixed-size integer inputs, not raw text. This chapter
builds the complete text processing pipeline using **Byte-Pair Encoding (BPE)**
— the algorithm used by GPT-2/3/4 and most modern LLMs.

Unlike word-level vocabularies, BPE operates on raw bytes so it handles any
Unicode text and has a bounded vocabulary size. Rare words decompose into
subword pieces; common words stay as single tokens.

```
raw text  →  pre-tokenization  →  BPE merges  →  token IDs  →  embeddings (Ch06)
```

## BPE Algorithm

```
Corpus: "the cat sat on the mat"

Initialise: character-level tokens
  ['t','h','e',' ','c','a','t',' ','s','a','t',' ','o','n',' ','t','h','e',' ','m','a','t']

Iteration 1: most frequent pair → ('t','h') → merge to 'th'
  ['th','e',' ','c','a','t',' ','s','a','t',' ','o','n',' ','th','e',' ','m','a','t']

Iteration 2: ('th','e') → 'the'  …  and so on until vocab_size is reached
```

Each merge reduces the token stream length while adding a new vocabulary entry.
After training, the learned merge table is applied greedily during encoding.

## GPT-2 Space Marker

GPT-2 encodes word boundaries using the Unicode character **Ġ** (U+0120, byte
`0xC4 0xA0`) prepended to words that follow whitespace:

```
encode("cat sat") → ['cat', 'Ġsat']
```

This means the tokenizer handles word boundaries without a dedicated symbol and
avoids the need for special whitespace tokens.

## API

```cpp
#include "sub0llm/tokenizer/bpe.hpp"
using namespace sub0llm;

// Train on a corpus
std::vector<std::string> corpus = {"the cat sat on the mat", ...};
BPETokenizer tok = BPETokenizer::train(corpus, /*vocab_size=*/100);

// Encode and decode
std::vector<int32_t> ids = tok.encode("the cat sat");
std::string text         = tok.decode(ids);

// Inspect vocabulary
tok.vocab_size();                     // total tokens including merges
tok.num_merges();                     // merge rules learned
tok.token_str(id);                    // token string for id
tok.eos_id();                         // <|endoftext|> id

// Special tokens (Ch16 thinking tokens)
int32_t think_id = tok.add_special_token("<think>");
int32_t id       = tok.token_id("<think>");

// Persist and reload
tok.save("/path/to/vocab_dir");
BPETokenizer tok2 = BPETokenizer::load("vocab.json", "merges.txt");
```

## Vocabulary Sizes

| Model | Vocabulary | Notes |
|-------|-----------|-------|
| GPT-2 | 50,257 | BPE on byte sequences |
| Llama 3 | 128,000 | SentencePiece BPE |
| This library | User-defined | `train(corpus, vocab_size)` |

## Files

| Path | Description |
|------|-------------|
| `include/sub0llm/tokenizer/bpe.hpp` | BPETokenizer class declaration |
| `src/tokenizer/bpe.cpp` | BPE training, encode, decode, save/load |
| `chapters/ch03_tokenization/main.cpp` | Demo (§1–§8) |
| `tests/test_tokenizer.cpp` | Encode/decode roundtrip tests |
