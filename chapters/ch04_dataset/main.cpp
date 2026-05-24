// Chapter 04 — Dataset & DataLoader
//
// Before a neural network can train, its input must be shaped into fixed-length
// integer sequences and delivered in randomised mini-batches.  This chapter
// builds that pipeline:
//
//   raw text  →  BPE tokenizer  →  token id stream
//             →  sliding-window chunks  →  (input_ids, target_ids)
//             →  shuffled mini-batches
//
// Topics covered:
//   1. Why fixed-length sequences?  Context window and attention cost.
//   2. Memory-mapped file reading (MappedFile)
//   3. Sliding-window chunking (TextDataset)
//   4. Input / target shift: target[t] = input[t+1]
//   5. DataLoader: batching, shuffle, epoch reset
//   6. What the training loop will see next chapter

#include <format>
#include <iostream>
#include <numeric>
#include <vector>

#include "sub0llm/compat/print.hpp"
#include "sub0llm/data/dataloader.hpp"
#include "sub0llm/data/dataset.hpp"
#include "sub0llm/tokenizer/bpe.hpp"
#include "sub0llm/version.hpp"

using namespace sub0llm;

static void section(std::string_view title) {
    std::println("\n── {} ──", title);
}

// ─────────────────────────────────────────────────────────────────────────────
int main() {
    std::println("sub0llm v{} — Chapter 04: Dataset & DataLoader", version_string);

    // ── 1. Why fixed-length sequences? ────────────────────────────────────────
    section("1. Why fixed-length sequences?");
    std::println(
        "Transformer attention is O(T²) in sequence length T.\n"
        "We therefore train on fixed-length chunks of L tokens.\n"
        "Each chunk generates L (input, target) pairs via a 1-token shift:\n"
        "  input  = tokens[0..L-1]\n"
        "  target = tokens[1..L]   ← each target[t] = next token after input[t]\n"
        "GPT-2 uses L=1024; we use smaller L for demonstration."
    );

    // ── 2. Build a small corpus and tokenize it ────────────────────────────────
    section("2. Tokenize a small corpus");

    const std::vector<std::string> corpus = {
        "the cat sat on the mat",
        "the cat ate the rat",
        "the mat the cat sat",
        "rats and cats and bats sat on mats",
        "a cat and a rat and a bat",
    };

    auto tok = BPETokenizer::train(corpus, 80);
    std::println("Vocabulary size: {}", tok.vocab_size());

    // Concatenate corpus into one token stream (real training uses mmap).
    std::vector<int32_t> all_tokens;
    for (const auto& text : corpus) {
        const auto ids = tok.encode(text);
        for (auto id : ids)
            all_tokens.push_back(static_cast<int32_t>(id));
        all_tokens.push_back(static_cast<int32_t>(tok.eos_id()));
    }
    std::println("Total tokens in stream: {}", all_tokens.size());

    // ── 3. TextDataset — sliding windows ──────────────────────────────────────
    section("3. TextDataset — sliding window");

    constexpr std::size_t CTX    = 8;   // context length
    constexpr std::size_t STRIDE = 4;   // window stride

    TextDataset ds(all_tokens, CTX, STRIDE);
    std::println("context_length={}, stride={}, samples={}", CTX, STRIDE, ds.size());

    // Show the first few samples.
    std::println("\nFirst 3 samples (input → target):");
    for (std::size_t i = 0; i < std::min<std::size_t>(3, ds.size()); ++i) {
        const auto s = ds[i];
        std::print("  [{}] input : [", i);
        for (std::size_t j = 0; j < s.input_ids.size(); ++j) {
            std::print("{}", s.input_ids[j]);
            if (j + 1 < s.input_ids.size()) std::print(", ");
        }
        std::print("]\n       target: [");
        for (std::size_t j = 0; j < s.target_ids.size(); ++j) {
            std::print("{}", s.target_ids[j]);
            if (j + 1 < s.target_ids.size()) std::print(", ");
        }
        std::println("]");
    }

    // ── 4. Target is input shifted by 1 ───────────────────────────────────────
    section("4. Input / target 1-token shift");
    {
        const auto s = ds[0];
        std::println(
            "For sample 0:\n"
            "  input[0]  = '{}' (id {})\n"
            "  target[0] = '{}' (id {})  ← the model must predict this from input[0]\n"
            "  input[1]  = '{}' (id {})\n"
            "  target[1] = '{}' (id {})  ← predicted from input[0..1]",
            tok.token_str(s.input_ids[0]),  s.input_ids[0],
            tok.token_str(s.target_ids[0]), s.target_ids[0],
            tok.token_str(s.input_ids[1]),  s.input_ids[1],
            tok.token_str(s.target_ids[1]), s.target_ids[1]
        );
    }

    // ── 5. DataLoader — mini-batches ──────────────────────────────────────────
    section("5. DataLoader — mini-batches");

    constexpr std::size_t BATCH = 4;
    DataLoader loader(ds, BATCH, /*shuffle=*/true, /*seed=*/42);
    std::println("batch_size={}, num_batches={}", loader.batch_size(), loader.num_batches());

    loader.reset();
    std::size_t batches_seen = 0;
    while (auto batch = loader.next()) {
        ++batches_seen;
        if (batches_seen == 1) {
            std::println("\nBatch 1: shape=({}, {})",
                batch->batch_size, batch->context_length);
            std::println("  input_ids[:{}] = [", CTX);
            for (std::size_t i = 0; i < CTX; ++i) {
                std::print("    '{}'({})", tok.token_str(batch->input_ids[i]),
                           batch->input_ids[i]);
                if (i + 1 < CTX) std::print(",");
                std::println("");
            }
            std::println("  ]");
        }
    }
    std::println("Total batches per epoch: {}", batches_seen);

    // ── 6. Epoch reset and reproducibility ────────────────────────────────────
    section("6. Epoch reset");

    // Run two epochs; with shuffle=true the first batches will differ.
    DataLoader loader2(ds, BATCH, /*shuffle=*/true, /*seed=*/7);
    loader2.reset();
    std::optional<Batch> first_e1 = loader2.next();

    loader2.reset();
    std::optional<Batch> first_e2 = loader2.next();

    const bool same = first_e1 && first_e2 &&
                      first_e1->input_ids == first_e2->input_ids;
    std::println("First-batch identical across epochs with shuffle: {} (expected: false)",
                 same ? "true" : "false");

    // ── 7. What training will see ──────────────────────────────────────────────
    section("7. What's next — Ch05: Autograd Engine");
    std::println(
        "The training loop will:\n"
        "  for epoch in range(N):\n"
        "    loader.reset()\n"
        "    while (auto batch = loader.next()):\n"
        "      logits = model.forward(batch.input_ids)\n"
        "      loss   = cross_entropy(logits, batch.target_ids)\n"
        "      loss.backward()\n"
        "      optimizer.step()\n"
        "\n"
        "Ch05 builds the autograd engine powering loss.backward().\n"
        "Ch06 adds token + positional embeddings that consume input_ids."
    );

    return 0;
}
