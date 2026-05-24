#pragma once

#include "sub0llm/data/dataset.hpp"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <random>
#include <vector>

namespace sub0llm {

// A Batch holds B samples, each of length context_length, packed into flat
// row-major buffers.
//
//   input_ids.size()  == batch_size * context_length
//   target_ids.size() == batch_size * context_length
//
// Row i occupies [i * context_length, (i+1) * context_length).
struct Batch {
    using TokenId = TextDataset::TokenId;

    std::vector<TokenId> input_ids;
    std::vector<TokenId> target_ids;
    std::size_t          batch_size;
    std::size_t          context_length;
};

// Iterates over a TextDataset in shuffled mini-batches.
//
// Typical usage:
//   DataLoader loader(dataset, /*batch_size=*/4, /*shuffle=*/true);
//   loader.reset();
//   while (auto batch = loader.next()) { ... }
class DataLoader {
public:
    DataLoader(const TextDataset& dataset,
               std::size_t        batch_size,
               bool               shuffle = false,
               uint64_t           seed    = 42);

    // Reset to the start of a new epoch (re-shuffles if shuffle=true).
    void reset();

    // Returns the next batch, or std::nullopt when the epoch is exhausted.
    // Drops the last incomplete batch if the dataset size is not divisible
    // by batch_size.
    [[nodiscard]] std::optional<Batch> next();

    [[nodiscard]] std::size_t batch_size()   const noexcept { return batch_size_; }
    [[nodiscard]] std::size_t num_batches()  const noexcept { return num_batches_; }
    [[nodiscard]] std::size_t current_batch() const noexcept { return current_; }

private:
    const TextDataset& dataset_;
    std::size_t        batch_size_;
    bool               shuffle_;
    std::size_t        num_batches_;
    std::size_t        current_ = 0;

    std::mt19937_64         rng_;
    std::vector<std::size_t> order_; // shuffled sample indices
};

} // namespace sub0llm
