#include "sub0llm/data/dataloader.hpp"

#include <algorithm>
#include <format>
#include <numeric>
#include <stdexcept>

namespace sub0llm {

DataLoader::DataLoader(const TextDataset& dataset,
                       std::size_t        batch_size,
                       bool               shuffle,
                       uint64_t           seed)
    : dataset_(dataset)
    , batch_size_(batch_size)
    , shuffle_(shuffle)
    , num_batches_(batch_size_ > 0 ? dataset_.size() / batch_size_ : 0)
    , rng_(seed)
{
    if (batch_size_ == 0)
        throw std::invalid_argument("DataLoader: batch_size must be > 0");

    order_.resize(dataset_.size());
    std::iota(order_.begin(), order_.end(), std::size_t{0});
    reset();
}

void DataLoader::reset() {
    current_ = 0;
    if (shuffle_)
        std::shuffle(order_.begin(), order_.end(), rng_);
}

std::optional<Batch> DataLoader::next() {
    if (current_ >= num_batches_)
        return std::nullopt;

    const std::size_t ctx  = dataset_.context_length();
    Batch b;
    b.batch_size     = batch_size_;
    b.context_length = ctx;
    b.input_ids.resize(batch_size_ * ctx);
    b.target_ids.resize(batch_size_ * ctx);

    for (std::size_t i = 0; i < batch_size_; ++i) {
        const std::size_t sample_idx = order_[current_ * batch_size_ + i];
        const auto in  = dataset_.input_span(sample_idx);
        const auto tgt = dataset_.target_span(sample_idx);

        std::copy(in.begin(),  in.end(),  b.input_ids.begin()  + static_cast<std::ptrdiff_t>(i * ctx));
        std::copy(tgt.begin(), tgt.end(), b.target_ids.begin() + static_cast<std::ptrdiff_t>(i * ctx));
    }

    ++current_;
    return b;
}

} // namespace sub0llm
