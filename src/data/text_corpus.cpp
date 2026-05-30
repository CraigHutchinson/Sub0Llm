#include "sub0llm/data/text_corpus.hpp"

#include <algorithm>
#include <format>
#include <fstream>
#include <nlohmann/json.hpp>
#include <numeric>
#include <random>
#include <stdexcept>

namespace sub0llm {

TextCorpus::TextCorpus(std::vector<std::string> texts, const BPETokenizer& tok,
                       int64_t seq_len, uint32_t seed)
    : seq_len_(seq_len), cursor_(0)
{
    if (seq_len_ <= 0)
        throw std::runtime_error(std::format("TextCorpus: seq_len must be > 0, got {}", seq_len_));

    // Encode each document and concatenate, inserting a newline between docs so
    // the tokenizer sees a clean sentence boundary.  BPETokenizer has no explicit
    // EOS token unless add_special_token() was called, so a newline is the safe separator.
    for (std::size_t i = 0; i < texts.size(); ++i) {
        auto ids = tok.encode(texts[i]);
        tokens_.insert(tokens_.end(), ids.begin(), ids.end());
        if (i + 1 < texts.size()) {
            // Insert a separator: if the tokenizer has an EOS, use it; otherwise
            // encode a newline to get a clean boundary token.
            auto sep_id = tok.eos_id();
            if (sep_id >= 0) {
                tokens_.push_back(static_cast<TokenId>(sep_id));
            } else {
                auto nl_ids = tok.encode("\n");
                tokens_.insert(tokens_.end(), nl_ids.begin(), nl_ids.end());
            }
        }
    }

    build_order(seed);
}

TextCorpus TextCorpus::from_file(const std::string& path, const BPETokenizer& tok,
                                  int64_t seq_len, uint32_t seed)
{
    std::ifstream f(path);
    if (!f.is_open())
        throw std::runtime_error(std::format("TextCorpus::from_file: cannot open '{}'", path));

    std::vector<std::string> texts;
    std::string line;
    while (std::getline(f, line)) {
        if (line.empty()) continue;
        if (line.front() == '{') {
            // JSONL: extract the "text" field
            try {
                auto j = nlohmann::json::parse(line);
                texts.push_back(j.at("text").get<std::string>());
            } catch (const std::exception& e) {
                throw std::runtime_error(
                    std::format("TextCorpus::from_file: JSON parse error in '{}': {}", path, e.what()));
            }
        } else {
            texts.push_back(line);
        }
    }

    return TextCorpus(std::move(texts), tok, seq_len, seed);
}

void TextCorpus::build_order(uint32_t seed)
{
    // Non-overlapping windows of size (seq_len+1): seq_len ids + 1 target overflow.
    // We need tokens_[i*(seq_len+1)] .. tokens_[i*(seq_len+1)+seq_len] inclusive.
    const std::size_t window = static_cast<std::size_t>(seq_len_) + 1;
    const std::size_t n = tokens_.size() >= window ? tokens_.size() / window : 0;

    order_.resize(n);
    std::iota(order_.begin(), order_.end(), std::size_t{0});

    std::mt19937 rng(seed);
    std::shuffle(order_.begin(), order_.end(), rng);
    cursor_ = 0;
}

bool TextCorpus::next_sample(Tensor& ids, Tensor& tgt)
{
    if (cursor_ >= order_.size())
        return false;

    const std::size_t idx    = order_[cursor_++];
    const std::size_t offset = idx * (static_cast<std::size_t>(seq_len_) + 1);

    // Allocate output tensors
    ids = Tensor({seq_len_}, DType::Int32);
    tgt = Tensor({seq_len_}, DType::Int32);

    auto ids_sp = ids.data_as<int32_t>();
    auto tgt_sp = tgt.data_as<int32_t>();

    for (int64_t i = 0; i < seq_len_; ++i) {
        ids_sp[static_cast<std::size_t>(i)] = tokens_[offset + static_cast<std::size_t>(i)];
        tgt_sp[static_cast<std::size_t>(i)] = tokens_[offset + static_cast<std::size_t>(i) + 1];
    }

    tokens_consumed_ += seq_len_;
    return true;
}

void TextCorpus::reset(uint32_t seed)
{
    build_order(seed);
    tokens_consumed_ = 0;
}

int64_t TextCorpus::total_tokens() const noexcept
{
    return static_cast<int64_t>(tokens_.size());
}

int64_t TextCorpus::tokens_consumed() const noexcept
{
    return tokens_consumed_;
}

std::size_t TextCorpus::n_samples() const noexcept
{
    return order_.size();
}

} // namespace sub0llm
