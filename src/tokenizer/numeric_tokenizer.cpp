#include "sub0llm/tokenizer/numeric_tokenizer.hpp"

#include <charconv>
#include <cmath>
#include <format>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>

namespace sub0llm {

NumericTokenizer::NumericTokenizer(BPETokenizer bpe)
    : bpe_(std::move(bpe)),
      numeric_start_(static_cast<TokenId>(bpe_.vocab_size())) {}

int64_t NumericTokenizer::total_vocab_size() const noexcept {
    return static_cast<int64_t>(bpe_.vocab_size()) + kIntRange + kExtraTokens;
}

int64_t NumericTokenizer::bpe_vocab_size() const noexcept {
    return static_cast<int64_t>(bpe_.vocab_size());
}

NumericTokenizer::TokenId NumericTokenizer::numeric_range_start() const noexcept {
    return numeric_start_;
}

bool NumericTokenizer::is_numeric(TokenId id) const noexcept {
    return id >= numeric_start_ &&
           id < static_cast<TokenId>(numeric_start_ + kIntRange + kExtraTokens);
}

bool NumericTokenizer::is_nan_token(TokenId id) const noexcept {
    return id == nan_token();
}

bool NumericTokenizer::is_overflow_token(TokenId id) const noexcept {
    return id == overflow_token();
}

float NumericTokenizer::numeric_value(TokenId id) const {
    if (!is_numeric(id))
        throw std::invalid_argument(
            std::format("NumericTokenizer::numeric_value: id={} is not a numeric token", id));
    if (is_nan_token(id))
        return std::numeric_limits<float>::quiet_NaN();
    if (is_overflow_token(id))
        return std::numeric_limits<float>::infinity();
    const int32_t val = static_cast<int32_t>(id - numeric_start_) + kIntMin;
    return static_cast<float>(val);
}

NumericTokenizer::TokenId NumericTokenizer::encode_int(int32_t value) const noexcept {
    if (value < kIntMin || value > kIntMax)
        return overflow_token();
    return static_cast<TokenId>(numeric_start_ + static_cast<int64_t>(value - kIntMin));
}

NumericTokenizer::TokenId NumericTokenizer::nan_token() const noexcept {
    return static_cast<TokenId>(numeric_start_ + kIntRange);
}

NumericTokenizer::TokenId NumericTokenizer::overflow_token() const noexcept {
    return static_cast<TokenId>(numeric_start_ + kIntRange + 1);
}

std::optional<int32_t> NumericTokenizer::try_parse_int16(std::string_view s) {
    if (s.empty()) return std::nullopt;

    // Only allow optional leading '-' followed by digits
    std::size_t start = 0;
    if (s[0] == '-') start = 1;
    if (start >= s.size()) return std::nullopt;
    for (std::size_t i = start; i < s.size(); ++i) {
        if (s[i] < '0' || s[i] > '9') return std::nullopt;
    }

    int32_t val{};
    const auto [ptr, ec] = std::from_chars(s.data(), s.data() + s.size(), val);
    if (ec != std::errc{} || ptr != s.data() + s.size()) return std::nullopt;
    if (val < kIntMin || val > kIntMax) return std::nullopt;
    return val;
}

std::vector<NumericTokenizer::TokenId> NumericTokenizer::encode(std::string_view text) const {
    std::vector<TokenId> result;

    std::size_t i = 0;
    while (i < text.size()) {
        // Skip whitespace
        while (i < text.size() && (text[i] == ' ' || text[i] == '\t' ||
                                    text[i] == '\n' || text[i] == '\r'))
            ++i;
        if (i >= text.size()) break;

        // Find end of word (non-whitespace)
        std::size_t j = i;
        while (j < text.size() && text[j] != ' ' && text[j] != '\t' &&
               text[j] != '\n' && text[j] != '\r')
            ++j;

        std::string_view word = text.substr(i, j - i);

        auto parsed = try_parse_int16(word);
        if (parsed.has_value()) {
            result.push_back(encode_int(*parsed));
        } else {
            auto bpe_ids = bpe_.encode(word);
            result.insert(result.end(), bpe_ids.begin(), bpe_ids.end());
        }

        i = j;
    }

    return result;
}

std::string NumericTokenizer::decode(std::span<const TokenId> ids) const {
    std::string result;
    std::vector<TokenId> bpe_buf;

    auto flush_bpe = [&]() {
        if (!bpe_buf.empty()) {
            result += bpe_.decode(bpe_buf);
            bpe_buf.clear();
        }
    };

    for (TokenId id : ids) {
        if (is_numeric(id)) {
            flush_bpe();
            if (!result.empty()) result += ' ';
            if (is_nan_token(id)) {
                result += "NaN";
            } else if (is_overflow_token(id)) {
                result += "Overflow";
            } else {
                const int32_t val = static_cast<int32_t>(id - numeric_start_) + kIntMin;
                result += std::to_string(val);
            }
        } else {
            bpe_buf.push_back(id);
        }
    }

    flush_bpe();
    return result;
}

BPETokenizer const& NumericTokenizer::bpe() const noexcept {
    return bpe_;
}

} // namespace sub0llm
