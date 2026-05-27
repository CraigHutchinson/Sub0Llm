#pragma once

#include "sub0llm/tokenizer/bpe.hpp"

#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace sub0llm {

// ── NumericTokenizer ──────────────────────────────────────────────────────────
//
// Wraps a BPETokenizer and extends its vocabulary with a contiguous range of
// numeric token IDs representing signed int16 values plus two sentinel tokens.
//
// ID layout (above the BPE vocab):
//   [numeric_start, numeric_start + 65536)  ↔  signed int16 [-32768, 32767]
//   numeric_start + 65536                   = NaN token
//   numeric_start + 65537                   = Overflow token
class NumericTokenizer {
public:
    using TokenId = BPETokenizer::TokenId;

    static constexpr int32_t  kIntMin     = -32768;
    static constexpr int32_t  kIntMax     =  32767;
    static constexpr int64_t  kIntRange   = 65536;
    static constexpr int64_t  kExtraTokens = 2;

    explicit NumericTokenizer(BPETokenizer bpe);

    [[nodiscard]] int64_t total_vocab_size() const noexcept;
    [[nodiscard]] int64_t bpe_vocab_size()   const noexcept;
    [[nodiscard]] TokenId numeric_range_start() const noexcept;

    [[nodiscard]] bool is_numeric(TokenId id)        const noexcept;
    [[nodiscard]] bool is_nan_token(TokenId id)      const noexcept;
    [[nodiscard]] bool is_overflow_token(TokenId id) const noexcept;

    // Returns float value for a numeric token.
    // NaN token → quiet NaN; overflow token → infinity.
    // Throws std::invalid_argument if id is not a numeric token.
    [[nodiscard]] float numeric_value(TokenId id) const;

    // Encode a signed integer value. Out-of-range → overflow_token().
    [[nodiscard]] TokenId encode_int(int32_t value) const noexcept;

    [[nodiscard]] TokenId nan_token()      const noexcept;
    [[nodiscard]] TokenId overflow_token() const noexcept;

    // Encode text: pure int16 words → numeric token; other words → BPE tokens.
    [[nodiscard]] std::vector<TokenId> encode(std::string_view text) const;

    // Decode token IDs: numeric tokens → integer string; others → BPE decode.
    [[nodiscard]] std::string decode(std::span<const TokenId> ids) const;

    [[nodiscard]] BPETokenizer const& bpe() const noexcept;

private:
    BPETokenizer bpe_;
    TokenId      numeric_start_;

    [[nodiscard]] static std::optional<int32_t> try_parse_int16(std::string_view s);
};

} // namespace sub0llm
