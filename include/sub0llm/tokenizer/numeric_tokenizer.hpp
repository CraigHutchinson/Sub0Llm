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
// numeric token IDs for integers in [int_min, int_max] plus sentinel tokens.
//
// ID layout (above the BPE vocab):
//   [numeric_start, numeric_start + int_range())  ↔  [int_min, int_max]
//   numeric_start + int_range()                   = NaN token
//   numeric_start + int_range() + 1               = Overflow token
class NumericTokenizer {
public:
    using TokenId = BPETokenizer::TokenId;

    // NaN + Overflow — the two tokens that sit inside the "numeric" predicate
    static constexpr int64_t  kIntExtraTokens  = 2;
    // Opaque algebraic symbol slots X0..X(kAlgSlots-1)
    static constexpr int64_t  kAlgSlots        = 4;
    // Total extra tokens beyond the integer range:
    //   kIntExtraTokens (NaN, Overflow) + 1 (NUM placeholder) + kAlgSlots (X0..X3)
    static constexpr int64_t  kExtraTokens     = kIntExtraTokens + 1 + kAlgSlots;

    // int_min/int_max define the representable integer range.
    // Defaults match the original signed int16 range for backward compatibility.
    explicit NumericTokenizer(BPETokenizer bpe,
                              int32_t int_min = -32768,
                              int32_t int_max =  32767);

    [[nodiscard]] int64_t total_vocab_size() const noexcept;
    [[nodiscard]] int64_t bpe_vocab_size()   const noexcept;
    [[nodiscard]] TokenId numeric_range_start() const noexcept;

    [[nodiscard]] int32_t int_min()   const noexcept;
    [[nodiscard]] int32_t int_max()   const noexcept;
    [[nodiscard]] int64_t int_range() const noexcept;

    [[nodiscard]] bool is_numeric(TokenId id)        const noexcept;
    [[nodiscard]] bool is_nan_token(TokenId id)      const noexcept;
    [[nodiscard]] bool is_overflow_token(TokenId id) const noexcept;

    // Returns float value for a numeric token.
    // NaN token → quiet NaN; overflow token → infinity.
    // Throws std::invalid_argument if id is not a numeric token.
    [[nodiscard]] float numeric_value(TokenId id) const;

    // Encode a signed integer value. Out-of-range → overflow_token().
    [[nodiscard]] TokenId encode_int(int32_t value) const noexcept;

    [[nodiscard]] TokenId nan_token()             const noexcept;
    [[nodiscard]] TokenId overflow_token()        const noexcept;
    // Full-anonymisation placeholder — all numeric tokens map to this single ID.
    [[nodiscard]] TokenId num_placeholder_token() const noexcept;
    // Algebraic symbol tokens X0..X(kAlgSlots-1) — slot must be in [0, kAlgSlots).
    [[nodiscard]] TokenId algebraic_token(int slot) const noexcept;

    [[nodiscard]] bool is_placeholder_token(TokenId id) const noexcept;
    [[nodiscard]] bool is_algebraic_token(TokenId id)   const noexcept;

    // Encode text: pure int16 words → numeric token; other words → BPE tokens.
    [[nodiscard]] std::vector<TokenId> encode(std::string_view text) const;

    // Decode token IDs: numeric tokens → integer string; others → BPE decode.
    [[nodiscard]] std::string decode(std::span<const TokenId> ids) const;

    [[nodiscard]] BPETokenizer const& bpe() const noexcept;

private:
    BPETokenizer bpe_;
    TokenId      numeric_start_;
    int32_t      int_min_;
    int32_t      int_max_;
    int64_t      int_range_;

    [[nodiscard]] std::optional<int32_t> try_parse_int(std::string_view s) const;
};

} // namespace sub0llm
