// sub0/casing.hpp — corpus-aware truecasing + quote normalization primitives.
//
// These pure functions are the single, shared definition of the reversible text
// transform that sits beneath tokenization. They are used by BOTH the build-time
// configurator (which trains the BPE vocabulary on the truecased stream) and the
// runtime engine (which truecases prompts and detokenizes output). Keeping one
// definition guarantees encode/decode round-trip: decode(encode(x)) reproduces
// normalize_quotes(x) by construction.
//
// The header is intentionally free of any dependency on the generated config, so
// the configurator — which must not depend on the engine — can include it too.

#pragma once

#include <string>
#include <unordered_set>
#include <vector>

namespace sub0::casing {

// Symbol codes for the two case markers, carried in the byte-symbol stream just
// above the 0..255 byte range.
constexpr int TOK_CAP = 256;  // next word: capitalize first letter  (<|cap|>)
constexpr int TOK_UP  = 257;  // next word: upper-case the whole word (<|up|>)

inline bool          is_alpha(unsigned char c) { return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z'); }
inline bool          is_lower(unsigned char c) { return c >= 'a' && c <= 'z'; }
inline bool          is_upper(unsigned char c) { return c >= 'A' && c <= 'Z'; }
inline bool          is_space(unsigned char c) { return c == ' ' || c == '\n' || c == '\t' || c == '\r'; }
inline unsigned char to_lower(unsigned char c) { return is_upper(c) ? static_cast<unsigned char>(c + 32) : c; }
inline unsigned char to_upper(unsigned char c) { return is_lower(c) ? static_cast<unsigned char>(c - 32) : c; }

// Per-corpus truecasing statistics (configurator reporting only).
struct TokStats {
    long words = 0;  // alpha runs seen
    long cap   = 0;  // collapsed to <|cap|> + lowercase
    long up    = 0;  // collapsed to <|up|>  + lowercase
    long names = 0;  // capitalized/upper words left verbatim (corpus-aware)
};

// Collapse space-adjacent "fancy" quote glyphs to a straight apostrophe. This
// shrinks the vocabulary: an opening quote (space before) and a closing quote
// (space after) both normalize to '\''. Handles the ASCII backtick and the
// UTF-8 curly single quotes U+2018/U+2019. Lossy by design — the specific glyph
// identity is intentionally discarded so dialogue marks share one token.
inline std::string normalize_quotes(const std::string& in, long& replaced) {
    std::string out;
    out.reserve(in.size());
    replaced = 0;
    const std::size_t n = in.size();
    for (std::size_t i = 0; i < n;) {
        const unsigned char c = static_cast<unsigned char>(in[i]);
        const bool curly = (c == 0xE2 && i + 2 < n &&
                            static_cast<unsigned char>(in[i + 1]) == 0x80 &&
                            (static_cast<unsigned char>(in[i + 2]) == 0x98 ||
                             static_cast<unsigned char>(in[i + 2]) == 0x99));
        const bool backtick = (c == '`');
        if (curly || backtick) {
            const std::size_t glyph_len    = curly ? 3 : 1;
            const bool        space_before = !out.empty() && is_space(static_cast<unsigned char>(out.back()));
            const std::size_t after        = i + glyph_len;
            const bool        space_after  = after < n && is_space(static_cast<unsigned char>(in[after]));
            if (space_before || space_after) {
                out.push_back('\'');
                ++replaced;
                i += glyph_len;
                continue;
            }
        }
        out.push_back(static_cast<char>(c));
        ++i;
    }
    return out;
}

// Corpus-aware truecasing. Each alpha word is classified:
//   - all lowercase            -> emitted verbatim
//   - Capitalized ("The")      -> <|cap|> + lowercase, *iff* the lowercase form
//                                 is attested (appears all-lowercase elsewhere)
//   - ALL-UPPER  ("HELLO")     -> <|up|>  + lowercase, under the same condition
//   - otherwise (names, mixed) -> emitted verbatim, so "Lily"/"Tom" stay atomic
// Non-alpha bytes pass through unchanged. Output is a stream of byte values plus
// the TOK_CAP/TOK_UP marker codes. `st` may be null when stats are not wanted.
inline std::vector<int> truecase_tokenize(const std::string& text,
                                          const std::unordered_set<std::string>& attested,
                                          TokStats* st) {
    std::vector<int> toks;
    toks.reserve(text.size());
    const std::size_t n = text.size();
    for (std::size_t i = 0; i < n;) {
        const unsigned char c = static_cast<unsigned char>(text[i]);
        if (!is_alpha(c)) { toks.push_back(c); ++i; continue; }

        std::size_t j = i;
        while (j < n && is_alpha(static_cast<unsigned char>(text[j]))) ++j;
        const std::string w = text.substr(i, j - i);
        if (st) ++st->words;

        std::string lw;
        lw.reserve(w.size());
        for (unsigned char ch : w) lw.push_back(static_cast<char>(to_lower(ch)));

        bool rest_lower = true, all_upper = true;
        for (std::size_t k = 1; k < w.size(); ++k)
            if (!is_lower(static_cast<unsigned char>(w[k]))) rest_lower = false;
        for (unsigned char ch : w)
            if (!is_upper(ch)) all_upper = false;

        const bool first_upper = is_upper(static_cast<unsigned char>(w[0]));
        const bool capitalized = first_upper && rest_lower;   // "The", single "I"
        const bool upper       = all_upper && w.size() >= 2;  // "HELLO"
        const bool attested_lw = attested.contains(lw);

        int marker = -1;
        if (upper && attested_lw)            marker = TOK_UP;
        else if (capitalized && attested_lw) marker = TOK_CAP;

        if (marker >= 0) {
            toks.push_back(marker);
            if (st) (marker == TOK_CAP ? st->cap : st->up) += 1;
            for (unsigned char ch : lw) toks.push_back(ch);
        } else {
            if (st && first_upper) ++st->names;  // capitalized but not collapsed -> kept verbatim
            for (unsigned char ch : w) toks.push_back(ch);
        }
        i = j;
    }
    return toks;
}

// Inverse of truecase_tokenize: markers retroactively re-case the following
// alpha run. detokenize(truecase_tokenize(x)) == x for any text x.
inline std::string detokenize(const std::vector<int>& toks) {
    std::string out;
    int pending = 0;  // 0 none, 1 cap (first letter), 2 up (whole run)
    for (int t : toks) {
        if (t == TOK_CAP) { pending = 1; continue; }
        if (t == TOK_UP)  { pending = 2; continue; }
        const unsigned char c = static_cast<unsigned char>(t);
        if (pending == 1 && is_alpha(c)) {
            out.push_back(static_cast<char>(to_upper(c)));
            pending = 0;  // capitalize first letter only
        } else if (pending == 2) {
            if (is_alpha(c)) {
                out.push_back(static_cast<char>(to_upper(c)));
            } else {
                pending = 0;  // run ended
                out.push_back(static_cast<char>(c));
            }
        } else {
            out.push_back(static_cast<char>(c));
        }
    }
    return out;
}

}  // namespace sub0::casing
