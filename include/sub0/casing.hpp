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

// JOIN-scheme spacing markers (symbol codes above the byte + case-marker range). The
// implicit-space tokenizer makes a single inter-token space free; these specialise the
// rest (see docs/TOKENIZER_DESIGN.md). Only minted when the join scheme is enabled.
constexpr int TOK_JOIN    = 258;  // suppress the implicit inter-token space (glue / intra-word)
constexpr int TOK_NEWLINE = 259;  // a single '\n'
constexpr int TOK_PARA    = 260;  // a paragraph break "\n\n"

inline bool          is_alpha(unsigned char c) { return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z'); }
inline bool          is_lower(unsigned char c) { return c >= 'a' && c <= 'z'; }
inline bool          is_upper(unsigned char c) { return c >= 'A' && c <= 'Z'; }
inline bool          is_space(unsigned char c) { return c == ' ' || c == '\n' || c == '\t' || c == '\r'; }
inline unsigned char to_lower(unsigned char c) { return is_upper(c) ? static_cast<unsigned char>(c + 32) : c; }
inline unsigned char to_upper(unsigned char c) { return is_lower(c) ? static_cast<unsigned char>(c - 32) : c; }

// A "word byte" for pre-tokenization: ASCII letters plus any UTF-8 multibyte byte
// (>= 0x80). Treating the continuation/lead bytes of accented letters as word
// material keeps loanwords like "piñata" or "café" as one BPE unit instead of
// shattering them at the accent. (Typographic punctuation -- curly quotes, dashes
// -- is folded to ASCII by normalize_text first, so the only multibyte sequences
// left in the stream are genuine letters and a few rare symbols.) Markers (>= 256)
// are deliberately excluded so they stay atomic case operators.
inline bool is_word_byte(int s) {
    return s >= 0 && s <= 0xFF &&
           (is_alpha(static_cast<unsigned char>(s)) || s >= 0x80);
}

// Per-corpus truecasing statistics (configurator reporting only).
struct TokStats {
    long long words = 0;  // alpha runs seen
    long long cap   = 0;  // collapsed to <|cap|> + lowercase
    long long up    = 0;  // collapsed to <|up|>  + lowercase
    long long names = 0;  // capitalized/upper words left verbatim (corpus-aware) (64-bit: a large corpus has >2^31 words)
};

// Fold "fancy" typographic glyphs to their ASCII equivalents. The corpus carries
// curly quotes, curly apostrophes (inside contractions like "don't"), en/em dashes
// and ellipses as UTF-8 multibyte sequences; left alone each one fragments into 2-3
// standalone byte tokens and splits the word it touches. Mapping them to ASCII
// removes those fragments and lets, e.g., "don't" stay a single unit. Lossy by
// design — the specific glyph identity (curly vs straight, en vs em) is discarded;
// the round-trip contract is decode(encode(x)) == normalize_text(x), not == x.
//   U+2018/U+2019 ' '  -> '      U+201C/U+201D " "  -> "
//   U+2013/U+2014 – —  -> -      U+2026 …            -> ...      ASCII ` -> '
inline std::string normalize_text(const std::string& in, long& replaced) {
    std::string out;
    out.reserve(in.size());
    replaced = 0;
    const std::size_t n = in.size();
    for (std::size_t i = 0; i < n;) {
        const unsigned char c = static_cast<unsigned char>(in[i]);
        // General-punctuation block U+2013..U+2026 is encoded E2 80 xx.
        if (c == 0xE2 && i + 2 < n && static_cast<unsigned char>(in[i + 1]) == 0x80) {
            const unsigned char t = static_cast<unsigned char>(in[i + 2]);
            const char* rep = nullptr;
            switch (t) {
                case 0x98: case 0x99: rep = "'";   break;  // ‘ ’ single quotes / apostrophe
                case 0x9C: case 0x9D: rep = "\"";  break;  // “ ” double quotes
                case 0x93: case 0x94: rep = "-";   break;  // – — dashes
                case 0xA6:            rep = "...";  break;  // … ellipsis
                default: break;
            }
            if (rep) { out += rep; ++replaced; i += 3; continue; }
        }
        if (c == '`') { out.push_back('\''); ++replaced; ++i; continue; }  // ASCII backtick
        out.push_back(static_cast<char>(c));
        ++i;
    }
    return out;
}

// End (exclusive) of the word unit beginning at `s[i]`, or `i` itself if `s[i]`
// does not start one. A unit is a maximal run of word bytes (see is_word_byte)
// with interior apostrophes kept — an apostrophe flanked by word bytes on both
// sides, so "don't"/"Lily's" stay whole while a leading/trailing ' splits off.
inline std::size_t word_unit_end(const std::vector<int>& s, std::size_t i) {
    if (i >= s.size() || !is_word_byte(s[i])) return i;
    std::size_t j = i + 1;
    while (j < s.size()) {
        if (is_word_byte(s[j])) { ++j; continue; }
        if (s[j] == '\'' && j + 1 < s.size() &&
            is_word_byte(s[j - 1]) && is_word_byte(s[j + 1])) { ++j; continue; }
        break;
    }
    return j;
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
