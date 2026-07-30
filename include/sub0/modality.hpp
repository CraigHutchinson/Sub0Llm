// sub0/modality.hpp — corpus spacing-modality calibration (engine-free).
//
// Accumulates, per non-alphanumeric CODEPOINT, how its neighbours are spaced across a corpus:
// space/glue BEFORE x space/glue AFTER (SS/SG/GS/GG). This is the ground-truth signal a v2 tokenizer
// needs to decide, per character, whether one baked "default spacing" suffices (a UNIMODAL char) or a
// dual open/close token is justified (a BIMODAL char) -- see docs/TOKENIZER_V2_IDEAS.md §4a/§4b.
//
// Built to COLLATE across many corpora: ModalityStats merges additively (order-free, like tok::Scan),
// serialises to a human-readable ledger (cf. data/tokenizer_calibration.txt), and can FLAG when a
// fresh corpus's dominant modality for a character CONTRADICTS the accumulated finding -- so a
// disagreeing corpus surfaces the mismatch instead of being silently averaged away, and the next
// scheme version can be derived from a decisive, multi-corpus picture.
//
// Engine-free (depends only on casing.hpp), so it lives in sub0_frontend with the rest of the
// tokenizer and is exercised by the engine-free tests.
#pragma once

#include "sub0/casing.hpp"

#include <array>
#include <cstdint>
#include <cstdlib>
#include <istream>
#include <map>
#include <ostream>
#include <string>
#include <string_view>
#include <vector>

namespace sub0::modality {

// Spacing combo: bit1 = glued-before, bit0 = glued-after.
enum Combo : int { SS = 0, SG = 1, GS = 2, GG = 3 };   // space/glue x before/after
inline const char* combo_name(int c) {
    switch (c) {
        case SS: return "SS-isolated";
        case SG: return "SG-opener";
        case GS: return "GS-closer";
        default: return "GG-interior";
    }
}

struct CharModality {
    std::array<std::uint64_t, 4> n{};   // counts indexed by Combo
    std::uint64_t total() const { return n[0] + n[1] + n[2] + n[3]; }
    int dominant() const { int d = 0; for (int k = 1; k < 4; ++k) if (n[k] > n[d]) d = k; return d; }
    // Share of the SECOND-largest combo; >= 0.25 => BIMODAL (no single default captures the char).
    double second_share() const {
        const std::uint64_t t = total();
        if (!t) return 0.0;
        const int d = dominant();
        int s = -1;
        for (int k = 0; k < 4; ++k) if (k != d && (s < 0 || n[k] > n[s])) s = k;
        return s < 0 ? 0.0 : static_cast<double>(n[s]) / static_cast<double>(t);
    }
    bool bimodal() const { return second_share() >= 0.25; }
};

// Per-codepoint modality counts plus a running byte total. std::map keeps codepoints sorted for a
// stable, diff-friendly ledger.
struct ModalityStats {
    std::map<std::uint32_t, CharModality> chars;
    std::uint64_t scanned_bytes = 0;
};

// --- UTF-8 decode ---
inline int utf8_len(unsigned char b) {
    if (b < 0x80) return 1;
    if ((b & 0xE0) == 0xC0) return 2;
    if ((b & 0xF0) == 0xE0) return 3;
    if ((b & 0xF8) == 0xF0) return 4;
    return 1;   // stray continuation / invalid lead: consume one byte
}
inline std::uint32_t utf8_cp(std::string_view s, std::size_t i, int len) {
    const unsigned char b0 = static_cast<unsigned char>(s[i]);
    if (len == 1) return b0;
    std::uint32_t cp = static_cast<std::uint32_t>(b0 & (0xFF >> (len + 1)));
    for (int k = 1; k < len; ++k) cp = (cp << 6) | (static_cast<unsigned char>(s[i + static_cast<std::size_t>(k)]) & 0x3F);
    return cp;
}

inline bool ascii_ws(unsigned char c) { return c == ' ' || c == '\t' || c == '\n' || c == '\r'; }

// A codepoint that is word MATERIAL (letter/digit) -- EXCLUDED from the punctuation/symbol scan.
// ASCII alnum plus the major Unicode letter blocks; everything else non-space (currency, maths,
// punctuation, box-drawing, emoji, ...) is a "symbol" we DO tally. Deliberately a pragmatic range
// check, not full UCD: it only has to keep the letter FLOOD (accented Latin, Greek, Cyrillic, CJK,
// Hangul, ...) out of the ledger so real symbols like £/€/©/° stay visible. Misclassifying a rare
// letter-ish codepoint as a symbol is harmless -- it just gets a low-count row.
inline bool is_word_cp(std::uint32_t cp) {
    if (cp < 0x80)
        return (cp >= 'a' && cp <= 'z') || (cp >= 'A' && cp <= 'Z') || (cp >= '0' && cp <= '9');
    return (cp >= 0x00C0 && cp <= 0x00FF && cp != 0x00D7 && cp != 0x00F7) ||  // Latin-1 letters (minus × ÷)
           (cp >= 0x0100 && cp <= 0x024F) ||   // Latin Extended-A/B
           (cp >= 0x0370 && cp <= 0x03FF) ||   // Greek
           (cp >= 0x0400 && cp <= 0x04FF) ||   // Cyrillic
           (cp >= 0x0590 && cp <= 0x05FF) ||   // Hebrew
           (cp >= 0x0600 && cp <= 0x06FF) ||   // Arabic
           (cp >= 0x3040 && cp <= 0x30FF) ||   // Hiragana/Katakana
           (cp >= 0x3400 && cp <= 0x9FFF) ||   // CJK Unified
           (cp >= 0xAC00 && cp <= 0xD7A3) ||   // Hangul syllables
           (cp >= 0xF900 && cp <= 0xFAFF);      // CJK compatibility
}

// Scan one chunk (normalize_text it first for the tokenizer's real view, or pass raw to see pre-fold
// glyphs). Tallies every non-word, non-whitespace codepoint's neighbour spacing at the BYTE boundary
// (a neighbour is "glued" iff the adjacent byte is not ASCII whitespace). Stream newline-aligned
// chunks: a codepoint split across a chunk boundary is a negligible miscount over a real corpus.
inline void add_modality(ModalityStats& st, std::string_view chunk) {
    st.scanned_bytes += chunk.size();
    const std::size_t n = chunk.size();
    std::size_t i = 0;
    while (i < n) {
        const unsigned char b = static_cast<unsigned char>(chunk[i]);
        int len = utf8_len(b);
        if (i + static_cast<std::size_t>(len) > n) len = 1;   // truncated multibyte at chunk end
        const std::uint32_t cp = utf8_cp(chunk, i, len);
        const bool control = cp < 0x20 || (cp >= 0x7F && cp <= 0x9F);   // C0/C1 controls: not characters
        if (!ascii_ws(b) && !control && !is_word_cp(cp)) {
            const bool gb = (i > 0) && !ascii_ws(static_cast<unsigned char>(chunk[i - 1]));
            const std::size_t after = i + static_cast<std::size_t>(len);
            const bool ga = (after < n) && !ascii_ws(static_cast<unsigned char>(chunk[after]));
            ++st.chars[cp].n[(gb ? 2 : 0) | (ga ? 1 : 0)];
        }
        i += static_cast<std::size_t>(len);
    }
}

// Fold `other` into `into` (additive, commutative -- collating corpora in any order gives the same
// ledger, exactly like tok::Scan's merges).
inline void merge(ModalityStats& into, const ModalityStats& other) {
    into.scanned_bytes += other.scanned_bytes;
    for (const auto& [cp, cm] : other.chars) {
        auto& dst = into.chars[cp];
        for (int k = 0; k < 4; ++k) dst.n[k] += cm.n[k];
    }
}

// --- serialize / load: a mergeable, human-readable ledger (cf. data/tokenizer_calibration.txt) ---
inline void serialize(const ModalityStats& st, std::ostream& os) {
    os << "# sub0 modality calibration ledger -- per-codepoint spacing modality, accumulated over corpora.\n";
    os << "# Merging a corpus ADDS counts; a fresh corpus whose dominant modality flips vs this ledger is flagged.\n";
    os << "# fields: U+HEX <TAB> SS <TAB> SG <TAB> GS <TAB> GG   (SG=opener, GS=closer, GG=interior, SS=isolated)\n";
    os << "scanned_bytes\t" << st.scanned_bytes << "\n";
    for (const auto& [cp, cm] : st.chars)
        os << "U+" << std::hex << std::uppercase << cp << std::dec
           << "\t" << cm.n[0] << "\t" << cm.n[1] << "\t" << cm.n[2] << "\t" << cm.n[3] << "\n";
}

inline bool deserialize(ModalityStats& st, std::istream& is) {
    st = ModalityStats{};
    std::string line;
    while (std::getline(is, line)) {
        if (line.empty() || line[0] == '#') continue;
        if (line.rfind("scanned_bytes\t", 0) == 0) {
            st.scanned_bytes = std::strtoull(line.c_str() + 14, nullptr, 10);
            continue;
        }
        if (line.rfind("U+", 0) != 0) continue;
        char* end = nullptr;
        const std::uint32_t cp = static_cast<std::uint32_t>(std::strtoul(line.c_str() + 2, &end, 16));
        CharModality cm{};
        for (int k = 0; k < 4 && end && *end; ++k) cm.n[k] = std::strtoull(end, &end, 10);
        st.chars[cp] = cm;
    }
    return true;
}

// A codepoint whose DOMINANT modality in `fresh` disagrees with the accumulated `prior` (both with
// enough samples to be meaningful). This is the mismatch signal: a corpus that would pull the derived
// default in a different direction, surfaced rather than averaged away.
struct Contradiction {
    std::uint32_t cp;
    int           prior_dom, fresh_dom;
    std::uint64_t prior_n, fresh_n;
};

inline std::vector<Contradiction> find_contradictions(const ModalityStats& prior, const ModalityStats& fresh,
                                                      std::uint64_t min_samples = 500) {
    std::vector<Contradiction> out;
    for (const auto& [cp, fcm] : fresh.chars) {
        if (fcm.total() < min_samples) continue;
        const auto it = prior.chars.find(cp);
        if (it == prior.chars.end() || it->second.total() < min_samples) continue;
        const int pd = it->second.dominant(), fd = fcm.dominant();
        if (pd != fd) out.push_back({cp, pd, fd, it->second.total(), fcm.total()});
    }
    return out;
}

}  // namespace sub0::modality
