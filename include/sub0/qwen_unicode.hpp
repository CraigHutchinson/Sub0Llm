// sub0/qwen_unicode.hpp -- the Unicode primitives the REAL Qwen tokenizer's pipeline needs:
// UTF-8 stepping, the four character classes its pre-tokenization regex tests, and NFC.
//
// SCOPE. This exists for ONE consumer -- sub0/qwen_tokenizer.hpp, this engine's reimplementation of
// Qwen3.8-Flash-Next's own byte-level BPE tokenizer (docs/QWEN_TOKENIZER.md). It is deliberately NOT
// a general Unicode library and deliberately NOT wired into this project's OWN tokenizer
// (sub0/tokenizer.hpp + sub0/casing.hpp), which has a different, from-scratch scheme and must keep
// behaving exactly as it does today. Nothing here is used by any non-Qwen build.
//
// WHY IT EXISTS AT ALL (the real obstacle, docs/QWEN_TOKENIZER.md sec 3): the real pre-tokenization
// regex is written with Unicode property escapes -- \p{L}, \p{N}, \p{M} -- and with a negative
// lookahead, neither of which C++'s <regex> supports in ANY standard-library implementation. There
// is no "just use std::regex" option, and approximating the classes with ASCII tests silently
// mis-tokenizes every non-English input. So the regex is transliterated by hand into
// qwen_tokenizer.hpp's pretokenize(), and this header supplies the class tests it needs.
//
// REUSE. UTF-8 stepping delegates to sub0::modality's utf8_len/utf8_cp (include/sub0/modality.hpp)
// rather than introducing a second decoder into this codebase; decode_utf8 below only ADDS the
// continuation-byte validation that a tokenizer facing arbitrary user input needs and that
// modality's corpus-scanning use never did.
//
// Engine-free (std + modality.hpp + the generated table header only), so it lives in sub0_frontend
// and is unit-testable with no compiled model, exactly like gguf.hpp and transplant.hpp.

#pragma once

#include "sub0/modality.hpp"             // utf8_len / utf8_cp -- the codebase's existing decoder
#include "sub0/qwen_unicode_tables.hpp"  // generated data; see its own header comment

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace sub0::qwen_uni {

// U+FFFD REPLACEMENT CHARACTER, as UTF-8. Emitted by decode_lossy for malformed byte sequences,
// matching the reference decoder's own `errors="replace"` (tokenizer_config.json's `errors` field).
inline constexpr std::string_view kReplacement = "\xEF\xBF\xBD";

// One decoded codepoint plus how many BYTES it occupied.
struct Cp {
    std::uint32_t cp   = 0;
    std::size_t   size = 1;
};

// Decode the codepoint at `i`. VALID UTF-8 is decoded via the existing sub0::modality decoder; a
// malformed sequence (truncated, or a lead byte not followed by the right number of continuation
// bytes) yields the single raw byte value with size 1, so the caller always makes progress and
// never reads past the end. The reference pipeline is fed a Rust `str`, i.e. UTF-8 by construction,
// so this fallback is OUTSIDE the reference-gated behaviour -- it exists so that a malformed prompt
// degrades predictably (each stray byte becomes its own pre-token chunk and then its own byte-level
// token) instead of reading out of bounds. See docs/QWEN_TOKENIZER.md sec 6.
inline Cp decode_utf8(std::string_view s, std::size_t i) {
    const auto b0 = static_cast<unsigned char>(s[i]);
    const int  n  = modality::utf8_len(b0);
    if (n == 1 || i + static_cast<std::size_t>(n) > s.size()) return {b0, 1};
    for (int k = 1; k < n; ++k)
        if ((static_cast<unsigned char>(s[i + static_cast<std::size_t>(k)]) & 0xC0) != 0x80) return {b0, 1};
    return {modality::utf8_cp(s, i, n), static_cast<std::size_t>(n)};
}

// Append one codepoint as UTF-8. (No encoder existed in the codebase -- modality.hpp only decodes.)
inline void append_utf8(std::string& out, std::uint32_t cp) {
    if (cp < 0x80) {
        out.push_back(static_cast<char>(cp));
    } else if (cp < 0x800) {
        out.push_back(static_cast<char>(0xC0 | (cp >> 6)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    } else if (cp < 0x10000) {
        out.push_back(static_cast<char>(0xE0 | (cp >> 12)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    } else {
        out.push_back(static_cast<char>(0xF0 | (cp >> 18)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    }
}

// --- character classes -------------------------------------------------------------------------
// Binary search over the generated sorted [lo, hi] range pairs. These four predicates are the ONLY
// thing the pre-tokenization regex asks about a codepoint; each was verified against the real
// reference engine for every non-surrogate codepoint (see qwen_unicode_tables.hpp).
inline bool in_ranges(const std::uint32_t* r, std::size_t n, std::uint32_t cp) {
    std::size_t lo = 0, hi = n;
    while (lo < hi) {
        const std::size_t mid = lo + (hi - lo) / 2;
        if (cp < r[2 * mid])            hi = mid;
        else if (cp > r[2 * mid + 1])   lo = mid + 1;
        else                            return true;
    }
    return false;
}

inline bool is_letter(std::uint32_t cp) { return in_ranges(tables::kLetter, tables::kLetterN, cp); }
inline bool is_number(std::uint32_t cp) { return in_ranges(tables::kNumber, tables::kNumberN, cp); }
inline bool is_mark(std::uint32_t cp)   { return in_ranges(tables::kMark,   tables::kMarkN,   cp); }
// The regex's `\s`. Deliberately the full Unicode White_Space property, NOT the ASCII six: the
// reference matches U+00A0, U+2000..U+200A, U+3000 and friends, confirmed by the codepoint sweep.
inline bool is_space(std::uint32_t cp)  { return in_ranges(tables::kWhiteSpace, tables::kWhiteSpaceN, cp); }

// --- NFC ---------------------------------------------------------------------------------------
// Hangul is composed/decomposed arithmetically (Unicode 3.12, "Hangul Syllable Composition"), so
// none of its 11,172 syllables appear in the decomposition table.
inline constexpr std::uint32_t kHangulSBase = 0xAC00, kHangulLBase = 0x1100;
inline constexpr std::uint32_t kHangulVBase = 0x1161, kHangulTBase = 0x11A7;
inline constexpr std::uint32_t kHangulLCount = 19, kHangulVCount = 21, kHangulTCount = 28;
inline constexpr std::uint32_t kHangulNCount = kHangulVCount * kHangulTCount;         // 588
inline constexpr std::uint32_t kHangulSCount = kHangulLCount * kHangulNCount;         // 11172

inline int combining_class(std::uint32_t cp) {
    std::size_t lo = 0, hi = tables::kCccN;
    while (lo < hi) {
        const std::size_t mid = lo + (hi - lo) / 2;
        if (cp < tables::kCccCp[mid])       hi = mid;
        else if (cp > tables::kCccCp[mid])  lo = mid + 1;
        else                                return tables::kCccVal[mid];
    }
    return 0;
}

// Fully (recursively) decompose `cp`, appending to `out`. Returns false if `cp` has no canonical
// decomposition, in which case nothing is appended.
inline bool canonical_decompose(std::uint32_t cp, std::vector<std::uint32_t>& out) {
    if (cp - kHangulSBase < kHangulSCount) {          // unsigned wrap makes this the range test
        const std::uint32_t s = cp - kHangulSBase, t = s % kHangulTCount;
        out.push_back(kHangulLBase + s / kHangulNCount);
        out.push_back(kHangulVBase + (s % kHangulNCount) / kHangulTCount);
        if (t) out.push_back(kHangulTBase + t);
        return true;
    }
    std::size_t lo = 0, hi = tables::kDecompN;
    while (lo < hi) {
        const std::size_t mid = lo + (hi - lo) / 2;
        if (cp < tables::kDecompCp[mid])       hi = mid;
        else if (cp > tables::kDecompCp[mid])  lo = mid + 1;
        else {
            for (std::uint32_t k = tables::kDecompOff[mid]; k < tables::kDecompOff[mid + 1]; ++k)
                out.push_back(tables::kDecompSeq[k]);
            return true;
        }
    }
    return false;
}

// The primary composite of (a, b), or 0 if the pair does not compose (including every pair barred
// by the Composition_Exclusion / non-starter-decomposition rules -- those are already absent from
// the generated table, which is why this needs no exclusion test of its own).
inline std::uint32_t canonical_compose(std::uint32_t a, std::uint32_t b) {
    if (a - kHangulLBase < kHangulLCount && b - kHangulVBase < kHangulVCount)
        return kHangulSBase + ((a - kHangulLBase) * kHangulVCount + (b - kHangulVBase)) * kHangulTCount;
    if (a - kHangulSBase < kHangulSCount && (a - kHangulSBase) % kHangulTCount == 0 &&
        b - kHangulTBase < kHangulTCount && b != kHangulTBase)
        return a + (b - kHangulTBase);
    const std::uint64_t key = (static_cast<std::uint64_t>(a) << 32) | b;
    std::size_t lo = 0, hi = tables::kComposeN;
    while (lo < hi) {
        const std::size_t mid = lo + (hi - lo) / 2;
        if (key < tables::kComposeKey[mid])       hi = mid;
        else if (key > tables::kComposeKey[mid])  lo = mid + 1;
        else                                      return tables::kComposeVal[mid];
    }
    return 0;
}

// Unicode Normalization Form C, over a UTF-8 string.
//
// The textbook three phases (UAX #15): full canonical decomposition, canonical ordering (a STABLE
// sort of each maximal non-starter run by combining class), then canonical composition (walk the
// starters, absorbing every following character that composes with the last starter, subject to the
// blocking rule). Written out rather than pulled from ICU because the only thing this project needs
// Unicode for is this one function, and an ICU dependency for it would dwarf the engine.
//
// Malformed UTF-8 passes through byte-for-byte (decode_utf8's fallback re-encodes each stray byte as
// its own codepoint, which is not byte-identical for bytes >= 0x80) -- see docs/QWEN_TOKENIZER.md
// sec 6. Callers that must preserve arbitrary bytes should not normalize.
inline std::string nfc(std::string_view in) {
    // Fast path: NFC is the identity for pure ASCII, which is the overwhelming majority of real
    // prompts, and skipping the decompose/sort/compose machinery for it is free.
    bool ascii = true;
    for (char c : in)
        if (static_cast<unsigned char>(c) >= 0x80) { ascii = false; break; }
    if (ascii) return std::string(in);

    std::vector<std::uint32_t> buf;
    buf.reserve(in.size());
    for (std::size_t i = 0; i < in.size();) {
        const Cp c = decode_utf8(in, i);
        i += c.size;
        if (!canonical_decompose(c.cp, buf)) buf.push_back(c.cp);
    }

    // Canonical ordering: insertion-sort each maximal run of non-starters by combining class. It is
    // a STABLE sort by definition of the algorithm, and the runs are a handful of marks long, so an
    // insertion sort is both the correct shape and the fast one.
    for (std::size_t i = 1; i < buf.size(); ++i) {
        const int cc = combining_class(buf[i]);
        if (cc == 0) continue;
        std::size_t j = i;
        while (j > 0) {
            const int prev = combining_class(buf[j - 1]);
            if (prev == 0 || prev <= cc) break;
            std::swap(buf[j - 1], buf[j]);
            --j;
        }
    }

    // Canonical composition. `last` indexes the most recent starter in the OUTPUT; `last_cc` is the
    // combining class of the last character KEPT after it (0 while nothing has been). A character C
    // is BLOCKED from that starter if anything still standing between them has a combining class >=
    // ccc(C) (UAX #15). Tracking only the last one is sufficient *because* the preceding pass left
    // each non-starter run sorted ascending by class, so the last survivor is the run's maximum; a
    // character that got absorbed into the starter is no longer "between" anything.
    std::string out;
    out.reserve(in.size());
    std::vector<std::uint32_t> res;
    res.reserve(buf.size());
    constexpr std::size_t kNone = static_cast<std::size_t>(-1);
    std::size_t last = kNone;
    int         last_cc = 0;
    for (std::uint32_t cp : buf) {
        const int cc = combining_class(cp);
        if (last != kNone && (last_cc == 0 || last_cc < cc)) {
            if (const std::uint32_t comp = canonical_compose(res[last], cp)) {
                res[last] = comp;
                continue;   // absorbed: the composite IS the starter, so `last_cc` is unchanged
            }
        }
        if (cc == 0) last = res.size();
        last_cc = cc;
        res.push_back(cp);
    }
    for (std::uint32_t cp : res) append_utf8(out, cp);
    return out;
}

// UTF-8 validation with replacement, matching the reference decoder's `errors="replace"`: every
// maximal ill-formed subsequence becomes ONE U+FFFD (the Unicode-recommended practice, UAX #16 /
// the WHATWG decoder both agree with Rust's String::from_utf8_lossy here). This is what makes
// decoding a TRUNCATED id sequence -- one that stops mid-character -- behave the same as the
// reference instead of emitting a raw partial byte.
inline std::string decode_lossy(std::string_view bytes) {
    std::string out;
    out.reserve(bytes.size());
    std::size_t i = 0;
    while (i < bytes.size()) {
        const auto b0 = static_cast<unsigned char>(bytes[i]);
        if (b0 < 0x80) { out.push_back(static_cast<char>(b0)); ++i; continue; }
        // Well-formedness per the Unicode table 3-7 ranges: rejects overlongs, surrogates and
        // anything above U+10FFFF, which a plain "lead byte says N, take N" reader would accept.
        std::size_t n = 0, lo = 0, hi = 0;
        if (b0 >= 0xC2 && b0 <= 0xDF)      { n = 2; lo = 0x80; hi = 0xBF; }
        else if (b0 == 0xE0)               { n = 3; lo = 0xA0; hi = 0xBF; }
        else if (b0 >= 0xE1 && b0 <= 0xEC) { n = 3; lo = 0x80; hi = 0xBF; }
        else if (b0 == 0xED)               { n = 3; lo = 0x80; hi = 0x9F; }
        else if (b0 >= 0xEE && b0 <= 0xEF) { n = 3; lo = 0x80; hi = 0xBF; }
        else if (b0 == 0xF0)               { n = 4; lo = 0x90; hi = 0xBF; }
        else if (b0 >= 0xF1 && b0 <= 0xF3) { n = 4; lo = 0x80; hi = 0xBF; }
        else if (b0 == 0xF4)               { n = 4; lo = 0x80; hi = 0x8F; }
        else                               { out += kReplacement; ++i; continue; }

        std::size_t k = 1;
        for (; k < n; ++k) {
            if (i + k >= bytes.size()) break;
            const auto b = static_cast<unsigned char>(bytes[i + k]);
            const std::size_t l = (k == 1) ? lo : 0x80, h = (k == 1) ? hi : 0xBF;
            if (b < l || b > h) break;
        }
        if (k == n) { out.append(bytes.substr(i, n)); i += n; }
        else        { out += kReplacement; i += k; }   // ONE U+FFFD per maximal ill-formed run
    }
    return out;
}

}  // namespace sub0::qwen_uni
