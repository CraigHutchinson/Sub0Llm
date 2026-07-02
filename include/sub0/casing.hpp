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
#include <string_view>
#include <unordered_set>
#include <vector>

namespace sub0::casing {

// Explicit end-of-document marker: the literal `<|endoftext|>` (the standard GPT-2/3 stop
// signal), inserted by the corpus extraction scripts BETWEEN documents (scripts/get_fineweb.py,
// scripts/get_tinystories.py) -- an unambiguous boundary, unlike a bare blank line, which also
// occurs mid-document as an ordinary paragraph break. Without this the model was never trained
// on the pair "last real content token -> what comes next": sample_window's document-boundary
// cap means the last trainable target inside a document is the document's own final token, so a
// document literally has no "successor" in the training signal unless that final token IS this
// marker. Collapses to one token like PARA/NEWLINE; the configurator's doc-boundary detection
// prefers it over the "\n\n" heuristic when present (see tools/configurator.cpp), and gen stops
// generating when it samples this token instead of running to a fixed budget.
//
// Deliberately the FIRST marker (id 256, right after the complete 0..255 byte range), not
// appended after the others: this keeps base id == raw byte value for every byte token, with NO
// offset to account for -- so char-level content stays directly discoverable in a raw dump of
// corpus.tok (any token id < 256 IS that literal byte, no translation needed) and any future
// tooling that masks/filters by id range (`id < 256` = raw byte, `id >= 256` = a marker/learned
// piece) keeps working without adjustment. NUL (0x00, the classic C-string terminator -- the
// same "0 means end" intuition EOS echoes) stays its own ordinary byte token at id 0, distinct
// from eos_id (256): a stray NUL leaking into messy/binary-contaminated corpus data is NOT the
// same event as an intentional document boundary, so it must not be silently treated as one.
constexpr int TOK_EOS = 256;

// Symbol codes for the two case markers, carried in the byte-symbol stream just
// above the 0..255 byte range.
constexpr int TOK_CAP = 257;  // next word: capitalize first letter  (<|cap|>)
constexpr int TOK_UP  = 258;  // next word: upper-case the whole word (<|up|>)

// JOIN-scheme spacing markers (symbol codes above the byte + case-marker range). The
// implicit-space tokenizer makes a single inter-token space free; these specialise the
// rest (see docs/TOKENIZER_DESIGN.md). Only minted when the join scheme is enabled.
constexpr int TOK_JOIN    = 259;  // suppress the implicit inter-token space (glue / intra-word)
constexpr int TOK_NEWLINE = 260;  // a single '\n'
constexpr int TOK_PARA    = 261;  // a paragraph break "\n\n"
constexpr int TOK_ODQUOTE = 262;  // opening double quote: ` "` (space-before, glue-after)
constexpr int TOK_CDQUOTE = 263;  // closing double quote: `" ` (glue-before, space-after)
constexpr int TOK_SPELL_START = 264;  // start of a spaceless group (N>=3 sub-token word; OOV/acronym/CamelCase)
constexpr int TOK_SPELL_END   = 265;  // end of the spaceless group
// Run-length whitespace tokens: a single inter-word space is free (implicit), but multi-space
// runs (indentation, alignment) and tab runs otherwise cost one verbatim byte EACH. These tile
// such runs greedily (4s before 2s, remainder as a verbatim byte) -- 2/4 cover the dominant
// 2/4/8-wide indentation. The encoder only emits them for runs >= 2; a lone space/tab stays a
// byte. See docs/TOKENIZER_DESIGN.md §6.
constexpr int TOK_SPACE2 = 266;  // "  "   (two spaces)
constexpr int TOK_SPACE4 = 267;  // "    " (four spaces)
constexpr int TOK_TAB2   = 268;  // "\t\t"
constexpr int TOK_TAB4   = 269;  // "\t\t\t\t"

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
    // TODO(simd, hot): this scans every corpus byte for two rare lead bytes (0xE2 UTF-8
    // punct, 0x60 backtick). The fast path is "byte is neither" -- a vectorized scan
    // (find the next 0xE2/0x60 with SSE/AVX, bulk-copy the run between) would cut this to
    // near-memcpy speed on the >99% of bytes that pass through unchanged. Runs over the
    // whole corpus during configuration.
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

// An "interior connector": a byte that binds two word bytes into ONE unit when flanked by
// them on both sides. The apostrophe keeps "don't"/"Lily's" whole; the underscore and hyphen
// keep snake_case ("save_scan_state") and hyphenated compounds ("well-known",
// "non-commercial") whole so BPE merges across them instead of paying a JOIN per separator
// (each glued separator otherwise costs two JOIN tokens). A leading/trailing connector still
// splits off (it is not flanked). See docs/TOKENIZER_DESIGN.md §8.
inline bool is_interior_connector(int c) { return c == '\'' || c == '_' || c == '-'; }

// End (exclusive) of the word unit beginning at `s[i]`, or `i` itself if `s[i]`
// does not start one. A unit is a maximal run of word bytes (see is_word_byte)
// with interior connectors kept (see is_interior_connector).
inline std::size_t word_unit_end(const std::vector<int>& s, std::size_t i) {
    if (i >= s.size() || !is_word_byte(s[i])) return i;
    std::size_t j = i + 1;
    while (j < s.size()) {
        if (is_word_byte(s[j])) { ++j; continue; }
        if (is_interior_connector(s[j]) && j + 1 < s.size() &&
            is_word_byte(s[j - 1]) && is_word_byte(s[j + 1])) { ++j; continue; }
        break;
    }
    return j;
}

// Split a mixed-case alpha run into CamelCase / PascalCase segments at case transitions, so
// each piece reuses the lowercase BPE merges + a CAP/UP marker instead of shattering into
// near-character SPELL tokens (interior capitals are distinct bytes that never match the
// lowercase merges). Boundaries: a lowercase->uppercase hump ("aA", e.g. "myFunc" -> my|Func)
// and the last upper of an acronym run before a lowercase ("AAa", e.g. "HTMLParser" ->
// HTML|Parser). Returns the per-segment lengths; a single element means the run is not
// CamelCase-decomposable (a plain word/acronym handled by the normal path). See §8.
inline std::vector<std::size_t> camel_segments(std::string_view w) {
    std::vector<std::size_t> lens;
    const std::size_t n = w.size();
    std::size_t start = 0;
    for (std::size_t k = 0; k + 1 < n; ++k) {
        const unsigned char a = static_cast<unsigned char>(w[k]), b = static_cast<unsigned char>(w[k + 1]);
        const bool hump = is_lower(a) && is_upper(b);
        const bool acro = is_upper(a) && is_upper(b) &&
                          k + 2 < n && is_lower(static_cast<unsigned char>(w[k + 2]));
        if (hump || acro) { lens.push_back(k + 1 - start); start = k + 1; }
    }
    lens.push_back(n - start);
    return lens;
}

// Corpus-aware truecasing. Each alpha word is classified:
//   - all lowercase            -> emitted verbatim
//   - Capitalized ("The")      -> <|cap|> + lowercase, *iff* the lowercase form
//                                 is attested (appears all-lowercase elsewhere)
//   - ALL-UPPER  ("HELLO")     -> <|up|>  + lowercase, under the same condition
//   - otherwise (names, mixed) -> emitted verbatim, so "Lily"/"Tom" stay atomic
// Non-alpha bytes pass through unchanged. Output is a stream of byte values plus
// the TOK_CAP/TOK_UP marker codes. `st` may be null when stats are not wanted.
// TODO(simd/cuda, hot): the configurator runs normalize_text -> truecase_tokenize ->
// encode over the WHOLE corpus (GBs). The pipeline is per-byte but EMBARRASSINGLY PARALLEL
// across newline-aligned chunks (a newline never splits a word unit or the casing look-back),
// which the configurator already exploits with std::thread. Two future levers: (1) SIMD the
// alpha-run scan below (classify 16/32 bytes at once: is_alpha / is_upper masks drive the
// run boundaries); (2) a CUDA tokenizer -- one block per chunk -- since the merge table +
// attested set are read-only and the work is regular. Flagged for later (see TOKENIZER_REVIEW.md).
inline std::vector<int> truecase_tokenize(const std::string& text,
                                          const std::unordered_set<std::string>& attested,
                                          TokStats* st) {
    std::vector<int> toks;
    toks.reserve(text.size());
    const std::size_t n = text.size();

    // Emit ONE (sub-)word: classify its case and either collapse to a CAP/UP marker + the
    // lowercase form, or keep it verbatim. `force_collapse` is set for CamelCase segments
    // (the capital is structural word-boundary signal, not a name), so they always collapse
    // and reuse the lowercase merges; a top-level word collapses only if its lowercase form
    // is `attested` (a name-dominated form like "Spot" stays a distinct verbatim token).
    auto emit_word = [&](std::string_view seg, bool force_collapse) {
        const bool first_upper = is_upper(static_cast<unsigned char>(seg[0]));
        bool rest_lower = true, all_upper = true;
        for (std::size_t k = 0; k < seg.size(); ++k) {
            const unsigned char ch = static_cast<unsigned char>(seg[k]);
            if (!is_upper(ch))         all_upper = false;
            if (k > 0 && !is_lower(ch)) rest_lower = false;
        }
        std::string lw;
        lw.reserve(seg.size());
        for (unsigned char ch : seg) lw.push_back(static_cast<char>(to_lower(ch)));

        const bool capitalized = first_upper && rest_lower;     // "The", single "I", "Non"
        const bool upper       = all_upper && seg.size() >= 2;  // "HELLO", "HTML"
        const bool collapse    = force_collapse || attested.contains(lw);

        int marker = -1;
        if (upper && collapse)            marker = TOK_UP;
        else if (capitalized && collapse) marker = TOK_CAP;

        if (marker >= 0) {
            toks.push_back(marker);
            if (st) (marker == TOK_CAP ? st->cap : st->up) += 1;
            for (unsigned char ch : lw) toks.push_back(ch);
        } else {
            if (st && first_upper) ++st->names;  // capitalized but not collapsed -> kept verbatim
            for (unsigned char ch : seg) toks.push_back(ch);
        }
    };

    for (std::size_t i = 0; i < n;) {
        const unsigned char c = static_cast<unsigned char>(text[i]);
        if (!is_alpha(c)) { toks.push_back(c); ++i; continue; }

        std::size_t j = i;
        bool any_upper = false;
        while (j < n && is_alpha(static_cast<unsigned char>(text[j]))) {
            if (is_upper(static_cast<unsigned char>(text[j]))) any_upper = true;
            ++j;
        }
        const std::string_view w(text.data() + i, j - i);
        if (st) ++st->words;

        if (!any_upper) {                                       // all-lowercase fast path
            for (unsigned char ch : w) toks.push_back(ch);
        } else if (const auto segs = camel_segments(w); segs.size() >= 2) {
            std::size_t off = 0;                                // CamelCase -> one collapsed word per segment
            for (std::size_t len : segs) { emit_word(w.substr(off, len), /*force_collapse=*/true); off += len; }
        } else {
            emit_word(w, /*force_collapse=*/false);             // plain word / acronym (attestation-gated)
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
