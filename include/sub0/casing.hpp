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

#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <unordered_set>
#include <vector>

namespace sub0::casing {

// The fixed token-id scheme: the complete 256-byte range (0..255, base id == byte value) plus
// these named markers. Unscoped + auto-incrementing (only TOK_EOS has an explicit value, the
// rest follow): auto-increment is what makes a duplicate value a compile-time impossibility
// (plain `constexpr int`/`enum class` with hand-typed values does not -- C++ only diagnoses
// duplicate *names*), and unscoped keeps every existing plain-`int` comparison site (elsewhere in
// this file, tokenizer.cpp, engine_core.cpp) compiling unchanged -- see
// docs/TOKENIZER_REVIEW.md §5.4/§5.5 for the full reasoning.
enum TokenId : int {
    // Deliberately the FIRST marker (id 256, right after the complete 0..255 byte range), not
    // appended after the others: this keeps base id == raw byte value for every byte token, with
    // NO offset to account for -- so char-level content stays directly discoverable in a raw dump
    // of corpus.tok (any token id < 256 IS that literal byte, no translation needed) and any
    // future tooling that masks/filters by id range (`id < 256` = raw byte, `id >= 256` = a
    // marker/learned piece) keeps working without adjustment. NUL (0x00, the classic C-string
    // terminator -- the same "0 means end" intuition EOS echoes) stays its own ordinary byte token
    // at id 0, distinct from TOK_EOS (256): a stray NUL leaking into messy/binary-contaminated
    // corpus data is NOT the same event as an intentional document boundary, so it must not be
    // silently treated as one.
    //
    // Explicit end-of-document marker: the literal `<|endoftext|>` (the standard GPT-2/3 stop
    // signal), inserted by the corpus extraction scripts BETWEEN documents
    // (scripts/get_fineweb.py, scripts/get_tinystories.py) -- an unambiguous boundary, unlike a
    // bare blank line, which also occurs mid-document as an ordinary paragraph break. Without
    // this the model was never trained on the pair "last real content token -> what comes next":
    // sample_window's document-boundary cap means the last trainable target inside a document is
    // the document's own final token, so a document literally has no "successor" in the training
    // signal unless that final token IS this marker. Collapses to one token like PARA/NEWLINE;
    // the configurator's doc-boundary detection prefers it over the "\n\n" heuristic when present
    // (see tools/configurator.cpp), and gen stops generating when it samples this token instead
    // of running to a fixed budget.
    TOK_EOS = 256,

    // The two case markers, carried in the byte-symbol stream just above the 0..255 byte range.
    TOK_CAP,  // next word: capitalize first letter  (<|cap|>)
    TOK_UP,   // next word: upper-case the whole word (<|up|>)

    // JOIN-scheme spacing markers (symbol codes above the byte + case-marker range). The
    // implicit-space tokenizer makes a single inter-token space free; these specialise the
    // rest (see docs/TOKENIZER_DESIGN.md).
    TOK_JOIN,         // suppress the implicit inter-token space (glue / intra-word)
    TOK_NEWLINE,      // a single '\n'
    TOK_PARA,         // a paragraph break "\n\n"
    TOK_ODQUOTE,      // opening double quote: ` "` (space-before, glue-after)
    TOK_CDQUOTE,      // closing double quote: `" ` (glue-before, space-after)
    TOK_SPELL_START,  // start of a spaceless group (N>=3 sub-token word; OOV/acronym/CamelCase)
    TOK_SPELL_END,    // end of the spaceless group

    // Run-length whitespace tokens: a single inter-word space is free (implicit), but multi-space
    // runs (indentation, alignment) and tab runs otherwise cost one verbatim byte EACH. These
    // tile such runs greedily (4s before 2s, remainder as a verbatim byte) -- 2/4 cover the
    // dominant 2/4/8-wide indentation. The encoder only emits them for runs >= 2; a lone
    // space/tab stays a byte. See docs/TOKENIZER_DESIGN.md §6.
    TOK_SPACE2,  // "  "   (two spaces)
    TOK_SPACE4,  // "    " (four spaces)
    TOK_TAB2,    // "\t\t"
    TOK_TAB4,    // "\t\t\t\t"

    // Conversational turn boundaries (Stage 2). Deliberately just 2 markers, not one per role:
    // matches both ChatML (`<|im_start|>`/`<|im_end|>`) and Gemma (`<start_of_turn>`/
    // `<end_of_turn>`) -- neither mints a per-role token id, the role name ("system"/"user"/
    // "assistant"/"model"/"tool") flows through as ordinary text right after TOK_TURN_START, which
    // the existing Unigram vocab already tokenizes fine with no new mechanism. Literal strings
    // adopted verbatim from ChatML (not invented) so corpora that already ship in ChatML (much of
    // SmolTalk/UltraChat/OASST) need no reformatting, and any future GGUF/HF export stays
    // interoperable. TOK_TURN_END is kept distinct from TOK_EOS (not reused) even though both can
    // end a document: matches ChatML/Llama-3 precedent and future instruction-tuning loss-masking
    // needs a clean assistant-span boundary that a document-boundary marker can't double as. See
    // docs/TOKENIZER_REVIEW.md §5.8.
    TOK_TURN_START,  // `<|im_start|>` -- glued to the role word that follows (no space)
    TOK_TURN_END,    // `<|im_end|>`   -- glued to the content that precedes it (no space)

    // WS5b: bracket-glue markers (6 of the 16 reserved slots, RENAMED not inserted -- see the
    // reserved-headroom comment this replaces below; n_base/TOK_MARKER_COUNT are unchanged by this,
    // only kSchemeVersion bumps, since the base alphabet layout didn't move). `(` `[` `{` `)` `]`
    // `}` are already unambiguous distinct bytes (unlike `"`), so unlike the quote markers these
    // don't exist to disambiguate direction -- they exist purely to collapse the JOIN tax:
    // `f(x)` cost 7 tokens (f,JOIN,(,JOIN,x,JOIN,)) before this, because EVERY zero-gap adjacency
    // pays a JOIN. Each open-bracket marker fires when the byte is glued to what precedes it
    // (mirrors TOK_CDQUOTE's trigger shape: `gap==0 && dps`) and, unlike a quote, ALSO clears `dps`
    // afterward (mirrors TOK_ODQUOTE's/TOK_SPELL_START's after-effect: the content immediately
    // inside typically glues too, e.g. "f(x" or "[i") -- so one marker absorbs BOTH the JOIN that
    // would precede the bracket AND the JOIN that would follow it. Each close-bracket marker fires
    // on the same glued-before condition but leaves `dps` true afterward (closing brackets are
    // typically followed by a space in real prose/code, ") the" / ") {"). A bracket preceded by a
    // real space (not glued) is NOT bundled -- that spacing was already free (implicit) before this
    // change and still falls through to the ordinary byte path unchanged, so "f( x )" still
    // round-trips, just without the extra savings this only targets the measured glued case. See
    // docs/TOKENIZER_REVIEW.md §5.9.
    TOK_GLUE_OPAREN, TOK_GLUE_CPAREN,      // `(` `)` glued to what precedes
    TOK_GLUE_OBRACKET, TOK_GLUE_CBRACKET,  // `[` `]` glued to what precedes
    TOK_GLUE_OBRACE, TOK_GLUE_CBRACE,      // `{` `}` glued to what precedes

    // Reserved headroom (Stage 2/3: reasoning delimiters like <think>/</think>, tool-call
    // structure -- see docs/ROADMAP.md). Since this enum's own extension is ALREADY a forced
    // re-tokenization (every learned piece id shifts, because pieces start at n_base), reserving
    // slots now means a future marker family is a rename of an existing enumerator (nothing
    // downstream shifts again), not another insertion (exactly what just happened above for the
    // bracket-glue markers). Rounds the marker region to ~32 total, not Llama-3's 256-slot
    // extravagance -- a reserved id is a dead vocab row, and at this project's tiny scale (vocab is
    // a large fraction of total params at d128-d768) that isn't free. Each reserved id gets a
    // base_symbol/expansion entry like any marker (required by the on-disk format) but no
    // encode-side literal match and no decode-side effect (detokenize_join's switch has no case for
    // it, and the fallback path explicitly no-ops any unassigned marker id rather than mis-decoding
    // it as a byte -- see detokenize_join) until a future workstream gives one meaning.
    //
    // COMBINE / UNCOMBINE region markers: four reserved slots RENAMED (not inserted -- the same
    // headroom-consuming pattern the bracket-glue markers used above) for the token-granularity
    // research spike (docs/ROADMAP.md; project memory spellspike-poc-results). They are CONTROL
    // tokens the model emits to REQUEST a deterministic tokenizer-layer operation, NOT text-encoding
    // markers: nothing in encode_join/detokenize_join emits or matches them (inert in the text path,
    // same as any reserved slot). The decode-loop interceptor (sub0/decode.hpp) fulfils the request:
    //   UNCOMBINE: the model emits `TOK_UNCOMBINE <token>`; the harness expands that token into its
    //     constituent byte/sub-token fragments (from the tokenizer's own `expansion` -- known, exact,
    //     free) and injects them followed by TOK_UNCOMBINE_END, so the model can then READ the
    //     characters it could not see inside one opaque token (spelling, Nth-char, char-counting).
    //   COMBINE: the model emits `TOK_COMBINE <fragments...> TOK_COMBINE_END`; the harness
    //     re-tokenizes that fragment span back into its minimal vocab token(s) -- the inverse op.
    // The point is NOT to memorise spellings in weights (the first spike proved that does not
    // generalise) but to learn to INVOKE these ops; round-trip fidelity (combine(uncombine(t))==t)
    // is the success measure. See sub0/spellspike.hpp for the curriculum and decode.hpp for the
    // interceptor.
    TOK_UNCOMBINE,       // request: expand the next token into its fragments
    TOK_UNCOMBINE_END,   // harness-injected: end of the injected fragment span
    TOK_COMBINE,         // request: contract the following fragment span into its token(s)
    TOK_COMBINE_END,     // end of the fragment span to contract
    TOK_RESERVED_4, TOK_RESERVED_5, TOK_RESERVED_6, TOK_RESERVED_7,
    TOK_RESERVED_8, TOK_RESERVED_9,

    TOK_MARKER_COUNT,  // sentinel: one past the last marker id; also == n_base - 256
};
static_assert(TOK_MARKER_COUNT - TOK_EOS == 32,
             "marker count drifted -- update learn()'s n_base assert and deserialize()'s "
             "verification loop bound together (both derive from this same enum)");

// Versions the encode_join/detokenize_join TRANSITION RULES (spacing FSM, quote directionality,
// SPELL threshold, ...), independent of the base-alphabet layout above. Written into serialize()'s
// existing output, so it rides the same fingerprint() hash -- no second, parallel reject mechanism
// alongside the magic number.
//   1 -> 2 (this bump): two transition-rule-only changes that do NOT touch n_base/base_symbol/
//   TOK_MARKER_COUNT (so the magic number alone wouldn't catch them): the line-initial opening-quote
//   fix (a quote right after a bare '\n' now fires TOK_ODQUOTE instead of falling back to a bare
//   byte) and the WS5b bracket-glue markers (TOK_GLUE_OPAREN etc., renamed from previously-reserved,
//   previously-inert slots into ones with real encode/decode effects -- see docs/TOKENIZER_REVIEW.md
//   §5.8/§5.9). Bundled into one bump since both landed in the same pre-release Stage 2 diff.
constexpr std::uint32_t kSchemeVersion = 2;

constexpr bool          is_alpha(unsigned char c) { return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z'); }
constexpr bool          is_lower(unsigned char c) { return c >= 'a' && c <= 'z'; }
constexpr bool          is_upper(unsigned char c) { return c >= 'A' && c <= 'Z'; }
constexpr bool          is_space(unsigned char c) { return c == ' ' || c == '\n' || c == '\t' || c == '\r'; }
constexpr unsigned char to_lower(unsigned char c) { return is_upper(c) ? static_cast<unsigned char>(c + 32) : c; }
constexpr unsigned char to_upper(unsigned char c) { return is_lower(c) ? static_cast<unsigned char>(c - 32) : c; }

// A "word byte" for pre-tokenization: ASCII letters plus any UTF-8 multibyte byte
// (>= 0x80). Treating the continuation/lead bytes of accented letters as word
// material keeps loanwords like "piñata" or "café" as one BPE unit instead of
// shattering them at the accent. (Typographic punctuation -- curly quotes, dashes
// -- is folded to ASCII by normalize_text first, so the only multibyte sequences
// left in the stream are genuine letters and a few rare symbols.) Markers (>= 256)
// are deliberately excluded so they stay atomic case operators.
constexpr bool is_word_byte(int s) {
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
    // WS6: this scans every corpus byte for two rare lead bytes (0xE2 UTF-8 punct, 0x60 backtick).
    // The fast path is "byte is neither" -- >99% of real prose. Two-pass, matching this project's
    // established `#pragma omp simd` auto-vectorization convention (see src/backend_cpu.cpp) rather
    // than hand-written intrinsics: pass 1 is a branchless per-byte classify the compiler can
    // vectorize (no data-dependent control flow, so it lowers to a compare+or over 16/32 bytes at
    // once); pass 2 walks the resulting flags and BULK-copies each pass-through run in one
    // `std::string::append` instead of one `push_back` per byte, then only re-examines the (rare)
    // flagged positions with the original scalar replacement logic. `flag` is a plain byte array,
    // not `std::vector<bool>` (bit-packed, defeats vectorization) -- it holds 0/1, not `bool`,
    // because `#pragma omp simd` needs a type the compiler can write in wide, uniform lanes.
    std::vector<unsigned char> flag(n);
    const unsigned char* ip = reinterpret_cast<const unsigned char*>(in.data());
    #pragma omp simd
    for (std::size_t i = 0; i < n; ++i)
        flag[i] = static_cast<unsigned char>((ip[i] == 0xE2) | (ip[i] == '`'));

    std::size_t i = 0, run_start = 0;
    while (i < n) {
        if (!flag[i]) { ++i; continue; }
        if (i > run_start) out.append(in, run_start, i - run_start);   // bulk-copy the pass-through run

        const unsigned char c = ip[i];
        // General-punctuation block U+2013..U+2026 is encoded E2 80 xx.
        if (c == 0xE2 && i + 2 < n && ip[i + 1] == 0x80) {
            const unsigned char t = ip[i + 2];
            const char* rep = nullptr;
            switch (t) {
                case 0x98: case 0x99: rep = "'";   break;  // ‘ ’ single quotes / apostrophe
                case 0x9C: case 0x9D: rep = "\"";  break;  // “ ” double quotes
                case 0x93: case 0x94: rep = "-";   break;  // – — dashes
                case 0xA6:            rep = "...";  break;  // … ellipsis
                default: break;
            }
            if (rep) { out += rep; ++replaced; i += 3; run_start = i; continue; }
            // An 0xE2 lead byte not followed by a recognized codepoint: falls through and is
            // copied as its own single byte below, exactly like the pre-WS6 scalar behavior did
            // (the loop there also just fell through to the unconditional push_back at the bottom).
        } else if (c == '`') {
            out.push_back('\''); ++replaced; ++i; run_start = i; continue;  // ASCII backtick
        }
        out.push_back(static_cast<char>(c));
        ++i; run_start = i;
    }
    if (n > run_start) out.append(in, run_start, n - run_start);   // final pass-through run
    return out;
}

// An "interior connector": a byte that binds two word bytes into ONE unit when flanked by
// them on both sides. The apostrophe keeps "don't"/"Lily's" whole; the underscore and hyphen
// keep snake_case ("save_scan_state") and hyphenated compounds ("well-known",
// "non-commercial") whole so BPE merges across them instead of paying a JOIN per separator
// (each glued separator otherwise costs two JOIN tokens). A leading/trailing connector still
// splits off (it is not flanked). See docs/TOKENIZER_DESIGN.md §8.
constexpr bool is_interior_connector(int c) { return c == '\'' || c == '_' || c == '-'; }

// End (exclusive) of the word unit beginning at `s[i]`, or `i` itself if `s[i]`
// does not start one. A unit is a maximal run of word bytes (see is_word_byte)
// with interior connectors kept (see is_interior_connector). Read-only view -- `s` is
// never mutated or resized here, so a span accepts a vector, a subrange, or any other
// contiguous int buffer without a copy.
inline std::size_t word_unit_end(std::span<const int> s, std::size_t i) {
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
// WS6/CUDA: the configurator runs normalize_text -> truecase_tokenize -> encode over the WHOLE
// corpus (GBs). The pipeline is per-byte but EMBARRASSINGLY PARALLEL across newline-aligned chunks
// (a newline never splits a word unit or the casing look-back), which the configurator already
// exploits with std::thread. A future lever: a CUDA tokenizer -- one block per chunk -- since the
// merge table + attested set are read-only and the work is regular. Flagged for later (see
// TOKENIZER_REVIEW.md).
//
// WS6 investigated SIMD-ing this function and deliberately did NOT change it -- worth recording
// why, so this isn't re-attempted blind. Measured on 500MB of real prose: the boundary-scan
// (finding where each alpha run starts/ends) is only ~30% of this function's total cost; the
// dominant cost is per-word case CLASSIFICATION (emit_word below: a lowercase copy, an `attested`
// hash lookup), which is inherently NOT a SIMD target (branchy, allocates, hashes). Two rewrites
// were tried and measured, not assumed: (1) a whole-text is_alpha/is_upper mask precomputed up
// front -- net SLOWER, since `sub0_frontend` (this header's actual compile target) gets no OpenMP
// flag by default (only `sub0_core` does, see cmake/OpenMP.cmake) so the `#pragma omp simd` was
// inert, and the extra two n-sized allocations were pure overhead; (2) bulk-appending each run via
// `resize`+store instead of one `push_back` per byte, even after adding `-fopenmp-simd` to
// `sub0_frontend` and gating it behind a length threshold (push_back wins for the length-1 runs
// that dominate real prose) -- an INTERLEAVED same-process A/B (this laptop has real thermal
// confounds on separate sequential process runs, see the project's own perf-testing notes) showed
// the "improved" version within +-3% of the original, i.e. noise, not a real win. Not worth the
// added complexity for a measured non-result. See docs/TOKENIZER_REVIEW.md's WS6 section.
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

}  // namespace sub0::casing
