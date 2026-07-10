// tok_interchange.cpp — implementation of sub0::tok::to_interchange (see tok_interchange.hpp).

#include "sub0/tok_interchange.hpp"

namespace sub0::tok {

using namespace sub0::casing;

namespace {

// Canonical text for each marker id. A `switch` over the enum TYPE (not plain int, unlike
// tokenizer.cpp's decode dispatch) so `-Wswitch` warns if a future marker is added here without an
// entry -- this table is meant to stay exhaustive, unlike the decode switch which deliberately
// tolerates unassigned/reserved ids falling through. Literal matches (EOS, the turn markers) use
// their REAL corpus literal; spacing/structural markers with no literal text of their own (JOIN,
// SPELL delimiters) get a descriptive bracketed tag instead; bracket-glue markers use the bare
// byte they expand to (the "glue" is a spacing decision, not part of the character itself).
std::string marker_literal(TokenId id) {
    switch (id) {
        case TOK_EOS:            return "<|endoftext|>";
        case TOK_CAP:            return "<|cap|>";
        case TOK_UP:              return "<|up|>";
        case TOK_JOIN:            return "<|join|>";
        case TOK_NEWLINE:         return "\n";
        case TOK_PARA:            return "\n\n";
        case TOK_ODQUOTE:         return "\"";
        case TOK_CDQUOTE:         return "\"";
        case TOK_SPELL_START:     return "<|spell_start|>";
        case TOK_SPELL_END:       return "<|spell_end|>";
        case TOK_SPACE2:          return "  ";
        case TOK_SPACE4:          return "    ";
        case TOK_TAB2:            return "\t\t";
        case TOK_TAB4:            return "\t\t\t\t";
        case TOK_TURN_START:      return "<|im_start|>";
        case TOK_TURN_END:        return "<|im_end|>";
        case TOK_GLUE_OPAREN:     return "(";
        case TOK_GLUE_CPAREN:     return ")";
        case TOK_GLUE_OBRACKET:   return "[";
        case TOK_GLUE_CBRACKET:   return "]";
        case TOK_GLUE_OBRACE:     return "{";
        case TOK_GLUE_CBRACE:     return "}";
        case TOK_UNCOMBINE:       return "<|uncombine|>";
        case TOK_UNCOMBINE_END:   return "<|/uncombine|>";
        case TOK_COMBINE:         return "<|combine|>";
        case TOK_COMBINE_END:     return "<|/combine|>";
        case TOK_RESERVED_4:      return "<|reserved_4|>";
        case TOK_RESERVED_5:      return "<|reserved_5|>";
        case TOK_RESERVED_6:      return "<|reserved_6|>";
        case TOK_RESERVED_7:      return "<|reserved_7|>";
        case TOK_RESERVED_8:      return "<|reserved_8|>";
        case TOK_RESERVED_9:      return "<|reserved_9|>";
        case TOK_MARKER_COUNT:    break;   // sentinel, never a real id -- unreachable
    }
    return "<|unknown_marker|>";   // unreachable given the exhaustive switch above; keeps this total
}

}  // namespace

Interchange to_interchange(const Tokenizer& t) {
    Interchange ic;
    ic.tokens.reserve(static_cast<std::size_t>(t.vocab));
    for (int id = 0; id < t.vocab; ++id) {
        TokenRecord rec;
        rec.id    = id;
        rec.score = id < static_cast<int>(t.piece_logp.size()) ? t.piece_logp[static_cast<std::size_t>(id)] : 0.0f;
        if (id < 256) {
            rec.kind = TokenKind::Byte;
            rec.text.assign(1, static_cast<char>(id));
        } else if (id < t.n_base) {
            rec.kind = TokenKind::Marker;
            rec.text = marker_literal(static_cast<TokenId>(id));
        } else {
            rec.kind = TokenKind::Piece;
            const std::vector<int>& e = t.expansion[static_cast<std::size_t>(id)];
            rec.text.reserve(e.size());
            for (int code : e) rec.text.push_back(static_cast<char>(code & 0xFF));
        }
        ic.tokens.push_back(std::move(rec));
    }
    ic.eos_id = TOK_EOS;   // the only special id this scheme has -- see the struct's own doc comment
    return ic;
}

}  // namespace sub0::tok
