// sub0/tok_interchange.hpp — a general, engine-free interchange shape for a learned Tokenizer:
// dense id -> (text, score, kind) records, matching HF tokenizer.json's "Unigram" model shape and
// GGUF's non-BPE vocab metadata (token+score pairs, no merge list -- WS2 made the runtime tokenizer
// Unigram-only, so this is a faithful export, not a lossy approximation of a BPE-shaped format).
//
// Deliberately export-only: converts a live sub0::tok::Tokenizer into a plain, tool-agnostic struct
// that JSON/TSV formatting (or any future consumer) builds on. Does NOT read/write GGUF itself --
// include/sub0/gguf.hpp stays fully decoupled from sub0::tok (see docs/TOKENIZER_REVIEW.md §5.7);
// any future weight-transplant/import work converts explicitly at that boundary, not through this
// header. See docs/TOKENIZER_REVIEW.md's WS7 section and docs/WORKFLOW_ARCHITECTURE.md's
// "Tokenizer / vocab as engine-free frontend tools" section (the `sub0llm-tokenizer export` tool
// this header exists to support).

#pragma once

#include "sub0/tokenizer.hpp"

#include <string>
#include <vector>

namespace sub0::tok {

enum class TokenKind { Byte, Marker, Piece };

// One vocabulary entry. `text` is the literal byte sequence this id decodes to for a Byte/Piece
// (raw bytes, not necessarily valid UTF-8/printable), or a canonical human-readable form for a
// Marker (its real literal match where one exists, e.g. "<|endoftext|>"; a descriptive bracketed
// tag for spacing/structural markers with no literal text of their own, e.g. "<|join|>"). `score`
// is the Unigram log-probability (`Tokenizer::piece_logp`) -- for markers this is always the floor
// value (markers are never in `piece_index`, so it isn't a meaningful ranking there, only a
// placeholder consistent with the rest of the id space).
struct TokenRecord {
    int         id    = 0;
    std::string text;
    float       score = 0.0f;
    TokenKind   kind  = TokenKind::Byte;
};

// A tokenizer's complete vocabulary as a flat, dense, id-ordered list, plus the handful of special
// ids a consumer conventionally wants named. sub0's scheme has no BOS/PAD/UNK concept (a byte-level
// base alphabet is never "unknown", and training uses fixed-length windows, never padding) -- those
// three are always -1, kept only so this shape matches what HF/GGUF-style consumers expect to find.
// Deliberately NO merges list: see this header's own top comment.
struct Interchange {
    std::vector<TokenRecord> tokens;                 // size == t.vocab, tokens[i].id == i
    int eos_id = -1, bos_id = -1, pad_id = -1, unk_id = -1;
};

[[nodiscard]] Interchange to_interchange(const Tokenizer& t);

}  // namespace sub0::tok
