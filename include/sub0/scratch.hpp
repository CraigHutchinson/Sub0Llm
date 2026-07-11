// sub0/scratch.hpp -- production binding table for scratch tokens (the context-translation layer).
//
// The STATEFUL side of combine/uncombine: a fragment sequence that is not a single vocab piece (an OOV
// construction) is bound to a free reserved SCRATCH SLOT for the life of ONE context; uncombining that
// slot expands it back. So an OOV that would cost N tokens on every mention collapses to ONE scratch
// token after being bound once -- the context-window + KV-cache saving (project memory
// scratch-tokens-context-translation-layer). The scratch spike validated that a model RESOLVES such
// slots and generalizes to unseen OOVs (scratchspike.hpp + scratchspike_engine_tests.cpp); this is the
// production form of that spike's ScratchOps, backing gen_stage's decode interceptor through
// decode.hpp's expand/combine callbacks (which are std::function, so this needs no engine change).
//
// Engine-free (tokenizer + std only). One instance per generation; reset() clears the per-context table.

#pragma once

#include "sub0/casing.hpp"      // TOK_RESERVED_* slot ids
#include "sub0/tokenizer.hpp"   // sub0::tok::Tokenizer (expansion / piece_index)

#include <string>
#include <vector>

namespace sub0 {

// The reserved scratch-slot pool. slot i == SCRATCH_SLOT_BASE + i. The spike used TOK_RESERVED_4..7; the
// production table exposes the full reserved headroom (TOK_RESERVED_4..9). A larger context-translation
// layer would carve a bigger reserved range (a tokenizer-budget change) -- this is the current headroom.
constexpr int SCRATCH_SLOT_BASE  = casing::TOK_RESERVED_4;
constexpr int SCRATCH_SLOT_COUNT = casing::TOK_MARKER_COUNT - casing::TOK_RESERVED_4;   // 4..9 -> 6 slots
constexpr int scratch_slot_id(int i) { return SCRATCH_SLOT_BASE + i; }
constexpr bool is_scratch_slot(int token) {
    return token >= SCRATCH_SLOT_BASE && token < SCRATCH_SLOT_BASE + SCRATCH_SLOT_COUNT;
}

// The per-context binding table + the two deterministic tokenizer-layer ops the decode interceptor
// calls. `expand` and `combine` subsume the plain vocab case (a spell-only model uses expand/combine
// over vocab tokens with allow_bind=false), so one table serves both the spell and scratch features.
struct ScratchTable {
    const tok::Tokenizer*         tk = nullptr;
    bool                          allow_bind = true;   // false => combine never MINTS a slot (vocab-only)
    std::vector<std::vector<int>> bindings;            // bindings[i] = fragments of slot SCRATCH_SLOT_BASE+i

    void reset() { bindings.clear(); }

    // Bind a slot to fragments explicitly (a caller-driven / encoder-driven binding). Slot must be in
    // the pool. Grows the table as needed.
    void bind(int slot, std::vector<int> frags) {
        const int i = slot - SCRATCH_SLOT_BASE;
        if (i < 0 || i >= SCRATCH_SLOT_COUNT) return;
        if (static_cast<int>(bindings.size()) <= i) bindings.resize(static_cast<std::size_t>(i) + 1);
        bindings[static_cast<std::size_t>(i)] = std::move(frags);
    }

    // Fulfil a TOK_UNCOMBINE: a scratch slot -> its bound fragments; a vocab word -> its expansion;
    // anything else -> identity (a bare byte/marker expands to itself).
    std::vector<int> expand(int token) const {
        const int s = token - SCRATCH_SLOT_BASE;
        if (s >= 0 && s < static_cast<int>(bindings.size()) && !bindings[static_cast<std::size_t>(s)].empty())
            return bindings[static_cast<std::size_t>(s)];
        if (tk && token >= tk->n_base && token < tk->vocab) return tk->expansion[static_cast<std::size_t>(token)];
        return {token};
    }

    // Fulfil a TOK_COMBINE region: an exact vocab piece -> that token; an already-bound OOV -> its slot;
    // else (allow_bind, pool not full) MINT a fresh slot for it; else identity (no compression).
    std::vector<int> combine(const std::vector<int>& frags) {
        std::string key;
        for (int f : frags) key.push_back(static_cast<char>(f & 0xFF));
        if (tk) {
            const auto it = tk->piece_index.find(key);
            if (it != tk->piece_index.end()) return {it->second};
        }
        for (std::size_t i = 0; i < bindings.size(); ++i)
            if (bindings[i] == frags) return {SCRATCH_SLOT_BASE + static_cast<int>(i)};
        if (allow_bind && static_cast<int>(bindings.size()) < SCRATCH_SLOT_COUNT) {
            bindings.push_back(frags);
            return {SCRATCH_SLOT_BASE + static_cast<int>(bindings.size()) - 1};
        }
        return frags;
    }
};

}  // namespace sub0
