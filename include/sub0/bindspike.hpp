// sub0/bindspike.hpp -- SPIKE: pillar 2 (numeric BIND) -- the op operates over BOUND SCRATCH SLOTS
// (algebraic variables) whose exact values live in the binding table, not the reasoning stream. Where
// node_frame's callback reads operands as inline digits (`A + B =`), this one dereferences the SLOTS in the
// preceding context via the active ScratchBindings: the model reasons over the symbol `S0 + S1` (each a
// single Scalar-embedded token carrying only magnitude), and the node reconstructs the exact operands from
// the bindings and returns the exact result. That is the pillar-2 claim: algebra + delegation + a 13-digit
// value collapsed to one token with its value in the binding (docs/DETERMINISTIC_MECHANISMS.md §2 BIND).
//
// SPIKE, not production: proves the mechanism before it folds into node_frame.hpp. Reuses the production
// registry (nodes.hpp), region frame (node_frame.hpp), and slot substrate (scratch_slots.hpp) unchanged.

#pragma once

#include "sub0/node_frame.hpp"     // nodes::FRAME_OPEN/CLOSE, scan_region, Registry, op_header
#include "sub0/scratch_slots.hpp"  // ScratchBindings, is_scratch_slot, SCRATCH_SLOT_BASE

#include <functional>
#include <string>
#include <vector>

namespace sub0::bind {

// A bound slot's value: its fragment token ids decoded to the decimal string they spell (byte tokens).
inline std::string slot_value(const ScratchBindings& b, int slot_id) {
    std::string s;
    for (int f : b.fragments(slot_id)) if (f >= 0 && f < 128) s.push_back(static_cast<char>(f));
    return s;
}

// A kv_decode_generate `compute` callback whose operands are the BOUND SLOTS in the preceding context,
// dereferenced through the active bindings (`*binds`). `binds` is a pointer-to-pointer so the caller can
// re-point it at each context's bindings without rebuilding the callback. Fires on TOK_TURN_END; on a
// non-op region (or the injected result region) it is inert. Falls back to inline digits when no bound slot
// precedes the op (so it degrades to node_frame's behaviour for a plain `A + B =`).
inline std::function<std::vector<int>(const std::vector<int>&)>
make_bind_compute_callback(nodes::Registry reg, const ScratchBindings* const* binds) {
    return [reg = std::move(reg), binds](const std::vector<int>& ctx) -> std::vector<int> {
        const int close = static_cast<int>(ctx.size()) - 1;                    // the TOK_TURN_END just fed
        int open = -1;
        for (int i = close - 1; i >= 0; --i) if (ctx[i] == nodes::FRAME_OPEN) { open = i; break; }
        if (open < 0) return {};
        const nodes::WordsNums in = nodes::scan_region(ctx, open + 1, close);
        if (in.words.size() < 2 || in.words[0] != "op") return {};             // not an OP region -> inert

        std::vector<std::string> operands;
        const ScratchBindings* b = binds ? *binds : nullptr;
        if (b)                                                                 // dereference the bound slots in scope
            for (int i = 0; i < open; ++i)
                if (is_scratch_slot(ctx[i]) && b->bound(ctx[i])) operands.push_back(slot_value(*b, ctx[i]));
        if (operands.empty()) operands = nodes::scan_region(ctx, 0, open).nums; // fallback: inline digits

        const std::string res = reg.run(in.words[1], operands);
        if (res.empty()) return {};
        std::vector<int> out; out.push_back(nodes::FRAME_OPEN);
        for (char c : res) out.push_back(static_cast<unsigned char>(c));
        out.push_back(nodes::FRAME_CLOSE);
        return out;
    };
}

}  // namespace sub0::bind
