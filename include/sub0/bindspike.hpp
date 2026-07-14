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
// re-point it at each context's bindings without rebuilding the callback. Thin wrapper over the unified
// production callback (node_frame::make_compute_callback), supplying a slot->value dereference over the
// ScratchBindings; the frame parse / inline-digit fallback / inertness all live in that one place.
inline std::function<std::vector<int>(const std::vector<int>&)>
make_bind_compute_callback(nodes::Registry reg, const ScratchBindings* const* binds) {
    auto deref = [binds](int slot_id) -> std::string {
        const ScratchBindings* b = binds ? *binds : nullptr;
        return (b && b->bound(slot_id)) ? slot_value(*b, slot_id) : std::string{};
    };
    return nodes::make_compute_callback(std::move(reg), std::move(deref));
}

}  // namespace sub0::bind
