// sub0/node_frame.hpp -- the PRODUCTION region-frame protocol: bridges the deterministic ComputeNode registry
// (nodes.hpp) to the decode interceptor (decode.hpp's `compute` seam) using the EXISTING chat turn markers, so
// a compute call costs ZERO new tokenizer ids (docs/DETERMINISTIC_MECHANISMS.md).
//
// An OP region is  `TOK_TURN_START op <op-name> [operands...] TOK_TURN_END`  -- exactly how chat models express
// a tool-call (a message with a tool role). The role word "op" and the op-name flow as ORDINARY text (nodespike
// proved WORD op-names route as well as dedicated tokens). On TOK_TURN_END the interceptor parses the region,
// dispatches on <op-name> via the registry over its operands (in-region, else read from the preceding inline
// context like `A + B =`), and injects `TOK_TURN_START <result> TOK_TURN_END`. Non-op regions -- ordinary chat
// turns, and the injected result region itself -- are inert (no injection, no re-trigger).
//
// Byte-level parse (this project's tokenizer is byte-heavy): letter-runs are words, digit-runs are numbers. A
// full-Unigram deployment would decode tokens to text first; the registry + protocol are unchanged.

#pragma once

#include "sub0/nodes.hpp"
#include "sub0/casing.hpp"         // TOK_TURN_START / TOK_TURN_END
#include "sub0/scratch_slots.hpp"  // is_scratch_slot -- bound-slot operand dereference (numeric BIND)

#include <functional>
#include <string>
#include <vector>

namespace sub0::nodes {

constexpr int FRAME_OPEN  = casing::TOK_TURN_START;
constexpr int FRAME_CLOSE = casing::TOK_TURN_END;   // the compute trigger (kv_decode_generate compute_marker)

// Scan a token span [lo,hi) into ordered letter-runs (words) and digit-runs (numbers).
struct WordsNums { std::vector<std::string> words, nums; };
inline WordsNums scan_region(const std::vector<int>& ctx, int lo, int hi) {
    WordsNums w; std::string cw, cn;
    for (int i = lo; i < hi; ++i) {
        const int t = ctx[i];
        if (t >= 'a' && t <= 'z') { if (!cn.empty()) { w.nums.push_back(cn); cn.clear(); } cw.push_back(static_cast<char>(t)); }
        else if (t >= '0' && t <= '9') { if (!cw.empty()) { w.words.push_back(cw); cw.clear(); } cn.push_back(static_cast<char>(t)); }
        else { if (!cw.empty()) { w.words.push_back(cw); cw.clear(); } if (!cn.empty()) { w.nums.push_back(cn); cn.clear(); } }
    }
    if (!cw.empty()) w.words.push_back(cw);
    if (!cn.empty()) w.nums.push_back(cn);
    return w;
}

// Render token span [lo,hi) as an expression string: ASCII bytes verbatim, each BOUND scratch slot
// SUBSTITUTED with the value string it holds (numeric BIND -- the slot's digits enter the expression, so
// `[op math S0+S1]` and `S0 + S1 = [op math]` both work). Unbound slots and non-ASCII tokens are skipped.
inline std::string region_expr(const std::vector<int>& ctx, int lo, int hi,
                               const std::function<std::string(int)>& slot_deref) {
    std::string s;
    for (int i = lo; i < hi; ++i) {
        const int t = ctx[i];
        if (slot_deref && is_scratch_slot(t)) s += slot_deref(t);
        else if (t >= 0 && t < 128)          s.push_back(static_cast<char>(t));
    }
    return s;
}

// A kv_decode_generate `compute` callback bound to `reg` (copied in -> self-contained). Fires on TOK_TURN_END.
// The op region is read as an EXPRESSION (operators `+ - * / ^ ( )` preserved), so the general `math` node
// evaluates a whole formula in one frame. Optional `slot_deref` substitutes a bound scratch-slot id with the
// value it holds, so a `math` expression can name bound symbols (numeric BIND). Operands are the expression
// after the op-name; if the frame carries none, the preceding posed expression is used (`A + B = [op math]`).
inline std::function<std::vector<int>(const std::vector<int>&)>
make_compute_callback(Registry reg, std::function<std::string(int)> slot_deref = {}) {
    return [reg = std::move(reg), slot_deref = std::move(slot_deref)](const std::vector<int>& ctx) -> std::vector<int> {
        const int close = static_cast<int>(ctx.size()) - 1;                 // the TOK_TURN_END just fed
        int open = -1;
        for (int i = close - 1; i >= 0; --i) if (ctx[i] == FRAME_OPEN) { open = i; break; }
        if (open < 0) return {};
        const std::vector<std::string> toks = lex_str(region_expr(ctx, open + 1, close, slot_deref));
        if (toks.size() < 2 || toks[0] != "op") return {};                  // not an OP region -> inert
        const std::string& name = toks[1];
        std::vector<std::string> operands(toks.begin() + 2, toks.end());    // the in-frame expression...
        if (operands.empty())                                              // ...else the preceding posed one
            operands = lex_str(region_expr(ctx, 0, open, slot_deref));
        const std::string res = reg.run(name, operands);
        if (res.empty()) return {};                                         // unknown op / bad expression -> inert
        std::vector<int> out; out.push_back(FRAME_OPEN);
        for (char c : res) out.push_back(static_cast<unsigned char>(c));
        out.push_back(FRAME_CLOSE);
        return out;
    };
}

// The op-region a model should emit to invoke `name` (role "op" + name, space-separated words). Operands are
// read from the inline context by the node, so a routing-only curriculum emits just this fixed header.
inline std::vector<int> op_header(const std::string& name) {
    std::vector<int> h; h.push_back(FRAME_OPEN);
    for (char c : std::string("op ")) h.push_back(static_cast<unsigned char>(c));
    for (char c : name) h.push_back(static_cast<unsigned char>(c));
    h.push_back(FRAME_CLOSE);
    return h;
}

}  // namespace sub0::nodes
