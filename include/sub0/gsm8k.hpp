// sub0/gsm8k.hpp -- convert GSM8K-style worked solutions into the op-curriculum (docs/ROADMAP.md §D, the
// deterministic-mechanisms connection). GSM8K's solutions carry inline calculator annotations
// `<<48/2=24>>` -- a ready-made op-frame marking exactly the arithmetic spans to DELEGATE (PROVIDE) and MASK
// (FILTER). This module parses those annotations into verifiable op calls: the model will learn to EMIT the
// routing `[op div 48 2]` while a deterministic node supplies the exact result, so it never learns the fuzzy
// arithmetic (which arithspike showed never generalises).
//
// Integer-op MVP: `+ - * /` over non-negative INTEGER operands (the bulk of grade-school annotations).
// Decimal/negative/multi-op annotations fail the parse and stay as ordinary literal text (a documented
// follow-on adds decimal/rational nodes). Engine + tokenizer free (nodes.hpp + std only) -- a FRONTEND module.

#pragma once

#include "sub0/nodes.hpp"

#include <string>
#include <string_view>
#include <vector>

namespace sub0::gsm8k {

// A parsed calc-annotation `<<a<sym>b=result>>`: the registry op-name for the symbol + operands + the stated
// result (kept for verification -- the node recomputes it).
struct Op { char sym = 0; std::string name, a, b, result; };

inline const char* op_name(char sym) {
    switch (sym) {
        case '+': return "add"; case '-': return "sub"; case '*': return "mul"; case '/': return "div";
        default:  return nullptr;
    }
}

// Parse EXPR = "a<sym>b" (a SINGLE binary op over non-negative integer operands) and its stated RESULT.
// False for anything else (decimals, negatives, spaces, multi-op) -- the integer-MVP filter.
inline bool parse_annotation(std::string_view expr, std::string_view result, Op& out) {
    auto all_digits = [](std::string_view s) {
        if (s.empty()) return false;
        for (char c : s) if (c < '0' || c > '9') return false;
        return true;
    };
    std::size_t oppos = std::string_view::npos; char sym = 0;
    for (std::size_t i = 1; i < expr.size(); ++i) {   // from 1: an operator at 0 would be a leading sign (unsupported)
        const char c = expr[i];
        if (c == '+' || c == '-' || c == '*' || c == '/') { oppos = i; sym = c; break; }
    }
    if (oppos == std::string_view::npos) return false;
    const std::string_view a = expr.substr(0, oppos), b = expr.substr(oppos + 1);
    if (!all_digits(a) || !all_digits(b) || !all_digits(result)) return false;
    out.sym = sym; out.name = op_name(sym); out.a = std::string(a); out.b = std::string(b); out.result = std::string(result);
    return true;
}

// A solution split into alternating literal text and parsed op annotations. A `<<...>>` that fails the parse
// is left in the literal text (the sentence is unchanged; it just isn't turned into an op).
struct Segment { std::string text; bool is_op = false; Op op; };

inline std::vector<Segment> segment(std::string_view sol) {
    std::vector<Segment> segs; std::string cur;
    std::size_t i = 0;
    while (i < sol.size()) {
        if (sol[i] == '<' && i + 1 < sol.size() && sol[i + 1] == '<') {
            const std::size_t close = sol.find(">>", i + 2);
            if (close != std::string_view::npos) {
                const std::string_view inner = sol.substr(i + 2, close - (i + 2));   // "expr=result"
                const std::size_t eq = inner.find('=');
                Op op;
                if (eq != std::string_view::npos &&
                    parse_annotation(inner.substr(0, eq), inner.substr(eq + 1), op)) {
                    if (!cur.empty()) { segs.push_back({std::move(cur), false, {}}); cur.clear(); }
                    segs.push_back({std::string(sol.substr(i, close + 2 - i)), true, op});
                    i = close + 2;
                    continue;
                }
            }
        }
        cur.push_back(sol[i++]);
    }
    if (!cur.empty()) segs.push_back({std::move(cur), false, {}});
    return segs;
}

// FILTER-pillar guard: does running the deterministic node actually reproduce the annotation's stated result?
// A mismatched annotation (bad label, or an op we compute differently) is dropped, not learned.
inline bool verify(const Op& op, const nodes::Registry& reg) {
    return reg.run(op.name, {op.a, op.b}) == op.result;
}

}  // namespace sub0::gsm8k
