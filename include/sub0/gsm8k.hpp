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
#include "sub0/casing.hpp"   // TOK_TURN_START / TOK_TURN_END (the op-frame markers)

#include <functional>
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

// The op-frame the model learns to EMIT for `op`: `TOK_TURN_START op <name> <a> <b> TOK_TURN_END`. Operands
// live in the frame (self-contained -- the node reads them there, not from ambiguous surrounding prose).
inline std::vector<int> op_frame(const Op& op) {
    std::vector<int> f;
    f.push_back(casing::TOK_TURN_START);
    const std::string hdr = "op " + op.name + " " + op.a + " " + op.b;
    for (char c : hdr) f.push_back(static_cast<unsigned char>(c));
    f.push_back(casing::TOK_TURN_END);
    return f;
}

// A tokenized training example: prose + op-frames are GRADED (mask 1 -- the model learns the reasoning and
// the routing); each verified annotation's RESULT is MASKED (mask 0 -- the node fills it, so the model never
// learns the fuzzy arithmetic). `ops` verified annotations became op-frames; `dropped` failed verification.
struct Example { std::vector<int> tokens; std::vector<std::uint8_t> mask; int ops = 0, dropped = 0; };

// Build one Example from a GSM8K solution. `encode` tokenizes prose -- injected so this stays
// tokenizer-agnostic and unit-testable (train_stage passes the real sub0::encode). GSM8K writes each
// annotation as `EXPR = <<EXPR=R>> R`, so the R that follows the annotation is stripped from the prose (it
// is delegated, not written): the op-frame is graded, then R is emitted once, masked.
inline Example build_stream(std::string_view sol,
                            const std::function<std::vector<int>(std::string_view)>& encode,
                            const nodes::Registry& reg) {
    Example ex;
    std::vector<Segment> segs = segment(sol);
    auto emit      = [&](int t, std::uint8_t m) { ex.tokens.push_back(t); ex.mask.push_back(m); };
    auto emit_text = [&](std::string_view s, std::uint8_t m) { for (int t : encode(s)) emit(t, m); };

    for (std::size_t k = 0; k < segs.size(); ++k) {
        const Segment& s = segs[k];
        if (!s.is_op)               { emit_text(s.text, 1); continue; }
        if (!verify(s.op, reg))     { ++ex.dropped; emit_text(s.text, 1); continue; }   // unverified -> plain prose
        for (int t : op_frame(s.op)) emit(t, 1);                    // GRADED: the routing the model must learn
        for (char c : s.op.result)   emit(static_cast<unsigned char>(c), 0);   // MASKED: the node supplies it
        ++ex.ops;
        // Strip the redundant result that GSM8K repeats right after the annotation (leading ws + R + word
        // boundary), so it is not ALSO written as graded prose.
        if (k + 1 < segs.size() && !segs[k + 1].is_op) {
            std::string& nx = segs[k + 1].text;
            std::size_t p = 0; while (p < nx.size() && (nx[p] == ' ' || nx[p] == '\t')) ++p;
            const std::string& R = s.op.result;
            const bool boundary = (p + R.size() >= nx.size()) ||
                                  !(nx[p + R.size()] >= '0' && nx[p + R.size()] <= '9');
            if (nx.compare(p, R.size(), R) == 0 && boundary) nx.erase(0, p + R.size());
        }
    }
    return ex;
}

}  // namespace sub0::gsm8k
