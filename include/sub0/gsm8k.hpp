// sub0/gsm8k.hpp -- convert GSM8K-style worked solutions into the op-curriculum (docs/ROADMAP.md §D, the
// deterministic-mechanisms connection). GSM8K's solutions carry inline calculator annotations `<<48/2=24>>`
// -- a ready-made op-frame marking exactly the arithmetic spans to DELEGATE (PROVIDE) and MASK (FILTER).
// `<<EXPR=RESULT>>` is GSM8K's OWN dataset format (see scripts/get_gsm8k.py's docstring for the ELI5 --
// short version: OpenAI's original solution text writes it as a "here's the calculator step" aside right
// before the number, e.g. "...= <<80/100*10=8>>8 more..."; the `<<`/`>>` are just this corpus's plain-text
// delimiter bytes, unrelated to C++ operators -- `segment()` below finds them by literal substring scan).
// Each annotation becomes `[op math EXPR]`: the model copies the expression verbatim and the general `math`
// node (nodes.hpp) evaluates it exactly -- so it never learns the fuzzy arithmetic (which arithspike showed
// never generalises). One general op, not a per-op frame per operator.
//
// Integer-op MVP: an annotation is kept only if `nodes::eval(EXPR)` reproduces the stated integer result --
// so decimals / non-integer division / malformed expressions are rejected (they stay as literal text). A
// decimal/rational `math` node is the documented follow-on. Engine + tokenizer free -- a FRONTEND module.

#pragma once

#include "sub0/nodes.hpp"
#include "sub0/casing.hpp"        // TOK_TURN_START / TOK_TURN_END (the op-frame markers)
#include "sub0/scratch_slots.hpp" // SCRATCH_SLOT_BASE (the collapsed-result slot)

#include <functional>
#include <string>
#include <string_view>
#include <vector>

namespace sub0::gsm8k {

// A parsed calc-annotation `<<EXPR=RESULT>>`: the arithmetic expression (verbatim) + its stated result.
struct Op { std::string expr, result; };

// Split the annotation body "EXPR=RESULT" on the FIRST '=', keep it only if the `math` node reproduces the
// result exactly (RESULT a non-negative integer). This single check IS the integer-MVP filter: `nodes::eval`
// declines decimals / non-integer division / malformed input, so those annotations are rejected here.
inline bool parse_annotation(std::string_view body, Op& out) {
    const std::size_t eq = body.find('=');
    if (eq == std::string_view::npos) return false;
    const std::string_view expr = body.substr(0, eq), result = body.substr(eq + 1);
    if (expr.empty() || result.empty()) return false;
    for (char c : result) if (c < '0' || c > '9') return false;         // result is a plain integer literal
    if (nodes::eval(expr) != result) return false;                      // FILTER: the node must reproduce it
    out.expr = std::string(expr); out.result = std::string(result);
    return true;
}

// A solution split into alternating literal text and parsed op annotations. A `<<...>>` that fails the
// parse/verify is left in the literal text (the sentence is unchanged; it just isn't turned into an op).
struct Segment { std::string text; bool is_op = false; Op op; };

inline std::vector<Segment> segment(std::string_view sol) {
    std::vector<Segment> segs; std::string cur;
    std::size_t i = 0;
    while (i < sol.size()) {
        if (sol[i] == '<' && i + 1 < sol.size() && sol[i + 1] == '<') {
            const std::size_t close = sol.find(">>", i + 2);
            if (close != std::string_view::npos) {
                Op op;
                if (parse_annotation(sol.substr(i + 2, close - (i + 2)), op)) {
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

// FILTER-pillar guard: does the `math` node reproduce the annotation's stated result? (parse_annotation
// already applies this; kept for callers that build an Op directly.)
inline bool verify(const Op& op) { return nodes::eval(op.expr) == op.result; }

// The op-frame the model learns to EMIT: a BARE `TOK_TURN_START op math TOK_TURN_END`. The expression is
// already in the prose right before it (GSM8K writes `EXPR = <<EXPR=R>>`), so the model only routes -- the
// node reads the posed expression from the preceding context (node_frame::preceding_expr). Emitting a bare
// marker avoids making the model COPY the multi-digit expression into the frame (the localization wall).
inline std::vector<int> op_frame() {
    std::vector<int> f;
    f.push_back(casing::TOK_TURN_START);
    for (char c : std::string("op math")) f.push_back(static_cast<unsigned char>(c));
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
//
// `collapse` (production/train use): the masked result is a single scratch SLOT token instead of R's digits,
// matching op_curriculum + gen's collapse (gen binds the live result to the slot and expands it for display).
// The chaining stays via the prose's literal numbers (the node reads the posed expression), so the slot is
// only the injected-result placeholder -- one slot id suffices (the model never emits it; it is masked).
inline Example build_stream(std::string_view sol,
                            const std::function<std::vector<int>(std::string_view)>& encode,
                            bool collapse = false) {
    Example ex;
    std::vector<Segment> segs = segment(sol);
    auto emit      = [&](int t, std::uint8_t m) { ex.tokens.push_back(t); ex.mask.push_back(m); };
    auto emit_text = [&](std::string_view s, std::uint8_t m) { for (int t : encode(s)) emit(t, m); };

    for (std::size_t k = 0; k < segs.size(); ++k) {
        const Segment& s = segs[k];
        if (!s.is_op)           { emit_text(s.text, 1); continue; }
        if (!verify(s.op))      { ++ex.dropped; emit_text(s.text, 1); continue; }   // unverified -> plain prose
        for (int t : op_frame())   emit(t, 1);                     // GRADED: `[op math]` (route; expr is in the prose)
        if (collapse) emit(SCRATCH_SLOT_BASE, 0);                  // MASKED: collapsed-result slot
        else for (char c : s.op.result) emit(static_cast<unsigned char>(c), 0);   // MASKED: the node's digits
        ++ex.ops;
        // Strip the redundant result GSM8K repeats right after the annotation (leading ws + R + word
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
