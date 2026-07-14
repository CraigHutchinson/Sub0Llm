// sub0/nodes.hpp -- the production-facing DETERMINISTIC COMPUTE-NODE registry (docs/DETERMINISTIC_MECHANISMS.md).
//
// The spikes proved the mechanism (arithspike: delegation 1.000 vs fuzzy 0.000; nodespike: WORD op-names route
// as well as dedicated tokens; chainspike: dynamic-turn composition length-generalises). This is the reusable
// substrate the real gen/train path builds on: an extensible name->function registry of EXACT nodes, plus the
// big-integer/decimal primitives they run on. The model learns to ROUTE to a node; the node supplies the exact
// answer -- so a new capability is ONE register_node() call and ZERO tokenizer budget (op-names are ordinary
// words; see nodespike).
//
// PRODUCTION MARKER PLAN (casing.hpp): a compute region reuses the EXISTING turn markers --
// `TOK_TURN_START <op-name> <operands> TOK_TURN_END` -- exactly how chat models express tool-calls (a message
// with a tool role). Zero new reserved ids. The decode interceptor (decode.hpp's `compute` seam) dispatches on
// the op-name word via this registry. Wiring that end-to-end (curriculum + gen interceptor + the turn-role
// convention) is the remaining integration; the nodes themselves are first-class and tested here.

#pragma once

#include <algorithm>
#include <functional>
#include <map>
#include <string>
#include <string_view>
#include <vector>

namespace sub0::nodes {

// --- Exact decimal-string big-number primitives (arbitrary length, no overflow) ------------------------
inline int cmp(const std::string& a, const std::string& b) {
    const std::size_t ia = a.find_first_not_of('0'), ib = b.find_first_not_of('0');
    const std::string na = (ia == std::string::npos) ? "0" : a.substr(ia);
    const std::string nb = (ib == std::string::npos) ? "0" : b.substr(ib);
    if (na.size() != nb.size()) return na.size() < nb.size() ? -1 : 1;
    return na < nb ? -1 : (na > nb ? 1 : 0);
}
inline std::string add(const std::string& a, const std::string& b) {
    std::string r;
    int i = static_cast<int>(a.size()) - 1, j = static_cast<int>(b.size()) - 1, carry = 0;
    while (i >= 0 || j >= 0 || carry) {
        const int s = carry + (i >= 0 ? a[i--] - '0' : 0) + (j >= 0 ? b[j--] - '0' : 0);
        r.push_back(static_cast<char>('0' + s % 10)); carry = s / 10;
    }
    std::reverse(r.begin(), r.end());
    return r;
}
inline std::string sub(const std::string& a, const std::string& b) {   // |a-b|, non-negative
    const bool swap = cmp(a, b) < 0; const std::string& hi = swap ? b : a; const std::string& lo = swap ? a : b;
    std::string r; int i = static_cast<int>(hi.size()) - 1, j = static_cast<int>(lo.size()) - 1, borrow = 0;
    while (i >= 0) {
        int d = (hi[i] - '0') - borrow - (j >= 0 ? lo[j] - '0' : 0);
        borrow = d < 0; if (borrow) d += 10;
        r.push_back(static_cast<char>('0' + d)); --i; --j;
    }
    while (r.size() > 1 && r.back() == '0') r.pop_back();
    std::reverse(r.begin(), r.end());
    return r;
}
// Strip leading zeros to the canonical form cmp() uses ("0" for all-zero).
inline std::string norm(const std::string& a) {
    const std::size_t i = a.find_first_not_of('0');
    return (i == std::string::npos) ? "0" : a.substr(i);
}
inline std::string mul(const std::string& a, const std::string& b) {   // schoolbook long multiplication
    if (a == "0" || b == "0" || a.empty() || b.empty()) return "0";
    std::vector<int> p(a.size() + b.size(), 0);
    for (int i = static_cast<int>(a.size()) - 1; i >= 0; --i)
        for (int j = static_cast<int>(b.size()) - 1; j >= 0; --j) {
            const int s = (a[i] - '0') * (b[j] - '0') + p[i + j + 1];
            p[i + j + 1] = s % 10; p[i + j] += s / 10;
        }
    std::string r; for (int d : p) { if (r.empty() && d == 0) continue; r.push_back(static_cast<char>('0' + d)); }
    return r.empty() ? "0" : r;
}
inline std::string div(const std::string& a, const std::string& b) {   // a/b, EXACT integer only -> "" otherwise
    if (cmp(b, "0") == 0) return {};                                    // division by zero: node declines
    std::string q, cur = "0";
    for (char c : a) {
        cur = norm(cur + c);                                            // bring down the next digit
        int d = 0;
        for (int t = 9; t >= 1; --t) if (cmp(mul(b, std::string(1, static_cast<char>('0' + t))), cur) <= 0) { d = t; break; }
        q.push_back(static_cast<char>('0' + d));
        cur = sub(cur, mul(b, std::string(1, static_cast<char>('0' + d))));
    }
    if (cmp(cur, "0") != 0) return {};                                 // non-zero remainder -> not exact, decline
    return norm(q);
}

// --- General expression evaluator (the `math` node) ----------------------------------------------------
// Parsing an arithmetic expression -- operator precedence, parentheses, associativity -- is itself a
// DETERMINISTIC job, so it belongs in the node, not in the model. `[op math 6 + 8 * 10 - 5]` resolves in ONE
// frame (= 81); the model just copies the expression it is reasoning about, and the node handles the rest.
// The per-op nodes (add/sub/mul/div) are this node's EXACT big-number primitives -- nothing here is `double`.
//
// Lex an expression string into ordered tokens: digit-runs (numbers), lowercase letter-runs (the op-name),
// and each operator / paren as its own single-char token. Whitespace and other bytes delimit. Operators are
// single ANSI bytes, so this is a trivial one-pass lexer.
inline std::vector<std::string> lex_str(std::string_view s) {
    std::vector<std::string> toks; std::string cur; char cls = 0;   // 'w' word, 'n' number
    auto flush = [&] { if (!cur.empty()) toks.push_back(cur); cur.clear(); cls = 0; };
    for (char c : s) {
        if (c >= 'a' && c <= 'z')      { if (cls == 'n') flush(); cur.push_back(c); cls = 'w'; }
        else if (c >= '0' && c <= '9') { if (cls == 'w') flush(); cur.push_back(c); cls = 'n'; }
        else if (c=='+'||c=='-'||c=='*'||c=='/'||c=='^'||c=='('||c==')') { flush(); toks.emplace_back(1, c); }
        else flush();
    }
    flush();
    return toks;
}

namespace detail {
// Exact signed big-integer for the evaluator (the primitives above are unsigned; expressions need signs
// because an intermediate like `5 - 8` is negative).
struct Big { bool neg = false; std::string mag = "0"; };
inline Big bnorm(Big x) { x.mag = norm(x.mag); if (x.mag == "0") x.neg = false; return x; }
inline Big bneg(Big x)  { x.neg = !x.neg; return bnorm(x); }
inline Big badd(Big a, Big b) {
    if (a.neg == b.neg) return bnorm({a.neg, add(a.mag, b.mag)});
    const int c = cmp(a.mag, b.mag);
    if (c == 0) return {false, "0"};
    return c > 0 ? bnorm({a.neg, sub(a.mag, b.mag)}) : bnorm({b.neg, sub(b.mag, a.mag)});
}
inline Big bmul(Big a, Big b) { return bnorm({a.neg != b.neg, mul(a.mag, b.mag)}); }

// Recursive-descent parser over a token list. Precedence: ^ (right-assoc) > * / > + -, with parens and unary
// +/-. `err` is set on any malformed input, unknown token, non-exact division, or a runaway power -> the
// node then declines (returns "") rather than emitting a wrong answer.
struct Ev {
    const std::vector<std::string>& t; std::size_t i = 0; bool err = false;
    explicit Ev(const std::vector<std::string>& toks) : t(toks) {}
    static bool isnum(const std::string& s) { if (s.empty()) return false; for (char c : s) if (c < '0' || c > '9') return false; return true; }
    Big pnum() { if (i >= t.size() || !isnum(t[i])) { err = true; return {}; } return bnorm({false, t[i++]}); }
    Big base() {
        if (err || i >= t.size()) { err = true; return {}; }
        if (t[i] == "(") { ++i; Big v = expr(); if (err) return {}; if (i >= t.size() || t[i] != ")") { err = true; return {}; } ++i; return v; }
        if (t[i] == "-") { ++i; return bneg(base()); }
        if (t[i] == "+") { ++i; return base(); }
        return pnum();
    }
    Big bdiv(Big a, Big b) { std::string q = div(a.mag, b.mag); if (q.empty()) { err = true; return {}; } return bnorm({a.neg != b.neg, q}); }
    Big bpow(Big a, Big b) {
        if (b.neg || b.mag.size() > 4) { err = true; return {}; }      // non-negative, bounded integer exponent
        int e = 0; for (char c : b.mag) e = e * 10 + (c - '0');
        if (e > 1000) { err = true; return {}; }                       // guard a runaway expansion
        Big r{false, "1"}; for (int k = 0; k < e && !err; ++k) r = bmul(r, a); return r;
    }
    Big factor() { Big v = base(); if (!err && i < t.size() && t[i] == "^") { ++i; Big r = factor(); if (err) return {}; return bpow(v, r); } return v; }
    Big term()   { Big v = factor(); while (!err && i < t.size() && (t[i] == "*" || t[i] == "/")) { const std::string op = t[i++]; Big r = factor(); if (err) break; v = (op == "*") ? bmul(v, r) : bdiv(v, r); } return v; }
    Big expr()   { Big v = term();   while (!err && i < t.size() && (t[i] == "+" || t[i] == "-")) { const std::string op = t[i++]; Big r = term();   if (err) break; v = (op == "+") ? badd(v, r) : badd(v, bneg(r)); } return v; }
};
}  // namespace detail

// Evaluate a tokenized expression (numbers + single-char operators + parens) to an EXACT decimal string.
// Every token must be consumed; any error -> "" (the node declines, never a wrong answer).
inline std::string eval_expr(const std::vector<std::string>& toks) {
    if (toks.empty()) return {};
    detail::Ev ev(toks);
    const detail::Big v = ev.expr();
    if (ev.err || ev.i != toks.size()) return {};
    return (v.neg ? "-" : "") + v.mag;
}
inline std::string eval(std::string_view s) { return eval_expr(lex_str(s)); }   // ergonomic: lex then evaluate

// --- The registry --------------------------------------------------------------------------------------
// A node maps operand strings -> an exact result string. N-ary (a vector), so it generalises past binary ops
// to expression-eval / CAS nodes later (same registry, same dispatch). Empty result = malformed operands.
using NodeFn = std::function<std::string(const std::vector<std::string>&)>;

// name -> node. Extensible: a new capability is one register_node() call. Dispatch is by the op-name WORD the
// model emits, so no reserved-id per op (nodespike proved word-naming carries no routing penalty).
class Registry {
public:
    void register_node(std::string name, NodeFn fn) { map_[std::move(name)] = std::move(fn); }
    const NodeFn* find(const std::string& name) const { const auto it = map_.find(name); return it == map_.end() ? nullptr : &it->second; }
    std::string run(const std::string& name, const std::vector<std::string>& operands) const {
        const NodeFn* fn = find(name);
        return fn ? (*fn)(operands) : std::string{};
    }
    std::size_t size() const { return map_.size(); }
private:
    std::map<std::string, NodeFn> map_;
};

// The built-in nodes. `math` is THE arithmetic op -- one general expression evaluator (precedence, parens,
// exact big-int) supersedes the per-op add/sub/mul/div FRAMES, so those are not registered as dispatch nodes
// (they remain this file's internal primitives that `eval_expr` runs on). `max`/`min` are selection ops
// `math` does not express, so they stay. Extend with a SymEngine symbolic node (solve/simplify) -- same
// registry (see docs/DETERMINISTIC_MECHANISMS.md).
inline Registry builtin() {
    Registry r;
    // The general expression node: operands are the lexed expression tokens (numbers + operators), evaluated
    // with full precedence -- `[op math 6 + 8 * 10 - 5]` -> 81 in ONE frame. The model copies the expression
    // it is reasoning about; the node does the parsing/precedence/arithmetic.
    r.register_node("math", [](const std::vector<std::string>& o) { return eval_expr(o); });
    r.register_node("max", [](const std::vector<std::string>& o) { return o.size() >= 2 ? (cmp(o[0], o[1]) >= 0 ? o[0] : o[1]) : std::string{}; });
    r.register_node("min", [](const std::vector<std::string>& o) { return o.size() >= 2 ? (cmp(o[0], o[1]) <= 0 ? o[0] : o[1]) : std::string{}; });
    return r;
}

}  // namespace sub0::nodes
