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

// The built-in exact nodes (binary arithmetic + comparison, matching the spikes). Extend with an exprtk
// numeric-expression node and a SymEngine symbolic node (solve/simplify) -- same registry (see the doc).
inline Registry builtin() {
    Registry r;
    r.register_node("add", [](const std::vector<std::string>& o) { return o.size() >= 2 ? add(o[0], o[1]) : std::string{}; });
    r.register_node("sub", [](const std::vector<std::string>& o) { return o.size() >= 2 ? sub(o[0], o[1]) : std::string{}; });
    r.register_node("max", [](const std::vector<std::string>& o) { return o.size() >= 2 ? (cmp(o[0], o[1]) >= 0 ? o[0] : o[1]) : std::string{}; });
    r.register_node("min", [](const std::vector<std::string>& o) { return o.size() >= 2 ? (cmp(o[0], o[1]) <= 0 ? o[0] : o[1]) : std::string{}; });
    return r;
}

}  // namespace sub0::nodes
