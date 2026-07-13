// sub0/chainspike.hpp -- the CHAINED / DYNAMIC-TURN delegation spike (docs/DETERMINISTIC_MECHANISMS.md,
// stage 4 "composition"). A single node call is proven (arithspike); the real prize is COMPOSING calls a
// DYNAMIC number of times until a problem is resolved. Task: reduce a variable-length sum `n1+n2+...+nk` to
// one value by repeatedly invoking a reduce node, one operator per call, and STOPPING when a single value
// remains. k varies per task, so the number of turns (k-1) is dynamic and the model must DECIDE termination.
//
// Each reduce call injects the new (shorter) expression as a STATE `( ... )`, so termination is a LOCAL
// decision: after a state, emit `{` (reduce again) if it still holds a `+`, else `;` (DONE). Because the
// per-step decision is local, correct dynamic iteration should GENERALISE to unseen lengths -- the test.
// Spike ASCII sentinels; production form is the region frame (nodespike / DETERMINISTIC_MECHANISMS.md).

#pragma once

#include "sub0/arithspike.hpp"   // add_ints, gen_int, Task/Dataset, detail::push*

#include <string>
#include <vector>

namespace sub0::chainspike {

using arithspike::add_ints;
using arithspike::gen_int;
using Task    = arithspike::Task;
using Dataset = arithspike::Dataset;
namespace detail = arithspike::detail;

constexpr int PLUS       = '+';
constexpr int SEP        = '=';
constexpr int FRAME_OPEN = '{';
constexpr int EXEC       = '}';   // the reduce trigger
constexpr int ST         = '(';   // node-injected: reduced-expression STATE open
constexpr int ST_END     = ')';   // node-injected: state close
constexpr int DONE       = ';';   // model: the expression is a single value -> stop

// Split a token span [lo,hi) into its '+'-joined decimal operands.
inline std::vector<std::string> split_ops(const std::vector<int>& ctx, int lo, int hi) {
    std::vector<std::string> ops; std::string cur;
    for (int i = lo; i < hi; ++i) {
        if (ctx[i] >= '0' && ctx[i] <= '9') cur.push_back(static_cast<char>(ctx[i]));
        else if (ctx[i] == PLUS && !cur.empty()) { ops.push_back(cur); cur.clear(); }
    }
    if (!cur.empty()) ops.push_back(cur);
    return ops;
}

inline std::vector<int> emit_state(const std::vector<std::string>& ops) {   // `( o0 + o1 + ... )`
    std::vector<int> out; out.push_back(ST);
    for (std::size_t i = 0; i < ops.size(); ++i) {
        if (i) out.push_back(PLUS);
        for (char c : ops[i]) out.push_back(static_cast<unsigned char>(c));
    }
    out.push_back(ST_END);
    return out;
}

// The reduce NODE (a kv_decode_generate `compute` callback, fires on EXEC): find the CURRENT expression
// (inside the last `( ... )` state, or the prompt before `=` on the first call), reduce the LEFTMOST `+`
// (r = o0 + o1), and inject the new state `( r + o2 + ... )` -- one operand shorter.
inline std::vector<int> reduce_node(const std::vector<int>& ctx) {
    int fo = -1;
    for (int j = static_cast<int>(ctx.size()) - 2; j >= 0; --j) if (ctx[j] == FRAME_OPEN) { fo = j; break; }
    if (fo < 0) return { ST_END };
    int b = -1; bool state = false;
    for (int j = fo - 1; j >= 0; --j) { if (ctx[j] == ST_END) { b = j; state = true; break; } if (ctx[j] == SEP) { b = j; break; } }
    if (b < 0) return { ST_END };
    int lo = 0, hi = b;
    if (state) { int op = -1; for (int j = b - 1; j >= 0; --j) if (ctx[j] == ST) { op = j; break; } if (op < 0) return { ST_END }; lo = op + 1; }
    const std::vector<std::string> ops = split_ops(ctx, lo, hi);
    if (ops.size() < 2) return { ST_END };                       // nothing to reduce (model shouldn't get here)
    std::vector<std::string> nxt = { add_ints(ops[0], ops[1]) };
    for (std::size_t i = 2; i < ops.size(); ++i) nxt.push_back(ops[i]);
    return emit_state(nxt);
}

// Build a task: reduce the sum of `nums` (k = nums.size()) via k-1 graded reduce calls, then DONE. The
// injected states are masked (node-supplied). `sum` = the true total.
inline Task make_task(const std::vector<std::string>& nums) {
    Task k;
    for (std::size_t i = 0; i < nums.size(); ++i) {
        if (i) detail::push_prompt(k, PLUS);
        for (char c : nums[i]) detail::push_prompt(k, static_cast<unsigned char>(c));
    }
    detail::push_prompt(k, SEP);
    std::vector<std::string> expr = nums;
    while (expr.size() >= 2) {
        detail::push(k, FRAME_OPEN, 1);                          // graded: request a reduction
        detail::push(k, EXEC, 1);                                // graded: run
        std::vector<std::string> nxt = { add_ints(expr[0], expr[1]) };
        for (std::size_t i = 2; i < expr.size(); ++i) nxt.push_back(expr[i]);
        for (int t : emit_state(nxt)) detail::push(k, t, 0);     // masked: node-injected state
        expr = std::move(nxt);
    }
    detail::push(k, DONE, 1);                                    // graded: a single value remains -> stop
    detail::push(k, casing::TOK_EOS, 1);                         // graded
    k.sum = expr[0];
    return k;
}

// The final value the produced context asserts: the digits in the LAST state `( ... )`. If it still holds a
// `+`, the model stopped early (a termination error) -> empty (a miss).
inline std::string extract(const std::vector<int>& ctx) {
    int e = -1; for (int i = static_cast<int>(ctx.size()) - 1; i >= 0; --i) if (ctx[i] == ST_END) { e = i; break; }
    if (e < 0) return {};
    int s = -1; for (int i = e - 1; i >= 0; --i) if (ctx[i] == ST) { s = i; break; }
    if (s < 0) return {};
    std::string r;
    for (int i = s + 1; i < e; ++i) { if (ctx[i] == PLUS) return {}; if (ctx[i] >= '0' && ctx[i] <= '9') r.push_back(static_cast<char>(ctx[i])); }
    return r;
}

inline std::vector<std::string> gen_list(std::mt19937_64& rng, int k, int min_d, int max_d) {
    std::uniform_int_distribution<int> dd(min_d, max_d);
    std::vector<std::string> v; v.reserve(static_cast<std::size_t>(k));
    for (int i = 0; i < k; ++i) v.push_back(gen_int(rng, dd(rng)));
    return v;
}

// Training set: random list lengths in [kmin,kmax], random small numbers, one task per doc.
inline Dataset build_dataset(std::mt19937_64& rng, int n_tasks, int kmin, int kmax, int min_d, int max_d) {
    Dataset ds; ds.doc_starts.push_back(0);
    std::uniform_int_distribution<int> kk(kmin, kmax);
    for (int i = 0; i < n_tasks; ++i) arithspike::append_doc(ds, make_task(gen_list(rng, kk(rng), min_d, max_d)));
    return ds;
}

}  // namespace sub0::chainspike
