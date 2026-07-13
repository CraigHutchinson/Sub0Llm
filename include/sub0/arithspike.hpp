// sub0/arithspike.hpp -- the ARITHMETIC-DELEGATION spike: the crispest test of the deterministic-mechanism
// thesis (docs/DETERMINISTIC_MECHANISMS.md). Big-number addition is trivial ALGEBRAICALLY but hopeless to
// APPROXIMATE. So we A/B two ways to answer `A + B = ?` over big integers:
//   * DELEGATION: the model emits a COMPUTE marker; a deterministic add NODE (reusing kv_decode_generate's
//     interceptor seam) parses the operands from the stream, adds them EXACTLY, and injects the sum. The
//     model learns only the ROUTING (emit COMPUTE), never the arithmetic -> it must GENERALISE perfectly to
//     held-out numbers, because the node is exact for any input.
//   * FUZZY: the model must PRODUCE the sum digits itself. This is internal, embedding-space arithmetic ->
//     it should FAIL to generalise to held-out numbers even after heavy training.
// If delegation ~1.0 on held-out and fuzzy ~0.0, the thesis holds: teach the model to USE the scalar, not
// to BE one. Everything is base byte tokens (digits, '+', '='), no learned pieces.

#pragma once

#include "sub0/casing.hpp"   // TOK_EOS

#include <algorithm>
#include <cstdint>
#include <random>
#include <string>
#include <vector>

namespace sub0::arithspike {

// Framing bytes (real byte tokens; opaque markers the model learns from consistent use, like spellspike's).
constexpr int OP_ADD      = '+';
constexpr int SEP         = '=';
constexpr int COMPUTE     = '$';   // the model REQUESTS the deterministic add node (the routing decision)
constexpr int COMPUTE_END = '#';   // node-injected: end of the injected result span

// A random non-negative integer as a `digits`-long decimal string (no leading zero).
inline std::string gen_int(std::mt19937_64& rng, int digits) {
    std::uniform_int_distribution<int> d0(1, 9), d(0, 9);
    std::string s(1, static_cast<char>('0' + d0(rng)));
    for (int i = 1; i < digits; ++i) s.push_back(static_cast<char>('0' + d(rng)));
    return s;
}

// Exact big-integer addition on decimal strings -- ANY length, no overflow. THIS is the deterministic node's
// core: a handful of scalar ops the model never has to learn.
inline std::string add_ints(const std::string& a, const std::string& b) {
    std::string r;
    int i = static_cast<int>(a.size()) - 1, j = static_cast<int>(b.size()) - 1, carry = 0;
    while (i >= 0 || j >= 0 || carry) {
        const int s = carry + (i >= 0 ? a[i--] - '0' : 0) + (j >= 0 ? b[j--] - '0' : 0);
        r.push_back(static_cast<char>('0' + s % 10));
        carry = s / 10;
    }
    std::reverse(r.begin(), r.end());
    return r;
}

// The COMPUTE NODE (a kv_decode_generate `compute` callback): given the context up to and including a just-
// emitted COMPUTE marker (`... A + B = $`), parse the two operands, add them EXACTLY, and return the sum
// digits followed by COMPUTE_END (the tokens the loop injects). Deterministic + exact -> generalises for free.
inline std::vector<int> add_node(const std::vector<int>& ctx) {
    int e = -1;
    for (int i = static_cast<int>(ctx.size()) - 1; i >= 0; --i) if (ctx[i] == SEP) { e = i; break; }
    if (e < 0) return {};
    int p = -1;
    for (int i = e - 1; i >= 0; --i) if (ctx[i] == OP_ADD) { p = i; break; }
    if (p < 0) return {};
    std::string a, b;
    for (int i = 0;     i < p; ++i) if (ctx[i] >= '0' && ctx[i] <= '9') a.push_back(static_cast<char>(ctx[i]));
    for (int i = p + 1; i < e; ++i) if (ctx[i] >= '0' && ctx[i] <= '9') b.push_back(static_cast<char>(ctx[i]));
    if (a.empty() || b.empty()) return {};
    std::vector<int> out;
    for (char c : add_ints(a, b)) out.push_back(static_cast<unsigned char>(c));
    out.push_back(COMPUTE_END);
    return out;
}

// One task = an eval PROMPT (`A + B =`), a full TRAINING trace, a parallel loss MASK, and the true sum.
// mask[i]=1 -> the model must PRODUCE trace[i] (graded); 0 -> injected/prompt (masked).
struct Task {
    std::vector<int>          prompt;
    std::vector<int>          trace;
    std::vector<std::uint8_t> mask;
    std::string               sum;
};
namespace detail {
inline void push(Task& k, int tok, std::uint8_t m) { k.trace.push_back(tok); k.mask.push_back(m); }
inline void push_prompt(Task& k, int tok) { push(k, tok, 0); k.prompt.push_back(tok); }
}

// `delegate` -> the DELEGATION trace: [A + B =] COMPUTE (graded routing) [<sum> COMPUTE_END] (masked, node-
// injected) EOS (graded). Else the FUZZY/FACT trace: [A + B =] <sum> EOS. `grade_result` controls whether the
// result digits are graded: true = a graded FUZZY target (the model must produce it); false = a FILTER-MASKED
// corpus FACT (the digits are present as context but the model is NOT rewarded for producing them -- what a
// corpus-filter pass does to a scalar-solvable span so it can't teach fuzzy memorisation).
inline Task make_task(const std::string& a, const std::string& b, bool delegate, bool grade_result = true) {
    Task k; k.sum = add_ints(a, b);
    for (char c : a) detail::push_prompt(k, static_cast<unsigned char>(c));
    detail::push_prompt(k, OP_ADD);
    for (char c : b) detail::push_prompt(k, static_cast<unsigned char>(c));
    detail::push_prompt(k, SEP);
    if (delegate) {
        detail::push(k, COMPUTE, 1);                                             // graded: the routing decision
        for (char c : k.sum) detail::push(k, static_cast<unsigned char>(c), 0);  // masked: node-injected
        detail::push(k, COMPUTE_END, 0);                                         // masked: node-injected
        detail::push(k, casing::TOK_EOS, 1);                                     // graded
    } else {
        const std::uint8_t g = grade_result ? 1 : 0;
        for (char c : k.sum) detail::push(k, static_cast<unsigned char>(c), g);  // graded FUZZY, or masked FACT
        detail::push(k, casing::TOK_EOS, 1);
    }
    return k;
}

// A flat token stream + parallel mask + per-document start index (one task per doc), like scratchspike::Dataset.
struct Dataset {
    std::vector<int>           tokens;
    std::vector<std::uint8_t>  mask;
    std::vector<std::uint64_t> doc_starts;
};
inline void append_doc(Dataset& ds, const Task& k) {
    ds.tokens.insert(ds.tokens.end(), k.trace.begin(), k.trace.end());
    ds.mask.insert(ds.mask.end(), k.mask.begin(), k.mask.end());
    ds.doc_starts.push_back(static_cast<std::uint64_t>(ds.tokens.size()));
}
inline Dataset build_dataset(std::mt19937_64& rng, int n_tasks, int min_d, int max_d, bool delegate) {
    Dataset ds; ds.doc_starts.push_back(0);
    std::uniform_int_distribution<int> dd(min_d, max_d);
    for (int i = 0; i < n_tasks; ++i)
        append_doc(ds, make_task(gen_int(rng, dd(rng)), gen_int(rng, dd(rng)), delegate));
    return ds;
}

// The answer a produced context asserts: the digit run after the last SEP (skipping the COMPUTE marker),
// stopping at COMPUTE_END / EOS / any non-digit. Works for both arms (injected sum or produced sum).
inline std::string extract_answer(const std::vector<int>& ctx) {
    int e = -1;
    for (int i = static_cast<int>(ctx.size()) - 1; i >= 0; --i) if (ctx[i] == SEP) { e = i; break; }
    if (e < 0) return {};
    std::string s;
    for (int i = e + 1; i < static_cast<int>(ctx.size()); ++i) {
        const int t = ctx[i];
        if (t == COMPUTE) continue;
        if (t >= '0' && t <= '9') s.push_back(static_cast<char>(t));
        else break;
    }
    return s;
}

}  // namespace sub0::arithspike
