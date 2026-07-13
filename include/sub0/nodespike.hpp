// sub0/nodespike.hpp -- the REGION-FRAME + COMPUTE-NODE-REGISTRY spike (docs/DETERMINISTIC_MECHANISMS.md,
// stage 1). Generalises the single-op arithspike into an OPEN-ENDED registry: the model opens ONE region
// frame and NAMES the op in ordinary tokens; a registry dispatches on that name and injects the exact result.
//
// The design claim under test (A/B, data as proof): naming an op in ORDINARY WORD TOKENS (which cost ZERO
// tokenizer budget -- the whole point) routes as accurately as a DEDICATED per-op marker token (one reserved
// id each). If WORD ~= TOKEN, the frame design is strictly better: free extensibility, no accuracy penalty.
//
// Extensibility: a new op is one `registry()` entry + curriculum -- no new token, no format change. Four ops
// (add/sub/max/min: arithmetic AND comparison) show the registry generalises past a single arithmetic node.
// The frame markers are spike ASCII sentinels; production uses ONE reserved region marker (mirroring
// casing.hpp's TOK_TURN_START/END) with the op-name as ordinary text.

#pragma once

#include "sub0/arithspike.hpp"   // gen_int, add_ints, Dataset shape, extract helpers

#include <array>
#include <string>
#include <vector>

namespace sub0::nodespike {

using arithspike::add_ints;
using arithspike::gen_int;

// --- Region frame (spike ASCII sentinels) --------------------------------------------------------------
constexpr int FRAME_OPEN  = '{';   // model: open the op region
constexpr int EXEC        = '}';   // model: header done -> RUN the node (the compute trigger)
constexpr int FRAME_CLOSE = '#';   // node-injected: end of the injected result
constexpr int SEP         = '=';

// Operators as they appear in the PROMPT -- what the model must MAP to an op-name (the routing decision):
constexpr int SYM_ADD = '+';
constexpr int SYM_SUB = '-';       // generated with A >= B so the result is non-negative
constexpr int SYM_MAX = '^';       // "the larger of"
constexpr int SYM_MIN = 'v';       // "the smaller of"

// Dedicated per-op marker tokens -- used ONLY by the TOKEN arm of the A/B (the "one reserved id per op"
// design the frame+word scheme replaces). Distinct ASCII, no clash with digits/symbols/frame.
constexpr int TOK_ADD = '@';
constexpr int TOK_SUB = '&';
constexpr int TOK_MAX = '%';
constexpr int TOK_MIN = '!';

// --- Exact big-integer nodes (decimal strings; any length) ---------------------------------------------
inline int cmp_ints(const std::string& a, const std::string& b) {
    std::size_t ia = a.find_first_not_of('0'), ib = b.find_first_not_of('0');
    const std::string na = (ia == std::string::npos) ? "0" : a.substr(ia);
    const std::string nb = (ib == std::string::npos) ? "0" : b.substr(ib);
    if (na.size() != nb.size()) return na.size() < nb.size() ? -1 : 1;
    return na < nb ? -1 : (na > nb ? 1 : 0);
}
inline std::string sub_ints(const std::string& a, const std::string& b) {   // requires a >= b
    std::string r; int i = static_cast<int>(a.size()) - 1, j = static_cast<int>(b.size()) - 1, borrow = 0;
    while (i >= 0) {
        int d = (a[i] - '0') - borrow - (j >= 0 ? b[j] - '0' : 0);
        borrow = d < 0; if (borrow) d += 10;
        r.push_back(static_cast<char>('0' + d)); --i; --j;
    }
    while (r.size() > 1 && r.back() == '0') r.pop_back();   // strip leading zeros (r is reversed)
    std::reverse(r.begin(), r.end());
    return r;
}

// A node = an op-name and the exact function over the two operand strings. Adding one is a single row here.
using NodeFn = std::string (*)(const std::string&, const std::string&);
inline std::string node_add(const std::string& a, const std::string& b) { return add_ints(a, b); }
inline std::string node_sub(const std::string& a, const std::string& b) { return sub_ints(a, b); }
inline std::string node_max(const std::string& a, const std::string& b) { return cmp_ints(a, b) >= 0 ? a : b; }
inline std::string node_min(const std::string& a, const std::string& b) { return cmp_ints(a, b) <= 0 ? a : b; }

struct Op { int sym; const char* name; int token; NodeFn fn; };
inline const std::array<Op, 4>& ops() {
    static const std::array<Op, 4> t = {{
        { SYM_ADD, "add", TOK_ADD, &node_add },
        { SYM_SUB, "sub", TOK_SUB, &node_sub },
        { SYM_MAX, "max", TOK_MAX, &node_max },
        { SYM_MIN, "min", TOK_MIN, &node_min },
    }};
    return t;
}
inline const Op* op_by_name(const std::string& n)  { for (const Op& o : ops()) if (n == o.name) return &o; return nullptr; }
inline const Op* op_by_token(int t)                { for (const Op& o : ops()) if (t == o.token) return &o; return nullptr; }

// --- Parsing the operands from the prompt (`<A> <sym> <B> = { header } ... `) --------------------------
// Given ctx up to a just-emitted EXEC, return (A, B) and the op the header NAMES (WORD: joined letters;
// TOKEN: the single marker). Dispatch is on the MODEL'S emitted header, so a misroute yields a wrong op.
struct Parsed { std::string a, b; const Op* op = nullptr; bool ok = false; };
inline Parsed parse(const std::vector<int>& ctx, bool word_encoding) {
    Parsed r;
    int ex = -1; for (int i = static_cast<int>(ctx.size()) - 1; i >= 0; --i) if (ctx[i] == EXEC) { ex = i; break; }
    if (ex < 0) return r;
    int fo = -1; for (int i = ex - 1; i >= 0; --i) if (ctx[i] == FRAME_OPEN) { fo = i; break; }
    if (fo < 0) return r;
    if (word_encoding) {
        std::string name; for (int i = fo + 1; i < ex; ++i) if (ctx[i] >= 'a' && ctx[i] <= 'z') name.push_back(static_cast<char>(ctx[i]));
        r.op = op_by_name(name);
    } else {
        r.op = (ex - fo == 2) ? op_by_token(ctx[fo + 1]) : nullptr;   // exactly one token in the header
    }
    int sep = -1; for (int i = fo - 1; i >= 0; --i) if (ctx[i] == SEP) { sep = i; break; }
    if (sep < 0) return r;
    int sy = -1; for (int i = sep - 1; i >= 0; --i)
        if (ctx[i] == SYM_ADD || ctx[i] == SYM_SUB || ctx[i] == SYM_MAX || ctx[i] == SYM_MIN) { sy = i; break; }
    if (sy < 0) return r;
    for (int i = 0;      i < sy;  ++i) if (ctx[i] >= '0' && ctx[i] <= '9') r.a.push_back(static_cast<char>(ctx[i]));
    for (int i = sy + 1; i < sep; ++i) if (ctx[i] >= '0' && ctx[i] <= '9') r.b.push_back(static_cast<char>(ctx[i]));
    r.ok = r.op && !r.a.empty() && !r.b.empty();
    return r;
}

// The registry dispatch as a kv_decode_generate `compute` callback (fires on EXEC): resolve the op the model
// named, run it EXACTLY on the parsed operands, inject `<result> FRAME_CLOSE`. A misrouted/garbage header
// yields no digits -> a wrong answer (so accuracy measures the model's ROUTING).
inline std::vector<int> dispatch(const std::vector<int>& ctx, bool word_encoding) {
    const Parsed p = parse(ctx, word_encoding);
    if (!p.ok) return { FRAME_CLOSE };
    std::vector<int> out;
    for (char c : p.op->fn(p.a, p.b)) out.push_back(static_cast<unsigned char>(c));
    out.push_back(FRAME_CLOSE);
    return out;
}

// --- Tasks ---------------------------------------------------------------------------------------------
using Task    = arithspike::Task;
using Dataset = arithspike::Dataset;
namespace detail = arithspike::detail;

// Build one task for operator `op`. `word_encoding` -> header names the op in letters; else the dedicated
// token. `delegate` here is always true (the whole point); a `direct` control is left to arithspike.
inline Task make_task(const Op& op, const std::string& a, const std::string& b, bool word_encoding) {
    Task k; k.sum = op.fn(a, b);
    for (char c : a) detail::push_prompt(k, static_cast<unsigned char>(c));
    detail::push_prompt(k, op.sym);
    for (char c : b) detail::push_prompt(k, static_cast<unsigned char>(c));
    detail::push_prompt(k, SEP);
    detail::push(k, FRAME_OPEN, 1);                                         // graded: open the region
    if (word_encoding) for (const char* s = op.name; *s; ++s) detail::push(k, static_cast<unsigned char>(*s), 1);  // graded: the op WORD
    else                detail::push(k, op.token, 1);                       // graded: the op TOKEN
    detail::push(k, EXEC, 1);                                               // graded: run
    for (char c : k.sum) detail::push(k, static_cast<unsigned char>(c), 0); // masked: node-injected result
    detail::push(k, FRAME_CLOSE, 0);                                        // masked
    detail::push(k, casing::TOK_EOS, 1);                                    // graded
    return k;
}

// A number whose two operands satisfy the op's precondition (sub needs A >= B).
inline void gen_operands(std::mt19937_64& rng, const Op& op, int min_d, int max_d, std::string& a, std::string& b) {
    std::uniform_int_distribution<int> dd(min_d, max_d);
    a = gen_int(rng, dd(rng)); b = gen_int(rng, dd(rng));
    if (op.sym == SYM_SUB && cmp_ints(a, b) < 0) std::swap(a, b);
}

// A mixed-op dataset: every op equally likely, one task per doc.
inline Dataset build_dataset(std::mt19937_64& rng, int n_tasks, int min_d, int max_d, bool word_encoding) {
    Dataset ds; ds.doc_starts.push_back(0);
    std::uniform_int_distribution<int> pick(0, static_cast<int>(ops().size()) - 1);
    for (int i = 0; i < n_tasks; ++i) {
        const Op& op = ops()[static_cast<std::size_t>(pick(rng))];
        std::string a, b; gen_operands(rng, op, min_d, max_d, a, b);
        arithspike::append_doc(ds, make_task(op, a, b, word_encoding));
    }
    return ds;
}

// The answer a produced context asserts: the digit run after EXEC (the injected result), up to FRAME_CLOSE.
inline std::string extract_result(const std::vector<int>& ctx) {
    int ex = -1; for (int i = static_cast<int>(ctx.size()) - 1; i >= 0; --i) if (ctx[i] == EXEC) { ex = i; break; }
    if (ex < 0) return {};
    std::string s;
    for (int i = ex + 1; i < static_cast<int>(ctx.size()); ++i) {
        const int t = ctx[i];
        if (t >= '0' && t <= '9') s.push_back(static_cast<char>(t)); else break;
    }
    return s;
}

}  // namespace sub0::nodespike
