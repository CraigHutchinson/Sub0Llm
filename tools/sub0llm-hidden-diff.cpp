// sub0llm-hidden-diff -- WP4f's per-layer divergence table (docs/WP4_SCOPE.md S6, WP4f).
//
// Reads TWO S0HD hidden-state dumps (include/sub0/hidden_dump.hpp) of the SAME token array and prints,
// per paired tensor, how far apart they are -- against a gate STATED HERE, before any result was seen.
//
// WHY THIS IS A SEPARATE TOOL FROM sub0llm-qwen4-forward. It links no engine and includes no generated
// config: it is a pure file differ, so it builds in ANY configuration of this repo, including the
// default one. That matters because the engine-linked harness can only be built against a real-axes
// configure run (43 GiB of weights, a 94s tokenizer learn), and a comparison tool that could only be
// built there would be unusable for checking two dumps someone hands you. It also means nothing in
// src/ links llama.cpp or anything from it, which is WP4f point 4's own requirement.
//
// WHY IT DOES NOT INVOKE llama.cpp ITSELF. The oracle is acquired as a binary/build artifact outside
// this repo (WP4f point 4). Coupling the differ to a particular llama.cpp CLI surface would make this
// tool rot the moment that surface changes, and would make the comparison unreproducible from the
// artifacts alone. Both sides are files; this reads files.
//
// --- THE GATE, STATED BEFORE THE RESULT ------------------------------------------------------------
//
// docs/WP4_SCOPE.md S6 WP4f point 3 proposes "a per-layer relative gate around 1e-4" and requires
// recording the ACTUAL number rather than a threshold picked to pass. This tool's default gate is
// therefore exactly 1e-4, on this quantity:
//
//     rel_l2 = ||a - b||_2 / ||b||_2
//
// and NOT on a max-elementwise-relative error. That choice is deliberate and is the kind of thing that
// silently decides an outcome if left implicit:
//
//   * A per-element relative error d/|b| is unbounded wherever b is near zero, which in a 2560-wide
//     hidden state happens at many coordinates on every row. WP4d already hit exactly this: its
//     forward-vs-forward_one check reported "max relative 0.135", which on inspection was a near-zero
//     logit against a +1e-6 denominator, not a real disagreement. A gate on that quantity would fail a
//     correct implementation and pass nothing useful.
//   * ||a-b|| / ||b|| is the natural relative size of the disagreement AS A VECTOR, which is what
//     "these two implementations compute the same hidden state" actually means, and it is the quantity
//     the reference agreements this project already records (0.0 / 1.5e-11 / 1.4e-09 / 4.4e-11) are
//     comparable to.
//
// max_abs, max_rel_scaled (max|a-b| divided by max|b|, i.e. relative to the tensor's own scale rather
// than to each element) and cosine similarity are all reported ALONGSIDE it, because a single number
// cannot distinguish "uniformly slightly off" from "one coordinate catastrophically off", and knowing
// which of those it is, is most of the diagnosis. The gate is on rel_l2; the rest is evidence.
//
// --- WHAT "PAIRED" MEANS ---------------------------------------------------------------------------
//
// By default tensors are paired by IDENTICAL NAME. llama.cpp's own tensor names are not this engine's,
// so a --map file supplies the correspondence, one pair per line: `<a_name> <b_name>`. Building that
// map is a judgement about which two tensors are the same mathematical object, and it belongs in a
// reviewable file next to the results, not hidden in this tool's source.
//
// --- WHAT THIS TOOL WILL NOT DO --------------------------------------------------------------------
//
// It will not reshape, re-order, re-scale, slice or otherwise massage either side to make a pair fit.
// A shape mismatch is reported as a shape mismatch. The ONE exception is --b-transpose, which exists
// because ggml declares tensor dimensions fastest-varying-first, so a [rows, cols] row-major tensor is
// declared [cols, rows] -- docs/WP4_SCOPE.md WP4c finding 7 records this exact trap ("no shape
// assertion can catch a missed transpose here, because the declared shape was never wrong"). It is a
// single explicit global flag precisely so that using it is a recorded decision rather than an
// automatic fixup that could quietly turn a real disagreement into an apparent match.

#include "sub0/hidden_dump.hpp"

#include <CLI/CLI.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <format>
#include <fstream>
#include <print>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

using sub0::hidden::Dump;
using sub0::hidden::Tensor;

namespace {

struct Metrics {
    double rel_l2 = 0;        // ||a-b|| / ||b||   <- THE GATE
    double max_abs = 0;
    double max_rel_scaled = 0;   // max|a-b| / max|b|
    double cosine = 0;
    double norm_a = 0, norm_b = 0;
    int    worst_row = -1, worst_col = -1;
    std::size_t nonfinite_a = 0, nonfinite_b = 0;
    int    worst_row_by_rel = -1;
    double worst_row_rel = 0;    // the largest per-row rel_l2, which localizes WHICH token position
};

// `b` is the reference side (the denominator). Both must already be the same shape.
Metrics compare(const Tensor& a, const Tensor& b, bool b_transposed) {
    Metrics m;
    const auto R = static_cast<std::size_t>(a.rows), C = static_cast<std::size_t>(a.cols);
    double da = 0, db = 0, dd = 0, dot = 0, maxb = 0;
    for (std::size_t r = 0; r < R; ++r) {
        double row_dd = 0, row_bb = 0;
        for (std::size_t c = 0; c < C; ++c) {
            const double av = a.data[r * C + c];
            // A transposed b is indexed [c][r] over its own declared [cols=R? ...] -- see the
            // --b-transpose note in the header comment. b.rows/b.cols are its OWN declared shape.
            const double bv = b_transposed ? b.data[c * R + r] : b.data[r * C + c];
            if (!std::isfinite(av)) { ++m.nonfinite_a; continue; }
            if (!std::isfinite(bv)) { ++m.nonfinite_b; continue; }
            const double d = av - bv;
            da += av * av; db += bv * bv; dd += d * d; dot += av * bv;
            row_dd += d * d; row_bb += bv * bv;
            maxb = std::max(maxb, std::abs(bv));
            if (std::abs(d) > m.max_abs) {
                m.max_abs = std::abs(d);
                m.worst_row = static_cast<int>(r);
                m.worst_col = static_cast<int>(c);
            }
        }
        const double rr = std::sqrt(row_dd) / (std::sqrt(row_bb) + 1e-30);
        if (rr > m.worst_row_rel) { m.worst_row_rel = rr; m.worst_row_by_rel = static_cast<int>(r); }
    }
    m.norm_a = std::sqrt(da);
    m.norm_b = std::sqrt(db);
    m.rel_l2 = std::sqrt(dd) / (m.norm_b + 1e-30);
    m.max_rel_scaled = m.max_abs / (maxb + 1e-30);
    m.cosine = dot / (m.norm_a * m.norm_b + 1e-30);
    return m;
}

// One `<a_name> <b_name>` pair per line; '#' starts a comment; blank lines ignored.
bool read_map(const std::string& path, std::vector<std::pair<std::string, std::string>>& out,
              std::string& err) {
    std::ifstream is(path);
    if (!is) { err = "cannot open map file '" + path + "'"; return false; }
    std::string line;
    int lineno = 0;
    while (std::getline(is, line)) {
        ++lineno;
        if (const auto h = line.find('#'); h != std::string::npos) line.erase(h);
        std::istringstream ss(line);
        std::string an, bn, extra;
        if (!(ss >> an)) continue;                       // blank / comment-only
        if (!(ss >> bn)) {
            err = path + ":" + std::to_string(lineno) + ": expected two names, found one";
            return false;
        }
        if (ss >> extra) {
            err = path + ":" + std::to_string(lineno) + ": expected exactly two names, found more";
            return false;
        }
        out.emplace_back(std::move(an), std::move(bn));
    }
    return true;
}

}  // namespace

int main(int argc, char** argv) {
    CLI::App app{"sub0llm-hidden-diff: WP4f -- per-layer divergence between two S0HD hidden-state dumps"};
    std::string a_path, b_path, map_path;
    double gate = 1e-4;                 // docs/WP4_SCOPE.md S6 WP4f point 3's own proposed value
    bool b_transpose = false;
    bool allow_token_mismatch = false;
    app.add_option("--a", a_path, "the S0HD dump under test (this engine's)")->required();
    app.add_option("--b", b_path, "the S0HD reference dump (llama.cpp's) -- the denominator")->required();
    app.add_option("--map", map_path,
                   "name correspondence file, one '<a_name> <b_name>' pair per line "
                   "(default: pair by identical name)");
    app.add_option("--gate", gate, "per-tensor relative-L2 gate")->capture_default_str();
    app.add_flag("--b-transpose", b_transpose,
                 "read every b tensor as [cols, rows] (ggml declares dims fastest-varying-first -- see "
                 "this tool's header comment; using this is a recorded decision, not a fixup)");
    app.add_flag("--allow-token-mismatch", allow_token_mismatch,
                 "continue even if the two dumps declare different input tokens (they are then NOT "
                 "comparable; the table is printed for diagnosis only)");
    CLI11_PARSE(app, argc, argv);
    std::setvbuf(stdout, nullptr, _IONBF, 0);

    Dump A, B;
    std::string err;
    if (!sub0::hidden::read(a_path, A, err)) { std::println(stderr, "FAIL (--a): {}", err); return 2; }
    if (!sub0::hidden::read(b_path, B, err)) { std::println(stderr, "FAIL (--b): {}", err); return 2; }

    std::println("a: {}  ({} tensors)", a_path, A.tensors.size());
    std::println("b: {}  ({} tensors){}", b_path, B.tensors.size(),
                 b_transpose ? "   [read transposed]" : "");

    // --- the premise, checked rather than assumed --------------------------------------------------
    // "Same tokens in" is the entire basis of WP4f. Two dumps of different inputs produce a divergence
    // table that looks exactly like a real disagreement, so this must be a hard stop by default.
    const auto tok_str = [](const std::vector<int>& v) {
        std::string s = "[";
        for (std::size_t i = 0; i < v.size(); ++i) s += (i ? ", " : "") + std::to_string(v[i]);
        return s + "]";
    };
    std::println("tokens a: {}", tok_str(A.tokens));
    std::println("tokens b: {}", tok_str(B.tokens));
    if (A.tokens != B.tokens) {
        std::println(stderr,
                     "\nFAIL: the two dumps were produced from DIFFERENT token arrays. Every number "
                     "below would be meaningless. (--allow-token-mismatch overrides, for diagnosis "
                     "only.)");
        if (!allow_token_mismatch) return 3;
    }

    // --- pairing -----------------------------------------------------------------------------------
    std::vector<std::pair<std::string, std::string>> pairs;
    if (!map_path.empty()) {
        if (!read_map(map_path, pairs, err)) { std::println(stderr, "FAIL: {}", err); return 2; }
        std::println("map: {} ({} pairs)", map_path, pairs.size());
    } else {
        for (const Tensor& t : A.tensors) pairs.emplace_back(t.name, t.name);
    }

    std::println("\ngate: rel_l2 = ||a-b||/||b|| <= {:.3g}   (docs/WP4_SCOPE.md S6 WP4f point 3, stated "
                 "before the result)\n", gate);
    std::println("{:<24} {:>12} {:>11} {:>11} {:>11} {:>9}  {}",
                 "tensor (a -> b)", "shape", "rel_l2", "max_abs", "max_rel_sc", "cosine", "verdict");
    std::println("{}", std::string(104, '-'));

    int compared = 0, passed = 0, unmatched_a = 0, unmatched_b = 0, shape_bad = 0;
    double worst = 0.0;
    std::string worst_name, first_fail;
    for (const auto& [an, bn] : pairs) {
        int dup_a = 0, dup_b = 0;
        const Tensor* ta = A.find(an, &dup_a);
        const Tensor* tb = B.find(bn, &dup_b);
        if (!ta) { std::println("{:<24} {:>12}  -- not present in a", an, ""); ++unmatched_a; continue; }
        if (!tb) { std::println("{:<24} {:>12}  -- not present in b", an, ""); ++unmatched_b; continue; }
        if (dup_a || dup_b)
            std::println("  note: '{}' occurs {}x in a, '{}' {}x in b -- first occurrence used",
                         an, dup_a + 1, bn, dup_b + 1);
        const bool shape_ok = b_transpose ? (tb->rows == ta->cols && tb->cols == ta->rows)
                                          : (tb->rows == ta->rows && tb->cols == ta->cols);
        if (!shape_ok) {
            std::println("{:<24} {:>12}  -- SHAPE MISMATCH: b is [{} x {}]{}", an,
                         std::format("[{}x{}]", ta->rows, ta->cols), tb->rows, tb->cols,
                         (!b_transpose && tb->rows == ta->cols && tb->cols == ta->rows)
                             ? "  (it is a's transpose -- try --b-transpose)" : "");
            ++shape_bad;
            continue;
        }
        const Metrics m = compare(*ta, *tb, b_transpose);
        const bool ok = m.rel_l2 <= gate && m.nonfinite_a == 0 && m.nonfinite_b == 0;
        ++compared;
        if (ok) ++passed; else if (first_fail.empty()) first_fail = an;
        if (m.rel_l2 > worst) { worst = m.rel_l2; worst_name = an; }
        std::println("{:<24} {:>12} {:>11.3e} {:>11.3e} {:>11.3e} {:>9.6f}  {}",
                     an, std::format("[{}x{}]", ta->rows, ta->cols),
                     m.rel_l2, m.max_abs, m.max_rel_scaled, m.cosine, ok ? "PASS" : "**FAIL**");
        if (!ok)
            std::println("    worst element [row {}, col {}]; worst row by rel_l2 is row {} at {:.3e}; "
                         "||a|| {:.6g} vs ||b|| {:.6g}{}",
                         m.worst_row, m.worst_col, m.worst_row_by_rel, m.worst_row_rel, m.norm_a,
                         m.norm_b,
                         (m.nonfinite_a || m.nonfinite_b)
                             ? std::format("; NON-FINITE: {} in a, {} in b", m.nonfinite_a, m.nonfinite_b)
                             : "");
    }

    // Anything in b that no pair mentioned -- the cheapest detector of a whole mechanism whose tensor
    // nobody thought to map, which is exactly how a comparison "passes" while covering half the model.
    std::vector<std::string> b_unused;
    for (const Tensor& t : B.tensors) {
        const bool used = std::ranges::any_of(pairs, [&](const auto& p) { return p.second == t.name; });
        if (!used) b_unused.push_back(t.name);
    }

    std::println("\n--- summary ------------------------------------------------------------");
    std::println("compared {} | passed {} | failed {} | shape mismatches {} | unmatched: {} in a, {} in b",
                 compared, passed, compared - passed, shape_bad, unmatched_a, unmatched_b);
    if (!b_unused.empty()) {
        std::string s;
        for (std::size_t i = 0; i < b_unused.size() && i < 12; ++i) s += (i ? ", " : "") + b_unused[i];
        std::println("b tensors no pair referenced ({}): {}{}", b_unused.size(), s,
                     b_unused.size() > 12 ? ", ..." : "");
    }
    if (compared == 0) {
        std::println("\nNOTHING WAS COMPARED. That is a result about the PAIRING, not about the models "
                     "-- supply a --map, or check the two dumps' tensor names.");
        return 4;
    }
    std::println("worst rel_l2: {:.6e} at '{}'  (gate {:.3g})", worst, worst_name, gate);
    if (passed == compared) {
        std::println("VERDICT: all {} paired tensors agree within the gate.", compared);
        return 0;
    }
    // WP4f's actual deliverable when they disagree: WHERE it first appears, not a global pass/fail.
    std::println("VERDICT: divergence exceeds the gate. First failing pair in table order: '{}'.",
                 first_fail);
    std::println("         That localization -- not the pass/fail -- is this stage's deliverable "
                 "(docs/WP4_SCOPE.md S6 WP4f gate).");
    return 1;
}
