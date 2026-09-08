// sub0llm-qwen4-gen -- WP5c: the generation loop. The FIRST time this repo runs the WHOLE pipeline
// end to end -- real Qwen tokenizer -> real transplanted Qwen3.8-Flash-Next weights -> real sampling
// -> real detokenization (docs/WP4_SCOPE.md S6, WP5c).
//
// WHAT THIS IS, AND WHAT IT DELIBERATELY IS NOT. Every piece it uses is already independently
// verified: WP5a gated `qwen_tok::Tokenizer` against the real reference (docs/QWEN_TOKENIZER.md),
// WP5b produced and ran the real 48-layer artifact (docs/WP4_SCOPE.md WP5b), and
// `sub0::sample_token` has been this engine's sampler since long before either. So this file adds NO
// new math. It is a WIRING harness, and the only thing it can establish is that the COMPOSITION is
// correct -- which is why its checks are composition checks (round trip, determinism, the decode
// prefix invariant, the stop condition) rather than another numerical gate on parts that already
// have one. There is no end-to-end oracle to diff against: sampling is stochastic and no independent
// Qwen inference engine exists on this machine. That is stated here rather than papered over.
//
// WHY A TOOL AND NOT A TEST -- the same reason sub0llm-qwen4-forward.cpp is one: it links sub0_core,
// so it must be built against the SAME generated config the engine was, i.e. a real
// `sub0llm-configure` run at the real Qwen4 axes. It cannot join sub0_tests, which is configured for
// something else entirely, and 18-46 GiB of resident weights is not a unit-test fixture. It
// additionally links sub0_frontend, where WP5a's engine-free tokenizer lives.
//
// ------------------------------------------------------------------------------------------------
// THE FOUR DESIGN DECISIONS, each with the reason it went the way it did
// ------------------------------------------------------------------------------------------------
//
// 1. PRIMING IS `forward_one` PER PROMPT TOKEN, NOT ONE BATCHED `forward()`. Not a performance
//    judgement -- a correctness one, and this project had already settled it. `sub0::kv_decode_generate`
//    (include/sub0/decode.hpp) primes with
//
//        kv_reset(); for (pos = 0; pos < ctx.size(); ++pos) logits = forward_one(ctx[pos], pos);
//
//    and the reason is visible in the backend: `forward()` is the Node-graph path and writes NOTHING
//    to the decode-path state. The KV cache (`g_kv`), the GDN recurrent accumulator (`g_gdn`) and the
//    QSA indexer's own key cache (`g_qsa_cache`) are thread_local state that ONLY `forward_one` reads
//    and writes (src/backends/cpu/backend.cpp). A batched `forward()` over the prompt would leave all
//    three empty, and the next `forward_one` would decode position P against an all-zero history.
//    There is no cheaper prefill available at this seam -- and at these axes there is nothing to gain
//    from one anyway: WP5b measured `forward` over 6 tokens at 50.35 s against `forward_one` over the
//    same 6 positions at 49.43 s, i.e. the same cost, because the per-layer GDN/QSA/MoE compute
//    dominates and attention re-scanning does not. (`forward` and `forward_one` agree to EXACTLY 0 at
//    these dims, so the choice costs no fidelity either.)
//
// 2. STREAMING DECODE IS "DECODE THE WHOLE PREFIX, PRINT WHAT IS NEW", HOLDING BACK A TRAILING
//    U+FFFD. A token can be a fragment of a multi-byte UTF-8 character, and `Tokenizer::decode`
//    finishes with `decode_lossy` (tokenizer_config.json's `errors: replace`), so decoding a prefix
//    that ends mid-character yields U+FFFD where the full sequence will later yield a real character.
//    Printing `decode({just this token})` per step would emit those replacement characters
//    permanently and produce output that is NOT `decode(all ids)`. So: decode the accumulated id list
//    each step, strip any trailing U+FFFD (it may still resolve), print only what is new. Everything
//    before a trailing incomplete character is byte-identical between a prefix decode and the full
//    decode, so the stripped prefix is stable forever -- the loop CHECKS that invariant rather than
//    assuming it, and flushes the held-back remainder at the end so a genuine U+FFFD in the output is
//    never swallowed. Streaming rather than buffering because one token costs seconds at 48 layers
//    and a run that printed only at the end would be indistinguishable from a hang.
//
// 3. NO CHAT TEMPLATE. The prompt goes through `encode()` exactly as given -- no `<|im_start|>user`
//    wrapping, no system turn, no generation prompt. A deliberate scope line (AGENTS.md S8: land the
//    stage that is actually wired up), and an honest limitation to state at the top of the file:
//    Qwen3.8-Flash-Next is an instruct model whose own chat_template.jinja sits in the same snapshot
//    directory these three tokenizer files come from, and raw-prompt continuation is NOT the input
//    distribution it was tuned for. Nothing here is a claim about the QUALITY of what it generates.
//
// 4. THE STACK IS ENLARGED AT THE LINK LINE, and that is a real finding rather than a nicety. See
//    the CMakeLists.txt comment on this target: `sub0::sample_token` (src/engine_core.cpp) declares
//    `std::array<float, VOCAB>` twice and `std::array<int, VOCAB>` once as ORDINARY LOCALS. At this
//    build's VOCAB of 248,320 that is 2.84 MiB of stack in a single frame, against Windows' 1 MiB
//    default. MEASURED, not argued: relinked without the /STACK option (PE stack reserve back to
//    0x100000) this tool dies at the FIRST sample_token call -- after the prefill lines print, before
//    the first continuation byte -- with exit status 0xC00000FD, STATUS_STACK_OVERFLOW. Nothing had
//    hit it before because no real-axes consumer had ever SAMPLED: WP4d/e/f and WP5b all stop at the
//    logits. Reported, not fixed; the engine-side fix belongs with whoever owns that path.
//
// ------------------------------------------------------------------------------------------------
// WHAT IT CHECKS, each printed with its actual value rather than a bare pass/fail
// ------------------------------------------------------------------------------------------------
//   1. The tokenizer loads the REAL three files, and its identities (vocab_size, eos, pad) are
//      printed against this build's own VOCAB axis, with the padding gap stated as a number.
//   2. ROUND TRIP: decode(encode(prompt)) against the prompt. WP5a's contract
//      (docs/QWEN_TOKENIZER.md S2.3) is `decode(encode(x)) == NFC(x)`, NOT byte identity -- the
//      reference NFC-normalizes before the pre-tokenization regex ever runs. For an ASCII prompt NFC
//      IS the identity, so that case is a HARD GATE; a non-ASCII difference is reported, not failed.
//   3. Every prompt id is inside [0, VOCAB). An id the engine cannot embed is a hard stop.
//   4. SEQ_LEN: prompt length + --n must fit, because forward_one requires pos < SEQ_LEN.
//   5. The generation itself, streamed, with prefill and per-token wall-clock.
//   6. A single DETERMINISM-KEY line carrying seed/temp/topk and the generated ids -- which is what
//      makes determinism checkable from OUTSIDE the process: same inputs, same line, every run.
//
// The tokenizer directory is found the same way tests/qwen_tokenizer_tests.cpp's model_files_dir()
// finds it (--tokenizer-dir first, then $SUB0_QWEN_TOKENIZER_DIR, then <repo>/data/qwen_tokenizer),
// so a machine already set up for the tokenizer tests needs no extra argument here.

#include "sub0/core.hpp"
#include "sub0/layout.hpp"
#include "sub0/qwen_tokenizer.hpp"

#include <CLI/CLI.hpp>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <print>
#include <random>
#include <string>
#include <string_view>
#include <vector>

#if defined(_WIN32)
#include <windows.h>
#include <psapi.h>
#endif

namespace fs = std::filesystem;

namespace {

double peak_working_set_gib() {
#if defined(_WIN32)
    PROCESS_MEMORY_COUNTERS pmc{};
    if (GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof pmc))
        return static_cast<double>(pmc.PeakWorkingSetSize) / (1024.0 * 1024.0 * 1024.0);
#endif
    return 0.0;
}

void report_memory(const char* when) {
    std::println("[mem] {:<28} peak working set {:.2f} GiB", when, peak_working_set_gib());
}

// The SAME lookup order tests/qwen_tokenizer_tests.cpp's model_files_dir() uses, with an explicit
// --tokenizer-dir ahead of it. Kept in step deliberately: a machine that can run the tokenizer tests
// can run this tool with no extra argument.
fs::path tokenizer_dir(const std::string& cli) {
    if (!cli.empty()) return fs::path(cli);
#if defined(__clang__) || defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"   // MSVC's UCRT pushes the non-portable _dupenv_s
#endif
    const char* env = std::getenv("SUB0_QWEN_TOKENIZER_DIR");
#if defined(__clang__) || defined(__GNUC__)
#pragma GCC diagnostic pop
#endif
    if (env) {
        const fs::path p(env);
        if (fs::exists(p / "vocab.json")) return p;
    }
    const fs::path repo = fs::path(SUB0_SOURCE_DIR) / "data" / "qwen_tokenizer";
    if (fs::exists(repo / "vocab.json")) return repo;
    return {};
}

constexpr std::string_view kReplacement = "\xEF\xBF\xBD";   // U+FFFD, decode_lossy's own output

// The longest prefix of `s` that no future token can change: everything up to (but not including) a
// run of trailing U+FFFD, which is exactly what decoding a sequence cut off mid-character produces.
std::string_view stable_prefix(std::string_view s) {
    std::size_t e = s.size();
    while (e >= kReplacement.size() && s.substr(e - kReplacement.size(), kReplacement.size()) == kReplacement)
        e -= kReplacement.size();
    return s.substr(0, e);
}

std::string ids_to_string(const std::vector<int>& ids) {
    std::string s;
    for (std::size_t i = 0; i < ids.size(); ++i) { if (i) s += ", "; s += std::to_string(ids[i]); }
    return s;
}

// Print text on ONE line with newlines/tabs/quotes escaped and non-ASCII bytes shown as \xNN. Written
// out rather than using std::format's `{:?}` because that specifier's availability still varies
// across the standard libraries this project builds against, and a diagnostic line that fails to
// compile on one of them is worse than four lines of escaping.
std::string escaped(std::string_view s) {
    std::string out = "\"";
    for (const unsigned char c : s) {
        switch (c) {
            case '\n': out += "\\n";  break;
            case '\r': out += "\\r";  break;
            case '\t': out += "\\t";  break;
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            default:
                if (c < 0x20 || c >= 0x7F) {
                    char buf[5];
                    std::snprintf(buf, sizeof buf, "\\x%02X", static_cast<unsigned>(c));
                    out += buf;
                } else {
                    out.push_back(static_cast<char>(c));
                }
        }
    }
    out += '"';
    return out;
}

bool is_ascii(std::string_view s) {
    return std::all_of(s.begin(), s.end(), [](char c) { return static_cast<unsigned char>(c) < 0x80; });
}

}  // namespace

int main(int argc, char** argv) {
    CLI::App app{"sub0llm-qwen4-gen: WP5c -- real Qwen tokenizer + real transplanted model + real "
                 "sampling, end to end"};
    std::string model_path, tok_dir_cli, prompt;
    int n = 20, topk = 40;
    float temp = 0.8f;
    unsigned seed = 1234;
    app.add_option("--model", model_path,
                   "the transplanted S0L5 artifact (the engine derives its .moeq sidecar path from this)")
       ->required();
    app.add_option("--prompt", prompt, "the prompt text, fed to encode() VERBATIM -- no chat template")
       ->required();
    app.add_option("--tokenizer-dir", tok_dir_cli,
                   "directory holding vocab.json / merges.txt / tokenizer_config.json "
                   "(default: $SUB0_QWEN_TOKENIZER_DIR, then <repo>/data/qwen_tokenizer)");
    app.add_option("--n", n, "maximum tokens to generate")->capture_default_str()
       ->check(CLI::PositiveNumber);
    app.add_option("--temp", temp, "sampling temperature")->capture_default_str();
    app.add_option("--topk", topk, "top-k cutoff (0 = no cutoff)")->capture_default_str();
    app.add_option("--seed", seed, "RNG seed -- fixing this makes the whole run deterministic")
       ->capture_default_str();
    CLI11_PARSE(app, argc, argv);
    // Unbuffered: this run is minutes long at 48 layers and streams as it goes, so a buffered stdout
    // redirected to a file would show nothing until the end -- and if it dies, the partial output IS
    // the finding. Same reasoning, and the same call, as sub0llm-qwen4-forward.
    std::setvbuf(stdout, nullptr, _IONBF, 0);

    std::println("--- build --------------------------------------------------------------");
    sub0::print_config();
    std::println("PARAM_FLOATS {} | NUM_PARAMS {} | VOCAB {} | SEQ_LEN {} | N_LAYERS {}",
                 sub0::PARAM_FLOATS, sub0::NUM_PARAMS, VOCAB, SEQ_LEN, N_LAYERS);

    // --- 1. the real tokenizer --------------------------------------------------------------------
    std::println("\n--- 1. the REAL Qwen tokenizer ------------------------------------------");
    const fs::path td = tokenizer_dir(tok_dir_cli);
    if (td.empty()) {
        std::println(stderr, "FAIL: no tokenizer directory. Pass --tokenizer-dir, or set "
                             "SUB0_QWEN_TOKENIZER_DIR, or put the files in <repo>/data/qwen_tokenizer.");
        return 2;
    }
    sub0::qwen_tok::Tokenizer tk;
    {
        std::string err;
        if (!tk.load((td / "vocab.json").string(), (td / "merges.txt").string(),
                     (td / "tokenizer_config.json").string(), &err)) {
            std::println(stderr, "FAIL: tokenizer at {} did not load: {}", td.string(), err);
            return 2;
        }
    }
    std::println("loaded from {}", td.string());
    std::println("vocab_size {} (base {}) | eos_id {} | pad_id {} | this build's VOCAB {}",
                 tk.vocab_size(), tk.base_vocab_size(), tk.eos_id(), tk.pad_id(), VOCAB);
    // The tokenizer names fewer tokens than the checkpoint has embedding rows. Sampling can land in
    // that gap and decode() is documented to survive it; stated as a NUMBER so a future axis change
    // that inverts the inequality is loud rather than silent.
    if (tk.vocab_size() > VOCAB) {
        std::println(stderr, "FAIL: the tokenizer defines {} tokens but this build's VOCAB is only {} "
                             "-- encode() can produce ids the engine cannot embed",
                     tk.vocab_size(), VOCAB);
        return 2;
    }
    std::println("padding rows the tokenizer does not name: {} (sampling may land there; decode skips them)",
                 VOCAB - tk.vocab_size());

    // --- 2. encode, and the round trip ------------------------------------------------------------
    std::println("\n--- 2. encode + round trip ----------------------------------------------");
    const std::vector<int> ids = tk.encode(prompt);
    std::println("prompt ({} bytes): {}", prompt.size(), escaped(prompt));
    std::println("prompt ids ({}): [{}]", ids.size(), ids_to_string(ids));
    const std::string prompt_decoded = tk.decode(ids);
    if (prompt_decoded == prompt) {
        std::println("round trip: EXACT -- {} bytes back, byte for byte", prompt_decoded.size());
    } else if (is_ascii(prompt)) {
        std::println(stderr, "FAIL: round trip differs on an ASCII prompt, where NFC is the identity."
                             "\n  in:  {}\n  out: {}", escaped(prompt), escaped(prompt_decoded));
        return 3;
    } else {
        std::println("round trip: DIFFERS, and the prompt is non-ASCII -- expected whenever "
                     "NFC(prompt) != prompt (docs/QWEN_TOKENIZER.md S2.3)\n  in:  {}\n  out: {}",
                     escaped(prompt), escaped(prompt_decoded));
    }
    if (ids.empty()) { std::println(stderr, "FAIL: the prompt encoded to zero tokens"); return 3; }
    for (const int id : ids)
        if (id < 0 || id >= VOCAB) {
            std::println(stderr, "FAIL: prompt id {} is outside this build's VOCAB [0, {})", id, VOCAB);
            return 3;
        }

    // --- 3. the window ----------------------------------------------------------------------------
    // forward_one requires pos < SEQ_LEN (include/sub0/core.hpp). The prompt occupies positions
    // 0..P-1 and generated token k is fed at position P+k, so the last position touched is P+n-1.
    std::println("\n--- 3. the window -------------------------------------------------------");
    const int P = static_cast<int>(ids.size());
    const int room = SEQ_LEN - P;
    std::println("SEQ_LEN {} | prompt {} tokens | room for {} more | --n {}", SEQ_LEN, P, room, n);
    if (room <= 0) {
        std::println(stderr, "FAIL: the prompt alone is {} tokens and SEQ_LEN is {} -- no room to "
                             "generate. Reconfigure with a larger --seq; see docs/WP4_SCOPE.md WP5b "
                             "for why the activation arena, not the weights, is what bounds that.",
                     P, SEQ_LEN);
        return 4;
    }
    if (n > room) {
        std::println("--n {} exceeds the window; clamping to {}", n, room);
        n = room;
    }

    // --- 4. the model -----------------------------------------------------------------------------
    std::println("\n--- 4. load the real transplanted model ---------------------------------");
    report_memory("before load");
    const auto t_load0 = std::chrono::steady_clock::now();
    if (!sub0::load_model(model_path.c_str())) {
        std::println(stderr, "FAIL: load_model rejected '{}' (see the reason printed above)", model_path);
        return 5;
    }
    const double load_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_load0).count();
    std::println("load_model: ACCEPTED in {:.1f}s ({:.2f} GiB of f32 parameters)", load_s,
                 static_cast<double>(sub0::PARAM_FLOATS) * 4.0 / (1024.0 * 1024.0 * 1024.0));
    report_memory("after load");
    // graph_reset() lays out this thread's parameter Nodes and allocates its Worker arenas. The
    // decode path does not execute the Node graph, but it does read those parameter Nodes, so this is
    // required before forward_one -- exactly as in sub0llm-qwen4-forward.
    sub0::graph_reset();
    report_memory("after graph_reset");

    // --- 5. generate ------------------------------------------------------------------------------
    std::println("\n--- 5. generate ---------------------------------------------------------");
    std::println("temp {:.3f} | topk {} | seed {} | max {} tokens | stop id {} (<|im_end|>)",
                 temp, topk, seed, n, tk.eos_id());
    std::mt19937 rng(seed);
    std::vector<int> gen;            // the sampled ids alone -- the determinism artifact
    std::vector<int> ctx = ids;      // prompt + generated, i.e. exactly what decode() is fed

    // PRIME -- forward_one per prompt token, for the reason in this file's header comment: the KV,
    // GDN and QSA decode caches are written by forward_one and by nothing else.
    sub0::kv_reset();
    const float* logits = nullptr;
    const auto t_prime0 = std::chrono::steady_clock::now();
    for (int pos = 0; pos < P; ++pos) {
        logits = sub0::forward_one(ids[static_cast<std::size_t>(pos)], pos);
        std::println("[prime] {}/{} (id {})", pos + 1, P, ids[static_cast<std::size_t>(pos)]);
    }
    const double prime_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_prime0).count();
    std::println("prefill: {} positions in {:.2f}s ({:.2f}s/token)", P, prime_s, P ? prime_s / P : 0.0);

    // `shown` is the decoded text already emitted. It starts at the PROMPT's own stable decode rather
    // than at prompt.size(), which is what keeps the streamed bytes right when the first generated
    // token completes a character the prompt's last token started.
    std::string shown(stable_prefix(prompt_decoded));
    bool hit_eos = false, prefix_violation = false;
    double gen_s = 0.0;
    std::print("\ncontinuation: ");
    for (int s = 0; s < n; ++s) {
        const auto t0 = std::chrono::steady_clock::now();
        const int next = sub0::sample_token(logits, temp, topk, rng);
        if (next == tk.eos_id()) {          // the learned stop signal -- never fed, never printed
            hit_eos = true;
            std::println("\n[stop] sampled eos_id {} as generated token {} -- stopping", next, s);
            break;
        }
        gen.push_back(next);
        ctx.push_back(next);
        // Stream what is now settled. decode() concatenates per-token byte spans and then makes one
        // lossy UTF-8 pass, so everything before a trailing incomplete character is identical between
        // this decode and every longer one -- CHECKED here, not assumed.
        const std::string full = tk.decode(ctx);
        if (full.size() < shown.size() || full.compare(0, shown.size(), shown) != 0) {
            if (!prefix_violation)
                std::println("\n[WARN] decode prefix invariant violated at generated token {} -- the "
                             "streamed text is then not a prefix of decode(ids)", s);
            prefix_violation = true;
        }
        const std::string_view stable = stable_prefix(full);
        if (stable.size() > shown.size()) {
            std::print("{}", stable.substr(shown.size()));
            shown.assign(stable);
        }
        logits = sub0::forward_one(next, P + s);      // this token's own window position
        gen_s += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    }
    // Flush whatever the trailing-U+FFFD hold-back is still sitting on: a GENUINE replacement
    // character in the output must not be swallowed by the streaming rule.
    const std::string full = tk.decode(ctx);
    if (full.size() > shown.size()) std::print("{}", full.substr(shown.size()));
    std::println("");

    // --- 6. the run, as numbers -------------------------------------------------------------------
    std::println("\n--- 6. results ----------------------------------------------------------");
    std::println("generated {} tokens in {:.2f}s ({:.2f}s/token){}", gen.size(), gen_s,
                 gen.empty() ? 0.0 : gen_s / static_cast<double>(gen.size()),
                 hit_eos ? "  [stopped on eos]" : "");
    // One line, nothing else on it, so a determinism check is a plain diff of two runs' stdout.
    std::println("DETERMINISM-KEY seed={} temp={:.6f} topk={} ids=[{}]", seed, temp, topk,
                 ids_to_string(gen));
    std::println("full text ({} bytes): {}", full.size(), escaped(full));
    std::println("continuation only: {}",
                 escaped(std::string_view(full).substr(std::min(full.size(), prompt_decoded.size()))));
    if (prefix_violation)
        std::println("NOTE: the decode prefix invariant was violated while streaming (see above), so "
                     "the 'full text' line is authoritative and the streamed bytes are not");
    report_memory("final");
    return 0;
}
