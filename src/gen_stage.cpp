// gen_stage.cpp — backend stage "gen" (libsub0_gen).
//
// Owns autoregressive sampling: load a trained model, encode the prompt, and
// repeatedly run the engine forward pass, sampling the next token with
// temperature and top-k. Prints the continuation. Exposes one C entry point.

#include "sub0/core.hpp"
#include "sub0/decode.hpp"  // shared KV-cache decode loop (GPU-first, CPU-fallback, EOS-stop)
#include "sub0/registry.hpp"    // read_config_json: is this model trained with the uncombine curriculum?
#include "sub0/blend_schedule.hpp" // ScheduleSpec/parse_blend_schedule_json -- the pinned recipe records
                                   // which curricula (hence which gen-time capabilities) a model was
                                   // actually trained with, replacing the old flat spell_mix/scratch_mix/
                                   // op_mix/content_embed RunConfig fields
#include "sub0/tokenizer.hpp"   // raw sub0::tok::Tokenizer for the uncombine/combine interceptor callbacks
#include "sub0/scratch.hpp"     // ScratchTable: binding table backing both spell + scratch resolution
#include "sub0/node_frame.hpp"  // op frame: dispatch deterministic compute nodes through the TOK_TURN region
#include "sub0/op_curriculum.hpp" // gen_eval: synthetic op problems for the native `eval` scorer

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <functional>
#include <print>
#include <random>
#include <string>
#include <vector>

// GPU KV-cache decode (Phase 2e follow-up to the training fast-path in train_stage.cpp). Parity-
// validated against a TRAINED model in cuda_tests.cpp ("CUDA forward_one decode matches full forward
// once the model is trained"): 100% top-1 agreement, ~1e-6 max rel diff at the d768/HD=96 config that
// diverges under random weights (a benign flash-vs-two-pass softmax amplification, not a kernel bug --
// see the memory notes on the d448 random-weight investigation). Device forward_one is FP32-only
// (matches the CPU path's dense assumption), so this mirrors USE_TERNARY gating below.

namespace {
// Is a source using generator `gen` given a positive weight in AT LEAST ONE schedule stage? Not just
// "declared" (a source the schedule never actually draws from needs no gen-time capability, per the
// parser's own unreferenced-source warning) and not just "active in the FINAL stage" (a schedule can
// legitimately bootstrap a generator hard early then taper it to zero later, curriculum-learning-style,
// and the model still needs the matching gen-time capability for what it learned during that stage).
// Shared by sub0_gen_stage and sub0_eval_stage below.
bool generator_ever_active(const sub0::ScheduleSpec& schedule, std::string_view gen) {
    for (const sub0::SourceSpec& s : schedule.sources) {
        if (s.generator != gen) continue;
        for (const sub0::ScheduleStage& stage : schedule.stages)
            for (const auto& [name, rate] : stage.weights)
                if (name == s.name && rate > 0.0) return true;
    }
    return false;
}
}  // namespace

extern "C" SUB0_API int sub0_gen_stage(const char* model_in, const char* prompt,
                                        int n, float temp, int topk, unsigned seed, int attn_sinks) {
    sub0::build_model();                 // establish parameter-node layout
    if (!sub0::load_model(model_in)) {   // overwrite with trained weights (also reads the vocab fingerprint)
        std::println(stderr, "gen: cannot load model '{}'", model_in);
        return 1;
    }
    // Decode with the vocab this model was TRAINED against, not whatever the build tree currently holds:
    // prefer the tokenizer.tok bundled beside the model, falling back to the baked default. The
    // fingerprint guard in load_tokenizer then rejects a mismatch loudly rather than emitting garble.
    std::string tok_path = sub0::default_tokenizer();
    if (model_in && *model_in) {
        const std::filesystem::path bundled = std::filesystem::path(model_in).parent_path() / "tokenizer.tok";
        std::error_code ec;
        if (std::filesystem::exists(bundled, ec)) tok_path = bundled.string();
    }
    if (!sub0::load_tokenizer(tok_path.c_str())) {
        std::println(stderr, "gen: cannot load tokenizer '{}'", tok_path);
        return 1;
    }

    std::vector<int> ctx = sub0::encode(prompt);
    if (ctx.empty()) ctx.push_back(0);
    std::mt19937 rng(seed);
    // The model's learned stop signal (see casing.hpp's TOK_EOS): sampling loops below check for it
    // BEFORE pushing, so it never enters ctx / gets printed -- generation just ends naturally instead
    // of always running to the fixed token budget `n`. -1 (no tokenizer eos_id, e.g. a pre-EOS model)
    // never matches a real sampled id, so this degrades to today's fixed-budget behavior transparently.
    const int eos_id = sub0::eos_token_id();

    // Combine/uncombine interceptor: enabled ONLY for a model trained with a curriculum that uses it
    // (config.json records spell_mix > 0 and/or scratch_mix > 0). A normal model never emits the markers,
    // and enabling interception forces the CPU KV-cache decode (the device-decode interceptor is a pending
    // follow-on), so gate it on the recipe rather than paying that cost for every model. The stateful
    // sub0::ScratchTable backs both features: expand/combine over vocab tokens (spell) AND over
    // dynamically-bound scratch slots (scratch). allow_bind mints a scratch slot for an unseen OOV only
    // for a scratch model -- a spell-only model's combine stays vocab-only. `raw_tok`/`scratch` live for
    // the whole function so the by-ref captures stay valid through decode.
    sub0::tok::Tokenizer raw_tok;
    sub0::ScratchTable   scratch;
    std::function<std::vector<int>(int)>                       expand;
    std::function<std::vector<int>(const std::vector<int>&)>   combine;
    // Op frame (deterministic compute nodes dispatched through the TOK_TURN region, node_frame.hpp). The
    // callback is INERT for any turn that is not `[op <name> ...]` (a normal chat/role turn scans and
    // returns nothing), so it is safe to enable alongside the interceptor with no extra recipe flag -- it
    // only ever fires for a model that emits the op role. Its slot dereference reads operands from the
    // ScratchTable bindings (numeric BIND: `[op add S0 S1]` resolves the bound slots), and falls back to
    // inline digits (`A + B =`). Like the interceptor, it forces the CPU KV-cache decode.
    std::function<std::vector<int>(const std::vector<int>&)>   compute;
    // HARNESS-DRIVEN (not model-requested) natural-word collapse (docs/SCRATCH_TOKENS.md, sub0::wordspike):
    // fires live when the model finishes a 3+-piece compound word (encode_join's TOK_SPELL_START..END).
    // Prompt-side recurrences are handled once, up front, by sub0::prefill_collapse -- see below.
    std::function<std::vector<int>(const std::vector<int>&)>   word_collapse;
    bool op_collapse = false;   // an --op-mix model: op results COLLAPSE to slots (expanded for display below)
    // content_embed: scratch slots embed as a live function of their bound fragments (mean-pool) instead
    // of a plain learned row -- meaningful for a --scratch-mix model AND/OR an --op-mix model (both now
    // record doc_bindings; train_stage.cpp requires scratch_mix>0 || op_mix>0 to ever persist
    // content_embed=on; see docs/DETERMINISTIC_MECHANISMS.md / smooth-noodling-kurzweil). `binds` lives
    // for the whole function so set_scratch_bindings' thread-local pointer stays valid through decode;
    // refreshed after `combine` (sp_on) or the op-collapse callback (op_on) mutates scratch.bindings --
    // the only two mutators reachable here -- since ScratchBindings' span is a snapshot, not a live view.
    bool content_embed_on = false;
    // Which SlotEncoding a content-embed model was actually TRAINED with (sub0::content_embed_encoding_of's
    // index space, persisted as cfg.content_embed_kind -- see train_stage.cpp's own comment on why the
    // struct default (0/MeanPool) must never silently become the new HRR default: a model saved before
    // this field existed must keep decoding with the plain reserved-id-free encoding it actually trained
    // with, not whatever the current build's default happens to be). Captured BY VALUE in the lambdas
    // below (never reassigned after this point, unlike `binds`), set once `cfg` is loaded.
    sub0::SlotEncoding content_embed_enc = sub0::SlotEncoding::MeanPool;
    sub0::ScratchBindings binds{};
    if (model_in && *model_in) {
        const std::filesystem::path mp(model_in);
        const std::filesystem::path dir = std::filesystem::is_directory(mp) ? mp : mp.parent_path();
        sub0::registry::RunConfig cfg;
        const bool have = !dir.empty() && sub0::registry::read_config_json(cfg, dir);
        if (have && cfg.config_schema < 2) {
            // Predates the blend-schedule redesign: its config.json still carries the now-removed
            // spell_mix/scratch_mix/op_mix/content_embed/content_embed_kind fields, silently ignored by
            // this build's RunConfig. Unlike train's resume path (which refuses outright -- continuing
            // to TRAIN under a guessed recipe risks real, silent corruption), gen degrades instead: the
            // model.bin itself is still valid, so generate WITHOUT any interceptor rather than refuse
            // useful output entirely -- just say so loudly, since a model that actually needs the
            // interceptor will visibly misbehave without it.
            std::println(stderr, "gen: '{}' predates the blend-schedule redesign (config_schema {}) -- "
                                 "cannot determine which curricula it needs; generating WITHOUT any "
                                 "interceptor (spell/scratch/op/content-embed). Re-train with this build "
                                 "to regain them, or use an older build to generate from this model "
                                 "exactly as trained.", dir.string(), cfg.config_schema);
        }
        sub0::ScheduleSpec schedule;
        std::string schedule_error;
        std::vector<std::string> schedule_warnings;   // unused here -- gen only needs the parsed shape
        bool have_schedule = false;
        // Only look for a schedule this model actually pinned. cfg.has_blend_schedule is 0 for a plain
        // single-corpus run, where no schedule file was ever written -- reading one and warning that it
        // is missing described normal operation as a fault on every such model.
        if (have && cfg.config_schema >= 2 && cfg.has_blend_schedule) {
            have_schedule = sub0::parse_blend_schedule_json(dir / "blend_schedule.json", schedule,
                                                             schedule_error, schedule_warnings);
            if (!have_schedule)
                std::println(stderr, "gen: '{}' pinned a blend_schedule.json but it is unreadable ({}) "
                                     "-- generating WITHOUT any interceptor "
                                     "(spell/scratch/op/content-embed)", dir.string(), schedule_error);
        }
        const bool sp_on = have_schedule && (generator_ever_active(schedule, "spellspike")
                                             || generator_ever_active(schedule, "scratchspike"));
        const bool scratch_on = have_schedule && generator_ever_active(schedule, "scratchspike");
        const bool op_on = have_schedule && generator_ever_active(schedule, "op_curriculum");
        const bool word_on = have_schedule && (generator_ever_active(schedule, "wordspike")
                                             || generator_ever_active(schedule, "corpus_collapse"));
        if (sp_on || op_on || word_on) {
            std::ifstream tis(tok_path, std::ios::binary);
            if (tis.good() && sub0::tok::deserialize(raw_tok, tis) && raw_tok.vocab == VOCAB) {
                scratch.tk = &raw_tok;
                content_embed_on = (scratch_on || op_on || word_on) && schedule.content_embed.has_value();
                content_embed_enc = content_embed_on
                    ? sub0::content_embed_encoding_of(static_cast<int>(*schedule.content_embed))
                    : sub0::SlotEncoding::MeanPool;
                if (sp_on) {
                    scratch.allow_bind = scratch_on;   // only a scratch model mints slots on OOVs
                    expand  = [&scratch](int token) { return scratch.expand(token); };
                    combine = [&scratch, &binds, content_embed_on, content_embed_enc](const std::vector<int>& frags) {
                        std::vector<int> r = scratch.combine(frags);
                        // refreshing the snapshot here keeps it current for every subsequent forward_one.
                        if (content_embed_on) binds = scratch.to_bindings(content_embed_enc);
                        return r;
                    };
                }
                // The op deref reads bound slots from the ScratchTable (numeric BIND + collapse-chain refs).
                auto op_deref = [&scratch](int slot) { return scratch.value(slot); };
                if (op_on) {
                    // COLLAPSE: resolve the op, bind the exact result to the next free scratch slot, inject
                    // that single slot token (not the digits) -- so a chained step references it as a symbol.
                    auto inner = sub0::nodes::make_compute_callback(sub0::nodes::builtin(), op_deref);
                    compute = [&scratch, &binds, content_embed_on, content_embed_enc, inner](const std::vector<int>& c) -> std::vector<int> {
                        const std::vector<int> r = inner(c);
                        if (r.empty()) return r;
                        std::string val;
                        for (int t : r) if (t != sub0::nodes::FRAME_OPEN && t != sub0::nodes::FRAME_CLOSE)
                            val.push_back(static_cast<char>(t));
                        const int idx = static_cast<int>(scratch.bindings.size());
                        if (val.empty() || idx >= sub0::SCRATCH_SLOT_COUNT) return r;   // pool full -> digits
                        std::vector<int> frags; for (char ch : val) frags.push_back(static_cast<unsigned char>(ch));
                        scratch.bind(sub0::SCRATCH_SLOT_BASE + idx, std::move(frags));
                        if (content_embed_on) binds = scratch.to_bindings(content_embed_enc);   // refresh the live snapshot
                        return { sub0::SCRATCH_SLOT_BASE + idx };
                    };
                    op_collapse = true;
                } else {
                    compute = sub0::nodes::make_compute_callback(sub0::nodes::builtin(), op_deref);
                }
                if (word_on) {
                    // Live (generation-side) collapse: reuses the SAME combine_recurrence + expand
                    // machinery prefill_collapse (below) uses for the prompt's own text, wrapped for
                    // kv_decode_generate's simpler "empty = not a recurrence" contract.
                    word_collapse = [&scratch, &binds, content_embed_on, content_embed_enc]
                                    (const std::vector<int>& piece_span) -> std::vector<int> {
                        std::vector<int> bytes;
                        for (int p : piece_span) {
                            const std::vector<int> e = scratch.expand(p);
                            bytes.insert(bytes.end(), e.begin(), e.end());
                        }
                        const sub0::ScratchTable::Recurrence r = scratch.combine_recurrence(bytes);
                        if (content_embed_on) binds = scratch.to_bindings(content_embed_enc);
                        return r.is_repeat ? r.tokens : std::vector<int>{};
                    };
                    // PREFILL-time collapse: the prompt's own text may already contain a recurring
                    // compound word (e.g. a name mentioned twice) -- collapse it once, up front,
                    // before generation starts (same mechanism as the live callback above, just
                    // applied to known text instead of a growing KV-cache).
                    ctx = sub0::prefill_collapse(scratch, ctx);
                    if (content_embed_on) binds = scratch.to_bindings(content_embed_enc);
                }
                {
                    std::vector<std::string> parts;
                    if (sp_on)   parts.push_back("spell/scratch");
                    if (op_on)   parts.push_back("op-dispatch");
                    if (word_on) parts.push_back("word-collapse");
                    std::string joined;
                    for (std::size_t i = 0; i < parts.size(); ++i) { if (i) joined += " + "; joined += parts[i]; }
                    std::println(stderr, "gen: interceptor enabled ({}{})", joined,
                                op_collapse ? ", collapse" : "");
                }
                if (content_embed_on)
                    std::println(stderr, "gen: content-embed {}",
                                sub0::content_embed_kind_name(static_cast<int>(*schedule.content_embed)));
            }
        }
    }
    const bool spell_enabled = static_cast<bool>(expand);
    // An --op-mix model's op results, and a wordspike model's collapsed name references, both COLLAPSE
    // to scratch slots -- expand each bound slot back to its value bytes for display (the model reasoned
    // over the symbol; the reader wants the number/name). is_scratch_slot + scratch.value() are generic
    // over WHO bound the slot, so one pass handles both.
    const bool expand_slots_for_display = op_collapse || static_cast<bool>(word_collapse);
    auto expand_op_slots = [&scratch, expand_slots_for_display](std::vector<int>& c) {
        if (!expand_slots_for_display) return;
        std::vector<int> out; out.reserve(c.size());
        for (int t : c) {
            if (sub0::is_scratch_slot(t)) { const std::string v = scratch.value(t);
                // TOK_JOIN between each expanded byte: detokenize_join (tokenizer.cpp) inserts an
                // implicit space before every "content" token unless the previous one was TOK_JOIN or
                // it's inside a TOK_SPELL group -- a bound value's bytes are pushed raw here (no
                // encode-time grouping happened for them), so without an explicit JOIN a multi-char
                // name renders as space-separated letters ("R u m p l e...") instead of one word. A
                // single-char value (most op-collapse digit results) needs no JOIN at all.
                if (!v.empty()) {
                    bool first = true;
                    for (char ch : v) {
                        if (!first) out.push_back(sub0::casing::TOK_JOIN);
                        out.push_back(static_cast<unsigned char>(ch));
                        first = false;
                    }
                    continue;
                }
            }
            out.push_back(t);
        }
        c.swap(out);
    };

    // Fast path: a KV-cache incremental decode (O(T) per token) whenever the whole context fits the
    // trained window and the weights are dense. forward_one's logits match the full forward to
    // fast-math tolerance, so the sampled continuation is the same -- just far cheaper. GPU decode is
    // tried first (device forward_one, faster per token); it falls back to the CPU KV-cache below on
    // any failure (no CUDA build, no device, or an upload error), so this always completes. The general
    // path (ternary weights, or a context that slides past SEQ_LEN) keeps the exact full-forward loop.
    if constexpr (!USE_TERNARY) {
        if (static_cast<int>(ctx.size()) + n <= SEQ_LEN) {
            sub0::DecodeSession sess;
            const bool intercept = spell_enabled || static_cast<bool>(compute) || static_cast<bool>(word_collapse);
            const bool gpu = sess.use_gpu && !intercept;   // interceptor + op dispatch are CPU-only for now
            std::println(stderr, "gen: decode backend {}{}",
                        gpu ? "GPU (device KV-cache)" : "CPU (KV-cache)",
                        intercept ? " + uncombine/op interceptor" : "");
            if (content_embed_on) { binds = scratch.to_bindings(content_embed_enc); sub0::set_scratch_bindings(&binds); }
            sub0::kv_decode_generate(ctx, n, temp, topk, rng, eos_id, gpu, {}, expand, combine,
                                     compute, sub0::nodes::FRAME_CLOSE, word_collapse);
            if (content_embed_on) sub0::set_scratch_bindings(nullptr);
            expand_op_slots(ctx);   // collapsed op results / word-collapsed names -> their bytes for display
            std::println("{}", sub0::detokenize(ctx));
            return 0;
        }
    }

    // SAY that we left the fast path, and why. This loop runs a FULL FORWARD PER TOKEN -- orders of
    // magnitude slower than the KV-cache decode above -- and the transition is invisible from the
    // command line: `--n 500` after a short prompt fits the window and returns in about a second,
    // `--n 512` misses it by the prompt's length and grinds, which is exactly the "gen freezes at
    // --n >= SEQ_LEN" report. Nothing is hung; it is doing ~100x the work per token. Sliding past the
    // trained window is a real capability (it is what --attn-sinks exists for), so this warns rather
    // than clamping -- unlike preview_at, whose callers only ever want a sample that fits.
    if constexpr (!USE_TERNARY)
        std::println(stderr,
                     "gen: prompt ({} tok) + --n {} exceeds the {}-token window -- using the "
                     "sliding-window full-forward path, which is MUCH slower per token (a KV-cache "
                     "decode needs prompt + n <= {}; try --n {}).",
                     ctx.size(), n, SEQ_LEN, SEQ_LEN,
                     std::max(0, SEQ_LEN - static_cast<int>(ctx.size())));

    // Sink tokens: clamped so the "recent" half of the window never shrinks below SEQ_LEN/2 even if
    // the caller asks for an unreasonably large --attn-sinks.
    const int n_sink = attn_sinks > 0 ? std::min(attn_sinks, SEQ_LEN / 2) : 0;
    std::vector<int> window;   // scratch, rebuilt each step once sinks are active; reused to avoid realloc
    for (int s = 0; s < n; ++s) {
        int T;
        const int* win;
        // Once ctx has grown past the trained window, a plain last-SEQ_LEN slice drops the sequence's
        // opening tokens entirely -- StreamingLLM's observation is that those early tokens soak up a
        // disproportionate share of attention mass regardless of their content, so dropping them can
        // degrade quality more than their content alone would suggest. Keep the first n_sink tokens
        // resident alongside the most recent (SEQ_LEN - n_sink), forming a contiguous SEQ_LEN window.
        // RoPE only encodes RELATIVE row position within a forward() call (row index, not the token's
        // true absolute step), so reassigning the sinks to rows [0,n_sink) and the recent tail to
        // [n_sink,SEQ_LEN) needs no change to op_rope/forward() at all -- both halves land in-
        // distribution positions every step, with no artificial position gap between them.
        if (n_sink > 0 && static_cast<int>(ctx.size()) > SEQ_LEN) {
            const int n_recent = SEQ_LEN - n_sink;
            window.assign(ctx.begin(), ctx.begin() + n_sink);
            window.insert(window.end(), ctx.end() - n_recent, ctx.end());
            T = SEQ_LEN;
            win = window.data();
        } else {
            T = std::min((int)ctx.size(), SEQ_LEN);
            win = ctx.data() + (ctx.size() - T);
        }
        sub0::graph_reset();
        sub0::Node* logits = sub0::forward(win, T);
        const int last = logits->rows - 1;
        const int next = sub0::sample_token(logits->data.data() + (size_t)last * VOCAB, temp, topk, rng);
        if (next == eos_id) break;   // learned stop signal: end here, don't print the marker
        ctx.push_back(next);
    }
    sub0::graph_reset();
    expand_op_slots(ctx);   // collapsed op results -> their value digits for display
    std::println("{}", sub0::detokenize(ctx));
    return 0;
}

// Native op-delegation eval: score a trained --op-mix model. Generate synthetic arithmetic problems,
// run the model through the SAME production collapse callback gen uses, and report the decomposed metrics a
// well-formed-math task affords -- because the node is exact, a miss is a ROUTING error, not arithmetic.
extern "C" SUB0_API int sub0_eval_stage(const char* model_in, unsigned seed, int n_problems) {
    sub0::build_model();
    if (!sub0::load_model(model_in)) { std::println(stderr, "eval: cannot load model '{}'", model_in); return 1; }
    std::string tok_path = sub0::default_tokenizer();
    if (model_in && *model_in) {
        const std::filesystem::path bundled = std::filesystem::path(model_in).parent_path() / "tokenizer.tok";
        std::error_code ec; if (std::filesystem::exists(bundled, ec)) tok_path = bundled.string();
    }
    if (!sub0::load_tokenizer(tok_path.c_str())) { std::println(stderr, "eval: cannot load tokenizer '{}'", tok_path); return 1; }
    const int eos_id = sub0::eos_token_id();

    bool op_on = false;
    if (model_in && *model_in) {
        const std::filesystem::path mp(model_in);
        const std::filesystem::path dir = std::filesystem::is_directory(mp) ? mp : mp.parent_path();
        sub0::registry::RunConfig cfg;
        // Same has_blend_schedule gate as gen above: a plain single-corpus model never pinned one, so
        // its absence is expected and silent. Only a model that DID pin one can have a broken one.
        if (!dir.empty() && sub0::registry::read_config_json(cfg, dir) && cfg.config_schema >= 2
            && cfg.has_blend_schedule) {
            sub0::ScheduleSpec schedule;
            std::string schedule_error;
            std::vector<std::string> schedule_warnings;
            if (sub0::parse_blend_schedule_json(dir / "blend_schedule.json", schedule, schedule_error,
                                                schedule_warnings))
                op_on = generator_ever_active(schedule, "op_curriculum");
            else
                std::println(stderr, "eval: '{}' pinned a blend_schedule.json but it is unreadable ({})",
                             dir.string(), schedule_error);
        }
    }
    if (!op_on)
        std::println(stderr, "eval: warning -- model's blend schedule has no active op_curriculum source "
                             "(not an op-delegation model; expect delegation ~0)");

    // The production collapse callback (mirrors gen_stage): resolve the op, bind the result to the next
    // scratch slot, inject that token. The last bound slot's value is the model's answer.
    sub0::ScratchTable scratch;
    auto inner = sub0::nodes::make_compute_callback(sub0::nodes::builtin(), [&scratch](int s) { return scratch.value(s); });
    auto compute = [&scratch, inner](const std::vector<int>& c) -> std::vector<int> {
        const std::vector<int> r = inner(c);
        if (r.empty()) return r;
        std::string val;
        for (int t : r) if (t != sub0::nodes::FRAME_OPEN && t != sub0::nodes::FRAME_CLOSE) val.push_back(static_cast<char>(t));
        const int idx = static_cast<int>(scratch.bindings.size());
        if (val.empty() || idx >= sub0::SCRATCH_SLOT_COUNT) return r;
        std::vector<int> frags; for (char ch : val) frags.push_back(static_cast<unsigned char>(ch));
        scratch.bind(sub0::SCRATCH_SLOT_BASE + idx, std::move(frags));
        return { sub0::SCRATCH_SLOT_BASE + idx };
    };

    const int n = n_problems > 0 ? n_problems : 300;
    std::mt19937_64 ev(seed ? seed : 4242u);
    std::mt19937    grng(0);
    int exact = 0, delegated = 0;
    for (int i = 0; i < n; ++i) {
        const sub0::op_curriculum::EvalProblem pr = sub0::op_curriculum::gen_eval(ev, 3);
        std::vector<int> ctx = sub0::encode(pr.prompt);
        if (static_cast<int>(ctx.size()) + 8 >= SEQ_LEN) continue;
        scratch.reset();
        sub0::kv_decode_generate(ctx, 24, 1.f, 1, grng, eos_id, false, {}, {}, {}, compute, sub0::nodes::FRAME_CLOSE);
        const bool did = !scratch.bindings.empty();
        const std::string got = did ? scratch.value(sub0::SCRATCH_SLOT_BASE + static_cast<int>(scratch.bindings.size()) - 1)
                                    : std::string{};
        delegated += did;
        exact     += (got == pr.gold);
    }
    const double dl = delegated / double(n), ex = exact / double(n);
    std::println("op-eval ({} problems): delegation-rate {:.3f}  exact-match {:.3f}  routing-accuracy {:.3f}",
                 n, dl, ex, delegated ? exact / double(delegated) : 0.0);
    std::println("  (arithmetic is exact by construction when delegation fires -- a miss is a ROUTING error)");
    return 0;
}

// Dump the build's BPE vocabulary as a readable table. Inspection only; no model
// is loaded. `tokenizer_path` may be null to use the baked-in tokenizer.tok, and
// `limit <= 0` prints every entry.
extern "C" SUB0_API int sub0_vocab_stage(const char* tokenizer_path, int limit) {
    const char* path = tokenizer_path ? tokenizer_path : sub0::default_tokenizer();
    if (!sub0::load_tokenizer(path)) {
        std::println(stderr, "vocab: cannot load tokenizer '{}'", path);
        return 1;
    }
    const std::vector<sub0::TokenEntry> rows = sub0::vocab_entries();

    int n_base = 0, n_merge = 0;
    for (const auto& e : rows)
        (e.kind == sub0::TokenEntry::Kind::Merge ? n_merge : n_base) += 1;
    std::println("tokenizer: {} | {} tokens ({} base, {} merges)",
                 path, rows.size(), n_base, n_merge);
    std::println("{:>6}  {:<6}  {:>3}  text", "id", "kind", "len");

    const std::size_t shown =
        (limit > 0) ? std::min(rows.size(), static_cast<std::size_t>(limit)) : rows.size();
    for (std::size_t i = 0; i < shown; ++i) {
        const sub0::TokenEntry& e = rows[i];
        const char* kind = e.kind == sub0::TokenEntry::Kind::Byte      ? "byte"
                         : e.kind == sub0::TokenEntry::Kind::CapMarker ? "cap"
                         : e.kind == sub0::TokenEntry::Kind::UpMarker  ? "up"
                         : e.kind == sub0::TokenEntry::Kind::Eos       ? "eos"
                                                                       : "merge";
        std::println("{:>6}  {:<6}  {:>3}  {}", e.id, kind, e.expansion_len, e.text);
    }
    if (shown < rows.size())
        std::println("... {} more (raise --limit to see all)", rows.size() - shown);
    return 0;
}
