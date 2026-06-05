// sub0llm-episodic — write/recall facts into a model's episodic delta
//
// Usage:
//   sub0llm-episodic write  --model <model.gguf> --fact <text>   --delta <out.epis> [opts]
//   sub0llm-episodic recall --model <model.gguf> --query <text>  --delta <in.epis>
//   sub0llm-episodic probe  --model <model.gguf> --fact <text>   [--query <text>] [opts]
//   sub0llm-episodic info   --model <model.gguf>
//
// probe: validity test — encodes a novel fact then checks:
//   1. Baseline NLL is high  (model doesn't already know this)
//   2. NLL drops after merge (delta is effective)
//   3. Related query also drops (generalises beyond exact tokens)
//   4. Control query unchanged (delta is specific, not corrupting)

#include "sub0llm/nn/episodic_memory.hpp"
#include "sub0llm/nn/gguf_loader.hpp"
#include "sub0llm/nn/modern_gpt.hpp"
#include "sub0llm/nn/sampler.hpp"
#include "sub0llm/tokenizer/bpe.hpp"

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <format>
#include <functional>
#include <iostream>
#include <numeric>
#include <random>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

using namespace sub0llm;
using namespace sub0llm::nn;

// ── Args ──────────────────────────────────────────────────────────────────────

struct Args {
    std::string subcommand;
    std::string model_path;
    std::string delta_path;
    std::string fact;
    std::string query;
    float       lr    = 5e-3f;   // conservative; gradients are clipped (see EpisodicConfig)
    int         steps = 10;
    int64_t     gen_tokens = 40; // length of the demonstrative query continuation (0 = off)
    int         train_layers = -1; // trainable trailing blocks: -1=last half, 0=full model
    int         angles    = 0;   // extra self-generated question->fact framings to rehearse (0 = off)
    int         lora_rank = 0;   // >0 = low-rank frozen-base partition (LoRA) instead of full delta
    float       locality  = 0.0f; // >0 = penalise drift on a generic anchor (keeps existing knowledge)
    bool        adaptive_steps = false; // scale rehearsal steps by fact novelty (fewer if familiar)
    bool        adaptive_lr = false; // scale learning rate by fact novelty (gentler if familiar)
    bool        token_novelty = false; // weight each token's loss by its surprisal
    bool        iterative  = false; // generate angles mid-training (from the improving model)
};

// Progress printer shared by write/probe: shows the comprehension summary and a
// one-line-per-step rehearsal trace so a long write visibly makes progress and
// the loss trend (falling = learning) is obvious.
static std::function<void(const EpisodicProgress&)> make_progress_printer() {
    return [](const EpisodicProgress& p) {
        using Phase = EpisodicProgress::Phase;
        if (p.phase == Phase::Comprehension) {
            std::cerr << std::format(
                "[comprehension] mean NLL = {:.4f} | {} span(s) to rehearse\n",
                p.mean_nll, p.span_count);
        } else {
            std::cerr << std::format(
                "\r[rehearse] span {}/{} (tok {}..{})  step {}/{}  loss={:.4f}  |g|={:.3f}   ",
                p.span_index + 1, p.span_count,
                p.span_start, p.span_start + p.span_len,
                p.step + 1, p.total_steps, p.loss, p.grad_norm);
            if (p.step + 1 == p.total_steps) std::cerr << "\n";
            std::cerr << std::flush;
        }
    };
}

static void usage(const char* argv0) {
    std::cerr <<
        "Usage:\n"
        "  " << argv0 << " write  --model <f.gguf> --fact <text>  --delta <out.epis> [--lr F] [--steps N]\n"
        "  " << argv0 << " recall --model <f.gguf> --query <text> --delta <in.epis>\n"
        "  " << argv0 << " probe  --model <f.gguf> --fact <text>  [--query <text>] [--lr F] [--steps N]\n"
        "  " << argv0 << " suite  --model <f.gguf>  [--steps N] [--angles N] [--locality W] ...\n"
        "  " << argv0 << " info   --model <f.gguf>\n"
        "\n"
        "suite: runs the probe over several built-in facts (model loaded once) and prints a\n"
        "       comparison table — variance across facts, with baseNLL as the novelty signal.\n"
        "\n"
        "Common options (write/probe/suite):\n"
        "  --lr F            rehearsal learning rate (default 5e-3; grads clipped to |g|<=1)\n"
        "  --steps N         rehearsal gradient steps per span (default 10)\n"
        "  --train-layers N  trailing blocks to update: -1=last half (default), 0=full model, N=last N\n"
        "  --angles N        rehearse N self-generated question->fact framings for retrieval linkage\n"
        "                    (default 0; each adds a rehearsal pass, so cost scales ~x(1+N))\n"
        "  --lora-rank R     write into low-rank, base-frozen adapters instead of full weights\n"
        "                    (default 0=off; e.g. 8 — keeps the base specific and the delta tiny)\n"
        "  --locality W      penalise drift on a generic anchor (W·MSE vs base logits), so the\n"
        "                    write stays specific; try 0.1-1.0 (default 0=off)\n"
        "  --adaptive-steps  scale rehearsal steps by fact novelty (fewer steps when the fact\n"
        "                    is close to what the model already knows)\n"
        "  --adaptive-lr     scale learning rate by fact novelty (gentler step for familiar facts)\n"
        "  --token-novelty   weight each token's loss by its surprisal (focus on the novel tokens)\n"
        "  --iterative       grow angles mid-training: each new question is generated from the\n"
        "                    partially-trained model, so it links to what's already embedded\n"
        "  --gen-tokens N    demonstrative query continuation length (default 40; 0=skip)\n"
        "\n"
        "probe: encodes --fact, then reports:\n"
        "  1. baseline NLL for the fact (should be HIGH - model shouldn't know it)\n"
        "  2. merged NLL for the fact   (should drop significantly)\n"
        "  3. merged NLL for --query    (related rephrasing - should also drop)\n"
        "  4. merged NLL for a control  (unrelated phrase - should be UNCHANGED)\n"
        "\nGood --fact examples (novel, not in training data):\n"
        "  \"sub0llm is a C++23 LLM framework built for education\"\n"
        "  \"Project Helix stores KV-cache shards on quantum tapes\"\n"
        "  \"The Zorblax-9 chip runs at 42 petaflops using ice cooling\"\n";
}

static Args parse_args(int argc, char** argv) {
    if (argc < 2) { usage(argv[0]); std::exit(1); }
    Args a;
    a.subcommand = argv[1];
    for (int i = 2; i < argc; ++i) {
        std::string flag = argv[i];
        auto next = [&]() -> std::string {
            if (++i >= argc) throw std::runtime_error(std::format("flag {} requires an argument", flag));
            return argv[i];
        };
        if      (flag == "--model") a.model_path = next();
        else if (flag == "--delta") a.delta_path = next();
        else if (flag == "--fact")  a.fact  = next();
        else if (flag == "--query") a.query = next();
        else if (flag == "--lr")    a.lr    = std::stof(next());
        else if (flag == "--steps") a.steps = std::stoi(next());
        else if (flag == "--gen-tokens")   a.gen_tokens   = std::stoll(next());
        else if (flag == "--train-layers") a.train_layers = std::stoi(next());
        else if (flag == "--angles")       a.angles       = std::stoi(next());
        else if (flag == "--lora-rank")    a.lora_rank    = std::stoi(next());
        else if (flag == "--locality")     a.locality     = std::stof(next());
        else if (flag == "--adaptive-steps") a.adaptive_steps = true;
        else if (flag == "--adaptive-lr")    a.adaptive_lr    = true;
        else if (flag == "--token-novelty")  a.token_novelty  = true;
        else if (flag == "--iterative")      a.iterative      = true;
        else { std::cerr << "Unknown flag: " << flag << "\n"; std::exit(1); }
    }
    return a;
}

// ── Helpers ───────────────────────────────────────────────────────────────────

// Model + its real BPE tokenizer, both built from one GGUF parse.  Using the
// model's own tokenizer (rather than raw bytes) is what makes the surprisal
// signal meaningful and lets us decode generated continuations to readable text.
struct LoadedModel {
    ModernGPT    model;
    BPETokenizer tok;
};

static LoadedModel load_model(const Args& args) {
    if (args.model_path.empty())
        throw std::runtime_error("--model is required");
    if (!args.model_path.ends_with(".gguf"))
        throw std::runtime_error(std::format("Only .gguf files are supported, got: '{}'", args.model_path));
    std::cerr << "[info] loading GGUF: " << args.model_path << "\n";
    GGUFReader reader(args.model_path);   // header parse only — cheap
    return LoadedModel{
        load_gguf_model(reader),
        BPETokenizer::from_vocab(reader.vocab().tokens, reader.vocab().merges),
    };
}

// Encode text with the model's BPE tokenizer; ensure at least one token so the
// downstream forward() never sees an empty sequence.
static std::vector<int32_t> tokenize(const BPETokenizer& tok, const std::string& text) {
    auto ids = tok.encode(text);
    if (ids.empty()) ids.push_back(0);
    return ids;
}

// Greedily continue `prompt` for max_tokens and return ONLY the generated text
// (prompt stripped).  Caller controls whether the episodic delta is merged.
static std::string generate_response(ModernGPT& model, const BPETokenizer& tok,
                                     const std::string& prompt, int64_t max_tokens) {
    auto ids = tokenize(tok, prompt);
    nn::SamplingConfig sc;
    sc.mode = nn::SamplingMode::Greedy;
    std::mt19937 rng(42);
    const auto all = nn::generate(model, ids, max_tokens, sc, rng);
    const std::vector<int32_t> gen(all.begin() + static_cast<std::ptrdiff_t>(ids.size()),
                                   all.end());
    return tok.decode(gen);
}

static float mean_nll(ModernGPT& model, const std::vector<int32_t>& tokens) {
    auto ls = comprehension_pass(model, tokens);
    return ls.empty() ? 0.0f :
           std::accumulate(ls.begin(), ls.end(), 0.0f) / static_cast<float>(ls.size());
}

static void waypoint(std::string_view msg);   // defined in Output helpers below

// ── Multi-angle elaboration ─────────────────────────────────────────────────────
// A single rehearsal only teaches the fact's exact surface form, so a reworded
// query doesn't benefit (the PASS3 failure). "Thinking from different angles" asks
// the base model to pose its own questions about the fact, then rehearses each
// "Question: ...\nAnswer: <fact>" framing — building several retrieval paths into
// the same knowledge. Questions (not paraphrases) are used so an imperfect small
// model can't assert a *false* restatement of the fact.

static std::string generate_question(ModernGPT& model, const BPETokenizer& tok,
                                     const std::string& fact, int seed) {
    const std::string prompt =
        fact + "\n\nWrite a question about the statement above.\nQuestion:";
    auto ids = tokenize(tok, prompt);
    nn::SamplingConfig sc;
    sc.mode        = nn::SamplingMode::TopK;
    sc.top_k       = 40;
    sc.temperature = 0.9f;   // vary phrasing across angles
    std::mt19937 rng(static_cast<std::uint32_t>(101 + seed));
    const auto all = nn::generate(model, ids, /*max_new=*/24, sc, rng);
    std::string q = tok.decode(std::vector<int32_t>(
        all.begin() + static_cast<std::ptrdiff_t>(ids.size()), all.end()));
    // Keep a single clean question: cut at first newline, end at first '?'.
    if (auto p = q.find('\n'); p != std::string::npos) q = q.substr(0, p);
    if (auto p = q.find('?');  p != std::string::npos) q = q.substr(0, p + 1);
    while (!q.empty() && q.front() == ' ') q.erase(q.begin());
    return q;
}

// Significant words (lowercased, length >= 5) — long enough to skip most
// stop/question words, so overlap is a cheap on-topic signal.  Split only on
// whitespace and drop intra-word punctuation, so "sub-0llm" still matches
// "sub0llm".
static std::set<std::string> content_words(const std::string& s) {
    std::set<std::string> out;
    std::string w;
    auto flush = [&] { if (w.size() >= 5) out.insert(w); w.clear(); };
    for (char c : s) {
        if (c == ' ' || c == '\n' || c == '\t' || c == '\r') flush();
        else if (std::isalnum(static_cast<unsigned char>(c)))
            w += static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        // intra-word punctuation (e.g. '-', '+') is dropped without splitting
    }
    flush();
    return out;
}

// Encode the fact, then a few self-generated question->fact framings, all
// accumulated into the same delta. Off-topic generations (no content word shared
// with the fact) are dropped, and the rehearsal step budget is split across the
// used documents so total gradient movement — hence drift — stays ~constant
// regardless of angle count. Angles are generated on the base model (the fact
// encode restores weights) and printed for transparency.
static void encode_with_angles(ModernGPT& model, const BPETokenizer& tok,
                               EpisodicState& state, const std::string& fact,
                               const EpisodicConfig& cfg, int n_angles,
                               bool verbose = true, bool iterative = false) {
    // ── Iterative (warm-up) mode ────────────────────────────────────────────────
    // Generate angles from a LIGHTLY-trained model: a short warm-up gives the model
    // partial grounding in the fact (so its questions are connected) WITHOUT
    // overfitting (which, measured, degrades its generation if it trains to
    // convergence first). Then the real joint rehearsal runs from base over the
    // fact + those grounded angles.
    if (iterative && n_angles > 0) {
        EpisodicConfig warm_cfg = cfg;
        warm_cfg.think_steps     = std::max(2, cfg.think_steps / 4);
        warm_cfg.locality_weight = 0.0f;          // keep warm-up light & fast
        warm_cfg.locality_anchor.clear();
        warm_cfg.adaptive_steps  = false;
        warm_cfg.on_progress     = nullptr;       // quiet

        EpisodicState warm = make_episodic_state(model);
        if (verbose)
            waypoint(std::format("warm-up: {} step(s) to ground angle generation",
                                 warm_cfg.think_steps));
        episodic_encode(model, warm, std::vector<std::vector<int32_t>>{ tokenize(tok, fact) }, warm_cfg);

        // Generate angles from the warmed (grounded, not overfit) model.
        warm.merge(model);
        const auto fact_words = content_words(fact);
        std::vector<std::vector<int32_t>> docs{ tokenize(tok, fact) };
        for (int i = 0; i < n_angles; ++i) {
            const std::string q = generate_question(model, tok, fact, 300 + i);
            bool on_topic = false;
            for (const auto& w : content_words(q))
                if (fact_words.count(w)) { on_topic = true; break; }
            if (verbose)
                std::cout << std::format("  iter-angle {}: {} Q: \"{}\"\n",
                                         i + 1, on_topic ? "[use] " : "[skip]", q);
            if (on_topic)
                docs.push_back(tokenize(tok, "Question: " + q + "\nAnswer: " + fact));
        }
        warm.unmerge(model);   // discard warm-up; real rehearsal starts from base

        if (verbose)
            waypoint(std::format("joint rehearsal of {} doc(s) for {} step(s)",
                                 docs.size(), cfg.think_steps));
        episodic_encode(model, state, docs, cfg);
        return;
    }

    // ── Up-front mode (default) ─────────────────────────────────────────────────
    // The fact is always document 0.
    std::vector<std::vector<int32_t>> docs{ tokenize(tok, fact) };

    if (n_angles > 0) {
        // Self-generate question framings; drop off-topic ones (no content-word
        // overlap with the fact). Kept ones become extra documents.
        const auto fact_words = content_words(fact);
        for (int i = 0; i < n_angles; ++i) {
            const std::string q = generate_question(model, tok, fact, i);
            if (q.empty()) continue;
            bool on_topic = false;
            for (const auto& w : content_words(q))
                if (fact_words.count(w)) { on_topic = true; break; }
            if (verbose)
                std::cout << std::format("  angle {}: {} Q: \"{}\"\n",
                                         i + 1, on_topic ? "[use] " : "[skip]", q);
            if (on_topic)
                docs.push_back(tokenize(tok, "Question: " + q + "\nAnswer: " + fact));
        }
    }

    // Rehearse fact + angles JOINTLY (one session): the framings co-train for
    // retrieval linkage, and the locality penalty constrains the combined write.
    if (verbose)
        waypoint(std::format("joint rehearsal of {} doc(s) for {} step(s)",
                             docs.size(), cfg.think_steps));
    episodic_encode(model, state, docs, cfg);
}

// If --lora-rank > 0, freeze the base and attach low-rank adapters to the late
// FFNs BEFORE make_episodic_state, so the write becomes a small, base-frozen,
// low-rank delta (specific by construction).  Returns the EpisodicConfig flag set.
static void maybe_enable_lora(ModernGPT& model, const Args& args) {
    if (args.lora_rank <= 0) return;
    waypoint(std::format(
        "partition: attaching LoRA rank {} to last-{} FFNs (base frozen)",
        args.lora_rank, args.train_layers < 0 ? std::string("half")
                            : std::to_string(args.train_layers)));
    model.enable_episodic_lora(args.train_layers, args.lora_rank);
}

// ── Output helpers ──────────────────────────────────────────────────────────────
// Episodic memory has no generated "answer" to print — the observable result is
// the model's surprisal (NLL) on a piece of text, before vs after the delta.
// These helpers echo the actual prompt text and report each waypoint clearly so a
// run shows what is being processed and whether it worked.

// Section marker for a significant waypoint (to stderr, alongside progress).
static void waypoint(std::string_view msg) {
    std::cerr << "-> " << msg << "\n" << std::flush;
}

// Echo a labelled prompt string the operation is about to process.
static void echo_prompt(std::string_view label, std::string_view text,
                        std::size_t n_tokens) {
    std::cout << std::format("  {:<8} \"{}\"  ({} tokens)\n", label, text, n_tokens);
}

// Classify an NLL change: lower surprisal after the delta = recall succeeded.
// `thresh_pct` is the minimum % drop to count as a clear success.
static std::string nll_verdict(float before, float after, float thresh_pct = 5.0f) {
    const float drop = 100.0f * (1.0f - after / before);   // +ve = improved
    if (drop >=  thresh_pct) return std::format("[OK]   recalled  (NLL -{:.1f}%)", drop);
    if (drop <= -thresh_pct) return std::format("[FAIL] degraded  (NLL +{:.1f}%)", -drop);
    return                          std::format("[~]    neutral   (NLL {:+.1f}%)", drop);
}

// ── Subcommands ───────────────────────────────────────────────────────────────

// Generic, neutral text the episodic write should NOT disturb — the locality
// regulariser anchors the model's logits on it to the base. Deliberately
// unrelated to the PASS4 control, so a low drift there is genuine transfer, not
// anchor overlap. (A longer/multi-sentence anchor dilutes the per-token pressure
// under a fixed weight, so a single focused sentence held tighter.)
static const std::string k_locality_anchor =
    "The river flowed gently past the old stone bridge while travellers rested nearby.";

// Unrelated phrase used to measure specificity (PASS4 control drift).
static const std::string k_control = "the quick brown fox jumps over the lazy dog";

// Pass thresholds (shared by single probe and suite so they never diverge).
static constexpr float kMinBaseline  = 2.5f;  // fact NLL must exceed this (novel)
static constexpr float kMinReduction = 5.0f;  // min % NLL drop on the fact
static constexpr float kMaxDrift     = 2.0f;  // max |%| control NLL change

struct Verdict { bool p1, p2, p3, p4; [[nodiscard]] bool core() const { return p2 && p3; } };

static Verdict verdict_of(float fb, float fm, float qm, float cb, float cm) {
    auto pct = [](float b, float a) { return 100.0f * (a - b) / b; };
    return Verdict{
        fb >= kMinBaseline,
        pct(fb, fm) <= -kMinReduction,
        qm < fb,
        std::abs(pct(cb, cm)) <= kMaxDrift,
    };
}

// Build an EpisodicConfig from CLI args (used by write / probe / suite).
static EpisodicConfig build_cfg(const Args& args, const BPETokenizer& tok, bool with_progress) {
    EpisodicConfig cfg;
    cfg.learning_rate         = args.lr;
    cfg.think_steps           = args.steps;
    cfg.surprise_threshold    = 0.0f;
    cfg.accumulate            = true;
    cfg.trainable_last_layers = args.train_layers;
    cfg.lora_rank             = args.lora_rank;
    cfg.adaptive_steps        = args.adaptive_steps;
    cfg.adaptive_lr           = args.adaptive_lr;
    cfg.token_novelty_weight  = args.token_novelty;
    cfg.locality_weight       = args.locality;
    if (args.locality > 0.0f) cfg.locality_anchor = tokenize(tok, k_locality_anchor);
    if (with_progress)        cfg.on_progress = make_progress_printer();
    return cfg;
}

// Measured NLLs for one probe case.
struct CaseMetrics { float fb, fm, qm, cb, cm; };

// Run one probe case end-to-end (fresh delta), returning the NLLs. `verbose`
// controls the per-step / angle chatter; the suite runs quiet.
static CaseMetrics run_probe_case(ModernGPT& model, const BPETokenizer& tok,
                                  const std::string& fact, const std::string& query,
                                  EpisodicConfig cfg, int angles, bool verbose,
                                  bool iterative) {
    auto fact_toks    = tokenize(tok, fact);
    auto query_toks   = query.empty() ? fact_toks : tokenize(tok, query);
    auto control_toks = tokenize(tok, k_control);
    if (!verbose) cfg.on_progress = nullptr;

    CaseMetrics m;
    m.fb = mean_nll(model, fact_toks);
    m.cb = mean_nll(model, control_toks);

    EpisodicState state = make_episodic_state(model);
    encode_with_angles(model, tok, state, fact, cfg, angles, verbose, iterative);

    state.merge(model);
    m.fm = mean_nll(model, fact_toks);
    m.qm = mean_nll(model, query_toks);
    m.cm = mean_nll(model, control_toks);
    state.unmerge(model);
    return m;
}

static int cmd_info(const Args& args) {
    if (args.model_path.empty()) throw std::runtime_error("--model is required");
    GGUFReader reader(args.model_path);
    const GGUFModelConfig cfg = reader.config();
    const int64_t head_dim = cfg.head_dim > 0
                                ? cfg.head_dim
                                : cfg.embed_dim / static_cast<int64_t>(cfg.n_heads);
    std::cout << std::format(
        "arch:       {}\nvocab_size: {}\nembed_dim:  {}\n"
        "n_heads:    {}\nn_kv_heads: {}\nhead_dim:   {}{}\nn_layers:   {}\n"
        "d_ff:       {}\ncontext:    {}\nrope_base:  {}\n"
        "lm_head:    {}\n",
        cfg.arch, cfg.vocab_size, cfg.embed_dim,
        cfg.n_heads, cfg.n_kv_heads,
        head_dim, cfg.head_dim > 0 ? " (explicit)" : " (derived)",
        cfg.n_layers,
        cfg.d_ff, cfg.context_len, cfg.rope_base,
        cfg.has_separate_lm_head ? "separate (output.weight)" : "tied (token_embd.weight)");
    auto model = load_gguf_model(reader);
    std::cout << std::format("n_params:   {}\n", model.parameters().size());
    return 0;
}

static int cmd_write(const Args& args) {
    if (args.fact.empty())       throw std::runtime_error("--fact is required for write");
    if (args.delta_path.empty()) throw std::runtime_error("--delta is required for write");

    auto loaded = load_model(args);
    auto& model = loaded.model;
    auto& tok   = loaded.tok;
    maybe_enable_lora(model, args);   // before make_episodic_state (adds params)

    EpisodicState state;
    if (std::filesystem::exists(args.delta_path)) {
        std::cerr << "[info] loading existing delta: " << args.delta_path << "\n";
        state = load_episodic_state(args.delta_path);
    } else {
        state = make_episodic_state(model);
    }

    auto tokens = tokenize(tok, args.fact);

    std::cout << "WRITE - encode a fact into the episodic delta\n";
    echo_prompt("fact:", args.fact, tokens.size());
    std::cout << std::format(
        "  config   lr={:.1e}  steps={}  train-layers={}  angles={}  lora-rank={}\n\n",
        args.lr, args.steps,
        args.train_layers < 0 ? "half"
            : (args.train_layers == 0 ? "all" : std::to_string(args.train_layers)),
        args.angles,
        args.lora_rank > 0 ? std::to_string(args.lora_rank) : std::string("off"));

    waypoint("measuring baseline surprisal");
    const float nll_before = mean_nll(model, tokens);

    waypoint("elaborative rehearsal (gradient delta)");
    EpisodicConfig cfg = build_cfg(args, tok, /*with_progress=*/true);
    encode_with_angles(model, tok, state, args.fact, cfg, args.angles, true, args.iterative);

    waypoint("measuring surprisal with delta merged");
    state.merge(model);
    const float nll_after = mean_nll(model, tokens);
    state.unmerge(model);

    save_episodic_state(state, args.delta_path);

    std::cout << std::format(
        "\nResult:\n"
        "  NLL  {:.4f} -> {:.4f}   {}\n"
        "  delta saved to: {}\n",
        nll_before, nll_after, nll_verdict(nll_before, nll_after),
        args.delta_path);
    return 0;
}

static int cmd_recall(const Args& args) {
    if (args.query.empty())      throw std::runtime_error("--query is required for recall");
    if (args.delta_path.empty()) throw std::runtime_error("--delta is required for recall");

    auto loaded = load_model(args);
    auto& model = loaded.model;
    auto& tok   = loaded.tok;

    auto state  = load_episodic_state(args.delta_path);
    auto tokens = tokenize(tok, args.query);

    std::cout << "RECALL - does the episodic delta change the model's response?\n";
    echo_prompt("query:", args.query, tokens.size());
    std::cout << std::format("  delta:   {}\n\n", args.delta_path);

    waypoint("measuring baseline surprisal (delta off)");
    const float nll_base = mean_nll(model, tokens);

    waypoint("measuring surprisal (delta merged)");
    state.merge(model);
    const float nll_merged = mean_nll(model, tokens);
    state.unmerge(model);

    std::cout << std::format(
        "\nSurprisal:\n"
        "  NLL  {:.4f} -> {:.4f}   {}\n",
        nll_base, nll_merged, nll_verdict(nll_base, nll_merged));

    // Show the actual generated continuation with the delta off vs on, so the
    // effect of the episodic memory is directly visible rather than only numeric.
    if (args.gen_tokens > 0) {
        waypoint(std::format("generating {}-token response (base, then episodic)",
                             args.gen_tokens));
        const std::string base_resp = generate_response(model, tok, args.query, args.gen_tokens);
        state.merge(model);
        const std::string epis_resp = generate_response(model, tok, args.query, args.gen_tokens);
        state.unmerge(model);

        std::cout << std::format(
            "\nResponse to \"{}\":\n"
            "  base: {}\n"
            "  epis: {}\n",
            args.query, base_resp, epis_resp);
    }
    return 0;
}

// ── probe ─────────────────────────────────────────────────────────────────────
// A self-contained validity test for episodic encoding.
// Checks four conditions that together prove the delta is working correctly:
//
//   PASS 1 — high baseline: model doesn't already know the novel fact
//   PASS 2 — NLL reduction: delta actually encodes the information
//   PASS 3 — query transfer: NLL also drops on a rephrasing (generalisation)
//   PASS 4 — specificity:   NLL is unchanged for a completely unrelated phrase
//
// Thresholds are conservative and intended for real models (NLL ≥ 3.0 typical).
// On the tiny synthetic test GGUF (V=64, D=32, L=2) all NLLs cluster around
// 4.2 and reductions are modest — that's expected.

static int cmd_probe(const Args& args) {
    if (args.fact.empty()) throw std::runtime_error("--fact is required for probe");

    auto loaded = load_model(args);
    auto& model = loaded.model;
    auto& tok   = loaded.tok;
    maybe_enable_lora(model, args);   // zero-effect at init, so baselines are unchanged

    auto fact_toks    = tokenize(tok, args.fact);
    auto query_toks   = args.query.empty()
                            ? fact_toks   // fall back to fact itself if no --query given
                            : tokenize(tok, args.query);
    auto control_toks = tokenize(tok, k_control);

    std::cout << "PROBE - validity test for episodic encoding\n";
    echo_prompt("fact:",    args.fact, fact_toks.size());
    echo_prompt("query:",   args.query.empty() ? "(same as fact)" : args.query,
                            query_toks.size());
    echo_prompt("control:", k_control, control_toks.size());
    std::cout << std::format(
        "  config   lr={:.1e}  steps={}  train-layers={}  angles={}  lora-rank={}\n\n",
        args.lr, args.steps,
        args.train_layers < 0 ? "half"
            : (args.train_layers == 0 ? "all" : std::to_string(args.train_layers)),
        args.angles,
        args.lora_rank > 0 ? std::to_string(args.lora_rank) : std::string("off"));

    // Baseline — before any encoding
    waypoint("measuring baseline surprisal (fact + control)");
    const float nll_fact_base    = mean_nll(model, fact_toks);
    const float nll_control_base = mean_nll(model, control_toks);

    // Encode the fact into a fresh delta
    waypoint("encoding fact into a fresh delta");
    EpisodicState state = make_episodic_state(model);
    EpisodicConfig cfg = build_cfg(args, tok, /*with_progress=*/true);
    encode_with_angles(model, tok, state, args.fact, cfg, args.angles, true, args.iterative);

    // Measure everything with the delta merged in
    waypoint("measuring surprisal with delta merged (fact + query + control)");
    state.merge(model);
    const float nll_fact_merged    = mean_nll(model, fact_toks);
    const float nll_query_merged   = mean_nll(model, query_toks);
    const float nll_control_merged = mean_nll(model, control_toks);
    state.unmerge(model);

    // ── Report ────────────────────────────────────────────────────────────────
    auto pct = [](float before, float after) {
        return 100.0f * (after - before) / before;
    };
    auto verdict = [](bool pass) { return pass ? "PASS" : "FAIL"; };

    // Thresholds — tunable; conservative defaults for real ~500M models.
    // The tiny synthetic GGUF will pass PASS2/3 but PASS1/4 depend on the model.
    const float k_min_baseline   = 2.5f;   // fact NLL must exceed this (model shouldn't know it)
    const float k_min_reduction  = 5.0f;   // minimum % NLL drop on the fact itself
    const float k_max_drift      = 2.0f;   // control NLL must not change by more than this %

    const bool p1 = nll_fact_base    >= k_min_baseline;
    const bool p2 = pct(nll_fact_base, nll_fact_merged) <= -k_min_reduction;
    const bool p3 = nll_query_merged  < nll_fact_base;     // query drops below fact baseline
    const bool p4 = std::abs(pct(nll_control_base, nll_control_merged)) <= k_max_drift;

    const bool core_ok = p2 && p3;   // p1/p4 are informational; p2+p3 are the core test

    std::cout << std::format(
        "\nResult:\n"
        "  PASS1 baseline high  NLL(fact,base)   = {:.4f}  (>={:.1f})          [{}]\n"
        "  PASS2 NLL reduction  {:.4f} -> {:.4f}  ({:+.1f}%)        [{}]\n"
        "  PASS3 query transfer NLL(query,merged) = {:.4f}  (<{:.4f})    [{}]\n"
        "  PASS4 specificity    control drift     = {:+.1f}%  (|x|<={:.1f}%)  [{}]\n"
        "\n"
        "  core test (PASS2 + PASS3): {}\n"
        "  all four checks:           {}\n",
        nll_fact_base, k_min_baseline,           verdict(p1),
        nll_fact_base, nll_fact_merged, pct(nll_fact_base, nll_fact_merged),  verdict(p2),
        nll_query_merged, nll_fact_base,          verdict(p3),
        pct(nll_control_base, nll_control_merged), k_max_drift,               verdict(p4),
        core_ok ? "SUCCESS" : "FAILURE",
        (p1 && p2 && p3 && p4) ? "ALL PASS" : "PARTIAL - see above");

    // Demonstrative generation: continue the query with the delta off vs on so
    // the encoded fact's effect on actual output is visible, not just the NLL.
    if (args.gen_tokens > 0) {
        const std::string& q = args.query.empty() ? args.fact : args.query;
        waypoint(std::format("generating {}-token response (base, then episodic)",
                             args.gen_tokens));
        const std::string base_resp = generate_response(model, tok, q, args.gen_tokens);
        state.merge(model);
        const std::string epis_resp = generate_response(model, tok, q, args.gen_tokens);
        state.unmerge(model);
        std::cout << std::format(
            "\nResponse to \"{}\":\n"
            "  base: {}\n"
            "  epis: {}\n",
            q, base_resp, epis_resp);
    }
    return core_ok ? 0 : 1;
}

// ── suite ─────────────────────────────────────────────────────────────────────
// Runs the probe over several diverse facts (model loaded once) so variance is
// visible — some facts encode/transfer far better than others, and the baseline
// NLL column is the novelty signal (low = the model already half-knows it, so it
// needs fewer steps; with --adaptive-steps the rehearsal scales down automatically).

static int cmd_suite(const Args& args) {
    auto loaded = load_model(args);
    auto& model = loaded.model;
    auto& tok   = loaded.tok;
    maybe_enable_lora(model, args);

    struct Case { std::string fact; std::string query; };
    const std::vector<Case> cases = {
        {"sub0llm is a C++23 educational LLM framework by CraigHutchinson",
         "what is sub0llm used for?"},
        {"Project Zephyr stores its data on crystal lattices at 9 kelvin",
         "where does Project Zephyr store its data?"},
        {"The Zorblax-9 chip runs at 42 petaflops using ice cooling",
         "how is the Zorblax-9 chip cooled?"},
        {"Aria Thornquist won the 2089 Galactic Chess Championship in Brussels",
         "who won the 2089 Galactic Chess Championship?"},
        {"Mount Everest is the tallest mountain on Earth",   // familiar → low baseline
         "what is the tallest mountain on Earth?"},
    };

    std::cout << std::format(
        "SUITE - {} cases | steps={} angles={} locality={:.1f} lora={} adaptive={}\n\n",
        cases.size(), args.steps, args.angles, args.locality,
        args.lora_rank > 0 ? std::to_string(args.lora_rank) : std::string("off"),
        args.adaptive_steps ? "yes" : "no");
    std::cout << "  baseNLL  fact-drop  queryNLL  P1 P2 P3 P4  core   fact\n";
    std::cout << "  -------  ---------  --------  -- -- -- --  ----   ----\n";

    EpisodicConfig cfg = build_cfg(args, tok, /*with_progress=*/false);
    int core_pass = 0, all_pass = 0;
    auto yn = [](bool b) { return b ? "Y " : "- "; };
    for (const auto& c : cases) {
        std::cerr << "-> case: " << c.fact << "\n" << std::flush;
        auto m = run_probe_case(model, tok, c.fact, c.query, cfg, args.angles,
                                /*verbose=*/false, args.iterative);
        auto v = verdict_of(m.fb, m.fm, m.qm, m.cb, m.cm);
        const float drop = 100.0f * (m.fm - m.fb) / m.fb;
        if (v.core())                     ++core_pass;
        if (v.p1 && v.p2 && v.p3 && v.p4) ++all_pass;
        std::cout << std::format(
            "  {:7.2f}  {:+8.0f}%  {:8.2f}  {}{}{}{}  {}   {:.46}\n",
            m.fb, drop, m.qm, yn(v.p1), yn(v.p2), yn(v.p3), yn(v.p4),
            v.core() ? "PASS" : "fail", c.fact);
    }
    std::cout << std::format(
        "\n  core (PASS2+PASS3): {}/{}    all-four: {}/{}\n",
        core_pass, cases.size(), all_pass, cases.size());
    return core_pass == static_cast<int>(cases.size()) ? 0 : 1;
}

// ── main ──────────────────────────────────────────────────────────────────────

int main(int argc, char** argv) {
    try {
        auto args = parse_args(argc, argv);
        if      (args.subcommand == "info")   return cmd_info(args);
        else if (args.subcommand == "write")  return cmd_write(args);
        else if (args.subcommand == "recall") return cmd_recall(args);
        else if (args.subcommand == "probe")  return cmd_probe(args);
        else if (args.subcommand == "suite")  return cmd_suite(args);
        else {
            std::cerr << "Unknown subcommand: " << args.subcommand << "\n";
            usage(argv[0]);
            return 1;
        }
    } catch (const std::exception& e) {
        std::cerr << "[error] " << e.what() << "\n";
        return 1;
    }
}
