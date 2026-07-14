// cli_stages.hpp — the per-stage CLI runners, factored out of the driver so each is defined ONCE and
// reused by both the umbrella `sub0llm` executable and the thin per-stage executables
// (sub0llm-train / -gen / -tune). Each runner builds its own CLI::App, parses, and dispatches into the
// stage shared libraries through their extern "C" entry points; it carries no model logic. Including
// this pulls in the generated config (DEFAULT_CORPUS / HAS_CUDA / tuned defaults), so anything that uses
// it is a post-configure (state-2) tool — the configurator itself never includes it.

#pragma once

#include "sub0/core.hpp"  // generated config: DEFAULT_CORPUS, HAS_CUDA, DEFAULT_THREADS, ...

#include <cmath>
#include <map>
#include <print>
#include <random>
#include <string>

#include <CLI/CLI.hpp>

// Entry points provided by the stage libraries (libsub0_train, libsub0_gen).
extern "C" int sub0_train_stage(const char* corpus, const char* model_out,
                                int steps, int batch, float lr, unsigned seed, int keep,
                                int optimizer, int resume_mode, float spell_mix, float scratch_mix,
                                float op_mix);
extern "C" int sub0_gen_stage(const char* model_in, const char* prompt,
                              int n, float temp, int topk, unsigned seed, int attn_sinks);
extern "C" int sub0_eval_stage(const char* model_in, unsigned seed, int n_problems);
extern "C" int sub0_vocab_stage(const char* tokenizer_path, int limit);
extern "C" int sub0_bench_stage(int iters, int threads, int windows_per_thread);
extern "C" int sub0_tune_stage(int max_threads, int verbose, int backend, int thorough, int budget_s);
extern "C" int sub0_autotemp_stage(const char* model_in, unsigned seed, int verbose);
extern "C" int sub0_report_stage(const char* model_in);
extern "C" int sub0_memplan_stage();
extern "C" int sub0_models_stage(int prune, int verbose, const char* corpus_filter,
                                 const char* sha_filter, const char* since, const char* until,
                                 int metrics, int refresh, int force);
extern "C" int sub0_bundle_stage(const char* model_in);
extern "C" int sub0_ckpt2model_stage(const char* ckpt_in, const char* model_out);

namespace sub0cli {

// --- train -----------------------------------------------------------------
// Default the minibatch to the tuned data-parallel width (threads x windows/thread, baked in by the
// configurator) so a default run saturates the cores. lr is the batch-8 default scaled by
// sqrt(batch/8) -- a bigger batch means fewer, lower-variance updates, so it wants a proportionally
// larger step (sqrt rule for Adam).
inline int run_train(int argc, char** argv) {
    constexpr int   LR_BASE_BATCH = 8;
    constexpr float LR_BASE       = 0.001f;
    CLI::App app{"sub0llm-train — train a model (resumes from its .ckpt); auto-names a registered "
                 "model dir when no path is given"};
    std::string train_model, train_corpus{DEFAULT_CORPUS};
    int   train_steps = 0, train_batch = 0;
    float train_lr    = LR_BASE;
    unsigned train_seed = 42;
    std::string train_keep = "3";
    int train_optimizer = 0;   // 0=adamw (default), 1=muon (hidden 2D matrices) + adamw (everything else)
    float train_spell_mix = 0.f;   // 0 = base corpus only; >0 blends in the uncombine curriculum
    float train_scratch_mix = 0.f; // 0 = off; >0 blends in the scratch-token (context-translation) curriculum
    float train_op_mix = 0.f;      // 0 = off; >0 blends in the op-delegation (math node routing) curriculum
    bool train_resume = false, train_fresh = false;
    app.add_option("model", train_model,
                   "Output model path (optional; omit to auto-name by corpus+dims -- see --resume/--fresh)");
    app.add_option("corpus", train_corpus, "Training corpus")->capture_default_str();
    app.add_option("--steps", train_steps,
                   "Training steps (0 = auto-size to corpus, stop on validation plateau)")->capture_default_str();
    app.add_option("--batch", train_batch,
                   "Minibatch size (0 = auto: tuned GPU batch on a CUDA build, else CPU data-parallel width)");
    auto* train_lr_opt =
        app.add_option("--lr", train_lr, "Learning rate (default: 0.001 scaled by sqrt(batch/8))")
           ->capture_default_str();
    app.add_option("--seed", train_seed, "RNG seed")->capture_default_str();
    app.add_option("--keep", train_keep,
                   "Progress-named checkpoints to retain (N, or ALL to keep every stage)")->capture_default_str();
    app.add_option("--optimizer", train_optimizer,
                   "adamw (default) | muon (hidden 2D weight matrices on Muon, everything else stays "
                   "AdamW -- CPU and GPU)")
       ->transform(CLI::CheckedTransformer(std::map<std::string, int>{{"adamw", 0}, {"muon", 1}}, CLI::ignore_case))
       ->default_str("adamw");
    auto* resume_flag = app.add_flag("--resume", train_resume,
                   "No explicit model path: force-resume the most recent matching-architecture model "
                   "dir, even if it already finished or the code version changed underneath it");
    app.add_flag("--fresh", train_fresh,
                "No explicit model path: force a brand-new dated model dir, even if a resumable one exists")
       ->excludes(resume_flag);
    app.add_option("--spell-mix", train_spell_mix,
                   "Fraction of training windows drawn from the synthetic uncombine/combine curriculum "
                   "(0 = base corpus only; e.g. 0.1 = ~10% curriculum). Teaches the model to REQUEST "
                   "character-level tokenizer ops (spell/count) that gen fulfils.")
       ->capture_default_str()->check(CLI::Range(0.0f, 0.9f));
    app.add_option("--scratch-mix", train_scratch_mix,
                   "Fraction of training windows drawn from the scratch-token (context-translation) "
                   "curriculum (0 = off). Teaches the model to RESOLVE a dynamically-bound scratch slot "
                   "for an OOV, so an OOV costs 1 token per mention (gen resolves via a binding table).")
       ->capture_default_str()->check(CLI::Range(0.0f, 0.9f));
    app.add_option("--op-mix", train_op_mix,
                   "Fraction of training windows drawn from the op-delegation curriculum (0 = off). Teaches "
                   "the model to ROUTE arithmetic to the deterministic `math` node (emit [op math]; the node "
                   "supplies the exact answer, which gen dispatches) rather than computing it -- including "
                   "multi-step chains via collapsed scratch slots. See docs/DETERMINISTIC_MECHANISMS.md.")
       ->capture_default_str()->check(CLI::Range(0.0f, 0.9f));
    CLI11_PARSE(app, argc, argv);

    if (train_batch <= 0)   // auto: the GPU-tuned batch on a CUDA build, else the CPU width
        train_batch = HAS_CUDA ? DEFAULT_GPU_BATCH : (DEFAULT_THREADS * DEFAULT_WINDOWS_PER_THREAD);
    if (train_lr_opt->count() == 0)   // couple lr to batch unless pinned (sqrt rule for Adam)
        train_lr = LR_BASE * std::sqrt(static_cast<float>(train_batch) / LR_BASE_BATCH);
    int keep = 3;                      // "ALL"/"all" -> keep every checkpoint (keep<0 sentinel)
    if (train_keep == "ALL" || train_keep == "all") keep = -1;
    else { try { keep = std::max(1, std::stoi(train_keep)); } catch (...) { keep = 3; } }
    const int resume_mode = train_resume ? 1 : train_fresh ? 2 : 0;   // 0=auto, 1=force resume, 2=force fresh
    return sub0_train_stage(train_corpus.c_str(), train_model.c_str(),
                            train_steps, train_batch, train_lr, train_seed, keep, train_optimizer,
                            resume_mode, train_spell_mix, train_scratch_mix, train_op_mix);
}

// --- gen -------------------------------------------------------------------
inline int run_gen(int argc, char** argv) {
    CLI::App app{"sub0llm-gen — generate text from a trained model"};
    std::string gen_model, gen_prompt;
    int   gen_n = 200, gen_topk = 20;
    float gen_temp = 0.8f;
    unsigned gen_seed = 0;
    int   gen_attn_sinks = 0;   // 0 = off (today's plain sliding-window fallback past SEQ_LEN)
    app.add_option("model",  gen_model,  "Trained model path")->required();
    app.add_option("prompt", gen_prompt, "Prompt text")->required();
    app.add_option("--n",    gen_n,    "Tokens to generate")->capture_default_str();
    app.add_option("--temp", gen_temp, "Sampling temperature")->capture_default_str();
    app.add_option("--topk", gen_topk, "Top-k sampling cutoff")->capture_default_str();
    auto* gen_seed_opt = app.add_option("--seed", gen_seed, "RNG seed (default: random)");
    app.add_option("--attn-sinks", gen_attn_sinks,
                   "StreamingLLM-style attention sinks: once the prompt+generation would exceed the "
                   "trained context window, keep this many tokens from the START of the sequence "
                   "resident alongside the most recent tokens, instead of a plain sliding window "
                   "that drops them (0 = off, plain sliding window)")
       ->capture_default_str();
    CLI11_PARSE(app, argc, argv);

    if (gen_seed_opt->count() == 0) gen_seed = std::random_device{}();  // fresh seed unless pinned
    return sub0_gen_stage(gen_model.c_str(), gen_prompt.c_str(), gen_n, gen_temp, gen_topk, gen_seed,
                          gen_attn_sinks);
}

// --- eval ------------------------------------------------------------------
// Score an --op-mix (op-delegation) model on synthetic arithmetic: delegation rate, exact-match, and
// routing accuracy (arithmetic is exact by construction, so a miss localises to routing).
inline int run_eval(int argc, char** argv) {
    CLI::App app{"sub0llm-eval — score an op-delegation model (delegation rate + exact-match + routing accuracy)"};
    std::string eval_model;
    unsigned    eval_seed = 4242;
    int         eval_n    = 300;
    app.add_option("model", eval_model, "Trained model path (a model trained with --op-mix)")->required();
    app.add_option("--seed", eval_seed, "RNG seed for the synthetic problems")->capture_default_str();
    app.add_option("--problems", eval_n, "Number of synthetic problems to score")->capture_default_str();
    CLI11_PARSE(app, argc, argv);
    return sub0_eval_stage(eval_model.c_str(), eval_seed, eval_n);
}

// --- tune ------------------------------------------------------------------
inline int run_tune(int argc, char** argv) {
    CLI::App app{"sub0llm-tune — auto-tune runtime knobs (threads, batch granularity) for throughput"};
    int  tune_max_threads = 0;   // 0 = hardware_concurrency
    bool tune_quiet = false, tune_thorough = false;
    int  tune_seconds = 0;       // 0 = profile default (fast 120s / thorough 600s)
    int  tune_backend = 0;       // 0=auto, 1=all, 2=cpu, 3=gpu -- sub0_tune_stage resolves auto to
                                 // gpu-only when a CUDA device is present (the CPU sweep is slow and
                                 // its result goes unused by a GPU run); `all` always runs both.
    app.add_option("--max-threads", tune_max_threads,
                   "Cap on threads to consider (0 = hardware_concurrency)")->capture_default_str();
    app.add_flag("--quiet", tune_quiet, "Print only the winning configuration, not the search trace");
    app.add_flag("--thorough", tune_thorough,
                 "Aggressive search: wider grid + more re-measurement (longer default budget)");
    app.add_option("--seconds", tune_seconds,
                   "Optional hard safety cap in seconds (0 = none; each sample is bounded per-test)");
    app.add_option("--backend", tune_backend,
                   "Backend(s) to tune: auto (GPU-only if present, else CPU) | all (always both) | cpu | gpu")
       ->transform(CLI::CheckedTransformer(std::map<std::string, int>{
           {"auto", 0}, {"all", 1}, {"cpu", 2}, {"gpu", 3}}, CLI::ignore_case))
       ->default_str("auto");
    CLI11_PARSE(app, argc, argv);

    return sub0_tune_stage(tune_max_threads, tune_quiet ? 0 : 1, tune_backend,
                           tune_thorough ? 1 : 0, tune_seconds);
}

// --- vocab -----------------------------------------------------------------
inline int run_vocab(int argc, char** argv) {
    CLI::App app{"sub0llm vocab — print the build's vocabulary table"};
    std::string vocab_tok;
    int vocab_limit = 0;
    app.add_option("tokenizer", vocab_tok, "Tokenizer path (defaults to the baked-in tokenizer)");
    app.add_option("--limit", vocab_limit, "Max entries to print (0 = all)")->capture_default_str();
    CLI11_PARSE(app, argc, argv);
    return sub0_vocab_stage(vocab_tok.empty() ? nullptr : vocab_tok.c_str(), vocab_limit);
}

// --- bench -----------------------------------------------------------------
inline int run_bench(int argc, char** argv) {
    CLI::App app{"sub0llm bench — cycle-accurate hot-path benchmark (forward/backward/optimizer)"};
    int bench_iters = 200, bench_threads = DEFAULT_THREADS, bench_wpt = DEFAULT_WINDOWS_PER_THREAD;
    app.add_option("--iters,--steps", bench_iters, "Iterations to time")->capture_default_str();
    app.add_option("--threads", bench_threads,
                   "Data-parallel worker threads (defaults to the tuned/hardware count; 1 = control)")
       ->capture_default_str();
    app.add_option("--windows-per-thread", bench_wpt,
                   "Windows each thread carries in the data-parallel minibatch")->capture_default_str();
    CLI11_PARSE(app, argc, argv);
    return sub0_bench_stage(bench_iters, bench_threads, bench_wpt);
}

// --- autotemp --------------------------------------------------------------
inline int run_autotemp(int argc, char** argv) {
    CLI::App app{"sub0llm autotemp — pick the sampling temperature matching held-out text perplexity"};
    std::string at_model;
    unsigned at_seed = 42;
    bool at_quiet = false;
    app.add_option("model", at_model, "Trained model path")->required();
    app.add_option("--seed", at_seed, "RNG seed for the generation sweep")->capture_default_str();
    app.add_flag("--quiet", at_quiet, "Print only the recommendation, not the temperature sweep");
    CLI11_PARSE(app, argc, argv);
    return sub0_autotemp_stage(at_model.c_str(), at_seed, at_quiet ? 0 : 1);
}

// --- models ----------------------------------------------------------------
inline int run_models(int argc, char** argv) {
    CLI::App app{"sub0llm models — list trained models; --prune removes ones incompatible with this build; "
                 "--metrics/--refresh show a filterable, cached cross-model comparison table"};
    bool models_prune = false, models_metrics = false, models_refresh = false, models_force = false;
    std::string models_corpus, models_sha, models_since, models_until;
    app.add_flag("--prune", models_prune, "Delete models whose architecture this build cannot load");
    app.add_option("--corpus", models_corpus, "Filter: only models trained on a corpus matching this substring");
    app.add_option("--sha", models_sha, "Filter: only models tagged with a git SHA starting with this");
    app.add_option("--since", models_since, "Filter: only models created on/after this ISO date (YYYY-MM-DD)");
    app.add_option("--until", models_until, "Filter: only models created on/before this ISO date (YYYY-MM-DD)");
    app.add_flag("--metrics", models_metrics,
                "Print a comparison table (dims/params/val_nelbo/bits-per-byte/autotemp temp) from each "
                "model's cached metrics.txt, read-only");
    app.add_flag("--refresh", models_refresh,
                "Like --metrics, but first compute+cache report/autotemp's raw numbers for any filtered, "
                "loadable model with a missing or stale cache");
    app.add_flag("--force", models_force, "With --refresh, recompute every filtered loadable model unconditionally")
       ->needs("--refresh");
    CLI11_PARSE(app, argc, argv);
    return sub0_models_stage(models_prune ? 1 : 0, 1,
                             models_corpus.c_str(), models_sha.c_str(), models_since.c_str(), models_until.c_str(),
                             models_metrics ? 1 : 0, models_refresh ? 1 : 0, models_force ? 1 : 0);
}

// --- report ----------------------------------------------------------------
inline int run_report(int argc, char** argv) {
    CLI::App app{"sub0llm report — diagnose model sizing vs its corpus; suggest dimension knobs"};
    std::string report_model;
    app.add_option("model", report_model,
                   "Trained model to include train/val loss diagnosis (optional; omit for structural-only)");
    CLI11_PARSE(app, argc, argv);
    return sub0_report_stage(report_model.empty() ? nullptr : report_model.c_str());
}

// --- memplan ---------------------------------------------------------------
inline int run_memplan(int argc, char** argv) {
    CLI::App app{"sub0llm memplan — predicted train/gen device memory footprints (vs VRAM)"};
    CLI11_PARSE(app, argc, argv);
    return sub0_memplan_stage();
}

// --- bundle ------------------------------------------------------------------
inline int run_bundle(int argc, char** argv) {
    CLI::App app{"sub0llm bundle — copy this build's exe+DLLs into a model dir so it can be run "
                 "later (gen/report) without rebuilding, even after this build's dims change. "
                 "Opt-in: not part of a normal train/gen workflow, meant for cross-model-size "
                 "comparisons where several incompatible builds need to stay runnable side by side."};
    std::string bundle_model;
    app.add_option("model", bundle_model, "Trained model path (dims must match THIS build)")->required();
    CLI11_PARSE(app, argc, argv);
    return sub0_bundle_stage(bundle_model.c_str());
}

// --- ckpt2model ----------------------------------------------------------------
inline int run_ckpt2model(int argc, char** argv) {
    CLI::App app{"sub0llm ckpt2model — extract weights from a training checkpoint (.ckpt, includes "
                 "optimizer/RNG state) into a model.bin gen/report can load directly. Lets you inspect "
                 "an OLDER checkpoint (e.g. the best-val_nelbo one, kept by prune_ckpts alongside the "
                 "newest N) without resuming a full training run."};
    std::string ck_in, ck_out;
    app.add_option("checkpoint", ck_in, "Input .ckpt path")->required();
    app.add_option("model_out", ck_out, "Output model.bin path")->required();
    CLI11_PARSE(app, argc, argv);
    return sub0_ckpt2model_stage(ck_in.c_str(), ck_out.c_str());
}

}  // namespace sub0cli
