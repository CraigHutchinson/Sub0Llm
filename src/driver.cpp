// driver.cpp — the thin "sub0llm" front-end executable.
//
// It parses the command line and dispatches into the backend stage shared
// libraries (libsub0_train, libsub0_gen) via their C entry points. It contains
// no model logic of its own; the heavy lifting lives behind the .so/.dylib
// boundary, which is where a future bundled-compiler / hot-swap design would
// regenerate and reload a stage.

#include "sub0/core.hpp"  // for DEFAULT_CORPUS (generated config)

#include <cmath>
#include <map>
#include <print>
#include <random>
#include <string>

#include <CLI/CLI.hpp>

// Entry points provided by the stage libraries we link against.
extern "C" int sub0_train_stage(const char* corpus, const char* model_out,
                                int steps, int batch, float lr, unsigned seed);
extern "C" int sub0_gen_stage(const char* model_in, const char* prompt,
                              int n, float temp, int topk, unsigned seed);
extern "C" int sub0_vocab_stage(const char* tokenizer_path, int limit);
extern "C" int sub0_bench_stage(int iters, int threads, int windows_per_thread);
extern "C" int sub0_tune_stage(int max_threads, int verbose, int backend);
extern "C" int sub0_autotemp_stage(const char* model_in, unsigned seed, int verbose);
extern "C" int sub0_report_stage(const char* model_in);
extern "C" int sub0_models_stage(int prune, int verbose);

int main(int argc, char** argv) {
    // CLI11 rejects unknown options and positionals by default (allow_extras is off),
    // so a mistyped flag is a hard error with a usage hint -- not a silent no-op that
    // runs with stale defaults. require_subcommand(1) enforces exactly one mode.
    CLI::App app{"sub0llm \u2014 staged CPU transformer LM (config baked in at build time)"};
    app.require_subcommand(1);
    app.set_help_all_flag("--help-all", "Show help for every subcommand");

    // --- train ---------------------------------------------------------------
    // Default the minibatch to the tuned data-parallel width (threads x windows/thread,
    // baked in by sub0-configure) so a default run saturates the cores. lr is the
    // batch-8 default scaled by sqrt(batch/8) -- a bigger batch means fewer, lower-
    // variance updates, so it wants a proportionally larger step (sqrt rule for Adam).
    constexpr int   LR_BASE_BATCH = 8;
    constexpr float LR_BASE       = 0.001f;
    std::string train_model, train_corpus{DEFAULT_CORPUS};
    int   train_steps = 0, train_batch = 0;     // 0 = auto (resolved from the baked defaults below)
    float train_lr    = LR_BASE;
    unsigned train_seed = 42;
    auto* train = app.add_subcommand("train",
        "Train a model (resumes from its .ckpt). With no path, auto-creates a structured, "
        "registered model directory under the models root; see `sub0llm models`.");
    train->add_option("model", train_model,
                      "Output model path (optional; omit to auto-name by corpus+dims+git SHA)");
    train->add_option("corpus", train_corpus, "Training corpus")->capture_default_str();
    train->add_option("--steps", train_steps,
                      "Training steps (0 = auto-size to corpus, stop on validation plateau)")->capture_default_str();
    train->add_option("--batch", train_batch,
                      "Minibatch size (0 = auto: tuned GPU batch on a CUDA build, else CPU data-parallel width)");
    auto* train_lr_opt =
        train->add_option("--lr", train_lr, "Learning rate (default: 0.001 scaled by sqrt(batch/8))")
             ->capture_default_str();
    train->add_option("--seed",  train_seed,  "RNG seed")->capture_default_str();

    // --- gen -----------------------------------------------------------------
    std::string gen_model, gen_prompt;
    int   gen_n = 200, gen_topk = 20;
    float gen_temp = 0.8f;
    unsigned gen_seed = 0;
    auto* gen = app.add_subcommand("gen", "Generate text from a trained model");
    gen->add_option("model",  gen_model,  "Trained model path")->required();
    gen->add_option("prompt", gen_prompt, "Prompt text")->required();
    gen->add_option("--n",    gen_n,    "Tokens to generate")->capture_default_str();
    gen->add_option("--temp", gen_temp, "Sampling temperature")->capture_default_str();
    gen->add_option("--topk", gen_topk, "Top-k sampling cutoff")->capture_default_str();
    auto* gen_seed_opt = gen->add_option("--seed", gen_seed, "RNG seed (default: random)");

    // --- vocab ---------------------------------------------------------------
    std::string vocab_tok;
    int vocab_limit = 0;
    auto* vocab = app.add_subcommand("vocab", "Print the build's BPE vocabulary table");
    vocab->add_option("tokenizer", vocab_tok, "Tokenizer path (defaults to the baked-in tokenizer)");
    vocab->add_option("--limit", vocab_limit, "Max entries to print (0 = all)")->capture_default_str();

    // --- bench ---------------------------------------------------------------
    int bench_iters = 200, bench_threads = DEFAULT_THREADS, bench_wpt = DEFAULT_WINDOWS_PER_THREAD;
    auto* bench = app.add_subcommand("bench", "Cycle-accurate hot-path benchmark (forward/backward/optimizer)");
    bench->add_option("--iters,--steps", bench_iters,
                      "Iterations to time (--steps is an alias for the natural reflex)")->capture_default_str();
    bench->add_option("--threads", bench_threads,
                      "Data-parallel worker threads (defaults to the tuned/hardware count; 1 = single-thread control)")
         ->capture_default_str();
    bench->add_option("--windows-per-thread", bench_wpt,
                      "Windows each thread carries in the data-parallel minibatch")->capture_default_str();

    // --- tune ----------------------------------------------------------------
    int  tune_max_threads = 0;   // 0 = hardware_concurrency
    bool tune_quiet = false;
    int  tune_backend = 0;       // 0=auto, 1=all, 2=cpu, 3=gpu (CheckedTransformer maps the string)
    auto* tune = app.add_subcommand("tune", "Auto-tune runtime knobs (threads, batch granularity) for peak throughput");
    tune->add_option("--max-threads", tune_max_threads,
                     "Cap on threads to consider (0 = hardware_concurrency)")->capture_default_str();
    tune->add_flag("--quiet", tune_quiet, "Print only the winning configuration, not the search trace");
    tune->add_option("--backend", tune_backend,
                     "Which backend(s) to tune: auto (CPU + GPU if present) | all | cpu | gpu (skip CPU, keep its cached tuning)")
        ->transform(CLI::CheckedTransformer(std::map<std::string, int>{
            {"auto", 0}, {"all", 1}, {"cpu", 2}, {"gpu", 3}}, CLI::ignore_case))
        ->default_str("auto");

    // --- autotemp ------------------------------------------------------------
    // The coherence analogue of `tune`: search the sampling temperature whose
    // generations are as in-distribution (by self-perplexity) as real held-out text.
    std::string at_model;
    unsigned at_seed = 42;
    bool at_quiet = false;
    auto* autotemp = app.add_subcommand("autotemp",
        "Auto-pick the sampling temperature whose generations match held-out text perplexity");
    autotemp->add_option("model", at_model, "Trained model path")->required();
    autotemp->add_option("--seed", at_seed, "RNG seed for the generation sweep")->capture_default_str();
    autotemp->add_flag("--quiet", at_quiet, "Print only the recommendation, not the temperature sweep");

    // --- models --------------------------------------------------------------
    // Discover trained models (the registry = the per-model meta.txt files) and flag which
    // load into this build; --prune reclaims architecture-incompatible ones.
    bool models_prune = false;
    auto* models = app.add_subcommand("models", "List trained models; --prune removes ones incompatible with this build");
    models->add_flag("--prune", models_prune, "Delete models whose architecture this build cannot load");

    // --- report --------------------------------------------------------------
    // Diagnose whether the baked architecture is correctly sized for its corpus and emit per-knob
    // retrain guidance (train/val gap, bits-per-byte, Chinchilla tokens/param, head_dim, aspect).
    std::string report_model;
    auto* report = app.add_subcommand("report",
        "Diagnose model sizing vs its corpus and suggest which dimension knobs to change");
    report->add_option("model", report_model,
                       "Trained model to include train/val loss diagnosis (optional; omit for structural-only)");

    CLI11_PARSE(app, argc, argv);

    if (*train) {
        if (train_batch <= 0)   // auto: the GPU-tuned batch on a CUDA build, else the CPU width
            train_batch = HAS_CUDA ? DEFAULT_GPU_BATCH : (DEFAULT_THREADS * DEFAULT_WINDOWS_PER_THREAD);
        if (train_lr_opt->count() == 0)   // couple lr to batch unless pinned (sqrt rule for Adam)
            train_lr = LR_BASE * std::sqrt(static_cast<float>(train_batch) / LR_BASE_BATCH);
        return sub0_train_stage(train_corpus.c_str(), train_model.c_str(),
                                train_steps, train_batch, train_lr, train_seed);
    }
    if (*gen) {
        if (gen_seed_opt->count() == 0) gen_seed = std::random_device{}();  // fresh seed unless pinned
        return sub0_gen_stage(gen_model.c_str(), gen_prompt.c_str(),
                              gen_n, gen_temp, gen_topk, gen_seed);
    }
    if (*vocab)
        return sub0_vocab_stage(vocab_tok.empty() ? nullptr : vocab_tok.c_str(), vocab_limit);
    if (*bench)
        return sub0_bench_stage(bench_iters, bench_threads, bench_wpt);
    if (*tune)
        return sub0_tune_stage(tune_max_threads, tune_quiet ? 0 : 1, tune_backend);
    if (*autotemp)
        return sub0_autotemp_stage(at_model.c_str(), at_seed, at_quiet ? 0 : 1);
    if (*models)
        return sub0_models_stage(models_prune ? 1 : 0, 1);
    if (*report)
        return sub0_report_stage(report_model.empty() ? nullptr : report_model.c_str());

    return 1;  // unreachable: require_subcommand(1) guarantees one of the above
}
