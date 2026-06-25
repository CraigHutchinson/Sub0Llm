// driver.cpp — the thin "sub0llm" front-end executable.
//
// It parses the command line and dispatches into the backend stage shared
// libraries (libsub0_train, libsub0_gen) via their C entry points. It contains
// no model logic of its own; the heavy lifting lives behind the .so/.dylib
// boundary, which is where a future bundled-compiler / hot-swap design would
// regenerate and reload a stage.

#include "sub0/core.hpp"  // for DEFAULT_CORPUS (generated config)

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
extern "C" int sub0_tune_stage(int max_threads, int verbose);

int main(int argc, char** argv) {
    // CLI11 rejects unknown options and positionals by default (allow_extras is off),
    // so a mistyped flag is a hard error with a usage hint -- not a silent no-op that
    // runs with stale defaults. require_subcommand(1) enforces exactly one mode.
    CLI::App app{"sub0llm \u2014 staged CPU transformer LM (config baked in at build time)"};
    app.require_subcommand(1);
    app.set_help_all_flag("--help-all", "Show help for every subcommand");

    // --- train ---------------------------------------------------------------
    std::string train_model, train_corpus{DEFAULT_CORPUS};
    int   train_steps = 0, train_batch = 8;
    float train_lr    = 0.001f;
    unsigned train_seed = 42;
    auto* train = app.add_subcommand("train", "Train a model into <model.bin> (resumes from <model.bin>.ckpt)");
    train->add_option("model", train_model, "Output model path")->required();
    train->add_option("corpus", train_corpus, "Training corpus")->capture_default_str();
    train->add_option("--steps", train_steps,
                      "Training steps (0 = auto-size to corpus, stop on validation plateau)")->capture_default_str();
    train->add_option("--batch", train_batch, "Minibatch size")->capture_default_str();
    train->add_option("--lr",    train_lr,    "Learning rate")->capture_default_str();
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
    auto* tune = app.add_subcommand("tune", "Auto-tune runtime knobs (threads, batch granularity) for peak throughput");
    tune->add_option("--max-threads", tune_max_threads,
                     "Cap on threads to consider (0 = hardware_concurrency)")->capture_default_str();
    tune->add_flag("--quiet", tune_quiet, "Print only the winning configuration, not the search trace");

    CLI11_PARSE(app, argc, argv);

    if (*train)
        return sub0_train_stage(train_corpus.c_str(), train_model.c_str(),
                                train_steps, train_batch, train_lr, train_seed);
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
        return sub0_tune_stage(tune_max_threads, tune_quiet ? 0 : 1);

    return 1;  // unreachable: require_subcommand(1) guarantees one of the above
}
