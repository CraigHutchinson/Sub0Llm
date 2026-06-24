// driver.cpp — the thin "sub0llm" front-end executable.
//
// It parses the command line and dispatches into the backend stage shared
// libraries (libsub0_train, libsub0_gen) via their C entry points. It contains
// no model logic of its own; the heavy lifting lives behind the .so/.dylib
// boundary, which is where a future bundled-compiler / hot-swap design would
// regenerate and reload a stage.

#include "sub0/core.hpp"  // for DEFAULT_CORPUS (generated config)

#include <cstdlib>
#include <cstring>
#include <print>
#include <random>

// Entry points provided by the stage libraries we link against.
extern "C" int sub0_train_stage(const char* corpus, const char* model_out,
                                int steps, int batch, float lr, unsigned seed);
extern "C" int sub0_gen_stage(const char* model_in, const char* prompt,
                              int n, float temp, int topk, unsigned seed);

static int arg_int(int c, char** v, const char* k, int d) {
    for (int i = 1; i < c - 1; ++i) if (!std::strcmp(v[i], k)) return std::atoi(v[i + 1]);
    return d;
}
static float arg_float(int c, char** v, const char* k, float d) {
    for (int i = 1; i < c - 1; ++i) if (!std::strcmp(v[i], k)) return (float)std::atof(v[i + 1]);
    return d;
}

static void usage() {
    std::println(
        "sub0llm \u2014 staged CPU transformer LM (config baked in at build time)\n\n"
        "  sub0llm train <model.bin> [corpus] [--steps N --batch N --lr F --seed N]\n"
        "      corpus defaults to the one this build was configured with:\n"
        "      {}\n\n"
        "  sub0llm gen   <model.bin> \"<prompt>\" [--n N --temp F --topk N --seed N]\n\n"
        "  Reconfigure dimensions/corpus/vocab by re-running cmake with -DSUB0_* and rebuilding.",
        DEFAULT_CORPUS);
}

int main(int argc, char** argv) {
    if (argc < 3) { usage(); return 1; }
    std::string mode = argv[1];

    if (mode == "train") {
        const char* model_out = argv[2];
        // Optional positional corpus (third arg if it isn't a flag); else default.
        const char* corpus = DEFAULT_CORPUS;
        if (argc >= 4 && argv[3][0] != '-') corpus = argv[3];
        int steps = arg_int(argc, argv, "--steps", 3000);
        int batch = arg_int(argc, argv, "--batch", 8);
        float lr  = arg_float(argc, argv, "--lr", 0.001f);
        unsigned seed = (unsigned)arg_int(argc, argv, "--seed", 42);
        return sub0_train_stage(corpus, model_out, steps, batch, lr, seed);
    }
    if (mode == "gen") {
        if (argc < 4) { usage(); return 1; }
        const char* model_in = argv[2];
        const char* prompt = argv[3];
        int n = arg_int(argc, argv, "--n", 200);
        float temp = arg_float(argc, argv, "--temp", 0.8f);
        int topk = arg_int(argc, argv, "--topk", 20);
        unsigned seed = (unsigned)arg_int(argc, argv, "--seed", (int)std::random_device{}());
        return sub0_gen_stage(model_in, prompt, n, temp, topk, seed);
    }
    usage();
    return 1;
}
