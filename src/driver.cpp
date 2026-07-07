// driver.cpp — the umbrella "sub0llm" executable: a thin dispatcher over the per-stage CLI runners in
// sub0/cli_stages.hpp. It parses only argv[1] (the command) and forwards the rest to the matching
// runner, which dispatches into the stage shared libraries (libsub0_train, libsub0_gen). The same
// runners back the dedicated per-stage executables (sub0llm-train / -gen / -tune), so the arg parsing
// and stage dispatch are defined ONCE. No model logic lives here.

#include "sub0/cli_stages.hpp"

#include <print>
#include <string_view>

namespace {
void usage() {
    std::println(stderr,
        "sub0llm — staged CPU/GPU transformer LM (config baked in at build time)\n"
        "Usage: sub0llm <command> [options]   (configure first with sub0llm-configure)\n\n"
        "Stages (also available as standalone executables):\n"
        "  train     train a model            (sub0llm-train)\n"
        "  gen       generate text            (sub0llm-gen)\n"
        "  tune      auto-tune runtime knobs  (sub0llm-tune)\n"
        "Diagnostics:\n"
        "  vocab     print the vocabulary table\n"
        "  bench     cycle-accurate hot-path benchmark\n"
        "  autotemp  pick a sampling temperature\n"
        "  models    list / prune trained models\n"
        "  report    diagnose model sizing vs corpus\n"
        "  memplan   predicted train/gen memory footprints\n"
        "  bundle    copy this build's binaries into a model dir (run later w/o rebuilding)\n"
        "  ckpt2model  extract weights from a .ckpt into a model.bin gen/report can load\n\n"
        "Run 'sub0llm <command> --help' for a command's options.");
}
}  // namespace

int main(int argc, char** argv) {
    using namespace sub0cli;
    if (argc < 2) { usage(); return 1; }

    const std::string_view cmd = argv[1];
    if (cmd == "-h" || cmd == "--help" || cmd == "help") { usage(); return 0; }

    // Forward argv[1..] so each runner sees the subcommand name as its argv[0] (program name) and its
    // own flags after it -- identical to invoking the standalone sub0llm-<cmd> executable.
    const int sub_argc = argc - 1;
    char** const sub_argv = argv + 1;

    if (cmd == "train")    return run_train(sub_argc, sub_argv);
    if (cmd == "gen")      return run_gen(sub_argc, sub_argv);
    if (cmd == "tune")     return run_tune(sub_argc, sub_argv);
    if (cmd == "vocab")    return run_vocab(sub_argc, sub_argv);
    if (cmd == "bench")    return run_bench(sub_argc, sub_argv);
    if (cmd == "autotemp") return run_autotemp(sub_argc, sub_argv);
    if (cmd == "models")   return run_models(sub_argc, sub_argv);
    if (cmd == "report")   return run_report(sub_argc, sub_argv);
    if (cmd == "memplan")  return run_memplan(sub_argc, sub_argv);
    if (cmd == "bundle")   return run_bundle(sub_argc, sub_argv);
    if (cmd == "ckpt2model") return run_ckpt2model(sub_argc, sub_argv);

    std::println(stderr, "sub0llm: unknown command '{}'\n", cmd);
    usage();
    return 1;
}
