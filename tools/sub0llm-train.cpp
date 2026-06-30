// sub0llm-train.cpp — the standalone train-stage executable. A thin main over sub0cli::run_train (the
// same runner the umbrella `sub0llm train` uses), gated on the configured engine being built.
#include "sub0/cli_stages.hpp"
int main(int argc, char** argv) { return sub0cli::run_train(argc, argv); }
