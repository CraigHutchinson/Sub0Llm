// sub0llm-gen.cpp — the standalone generate-stage executable. A thin main over sub0cli::run_gen (the
// same runner the umbrella `sub0llm gen` uses), gated on the configured engine being built.
#include "sub0/cli_stages.hpp"
int main(int argc, char** argv) { return sub0cli::run_gen(argc, argv); }
