// sub0llm-tune.cpp — the standalone tune-stage executable. A thin main over sub0cli::run_tune (the same
// runner the umbrella `sub0llm tune` uses); tune lives in the train stage lib, so it links sub0_train.
#include "sub0/cli_stages.hpp"
int main(int argc, char** argv) { return sub0cli::run_tune(argc, argv); }
