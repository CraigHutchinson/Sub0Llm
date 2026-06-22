#pragma once

// model_io.hpp (Ch30) — load a trained denoiser from a Ch29 model directory:
//   config.json   architecture (vocab_size, embed_dim, n_layers, heads, d_ff, seq_len)
//   tokenizer/    GPT-2 format vocab.json + merges.txt
//   step_*.ckpt   weights (latest = the early-stopping winner Ch29 reloads at exit)

#include "sub0diff/nn/denoiser.hpp"
#include "sub0diff/nn/mera_denoiser.hpp"

#include "sub0llm/tokenizer/bpe.hpp"

#include <cstdint>
#include <memory>
#include <string>

namespace sub0diff::nn {

// A loaded model directory. `model_type` selects which denoiser pointer is populated: "flat" → `model`
// (the vanilla Denoiser), "mera" → `mera` (the Ch32 hierarchical MeraDenoiser). config.json with no
// model_type loads as flat (back-compat with pre-Ch32 model dirs). Both denoisers satisfy the templated
// sampler/loss, so a consumer dispatches on model_type and runs the populated one.
struct LoadedModel {
    std::string                             model_type = "flat";
    std::unique_ptr<Denoiser>               model;     // set iff model_type == "flat"
    std::unique_ptr<MeraDenoiser>           mera;      // set iff model_type == "mera"
    std::unique_ptr<sub0llm::BPETokenizer>  tokenizer;
    std::int64_t                            seq_len = 0;   // training window length T
    std::int64_t                            mera_coarsen = 0, mera_window = 0;  // MERA geometry (if mera)
    bool                                    mera_gated_pool = false;            // MERA content-weighted pool
    std::int64_t                            step    = 0;   // checkpoint step loaded
};

// Throws std::runtime_error with a precise message on any missing piece.
[[nodiscard]] LoadedModel load_model_dir(const std::string& dir);

} // namespace sub0diff::nn
