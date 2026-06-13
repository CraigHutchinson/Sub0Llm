#pragma once

// model_io.hpp (Ch30) — load a trained denoiser from a Ch29 model directory:
//   config.json   architecture (vocab_size, embed_dim, n_layers, heads, d_ff, seq_len)
//   tokenizer/    GPT-2 format vocab.json + merges.txt
//   step_*.ckpt   weights (latest = the early-stopping winner Ch29 reloads at exit)

#include "sub0diff/nn/denoiser.hpp"

#include "sub0llm/tokenizer/bpe.hpp"

#include <cstdint>
#include <memory>
#include <string>

namespace sub0diff::nn {

struct LoadedModel {
    std::unique_ptr<Denoiser>               model;
    std::unique_ptr<sub0llm::BPETokenizer>  tokenizer;
    std::int64_t                            seq_len = 0;   // training window length
    std::int64_t                            step    = 0;   // checkpoint step loaded
};

// Throws std::runtime_error with a precise message on any missing piece.
[[nodiscard]] LoadedModel load_model_dir(const std::string& dir);

} // namespace sub0diff::nn
