#include "sub0diff/nn/model_io.hpp"

#include "sub0llm/nn/checkpoint.hpp"

#include <nlohmann/json.hpp>

#include <filesystem>
#include <format>
#include <fstream>
#include <stdexcept>

namespace sub0diff::nn {

namespace fs = std::filesystem;
using sub0llm::BPETokenizer;

LoadedModel load_model_dir(const std::string& dir) {
    const fs::path root(dir);
    const fs::path cfg_path = root / "config.json";
    std::ifstream cf(cfg_path);
    if (!cf)
        throw std::runtime_error(std::format("load_model_dir: cannot open {}", cfg_path.string()));
    nlohmann::json j;
    cf >> j;

    const auto vocab_size = j.at("vocab_size").get<std::int64_t>();
    const auto embed_dim  = j.at("embed_dim").get<std::int64_t>();
    const auto n_layers   = j.at("n_layers").get<std::int64_t>();
    const auto n_heads    = j.at("n_heads").get<std::size_t>();
    const auto n_kv_heads = j.at("n_kv_heads").get<std::size_t>();
    const auto d_ff       = j.at("d_ff").get<std::int64_t>();

    LoadedModel out;
    out.seq_len    = j.value("seq_len", std::int64_t{64});
    out.model_type = j.value("model_type", std::string("flat"));

    // Build the architecture named in config.json. Both denoisers expose parameters()/forward() the
    // sampler/loss templates accept; the checkpoint loads into whichever one we built.
    std::vector<sub0llm::autograd::Variable*> param_ptrs;
    if (out.model_type == "mera") {
        out.mera_coarsen = j.value("mera_coarsen", std::int64_t{4});
        out.mera_window  = j.value("mera_window",  std::int64_t{64});
        out.mera = std::make_unique<MeraDenoiser>(vocab_size, embed_dim, n_heads, n_kv_heads,
                                                  out.mera_coarsen, out.mera_window, out.seq_len,
                                                  d_ff, /*seed=*/42);
        param_ptrs = out.mera->parameters();
    } else {
        out.model = std::make_unique<Denoiser>(vocab_size, embed_dim, n_heads, n_kv_heads,
                                               n_layers, d_ff, /*seed=*/42);
        param_ptrs = out.model->parameters();
    }

    const std::string ckpt = sub0llm::latest_checkpoint_path(dir);
    if (ckpt.empty())
        throw std::runtime_error(std::format("load_model_dir: no step_*.ckpt in {}", dir));
    std::vector<sub0llm::autograd::Variable> params;
    params.reserve(param_ptrs.size());
    for (auto* p : param_ptrs) params.push_back(*p);
    out.step = sub0llm::load_checkpoint(params, ckpt);

    const fs::path tok_dir = root / "tokenizer";
    out.tokenizer = std::make_unique<BPETokenizer>(
        BPETokenizer::load(tok_dir / "vocab.json", tok_dir / "merges.txt"));
    if (static_cast<std::int64_t>(out.tokenizer->vocab_size()) != vocab_size)
        throw std::runtime_error(std::format(
            "load_model_dir: tokenizer vocab {} != config vocab_size {}",
            out.tokenizer->vocab_size(), vocab_size));
    return out;
}

} // namespace sub0diff::nn
