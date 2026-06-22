// mera_train.cpp (Ch32) — a RESUMABLE, CUDA-capable trainer for the hierarchical MeraDenoiser (and the
// flat Denoiser), reusing the Ch29 consistent config layer + binary checkpoint format.
//
// Robust resume (the ask): `--ckpt-dir X` ALONE reconstructs the exact architecture from the
// run_config.json saved in X (model_type / coarsen / window / D / layers / vocab — all BuildTime fields
// of the consistent layer), reloads the latest step_*.ckpt weights AND the matching step_*.opt Adam
// moments, and continues. Every flag is auto-derived from the schema; an explicit flag overrides just
// that field. Train on the GPU with `--device cuda` (single CUDA stream via the templated GpuTrainer).
//
// This is intentionally lean vs ch29 (no curriculum / recall sweep / worker pool) — its job is the
// MERA train↔checkpoint↔resume loop on the CUDA backend. Build:
//   cmake --build build-cuda --target ch32_mera_train          (GPU)
//   ./build-cuda/.../ch32_mera_train --model-type mera --device cuda --ckpt-dir /tmp/mera --steps 3000

#include "sub0diff/config/run_config.hpp"
#include "sub0diff/nn/denoiser.hpp"
#include "sub0diff/nn/mera_denoiser.hpp"
#include "sub0diff/train/diffusion_loss.hpp"

#include "sub0llm/core/ops.hpp"
#include "sub0llm/core/runtime.hpp"   // init_cpu_compute (FTZ+DAZ — training-throughput prerequisite)
#include "sub0llm/nn/checkpoint.hpp"
#include "sub0llm/nn/optimizer.hpp"
#include "sub0llm/tokenizer/bpe.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <format>
#include <fstream>
#include <memory>
#include <print>
#include <random>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#ifndef SUB0DIFF_CODE_SHA
#define SUB0DIFF_CODE_SHA "dev"
#endif

namespace fs = std::filesystem;
namespace cfgm = sub0diff::config;
namespace dn = sub0diff::nn;
namespace dt = sub0diff::train;
using sub0llm::BPETokenizer;

namespace {

std::vector<std::string> read_paragraphs(const std::string& path, std::int64_t limit) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error(std::format("cannot open corpus: {}", path));
    std::vector<std::string> out;
    std::string line;
    while (std::getline(f, line)) {
        while (!line.empty() && std::isspace(static_cast<unsigned char>(line.back()))) line.pop_back();
        std::size_t b = 0;
        while (b < line.size() && std::isspace(static_cast<unsigned char>(line[b]))) ++b;
        if (b >= line.size()) continue;
        out.push_back(line.substr(b));
        if (limit > 0 && static_cast<std::int64_t>(out.size()) >= limit) break;
    }
    return out;
}

// Average NELBO over `n_batches` random eval windows (no backward — a held-out quality read).
template <class Model>
double eval_nelbo(Model& model, std::span<const std::int32_t> stream, const cfgm::RunConfig& cfg,
                  std::mt19937& rng, int n_batches) {
    const std::int64_t N = cfg.model.seq_len, B = std::max<std::int64_t>(1, cfg.optim.batch);
    if (static_cast<std::int64_t>(stream.size()) <= N + B) return 0.0;
    dt::BatchedDiffusionLossContext ctx(B, N);
    std::uniform_int_distribution<std::size_t> off(0, stream.size() - static_cast<std::size_t>(N));
    double sum = 0.0;
    for (int i = 0; i < n_batches; ++i) {
        std::vector<std::size_t> offs(static_cast<std::size_t>(B));
        for (auto& o : offs) o = off(rng);
        auto res = dt::batched_diffusion_loss(model, stream, offs, rng, ctx, 0.02f, cfg.optim.t_max);
        sum += static_cast<double>(res.loss.data().to(sub0llm::Device::cpu()).template item<float>());
    }
    return sum / std::max(1, n_batches);
}

// Snapshot the live param Variables (copies sharing storage with the model AT THE CALL TIME).
// Must be re-taken whenever the model's device changes: Variable::to() swaps in a NEW storage tensor, so
// a snapshot taken before to(cuda) goes stale. load BEFORE to(cuda) (CPU weights) and re-snapshot at each
// save (current on-device weights — save_checkpoint D2H's them).
std::vector<sub0llm::autograd::Variable> snapshot_params(
    const std::vector<sub0llm::autograd::Variable*>& ptrs) {
    std::vector<sub0llm::autograd::Variable> v;
    v.reserve(ptrs.size());
    for (auto* p : ptrs) v.push_back(*p);
    return v;
}

// The model-agnostic training loop: resume weights+optimizer, train on GPU/CPU, checkpoint periodically.
template <class Model>
int run(Model& model, cfgm::RunConfig& cfg, std::span<const std::int32_t> train_ids,
        std::span<const std::int32_t> eval_ids, std::span<const std::uint8_t> ws_span,
        const std::string& resume_ckpt) {
    auto param_ptrs = model.parameters();

    // 1) Resume weights on CPU (the snapshot shares storage with the model, still on CPU here).
    std::uint64_t start_step = 0;
    if (!resume_ckpt.empty()) {
        auto cpu_view = snapshot_params(param_ptrs);
        start_step = static_cast<std::uint64_t>(sub0llm::load_checkpoint(cpu_view, resume_ckpt));
        std::println("resumed weights from {} (step {})", resume_ckpt, start_step);
    }

    // 2) Move to the device BEFORE building the optimizer, so Adam's (m,v) allocate on-device with the
    //    params (constructing the optimizer first leaves its state on the CPU → device-mismatch crash).
    const bool use_cuda = (cfg.optim.device == "cuda" || cfg.optim.device == "gpu");
    if (use_cuda) {
#ifdef SUB0LLM_CUDA
        model.to(sub0llm::Device::cuda());
        std::println("device: CUDA (single GPU stream)");
#else
        throw std::runtime_error("--device cuda requested but this binary was built without CUDA");
#endif
    } else {
        std::println("device: CPU");
    }

    // 3) Optimizer over the now-on-device params; restore Adam moments from the matching .opt on resume.
    auto opt = sub0llm::nn::make_optimizer(cfg.optim.optimizer, param_ptrs, cfg.optim.lr);
    if (!resume_ckpt.empty()) {
        fs::path opt_path = resume_ckpt; opt_path.replace_extension(".opt");
        if (fs::exists(opt_path) && opt->load_state(opt_path.string()))
            std::println("restored optimizer state from {}", opt_path.string());
        else
            std::println("note: no optimizer state at {} — Adam moments start fresh", opt_path.string());
    }

    const std::int64_t N = cfg.model.seq_len, B = std::max<std::int64_t>(1, cfg.optim.batch);
    const std::uint64_t steps = cfg.train.steps ? cfg.train.steps : 3000;
    const std::uint64_t ckpt_every = cfg.train.eval_every ? cfg.train.eval_every : 500;
    std::mt19937 rng(static_cast<std::uint32_t>(cfg.optim.seed) + static_cast<std::uint32_t>(start_step));
    std::uniform_int_distribution<std::size_t> off(0, train_ids.size() - static_cast<std::size_t>(N));
    (void)ws_span;  // whole-word / contiguous masking not wired in this lean trainer yet (defaults off)

    // One forward+backward+step on `offs`. The SAME batched loss drives CPU and the GPU single stream
    // (the model already sits on the chosen device; batched_diffusion_loss dispatches). Validated on
    // CUDA for MERA via ch32_viz_train. The optimizer steps here on the same Variables it writes grads to.
    dt::BatchedDiffusionLossContext loss_ctx(B, N);
    auto train_step = [&](const std::vector<std::size_t>& offs) {
        for (auto* p : param_ptrs)
            if (p->grad().numel() > 0)
                p->grad() = sub0llm::zeros(p->data().shape(), p->data().dtype(), p->data().device());
        auto res = dt::batched_diffusion_loss(model, train_ids, offs, rng, loss_ctx, 0.02f, cfg.optim.t_max);
        res.loss.backward();
        (void)sub0llm::nn::clip_grad_norm(param_ptrs, 5.0f);
        opt->step();
        return static_cast<double>(res.loss.data().to(sub0llm::Device::cpu()).template item<float>());
    };

    auto save = [&](std::uint64_t step, double nelbo) {
        auto view = snapshot_params(param_ptrs);   // current live params (on device) — save_checkpoint D2H's
        sub0llm::save_checkpoint(view, cfg.data.ckpt_dir, static_cast<std::int64_t>(step));
        const std::string ck = sub0llm::latest_checkpoint_path(cfg.data.ckpt_dir);
        if (!ck.empty()) { fs::path op = ck; op.replace_extension(".opt"); opt->save_state(op.string()); }
        std::println("  ckpt @ step {:>6}  eval_nelbo={:.4f}  -> {}", step, nelbo, ck);
    };

    std::println("training {} → {} steps (N={}, B={}, model={})", start_step, steps, N, B, cfg.model.model_type);
    const auto t0 = std::chrono::steady_clock::now();
    double last_loss = 0.0;
    for (std::uint64_t s = start_step; s < steps; ++s) {
        std::vector<std::size_t> offs(static_cast<std::size_t>(B));
        for (auto& o : offs) o = off(rng);
        last_loss = train_step(offs);
        if ((s + 1) % 200 == 0) {
            const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            std::println("  step {:>6}  train_nelbo={:.4f}  ({:.0f} steps/s)", s + 1, last_loss, (s + 1 - start_step) / std::max(1e-9, secs));
        }
        if ((s + 1) % ckpt_every == 0) save(s + 1, eval_nelbo(model, eval_ids, cfg, rng, 4));
    }
    // final checkpoint (so a resume always finds the last step)
    save(steps, eval_nelbo(model, eval_ids, cfg, rng, 8));
    std::println("done. checkpoints in {}", cfg.data.ckpt_dir);
    return 0;
}

}  // namespace

int run_main(int argc, char** argv) {
    cfgm::RunConfig cfg = cfgm::resolve(argc, argv);
    std::println("== Ch32 MERA trainer (resumable, CUDA) ==");

    // corpus → train / eval split (last 10% held out, min 1)
    auto paras = read_paragraphs(cfg.data.corpus, cfg.data.paragraphs);
    if (paras.size() < 20) throw std::runtime_error("need >=20 paragraphs");
    const std::size_t n_eval = std::max<std::size_t>(1, paras.size() / 10);
    std::vector<std::string> eval_paras(paras.end() - static_cast<std::ptrdiff_t>(n_eval), paras.end());
    paras.resize(paras.size() - n_eval);

    // tokenizer — reload from the ckpt dir if present (so a resume is deterministic + instant), else build.
    const fs::path tok_dir = fs::path(cfg.data.ckpt_dir) / "tokenizer";
    const fs::path vocab_json = tok_dir / "vocab.json", merges_txt = tok_dir / "merges.txt";
    BPETokenizer tok = [&] {
        if (fs::exists(vocab_json) && fs::exists(merges_txt)) {
            std::println("loaded tokenizer from {}", tok_dir.string());
            return BPETokenizer::load(vocab_json, merges_txt);
        }
        if (cfg.data.char_level) { std::print("char tokenizer... "); auto t = BPETokenizer::char_level(paras); std::println("{} vocab", t.vocab_size()); return t; }
        if (cfg.data.word_level) { std::print("word tokenizer... "); auto t = BPETokenizer::word_level(paras); std::println("{} vocab", t.vocab_size()); return t; }
        std::print("BPE tokenizer (vocab {})... ", cfg.model.vocab_size);
        auto t = BPETokenizer::train(paras, cfg.model.vocab_size); std::println("done"); return t;
    }();
    const std::int64_t V = static_cast<std::int64_t>(tok.vocab_size());
    cfg.model.vocab_size = V;   // pin REAL post-tokenizer vocab so a resume rebuilds the exact arch

    std::vector<std::uint8_t> is_word_start(tok.vocab_size(), 0);
    for (std::size_t id = 0; id < tok.vocab_size(); ++id) {
        const auto s = tok.token_str(static_cast<BPETokenizer::TokenId>(id));
        is_word_start[id] = (s.size() >= 2 && static_cast<unsigned char>(s[0]) == 0xC4 &&
                             static_cast<unsigned char>(s[1]) == 0xA0) ? 1 : 0;
    }

    auto flatten = [&](const std::vector<std::string>& texts) {
        std::vector<std::int32_t> ids;
        for (const auto& p : texts) { auto v = tok.encode(p); ids.insert(ids.end(), v.begin(), v.end()); }
        return ids;
    };
    auto train_ids = flatten(paras);
    auto eval_ids  = flatten(eval_paras);
    if (static_cast<std::int64_t>(train_ids.size()) < cfg.model.seq_len + cfg.optim.batch + 8)
        throw std::runtime_error("corpus too short for seq_len/batch");
    std::println("train {} tok / eval {} tok, seq_len {}", train_ids.size(), eval_ids.size(), cfg.model.seq_len);

    // resume detection BEFORE we (maybe) write config.json/tokenizer
    fs::create_directories(cfg.data.ckpt_dir);
    const std::string resume_ckpt = sub0llm::latest_checkpoint_path(cfg.data.ckpt_dir);
    if (resume_ckpt.empty()) {
        tok.save(tok_dir);
        std::ofstream cj(fs::path(cfg.data.ckpt_dir) / "config.json");
        cj << std::format(
            "{{\n  \"model\": \"sub0diff-{}\",\n  \"model_type\": \"{}\",\n  \"vocab_size\": {},\n"
            "  \"embed_dim\": {},\n  \"n_layers\": {},\n  \"n_heads\": {},\n  \"n_kv_heads\": {},\n"
            "  \"d_ff\": {},\n  \"seq_len\": {},\n  \"mera_coarsen\": {},\n  \"mera_window\": {}\n}}\n",
            cfg.model.model_type, cfg.model.model_type, V, cfg.model.embed_dim, cfg.model.n_layers,
            cfg.model.n_heads, cfg.model.n_kv_heads, cfg.model.d_ff, cfg.model.seq_len,
            cfg.model.mera_coarsen, cfg.model.mera_window);
        std::println("fresh run — wrote tokenizer/ + config.json to {}", cfg.data.ckpt_dir);
    } else {
        std::println("resuming from {}", resume_ckpt);
    }
    // persist the FULL resolved config every start — `--ckpt-dir X` alone then resumes this exact run
    cfgm::write_run_config(cfg.data.ckpt_dir, cfg, SUB0DIFF_CODE_SHA);

    const std::span<const std::uint8_t> ws_span(is_word_start);
    if (cfg.model.model_type == "mera") {
        dn::MeraDenoiser model(V, cfg.model.embed_dim, static_cast<std::size_t>(cfg.model.n_heads),
                               static_cast<std::size_t>(cfg.model.n_kv_heads), cfg.model.mera_coarsen,
                               cfg.model.mera_window, cfg.model.seq_len, cfg.model.d_ff, cfg.optim.seed);
        std::println("MeraDenoiser: V={} D={} c={} w={} levels={}", V, cfg.model.embed_dim,
                     cfg.model.mera_coarsen, cfg.model.mera_window,
                     [&]{ std::string s; for (auto l : model.level_lens(cfg.model.seq_len)) s += std::format("{} ", l); return s; }());
        return run(model, cfg, train_ids, eval_ids, ws_span, resume_ckpt);
    }
    dn::Denoiser model(V, cfg.model.embed_dim, static_cast<std::size_t>(cfg.model.n_heads),
                       static_cast<std::size_t>(cfg.model.n_kv_heads), cfg.model.n_layers,
                       cfg.model.d_ff, cfg.optim.seed);
    std::println("Denoiser (flat): V={} D={} layers={}", V, cfg.model.embed_dim, cfg.model.n_layers);
    return run(model, cfg, train_ids, eval_ids, ws_span, resume_ckpt);
}

int main(int argc, char** argv) {
    sub0llm::init_cpu_compute();   // FTZ+DAZ on the main thread (Adam state goes denormal as loss falls)
    try {
        return run_main(argc, argv);
    } catch (const std::exception& e) {
        std::fflush(stdout);
        std::println(stderr, "ERROR: {}", e.what());
        return 1;
    }
}
