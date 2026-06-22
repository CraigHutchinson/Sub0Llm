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

#include <simdjson.h>   // train_state.json read (forward on-demand)

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <ctime>
#include <filesystem>
#include <format>
#include <fstream>
#include <iterator>
#include <limits>
#include <memory>
#include <print>
#include <random>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
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

bool has_flag(int argc, char** argv, const std::string& k) {
    for (int i = 1; i < argc; ++i) if (k == argv[i]) return true;
    return false;
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

// The dynamic-progress sidecar for an HONEST resume: weights live in step_*.ckpt and Adam moments in
// step_*.opt; this train_state.json holds the REST — the early-stop history — so a resume continues
// mid-climb instead of restarting the best-tracking. `step` is matched against the resumed checkpoint so
// a stale sidecar is ignored. Written as JSON (per the project model-artifact convention); READ with
// simdjson on-demand (forward, single pass — the project JSON-read direction).
struct TrainState {
    bool          have = false;
    std::uint64_t step = 0;
    double        best_nelbo = std::numeric_limits<double>::max();
    std::uint64_t evals_since_best = 0;
};

void save_train_state(const fs::path& path, const TrainState& s, std::string_view code_sha,
                      std::uint64_t config_sha) {
    std::ofstream f(path);
    if (!f) return;
    f << std::format(
        "{{\n  \"step\": {},\n  \"best_nelbo\": {:.9g},\n  \"evals_since_best\": {},\n"
        "  \"code_sha\": \"{}\",\n  \"config_sha\": \"{:016x}\",\n  \"updated_unix\": {}\n}}\n",
        s.step, s.best_nelbo, s.evals_since_best, code_sha, config_sha,
        static_cast<std::int64_t>(std::time(nullptr)));
}

// Load + validate against the resumed step; have=false on absence/mismatch (a stale sidecar is ignored).
[[nodiscard]] TrainState load_train_state(const fs::path& path, std::uint64_t expect_step) {
    TrainState s;
    std::ifstream f(path, std::ios::binary);
    if (!f) return s;
    std::string body((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    if (body.empty()) return s;
    simdjson::ondemand::parser parser;
    simdjson::padded_string padded(body.data(), body.size());
    auto doc = parser.iterate(padded);
    if (doc.error()) return s;
    auto obj = doc.get_object();
    if (obj.error()) return s;
    for (auto field : obj) {
        auto k = field.unescaped_key(); if (k.error()) continue;
        auto v = field.value();         if (v.error()) continue;
        const std::string_view key = k.value_unsafe();
        if      (key == "step")             { std::uint64_t x; if (!v.get(x)) s.step = x; }
        else if (key == "best_nelbo")       { double x;        if (!v.get(x)) s.best_nelbo = x; }
        else if (key == "evals_since_best") { std::uint64_t x; if (!v.get(x)) s.evals_since_best = x; }
    }
    s.have = (s.step == expect_step);
    return s;
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

    // 4) Dynamic training-state sidecar — the early-stop history, for an HONEST resume (continues the
    //    best-tracking instead of restarting it). step is validated against the resumed checkpoint.
    const std::string code_sha = SUB0DIFF_CODE_SHA;
    const std::uint64_t cfg_sha = cfgm::config_sha(cfg);
    const fs::path state_path = fs::path(cfg.data.ckpt_dir) / "train_state.json";
    TrainState state;
    if (!resume_ckpt.empty()) {
        state = load_train_state(state_path, start_step);
        if (state.have)
            std::println("rehydrated train_state.json: best_nelbo={:.4f}, evals_since_best={}",
                         state.best_nelbo, state.evals_since_best);
        else
            std::println("note: no matching train_state.json @ step {} — best-tracking starts fresh", start_step);
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

    // Checkpoint = eval + update best-tracking + write weights/.opt/train_state.json (the full honest
    // resume set). Returns true if early-stop should fire (patience stalled evals with no improvement).
    auto checkpoint = [&](std::uint64_t step, int eval_batches) {
        const double nelbo = eval_nelbo(model, eval_ids, cfg, rng, eval_batches);
        constexpr double kMinImprove = 0.01;
        if (nelbo < state.best_nelbo - kMinImprove) { state.best_nelbo = nelbo; state.evals_since_best = 0; }
        else ++state.evals_since_best;
        state.step = step;
        auto view = snapshot_params(param_ptrs);   // current live params (on device) — save_checkpoint D2H's
        sub0llm::save_checkpoint(view, cfg.data.ckpt_dir, static_cast<std::int64_t>(step));
        const std::string ck = sub0llm::latest_checkpoint_path(cfg.data.ckpt_dir);
        if (!ck.empty()) { fs::path op = ck; op.replace_extension(".opt"); opt->save_state(op.string()); }
        save_train_state(state_path, state, code_sha, cfg_sha);   // honest-resume sidecar
        const bool stop = cfg.train.patience > 0 && state.evals_since_best >= cfg.train.patience;
        std::println("  ckpt @ step {:>6}  eval_nelbo={:.4f}  best={:.4f}  stalls={}{}  -> {}",
                     step, nelbo, state.best_nelbo, state.evals_since_best, stop ? "  EARLY-STOP" : "", ck);
        return stop;
    };

    std::println("training {} → {} steps (N={}, B={}, model={})", start_step, steps, N, B, cfg.model.model_type);
    const auto t0 = std::chrono::steady_clock::now();
    double last_loss = 0.0;
    bool stopped = false;
    std::uint64_t last_step = start_step;
    for (std::uint64_t s = start_step; s < steps; ++s) {
        std::vector<std::size_t> offs(static_cast<std::size_t>(B));
        for (auto& o : offs) o = off(rng);
        last_loss = train_step(offs);
        last_step = s + 1;
        if ((s + 1) % 200 == 0) {
            const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            std::println("  step {:>6}  train_nelbo={:.4f}  ({:.0f} steps/s)", s + 1, last_loss, (s + 1 - start_step) / std::max(1e-9, secs));
        }
        if ((s + 1) % ckpt_every == 0 && checkpoint(s + 1, 4)) { stopped = true; break; }
    }
    // final checkpoint so a resume always finds the last step (skip if already saved this step)
    if (!stopped && last_step > start_step && last_step % ckpt_every != 0) checkpoint(last_step, 8);
    std::println("done. checkpoints + train_state.json in {}", cfg.data.ckpt_dir);
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

    // --name defers the real ckpt dir until the vocab is known (the dir name embeds config_sha, which
    // depends on vocab) — so with --name we must NOT reload a tokenizer from the *default* dir (a stale
    // one there would silently override --word-level/--char-level). Build fresh, then reload from the
    // finalized name dir only.
    const bool use_name = !cfg.data.name.empty() && !has_flag(argc, argv, "--ckpt-dir");

    auto build_tokenizer = [&](bool allow_reload, const fs::path& dir) -> BPETokenizer {
        const fs::path vj = dir / "tokenizer" / "vocab.json", mt = dir / "tokenizer" / "merges.txt";
        if (allow_reload && fs::exists(vj) && fs::exists(mt)) {
            std::println("loaded tokenizer from {}", (dir / "tokenizer").string());
            return BPETokenizer::load(vj, mt);
        }
        if (cfg.data.char_level) { std::print("char tokenizer... "); auto t = BPETokenizer::char_level(paras); std::println("{} vocab", t.vocab_size()); return t; }
        if (cfg.data.word_level) { std::print("word tokenizer... "); auto t = BPETokenizer::word_level(paras); std::println("{} vocab", t.vocab_size()); return t; }
        std::print("BPE tokenizer (vocab {})... ", cfg.model.vocab_size);
        auto t = BPETokenizer::train(paras, cfg.model.vocab_size); std::println("done"); return t;
    };
    BPETokenizer tok = build_tokenizer(/*allow_reload=*/!use_name, cfg.data.ckpt_dir);
    cfg.model.vocab_size = static_cast<std::int64_t>(tok.vocab_size());  // pin REAL vocab → exact-arch resume

    // In-repo, provenance-tagged model dir: models/<name>_g<gitSHA>_c<configSHA> (config_sha needs the
    // now-pinned vocab). Deterministic tokenization ⇒ same name+config reproduces the SAME dir → resume.
    if (use_name) {
        cfg.data.ckpt_dir = std::format("models/{}_g{}_c{:016x}", cfg.data.name,
                                        std::string(SUB0DIFF_CODE_SHA), cfgm::config_sha(cfg));
        std::println("model dir (from --name): {}", cfg.data.ckpt_dir);
        const fs::path vj = fs::path(cfg.data.ckpt_dir) / "tokenizer" / "vocab.json";
        const fs::path mt = fs::path(cfg.data.ckpt_dir) / "tokenizer" / "merges.txt";
        if (fs::exists(vj) && fs::exists(mt)) {   // a prior run of this exact config → reload for exactness
            tok = BPETokenizer::load(vj, mt);
            cfg.model.vocab_size = static_cast<std::int64_t>(tok.vocab_size());
        }
    }
    const std::int64_t V = cfg.model.vocab_size;

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
        tok.save(fs::path(cfg.data.ckpt_dir) / "tokenizer");   // final dir (--name may have moved it)
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
