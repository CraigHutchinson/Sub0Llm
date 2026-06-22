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
#include "sub0diff/train/checkpointer.hpp"   // reusable checks: coverage cadence + early-stop + resume I/O
#include "sub0diff/train/diffusion_loss.hpp"
#include "sub0diff/train/schedule.hpp"        // make_schedule — coverage-rule eval cadence + sample size
#include "sub0diff/util/heartbeat.hpp"        // wall-clock progress log + safety-checkpoint cadence

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

bool has_flag(int argc, char** argv, const std::string& k) {
    for (int i = 1; i < argc; ++i) if (k == argv[i]) return true;
    return false;
}

// Held-out NELBO AVERAGED over ~`target_windows` random eval windows — the coverage rule's stable
// signal (a too-small sample makes consecutive evals differ by noise, tripping early-stop on noise).
// target_windows comes from make_schedule (clamp(sliding_eval/500, 64, 512)).
template <class Model>
double eval_nelbo(Model& model, std::span<const std::int32_t> stream, const cfgm::RunConfig& cfg,
                  std::mt19937& rng, std::size_t target_windows) {
    const std::int64_t N = cfg.model.seq_len, B = std::max<std::int64_t>(1, cfg.optim.batch);
    if (static_cast<std::int64_t>(stream.size()) <= N + B) return 0.0;
    dt::BatchedDiffusionLossContext ctx(B, N);
    std::uniform_int_distribution<std::size_t> off(0, stream.size() - static_cast<std::size_t>(N));
    const int n_batches = std::max<int>(1, static_cast<int>((target_windows + static_cast<std::size_t>(B) - 1) / static_cast<std::size_t>(B)));
    double sum = 0.0;
    for (int i = 0; i < n_batches; ++i) {
        std::vector<std::size_t> offs(static_cast<std::size_t>(B));
        for (auto& o : offs) o = off(rng);
        auto res = dt::batched_diffusion_loss(model, stream, offs, rng, ctx, 0.02f, cfg.optim.t_max);
        sum += static_cast<double>(res.loss.data().to(sub0llm::Device::cpu()).template item<float>());
    }
    return sum / n_batches;
}

// The model-agnostic training loop: resume weights+optimizer, train on GPU/CPU, checkpoint periodically.
// All the reusable CHECKS (coverage-rule cadence, early-stop, honest-resume I/O) live in dt::Checkpointer.
template <class Model>
int run(Model& model, cfgm::RunConfig& cfg, std::span<const std::int32_t> train_ids,
        std::span<const std::int32_t> eval_ids, std::span<const std::uint8_t> ws_span) {
    auto param_ptrs = model.parameters();
    const std::int64_t N = cfg.model.seq_len, B = std::max<std::int64_t>(1, cfg.optim.batch);
    (void)ws_span;  // whole-word / contiguous masking not wired in this lean trainer yet (defaults off)

    // THE COVERAGE RULE (ch29 schedule.hpp): never eval more often than ~½ epoch of coverage, and average
    // over a corpus-scaled sample — else consecutive evals differ only by noise and early-stop trips on it.
    const double mean_t = 0.5 * (0.02 + static_cast<double>(cfg.optim.t_max));
    const auto sched = dt::make_schedule(
        static_cast<std::uint64_t>(train_ids.size()), static_cast<std::uint64_t>(eval_ids.size()),
        N, static_cast<std::size_t>(B), mean_t * static_cast<double>(N), {},
        cfg.train.eval_every, cfg.train.steps);
    std::println("schedule: epoch={} windows ({} steps/epoch); eval every {} steps = {:.0f}% epoch "
                 "coverage, averaged over {} eval windows; steps bound {}",
                 sched.epoch_windows, B > 0 ? sched.epoch_windows / static_cast<std::uint64_t>(B) : 0,
                 sched.eval_every, 100.0 * sched.coverage_per_eval, sched.eval_nelbo_windows, sched.steps_bound);

    // The reusable training-loop checks: cadence + early-stop + honest-resume I/O in one tested place.
    dt::Checkpointer ck(sched, cfg.data.ckpt_dir, SUB0DIFF_CODE_SHA, cfgm::config_sha(cfg), cfg.train.patience);

    // 1) Resume weights on CPU (the snapshot shares storage with the model, still on CPU here).
    const std::uint64_t start_step = ck.load_weights(param_ptrs);
    if (start_step) std::println("resumed weights from {} (step {})", ck.resumed_from(), start_step);

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

    // 3) Optimizer over the now-on-device params, then restore Adam moments + the early-stop history.
    auto opt = sub0llm::nn::make_optimizer(cfg.optim.optimizer, param_ptrs, cfg.optim.lr);
    ck.restore(*opt);
    if (start_step && ck.progress().have)
        std::println("rehydrated train_state.json: best_nelbo={:.4f}, evals_since_best={}",
                     ck.progress().best, ck.progress().stalls);

    const std::uint64_t steps = ck.steps_bound();
    std::mt19937 rng(static_cast<std::uint32_t>(cfg.optim.seed) + static_cast<std::uint32_t>(start_step));
    std::uniform_int_distribution<std::size_t> off(0, train_ids.size() - static_cast<std::size_t>(N));

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

    // Eval + checkpoint via the reusable guard: averaged held-out NELBO over the schedule's sample, then
    // ck.record saves weights/.opt/train_state.json, updates best/stalls, and reports early-stop.
    auto checkpoint = [&](std::uint64_t step) {
        const double nelbo = eval_nelbo(model, eval_ids, cfg, rng, ck.eval_windows());
        const bool stop = ck.record(step, nelbo, param_ptrs, *opt);
        std::println("  ckpt @ step {:>6}  eval_nelbo={:.4f}  best={:.4f}  stalls={}{}  -> {}",
                     step, nelbo, ck.progress().best, ck.progress().stalls,
                     stop ? "  EARLY-STOP" : "", sub0llm::latest_checkpoint_path(cfg.data.ckpt_dir));
        return stop;
    };

    std::println("training {} → {} steps (N={}, B={}, model={})", start_step, steps, N, B, cfg.model.model_type);
    double last_loss = 0.0;
    bool stopped = false;
    std::uint64_t last_step = start_step, log_step = start_step;
    // Wall-clock cadence (decoupled from the deliberately-rare ½-epoch eval): a progress line every 30 s
    // with the INSTANTANEOUS rate over the interval, and a rolling safety checkpoint every 3 min so a
    // crash mid-epoch loses minutes, not hours. The eval-driven best checkpoint still rides ck.due().
    sub0diff::util::Heartbeat log_hb(30.0), save_hb(180.0);
    for (std::uint64_t s = start_step; s < steps; ++s) {
        std::vector<std::size_t> offs(static_cast<std::size_t>(B));
        for (auto& o : offs) o = off(rng);
        last_loss = train_step(offs);
        last_step = s + 1;
        if (const double dt = log_hb.due(); dt > 0.0) {
            std::println("  step {:>8}  train_nelbo={:.4f}  ({:.0f} steps/s)", s + 1, last_loss,
                         static_cast<double>(s + 1 - log_step) / dt);
            log_step = s + 1;
        }
        if (save_hb.due() > 0.0) ck.save_safety(s + 1, param_ptrs, *opt);   // crash insurance between evals
        if (ck.due(s + 1) && checkpoint(s + 1)) { stopped = true; break; }
    }
    // final checkpoint so a resume always finds the last step (skip if already saved this step)
    if (!stopped && last_step > start_step && !ck.due(last_step)) checkpoint(last_step);
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
            "  \"d_ff\": {},\n  \"seq_len\": {},\n  \"mera_coarsen\": {},\n  \"mera_window\": {},\n"
            "  \"mera_gated_pool\": {}\n}}\n",
            cfg.model.model_type, cfg.model.model_type, V, cfg.model.embed_dim, cfg.model.n_layers,
            cfg.model.n_heads, cfg.model.n_kv_heads, cfg.model.d_ff, cfg.model.seq_len,
            cfg.model.mera_coarsen, cfg.model.mera_window, cfg.model.mera_gated_pool);
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
                               cfg.model.mera_window, cfg.model.seq_len, cfg.model.d_ff, cfg.optim.seed,
                               cfg.model.mera_gated_pool);
        std::println("MeraDenoiser: V={} D={} c={} w={} pool={} levels={}", V, cfg.model.embed_dim,
                     cfg.model.mera_coarsen, cfg.model.mera_window,
                     cfg.model.mera_gated_pool ? "gated" : "mean",
                     [&]{ std::string s; for (auto l : model.level_lens(cfg.model.seq_len)) s += std::format("{} ", l); return s; }());
        // Footgun guard: MERA only coarsens while a level length exceeds the window. seq_len ≤ window
        // ⇒ ONE level ⇒ no hierarchy at all (a single full-attention block mislabelled "mera"). The
        // whole point of MERA is the recursive pyramid, so warn loudly — use seq_len ≫ window (e.g.
        // seq_len 256, window 16, coarsen 4 → 256·64·16) to actually exercise it.
        if (model.n_levels_for(cfg.model.seq_len) <= 1)
            std::println("  [WARN] seq_len {} ≤ mera_window {} → a SINGLE level (no coarsening). This "
                         "trains a flat block, not a hierarchy. Set seq_len ≫ window for real MERA.",
                         cfg.model.seq_len, cfg.model.mera_window);
        return run(model, cfg, train_ids, eval_ids, ws_span);
    }
    dn::Denoiser model(V, cfg.model.embed_dim, static_cast<std::size_t>(cfg.model.n_heads),
                       static_cast<std::size_t>(cfg.model.n_kv_heads), cfg.model.n_layers,
                       cfg.model.d_ff, cfg.optim.seed);
    std::println("Denoiser (flat): V={} D={} layers={}", V, cfg.model.embed_dim, cfg.model.n_layers);
    return run(model, cfg, train_ids, eval_ids, ws_span);
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
