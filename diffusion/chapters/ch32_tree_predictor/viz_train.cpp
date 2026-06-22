// viz_train.cpp (Ch32 Viz, Phase D) — TRAINING-TIME trajectory: watch the engine learn to denoise.
//
// Trains a MERA and, every `--snap-every` steps, generates ONE traced passage with a FIXED prompt + seed
// and records it tagged with the training step + current NELBO. Because the prompt and seed are held
// constant, the ONLY thing that changes across snapshots is the model — so scrubbing the "training step"
// axis in the viewer shows the same prompt's reverse process sharpening as the model learns (early: noisy
// unigram mush; later: structured prose). Writes tools/viz/trajectory.json, a list of {step, nelbo, trace}
// where each trace is the same GenerationTrace the scrubber already understands.
//
// Build: cmake --build build-native --target ch32_viz_train   (Release CPU links everywhere; trains fast)
// Run:   ./build-native/diffusion/chapters/ch32_tree_predictor/ch32_viz_train.exe --device cpu

#include "sub0diff/nn/denoiser.hpp"
#include "sub0diff/nn/mera_denoiser.hpp"
#include "sub0diff/nn/sampler.hpp"
#include "sub0diff/train/diffusion_loss.hpp"
#include "sub0diff/viz/trace.hpp"
#include "sub0diff/viz/trace_json.hpp"

#include "sub0llm/core/ops.hpp"
#include "sub0llm/nn/optimizer.hpp"
#include "sub0llm/tokenizer/bpe.hpp"

#include <chrono>
#include <cstdint>
#include <fstream>
#include <format>
#include <print>
#include <random>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

using sub0llm::BPETokenizer;
namespace dn = sub0diff::nn;
namespace dt = sub0diff::train;
namespace dv = sub0diff::viz;

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

std::int64_t arg_i(int argc, char** argv, const std::string& k, std::int64_t d) {
    for (int i = 1; i + 1 < argc; ++i) if (k == argv[i]) return std::stoll(argv[i + 1]);
    return d;
}
std::string arg_s(int argc, char** argv, const std::string& k, const std::string& d) {
    for (int i = 1; i + 1 < argc; ++i) if (k == argv[i]) return argv[i + 1];
    return d;
}

}  // namespace

int main(int argc, char** argv) {
    const std::string corpus  = arg_s(argc, argv, "--corpus", "data/tinystories_clean.txt");
    const std::string out     = arg_s(argc, argv, "--out", "tools/viz/trajectory.json");
    const std::string prompt_text = arg_s(argc, argv, "--prompt", "once there was a");
    const std::string devs    = arg_s(argc, argv, "--device", "cuda");
    const std::int64_t plimit = arg_i(argc, argv, "--paragraphs", 600);
    const std::int64_t steps  = arg_i(argc, argv, "--steps", 3000);
    const std::int64_t snap   = arg_i(argc, argv, "--snap-every", 300);
    const std::int64_t D      = arg_i(argc, argv, "--embed_dim", 256);
    const std::int64_t N      = arg_i(argc, argv, "--seq_len", 128);
    const std::int64_t w      = arg_i(argc, argv, "--window", 64);
    const std::int64_t c      = arg_i(argc, argv, "--coarsen", 4);
    const std::int64_t B      = arg_i(argc, argv, "--batch", 8);
    const float temp          = static_cast<float>(arg_i(argc, argv, "--temp_x100", 80)) / 100.0f;
    const std::uint64_t seed  = static_cast<std::uint64_t>(arg_i(argc, argv, "--seed", 7));
    const sub0llm::Device train_dev = devs == "cpu" ? sub0llm::Device::cpu() : sub0llm::Device::cuda();

    std::println("== Ch32 Viz Phase D — training-time trajectory ==");
    auto paras = read_paragraphs(corpus, plimit);
    if (paras.size() < 20) throw std::runtime_error("need >=20 paragraphs");
    std::print("building word tokenizer over {} paragraphs... ", paras.size());
    BPETokenizer tok = BPETokenizer::word_level(paras);
    const std::int64_t Vr = static_cast<std::int64_t>(tok.vocab_size());
    std::println("done — {} word vocab", Vr);

    std::vector<std::int32_t> ids;
    for (const auto& p : paras) { auto v = tok.encode(p); ids.insert(ids.end(), v.begin(), v.end()); }
    if (static_cast<std::int64_t>(ids.size()) < N + B + 8) throw std::runtime_error("corpus too short");

    auto decode = [&tok, Vr](std::int32_t id) {
        return (id >= 0 && id < Vr) ? std::string(tok.token_str(static_cast<BPETokenizer::TokenId>(id)))
                                    : std::string("?");
    };

    dn::MeraDenoiser model(Vr, D, 8, 4, c, w, N, 0, seed);
    model.to(train_dev);
    auto params = model.parameters();
    sub0llm::nn::Adam opt(params, 1e-3f);
    dt::BatchedDiffusionLossContext ctx(B, N);
    std::mt19937 rng(static_cast<std::uint32_t>(seed));
    std::uniform_int_distribution<std::size_t> off(0, ids.size() - static_cast<std::size_t>(N));

    // The fixed prompt (token ids) used at EVERY snapshot — only the model changes between snapshots.
    std::vector<std::int32_t> prompt;
    { auto enc = tok.encode(prompt_text);
      for (auto t : enc) if (t >= 0 && t < Vr && static_cast<std::int64_t>(prompt.size()) < N) prompt.push_back(t); }

    dn::SamplerConfig cfg;
    cfg.temperature = temp;
    cfg.commit_order = dn::CommitOrder::Spread;
    const std::int32_t mask_id = model.mask_id();

    // A snapshot = the step, the NELBO there, and the serialized GenerationTrace for the fixed prompt.
    struct Snap { std::int64_t step; double nelbo; std::string trace_json; };
    std::vector<Snap> snaps;

    auto take_snapshot = [&](std::int64_t step, double nelbo) {
        model.to(sub0llm::Device::cpu());                 // generation runs on CPU (reads host logits)
        dv::GenerationTrace trace;
        trace.T = N; trace.model = "mera"; trace.N = N; trace.c = c; trace.w = w;
        trace.commit_order = "spread"; trace.temperature = cfg.temperature;
        trace.conf_threshold = cfg.conf_threshold; trace.min_commit_frac = cfg.min_commit_frac;
        trace.remask_threshold = cfg.remask_threshold; trace.prompt = prompt_text;
        trace.levels = model.level_lens(N);
        std::mt19937 grng(seed * 131 + 5);                // SAME seed every snapshot
        auto canvas = dn::make_canvas(model, N, prompt);
        dn::refine_canvas(model, canvas, cfg, grng, {}, &trace);
        snaps.push_back({step, nelbo, dv::serialize_trace_json(trace, decode, mask_id)});
        model.to(train_dev);
        std::string txt; for (auto t : canvas) { txt += decode(t); txt += ' '; }
        std::println("  snap @ step {:>5}  nelbo={:.3f}  iters={}  text: {}",
                     step, nelbo, trace.frames.size(), txt.substr(0, 90));
    };

    std::println("\ntraining {} steps (snapshot every {}), prompt=\"{}\"...", steps, snap, prompt_text);
    double last_nelbo = 0.0;
    for (std::int64_t s = 0; s < steps; ++s) {
        std::vector<std::size_t> offs(static_cast<std::size_t>(B));
        for (auto& o : offs) o = off(rng);
        for (auto* p : params)
            if (p->grad().numel() > 0)
                p->grad() = sub0llm::zeros(p->data().shape(), p->data().dtype(), p->data().device());
        auto res = dt::batched_diffusion_loss(model, ids, offs, rng, ctx, 0.02f, 1.0f);
        res.loss.backward();
        (void)sub0llm::nn::clip_grad_norm(params, 5.0f);
        opt.step();
        last_nelbo = static_cast<double>(res.loss.data().to(sub0llm::Device::cpu()).item<float>());
        if (s == 0 || (s + 1) % snap == 0) take_snapshot(s + 1, last_nelbo);
    }

    // assemble the trajectory artifact: { meta, snapshots:[{step,nelbo,trace}, …] }
    std::string lv; for (auto l : model.level_lens(N)) { if (!lv.empty()) lv += ','; lv += std::format("{}", l); }
    std::string j = std::format(
        R"({{"version":1,"kind":"training_trajectory","meta":{{"model":"mera","N":{},"coarsen":{},"window":{},"levels":[{}],"prompt":"{}","seed":{},"snap_every":{},"n_snapshots":{}}},"snapshots":[)",
        N, c, w, lv, prompt_text, seed, snap, snaps.size());
    for (std::size_t i = 0; i < snaps.size(); ++i) {
        if (i) j += ',';
        j += std::format(R"({{"step":{},"nelbo":{:.4f},"trace":{}}})", snaps[i].step, snaps[i].nelbo, snaps[i].trace_json);
    }
    j += "]}";
    { std::ofstream os(out); if (!os) throw std::runtime_error("cannot write " + out); os << j; }

    std::println("\n── self-check ──");
    std::println("  snapshots   : {}", snaps.size());
    std::println("  steps       : {}", [&]{ std::string s; for (auto& sn : snaps) s += std::format("{} ", sn.step); return s; }());
    std::println("  nelbo first→last : {:.3f} → {:.3f}{}", snaps.front().nelbo, snaps.back().nelbo,
                 snaps.back().nelbo < snaps.front().nelbo ? "  (learning ✓)" : "");
    std::println("  trajectory  : {}", out);
    if (snaps.size() < 2) { std::println("  STATUS: FAIL (need >=2 snapshots)"); return 2; }
    std::println("  STATUS: OK");
    std::println("\n── how to view ──");
    std::println("  serve repo root (python -m http.server 8000) then open http://localhost:8000/tools/viz/");
    std::println("  load {} → a 'training step' slider appears above the iteration scrubber.", out);
    return 0;
}
