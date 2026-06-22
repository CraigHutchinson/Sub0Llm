// viz_gen.cpp (Ch32 Viz, Phase A) — generate ONE traced passage and write the GenerationTrace JSON.
//
// Trains a small MERA (or flat) denoiser, generates a single canvas with a GenerationTrace attached,
// and writes it to tools/viz/trace.json for the web scrubber. Includes a self-check (frames present,
// canvas filled) and prints exactly how to view it. Build: cmake --build build-cuda --target ch32_viz_gen.

#include "sub0diff/nn/denoiser.hpp"
#include "sub0diff/nn/mera_denoiser.hpp"
#include "sub0diff/nn/sampler.hpp"
#include "sub0diff/train/diffusion_loss.hpp"
#include "sub0diff/viz/trace.hpp"
#include "sub0diff/viz/trace_json.hpp"

#include "sub0llm/core/ops.hpp"
#include "sub0llm/core/runtime.hpp"   // init_cpu_compute (FTZ+DAZ)
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

template <class Model>
void train(Model& model, std::span<const std::int32_t> stream, int steps, std::int64_t B,
           std::int64_t N, std::uint64_t seed) {
    model.to(sub0llm::Device::cuda());
    auto params = model.parameters();
    sub0llm::nn::Adam opt(params, 1e-3f);
    dt::BatchedDiffusionLossContext ctx(B, N);
    std::mt19937 rng(static_cast<std::uint32_t>(seed));
    std::uniform_int_distribution<std::size_t> off(0, stream.size() - static_cast<std::size_t>(N));
    const auto t0 = std::chrono::steady_clock::now();
    for (int s = 0; s < steps; ++s) {
        std::vector<std::size_t> offs(static_cast<std::size_t>(B));
        for (auto& o : offs) o = off(rng);
        for (auto* p : params)
            if (p->grad().numel() > 0)
                p->grad() = sub0llm::zeros(p->data().shape(), p->data().dtype(), p->data().device());
        auto res = dt::batched_diffusion_loss(model, stream, offs, rng, ctx, 0.02f, 1.0f);
        res.loss.backward();
        (void)sub0llm::nn::clip_grad_norm(params, 5.0f);
        opt.step();
        if ((s + 1) % 1000 == 0)
            std::println("  step {:>5}  nelbo={:.4f}  ({:.1f}s)", s + 1,
                         static_cast<double>(res.loss.data().to(sub0llm::Device::cpu()).item<float>()),
                         std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
    }
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
    sub0llm::init_cpu_compute();   // FTZ+DAZ (training-throughput prerequisite)
    const std::string corpus = arg_s(argc, argv, "--corpus", "data/tinystories_clean.txt");
    const std::string which  = arg_s(argc, argv, "--model", "mera");      // mera | flat
    const std::string out    = arg_s(argc, argv, "--out", "tools/viz/trace.json");
    const std::int64_t plimit = arg_i(argc, argv, "--paragraphs", 600);
    const std::int64_t steps  = arg_i(argc, argv, "--steps", 2000);
    const std::int64_t D      = arg_i(argc, argv, "--embed_dim", 256);
    const std::int64_t N      = arg_i(argc, argv, "--seq_len", 128);
    const std::int64_t w      = arg_i(argc, argv, "--window", 64);
    const std::int64_t c      = arg_i(argc, argv, "--coarsen", 4);
    const std::int64_t L      = arg_i(argc, argv, "--n_layers", 4);
    const std::int64_t B      = arg_i(argc, argv, "--batch", 8);
    const float temp          = static_cast<float>(arg_i(argc, argv, "--temp_x100", 90)) / 100.0f;
    const std::uint64_t seed  = static_cast<std::uint64_t>(arg_i(argc, argv, "--seed", 7));

    std::println("== Ch32 Viz — generate a traced passage ({}) ==", which);
    auto paras = read_paragraphs(corpus, plimit);
    if (paras.size() < 20) throw std::runtime_error("need >=20 paragraphs");
    std::print("building word tokenizer over {} paragraphs... ", paras.size());
    BPETokenizer tok = BPETokenizer::word_level(paras);
    const std::int64_t Vr = static_cast<std::int64_t>(tok.vocab_size());
    std::println("done — {} word vocab", Vr);

    std::vector<std::int32_t> ids;
    for (const auto& p : paras) { auto v = tok.encode(p); ids.insert(ids.end(), v.begin(), v.end()); }
    if (static_cast<std::int64_t>(ids.size()) < N + B + 8) throw std::runtime_error("corpus too short");

    dv::GenerationTrace trace;
    dn::SamplerConfig cfg;
    cfg.temperature = temp;
    cfg.commit_order = dn::CommitOrder::Spread;
    auto decode = [&tok, Vr](std::int32_t id) {
        return (id >= 0 && id < Vr) ? std::string(tok.token_str(static_cast<BPETokenizer::TokenId>(id)))
                                    : std::string("?");
    };

    std::int32_t mask_id = 0;
    std::println("\ntraining {} ({} steps, N={})...", which, steps, N);
    std::mt19937 grng(seed * 131 + 5);
    std::vector<std::int32_t> canvas;
    if (which == "flat") {
        dn::Denoiser m(Vr, D, 8, 4, L, 0, seed);
        train(m, ids, static_cast<int>(steps), B, N, seed);
        m.to(sub0llm::Device::cpu());
        mask_id = m.mask_id();
        trace.levels = {N};                                  // flat = one level
        canvas = dn::make_canvas(m, N);
        dn::refine_canvas(m, canvas, cfg, grng, {}, &trace);
    } else {
        dn::MeraDenoiser m(Vr, D, 8, 4, c, w, N, 0, seed);
        train(m, ids, static_cast<int>(steps), B, N, seed);
        m.to(sub0llm::Device::cpu());
        mask_id = m.mask_id();
        trace.levels = m.level_lens(N);                      // [N, N/c, …, top]
        canvas = dn::make_canvas(m, N);
        dn::refine_canvas(m, canvas, cfg, grng, {}, &trace);
    }

    // fill meta + write
    trace.T = N; trace.model = which; trace.N = N; trace.c = (which == "mera") ? c : 0;
    trace.w = (which == "mera") ? w : 0; trace.commit_order = "spread";
    trace.temperature = cfg.temperature; trace.conf_threshold = cfg.conf_threshold;
    trace.min_commit_frac = cfg.min_commit_frac; trace.remask_threshold = cfg.remask_threshold;
    const std::size_t nf = dv::write_trace_json(trace, out, decode, mask_id);

    // ── self-check ────────────────────────────────────────────────────────────────────────────
    bool ok = nf > 0;
    std::size_t still_masked = 0;
    for (auto t : canvas) if (t == mask_id) ++still_masked;
    std::string text;
    for (auto t : canvas) { text += decode(t); text += ' '; }

    // timing / settled tok/s from the trace
    double total_ms = 0.0; std::int64_t total_committed = 0;
    for (const auto& f : trace.frames) { total_ms += f.ms; total_committed += f.committed_count; }
    const double settled_tps = total_ms > 0 ? total_committed / (total_ms / 1000.0) : 0.0;

    std::println("\n── self-check ──");
    std::println("  frames written : {}", nf);
    std::println("  final masked   : {} / {}", still_masked, N);
    std::println("  levels (pyramid): {}", [&]{ std::string s; for (auto l : trace.levels) s += std::format("{} ", l); return s; }());
    std::println("  settle time    : {:.1f} ms over {} iters", total_ms, trace.frames.size());
    std::println("  settled tok/s  : {:.0f}  ({} tokens settled)", settled_tps, total_committed);
    std::println("  trace file     : {}", out);
    std::println("  final text     : {}", text.substr(0, 240));
    if (!ok) { std::println("  STATUS: FAIL (no frames)"); return 2; }
    std::println("  STATUS: OK");
    std::println("\n── how to view ──");
    std::println("  1) serve the repo root:   python -m http.server 8000");
    std::println("     (or any static server; needed so the page can fetch ./{} and ./ACRONYMS.json)", out);
    std::println("  2) open:                  http://localhost:8000/tools/viz/");
    std::println("     (or open tools/viz/index.html directly and use the 'Load trace' button)");
    return 0;
}
