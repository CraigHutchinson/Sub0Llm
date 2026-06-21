// hier_gen.cpp (Ch32 P2 2e/2c) — generate from the coarse-to-fine HierDenoiser and score M2.
//
// The 2e/2c benchmarks scored NLL (per-token prediction). The ACTUAL P2 goal is topic coherence in
// GENERATION (M2 content-recurrence; M2_RESULTS). This runner trains a flat Denoiser and a mask-aware
// HierDenoiser at length N, GENERATES passages from each with the SAME templated sampler, and measures
// M2 (content recurrence + distinct-n) on flat-gen vs hier-gen vs the corpus and a unigram-chance
// floor. Question: does the coarse plan make generations reuse entities more (closer to the corpus)?
// Build: cmake --build build-cuda --target ch32_hier_gen.

#include "sub0diff/eval/topic_drift.hpp"
#include "sub0diff/nn/denoiser.hpp"
#include "sub0diff/nn/hier_denoiser.hpp"
#include "sub0diff/nn/sampler.hpp"
#include "sub0diff/train/diffusion_loss.hpp"

#include "sub0llm/core/ops.hpp"
#include "sub0llm/nn/optimizer.hpp"
#include "sub0llm/tokenizer/bpe.hpp"

#include <algorithm>
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
namespace de = sub0diff::eval;

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

std::vector<std::int32_t> encode_all(const BPETokenizer& tok, std::span<const std::string> texts) {
    std::vector<std::int32_t> ids;
    for (const auto& t : texts) { auto v = tok.encode(t); ids.insert(ids.end(), v.begin(), v.end()); }
    return ids;
}

template <class Model>
void train(Model& model, std::span<const std::int32_t> stream, int steps, std::int64_t B,
           std::int64_t N, float lr, std::uint64_t seed, const char* name) {
    model.to(sub0llm::Device::cuda());
    auto params = model.parameters();
    sub0llm::nn::Adam opt(params, lr);
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
        if ((s + 1) % 500 == 0)
            std::println("  [{}] step {:>5}  nelbo={:.4f}  ({:.1f}s)", name, s + 1,
                         static_cast<double>(res.loss.data().to(sub0llm::Device::cpu()).item<float>()),
                         std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
    }
}

template <class Model>
std::vector<std::vector<std::int32_t>> generate(Model& model, std::int64_t N, std::int64_t K,
                                                float temp, std::uint64_t seed) {
    model.to(sub0llm::Device::cpu());            // sampler reads host logits
    dn::SamplerConfig cfg;
    cfg.temperature = temp;
    cfg.commit_order = dn::CommitOrder::Spread;
    std::vector<std::vector<std::int32_t>> out;
    std::mt19937 rng(static_cast<std::uint32_t>(seed));
    for (std::int64_t k = 0; k < K; ++k) {
        auto canvas = dn::make_canvas(model, N);
        dn::refine_canvas(model, canvas, cfg, rng);
        out.push_back(std::move(canvas));
    }
    return out;
}

std::vector<std::span<const std::int32_t>> spans(const std::vector<std::vector<std::int32_t>>& v) {
    std::vector<std::span<const std::int32_t>> s;
    for (const auto& p : v) s.emplace_back(p);
    return s;
}

void m2row(const char* tag, const de::TopicDriftResult& r) {
    std::println("  {:<16} recurrence {:.3f}   distinct-2/3/4 {:.3f}/{:.3f}/{:.3f}",
                 tag, r.content_recurrence, r.distinct2, r.distinct3, r.distinct4);
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
    const std::string corpus = arg_s(argc, argv, "--corpus", "data/tinystories_clean.txt");
    const std::int64_t plimit = arg_i(argc, argv, "--paragraphs", 600);
    const std::int64_t steps  = arg_i(argc, argv, "--steps", 2000);
    const std::int64_t D      = arg_i(argc, argv, "--embed_dim", 256);
    const std::int64_t N      = arg_i(argc, argv, "--seq_len", 256);
    const std::int64_t w      = arg_i(argc, argv, "--window", 64);
    const std::int64_t c      = arg_i(argc, argv, "--coarsen", 4);
    const std::int64_t L      = arg_i(argc, argv, "--n_layers", 4);
    const std::int64_t Bsz    = arg_i(argc, argv, "--batch", 8);
    const std::int64_t K      = arg_i(argc, argv, "--gen", 64);
    const std::int64_t stop_k = arg_i(argc, argv, "--stop_k", 100);
    const float temp          = static_cast<float>(arg_i(argc, argv, "--temp_x100", 90)) / 100.0f;
    const std::int64_t win    = arg_i(argc, argv, "--m2_window", 16);
    const std::uint64_t seed  = static_cast<std::uint64_t>(arg_i(argc, argv, "--seed", 7));

    std::println("== Ch32 P2 — HierDenoiser generation + M2 (flat vs coarse-to-fine) ==");
    std::println("N={} window={} coarsen={} gen={} temp={:.2f}", N, w, c, K, temp);
    auto paras = read_paragraphs(corpus, plimit);
    if (paras.size() < 20) throw std::runtime_error("need >=20 paragraphs");
    const std::size_t n_eval = std::max<std::size_t>(4, paras.size() / 10);
    std::vector<std::string> train_p(paras.begin(), paras.end() - n_eval);
    std::vector<std::string> eval_p(paras.end() - n_eval, paras.end());

    std::print("building word tokenizer over {} paragraphs... ", train_p.size());
    BPETokenizer tok = BPETokenizer::word_level(train_p);
    const std::int64_t Vr = static_cast<std::int64_t>(tok.vocab_size());
    std::println("done — {} word vocab", Vr);

    const auto train_ids = encode_all(tok, train_p);
    const auto is_content = de::content_type_mask(train_ids, Vr, stop_k);
    if (static_cast<std::int64_t>(train_ids.size()) < N + 1) throw std::runtime_error("corpus too short");

    std::println("\n-- training FLAT Denoiser --");
    dn::Denoiser flat(Vr, D, 8, 4, L, 0, /*seed=*/seed);
    train(flat, train_ids, static_cast<int>(steps), Bsz, N, 1e-3f, seed, "flat");

    std::println("\n-- training HIER Denoiser (mask-aware) --");
    dn::HierDenoiser hier(Vr, D, 8, 4, L / 2, L - L / 2, c, w, 0, /*seed=*/seed, /*mask_aware=*/true);
    train(hier, train_ids, static_cast<int>(steps), Bsz, N, 1e-3f, seed, "hier");

    std::println("\ngenerating {} passages from each (N={}, temp {:.2f}, spread)...", K, N, temp);
    auto flat_gen = generate(flat, N, K, temp, seed * 131 + 17);
    auto hier_gen = generate(hier, N, K, temp, seed * 131 + 17);

    // corpus passages (per held-out paragraph) + unigram-chance floor
    std::vector<std::vector<std::int32_t>> corpus_p;
    for (const auto& p : eval_p) { auto v = tok.encode(p); if (static_cast<std::int64_t>(v.size()) >= 2 * win) corpus_p.push_back(std::move(v)); }
    std::vector<double> freq(static_cast<std::size_t>(Vr), 0.0);
    for (auto id : train_ids) if (id >= 0 && id < Vr) freq[static_cast<std::size_t>(id)] += 1.0;
    std::mt19937 urng(99);
    std::discrete_distribution<std::int32_t> uni(freq.begin(), freq.end());
    std::vector<std::vector<std::int32_t>> unigram;
    for (const auto& p : corpus_p) { std::vector<std::int32_t> q(p.size()); for (auto& x : q) x = uni(urng); unigram.push_back(std::move(q)); }

    const auto rc = de::evaluate_topic_drift(spans(corpus_p), is_content, win, 1, 2);
    const auto ru = de::evaluate_topic_drift(spans(unigram),  is_content, win, 1, 2);
    const auto rf = de::evaluate_topic_drift(spans(flat_gen), is_content, win, 1, 2);
    const auto rh = de::evaluate_topic_drift(spans(hier_gen), is_content, win, 1, 2);

    std::println("\n============= M2 on generations (window {}) =============", win);
    m2row("corpus",         rc);
    m2row("unigram-floor",  ru);
    m2row("flat-gen",       rf);
    m2row("hier-gen",       rh);
    const double span = rc.content_recurrence - ru.content_recurrence;   // corpus - chance
    auto frac = [&](double v) { return span > 0 ? (v - ru.content_recurrence) / span * 100.0 : 0.0; };
    std::println("\n  content recurrence — corpus {:.3f}, chance floor {:.3f} (gap {:.3f})",
                 rc.content_recurrence, ru.content_recurrence, span);
    std::println("  flat-gen {:.3f} ({:.0f}% of the gap closed);  hier-gen {:.3f} ({:.0f}% of the gap closed)",
                 rf.content_recurrence, frac(rf.content_recurrence), rh.content_recurrence, frac(rh.content_recurrence));
    std::println("  VERDICT: {}",
                 rh.content_recurrence > rf.content_recurrence + 0.005
                     ? "coarse-to-fine generations reuse entities MORE than flat (toward corpus)"
                     : "no M2 recurrence gain from the hierarchy — see numbers");
    return 0;
}
