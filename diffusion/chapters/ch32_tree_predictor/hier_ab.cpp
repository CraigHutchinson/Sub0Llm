// hier_ab.cpp (Ch32 P2 step 2e) — coarse-to-fine vs flat: the efficiency/context benchmark.
//
// Trains a FLAT Denoiser and a HierDenoiser (coarse global pass + fine window-local pass) on the SAME
// word corpus at sequence length N ≫ w, then reports (a) held-out masked NLL (accuracy gap) and (b)
// wall-clock per step + the analytic attention-op ratio (compute win). The gist-as-coarsening design
// passes iff the compute/memory win is large while the NLL gap is small (DESIGN_REVIEW_3 §6).
// Build: cmake --build build-cuda --target ch32_hier_ab.

#include "sub0diff/eval/oov_cliff.hpp"
#include "sub0diff/eval/topic_drift.hpp"
#include "sub0diff/nn/denoiser.hpp"
#include "sub0diff/nn/hier_denoiser.hpp"
#include "sub0diff/train/diffusion_loss.hpp"

#include "sub0llm/core/ops.hpp"
#include "sub0llm/nn/optimizer.hpp"
#include "sub0llm/tokenizer/bpe.hpp"

#include <array>
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
    for (const auto& t : texts) {
        auto v = tok.encode(t);
        ids.insert(ids.end(), v.begin(), v.end());
    }
    return ids;
}

template <class Model>
double train(Model& model, std::span<const std::int32_t> stream, int steps, std::int64_t B,
             std::int64_t N, float lr, std::uint64_t seed, const char* name) {
    model.to(sub0llm::Device::cuda());
    auto params = model.parameters();
    sub0llm::nn::Adam opt(params, lr);
    dt::BatchedDiffusionLossContext ctx(B, N);
    std::mt19937 rng(static_cast<std::uint32_t>(seed));
    const std::size_t n_pos = stream.size() - static_cast<std::size_t>(N) + 1;
    std::uniform_int_distribution<std::size_t> off_dist(0, n_pos - 1);
    const auto t0 = std::chrono::steady_clock::now();
    for (int s = 0; s < steps; ++s) {
        std::vector<std::size_t> offsets(static_cast<std::size_t>(B));
        for (auto& o : offsets) o = off_dist(rng);
        for (auto* p : params)
            if (p->grad().numel() > 0)
                p->grad() = sub0llm::zeros(p->data().shape(), p->data().dtype(), p->data().device());
        auto res = dt::batched_diffusion_loss(model, stream, offsets, rng, ctx, 0.02f, 1.0f);
        res.loss.backward();
        (void)sub0llm::nn::clip_grad_norm(params, 5.0f);
        opt.step();
        if ((s + 1) % 500 == 0)
            std::println("  [{}] step {:>5}  nelbo={:.4f}  ({:.1f}s)", name, s + 1,
                         static_cast<double>(res.loss.data().to(sub0llm::Device::cpu()).item<float>()),
                         std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
    }
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

template <class Model>
std::array<double, 3> nll(const Model& model, std::span<const std::int32_t> eval_ids,
                          std::span<const std::uint8_t> is_content, std::int64_t N,
                          std::size_t windows) {
    std::mt19937 rng(123);
    const auto r = de::evaluate_oov_cliff(model, eval_ids, is_content, N, 0.5f, rng, windows);
    const double overall = (r.n_rare + r.n_common)
        ? (r.sum_ce_rare + r.sum_ce_common) / static_cast<double>(r.n_rare + r.n_common) : 0.0;
    return {r.nll_rare(), r.nll_common(), overall};   // rare bucket == content
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
    const std::int64_t L      = arg_i(argc, argv, "--n_layers", 4);   // flat depth; hier splits L/2+L/2
    const std::int64_t B      = arg_i(argc, argv, "--batch", 8);
    const std::int64_t stop_k = arg_i(argc, argv, "--stop_k", 100);
    const std::size_t windows = static_cast<std::size_t>(arg_i(argc, argv, "--eval_windows", 2000));

    std::println("== Ch32 P2 2e — coarse-to-fine vs flat (efficiency/context benchmark) ==");
    std::println("N={} window={} (M={} fine windows) coarsen={} (Nc={} coarse slots)", N, w, N / w, c, N / c);
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
    const auto eval_ids  = encode_all(tok, eval_p);
    const auto is_content = de::content_type_mask(train_ids, Vr, stop_k);
    std::println("train {} tok, eval {} tok; D={} L={} steps={} batch={}",
                 train_ids.size(), eval_ids.size(), D, L, steps, B);
    if (static_cast<std::int64_t>(train_ids.size()) < N + 1)
        throw std::runtime_error("corpus too short for this seq_len");

    std::println("\n-- training FLAT Denoiser (N={}) --", N);
    dn::Denoiser flat(Vr, D, 8, 4, L, 0, /*seed=*/7);
    const double t_flat = train(flat, train_ids, static_cast<int>(steps), B, N, 1e-3f, 7, "flat");
    const auto nf = nll(flat, eval_ids, is_content, N, windows);

    std::println("\n-- training HIER Denoiser, UNIFORM pool (coarse {}L over Nc={} + fine {}L over w={}) --",
                 L / 2, N / c, L - L / 2, w);
    dn::HierDenoiser hier(Vr, D, 8, 4, L / 2, L - L / 2, c, w, 0, /*seed=*/7, /*mask_aware=*/false);
    const double t_hier = train(hier, train_ids, static_cast<int>(steps), B, N, 1e-3f, 7, "hier");
    const auto nh = nll(hier, eval_ids, is_content, N, windows);

    std::println("\n-- training HIER Denoiser, MASK-AWARE pool (2c) --");
    dn::HierDenoiser hierm(Vr, D, 8, 4, L / 2, L - L / 2, c, w, 0, /*seed=*/7, /*mask_aware=*/true);
    const double t_hierm = train(hierm, train_ids, static_cast<int>(steps), B, N, 1e-3f, 7, "hierM");
    const auto nm = nll(hierm, eval_ids, is_content, N, windows);

    // analytic attention-op ratio (per sequence, ignoring D and heads): flat L·N² vs coarse+fine.
    const double flat_ops = static_cast<double>(L) * static_cast<double>(N) * static_cast<double>(N);
    const double hier_ops = static_cast<double>(L / 2) * static_cast<double>(N / c) * static_cast<double>(N / c)
                          + static_cast<double>(L - L / 2) * static_cast<double>(N) * static_cast<double>(w);

    auto gap = [](double h, double f) { return f > 0 ? (h - f) / f * 100.0 : 0.0; };
    std::println("\n============== coarse-to-fine vs flat (N={}) ==============", N);
    std::println("  FLAT       NLL content {:.3f} function {:.3f} overall {:.3f}   train {:.1f}s", nf[0], nf[1], nf[2], t_flat);
    std::println("  HIER-unif  NLL content {:.3f} function {:.3f} overall {:.3f}   train {:.1f}s", nh[0], nh[1], nh[2], t_hier);
    std::println("  HIER-mask  NLL content {:.3f} function {:.3f} overall {:.3f}   train {:.1f}s", nm[0], nm[1], nm[2], t_hierm);
    std::println("\n  ACCURACY gap vs flat (overall NLL):  uniform {:+.1f}%  ->  mask-aware {:+.1f}%   (content {:+.1f}% -> {:+.1f}%)",
                 gap(nh[2], nf[2]), gap(nm[2], nf[2]), gap(nh[0], nf[0]), gap(nm[0], nf[0]));
    std::println("  mask-aware pooling closes the gap by: {:.0f}% of it (overall)",
                 gap(nh[2], nf[2]) != 0 ? (gap(nh[2], nf[2]) - gap(nm[2], nf[2])) / gap(nh[2], nf[2]) * 100.0 : 0.0);
    std::println("  COMPUTE: attention-ops flat {:.0f} vs hier {:.0f}  ({:.1f}x fewer);  wall-time {:.2f}x of flat",
                 flat_ops, hier_ops, hier_ops > 0 ? flat_ops / hier_ops : 0.0,
                 t_flat > 0 ? t_hier / t_flat : 0.0);
    std::println("  attention memory/window: flat N²={} vs hier w²={}  ({:.0f}x smaller working set)",
                 N * N, w * w, w > 0 ? static_cast<double>(N * N) / static_cast<double>(w * w) : 0.0);
    return 0;
}
