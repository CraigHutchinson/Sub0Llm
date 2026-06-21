// gist_ab.cpp (Ch32 P2 step 2b) — the gist-conditioning kill-test.
//
// Trains a flat BASELINE Denoiser and a GistDenoiser (content-word gist broadcast via a learned
// projection, init 0 = no-regression) on the SAME word-level corpus + budget, then measures held-out
// masked-token NLL bucketed by CONTENT vs function words. The gist passes the kill-test iff it LOWERS
// content-word NLL (and does not regress overall) — i.e. the gist carries predictive info the worker
// uses (the feudal signal, BUILD_PLAN §P2). Build: cmake --build build-cuda --target ch32_gist_ab.

#include "sub0diff/eval/oov_cliff.hpp"
#include "sub0diff/eval/topic_drift.hpp"
#include "sub0diff/nn/denoiser.hpp"
#include "sub0diff/nn/gist_denoiser.hpp"
#include "sub0diff/train/diffusion_loss.hpp"

#include "sub0llm/core/ops.hpp"
#include "sub0llm/nn/optimizer.hpp"
#include "sub0llm/tokenizer/bpe.hpp"

#include <array>
#include <chrono>
#include <cmath>
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
void train(Model& model, std::span<const std::int32_t> stream, int steps, std::int64_t B,
           std::int64_t T, float lr, std::uint64_t seed, const char* name) {
    model.to(sub0llm::Device::cuda());
    auto params = model.parameters();
    sub0llm::nn::Adam opt(params, lr);
    dt::BatchedDiffusionLossContext ctx(B, T);
    std::mt19937 rng(static_cast<std::uint32_t>(seed));
    const std::size_t n_pos = stream.size() - static_cast<std::size_t>(T) + 1;
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
}

// Held-out masked NLL bucketed by content/function (is_content as the bucket mask). Returns
// {nll_content, nll_function, nll_overall}.
template <class Model>
std::array<double, 3> nll(const Model& model, std::span<const std::int32_t> eval_ids,
                          std::span<const std::uint8_t> is_content, std::int64_t T,
                          std::size_t windows) {
    std::mt19937 rng(123);
    const auto r = de::evaluate_oov_cliff(model, eval_ids, is_content, T, 0.5f, rng, windows);
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
    const std::int64_t L      = arg_i(argc, argv, "--n_layers", 4);
    const std::int64_t T      = arg_i(argc, argv, "--seq_len", 64);
    const std::int64_t B      = arg_i(argc, argv, "--batch", 16);
    const std::int64_t stop_k = arg_i(argc, argv, "--stop_k", 100);
    const std::size_t windows = static_cast<std::size_t>(arg_i(argc, argv, "--eval_windows", 4000));
    const std::uint64_t seed  = static_cast<std::uint64_t>(arg_i(argc, argv, "--seed", 7));

    std::println("== Ch32 P2 2b — gist-conditioning A/B (flat Denoiser vs GistDenoiser) ==");
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
    std::uint64_t n_content = 0;
    for (auto c : is_content) n_content += c;
    std::println("train {} tok, eval {} tok; content types {}/{} (stop_k={}); D={} L={} T={} steps={}",
                 train_ids.size(), eval_ids.size(), n_content, Vr, stop_k, D, L, T, steps);

    std::println("\n-- training BASELINE Denoiser --");
    dn::Denoiser base(Vr, D, 8, 4, L, 0, /*seed=*/seed);
    train(base, train_ids, static_cast<int>(steps), B, T, 1e-3f, seed, "base");
    const auto nb = nll(base, eval_ids, is_content, T, windows);

    std::println("\n-- training GistDenoiser (real gist) --");
    dn::GistDenoiser gist(Vr, D, 8, 4, L, 0, is_content, /*seed=*/seed);
    train(gist, train_ids, static_cast<int>(steps), B, T, 1e-3f, seed, "gist");
    const auto ng = nll(gist, eval_ids, is_content, T, windows);

    std::println("\n-- training GistDenoiser (SHUFFLED-gist control: same params, signal destroyed) --");
    dn::GistDenoiser shuf(Vr, D, 8, 4, L, 0, is_content, /*seed=*/seed, /*shuffle_gist=*/true);
    train(shuf, train_ids, static_cast<int>(steps), B, T, 1e-3f, seed, "shuf");
    const auto ns = nll(shuf, eval_ids, is_content, T, windows);

    std::println("\n============== gist A/B (held-out masked NLL, nats) ==============");
    std::println("  BASELINE      content {:.3f}  function {:.3f}  overall {:.3f}", nb[0], nb[1], nb[2]);
    std::println("  GIST (real)   content {:.3f}  function {:.3f}  overall {:.3f}", ng[0], ng[1], ng[2]);
    std::println("  GIST-shuffled content {:.3f}  function {:.3f}  overall {:.3f}  (capacity control)",
                 ns[0], ns[1], ns[2]);
    const double dc  = nb[0] > 0 ? (ng[0] - nb[0]) / nb[0] * 100.0 : 0.0;
    const double doa = nb[2] > 0 ? (ng[2] - nb[2]) / nb[2] * 100.0 : 0.0;
    const double sig = ns[2] > 0 ? (ng[2] - ns[2]) / ns[2] * 100.0 : 0.0;   // real vs shuffled
    std::println("\n  GIST vs baseline:  content {:+.1f}%, overall {:+.1f}%", dc, doa);
    std::println("  GIST vs shuffled (the SIGNAL, capacity-matched):  overall {:+.1f}%", sig);
    const bool real_signal = ng[2] < ns[2] * 0.99;   // real beats shuffled by >1%
    std::println("  VERDICT: {}",
                 (ng[0] < nb[0] && ng[2] <= nb[2] * 1.01 && real_signal)
                     ? "PASS — gist lowers NLL AND beats the capacity-matched shuffled control"
                     : (ng[2] < nb[2] && !real_signal)
                           ? "CONFOUNDED — gist helps but ties shuffled control (capacity, not signal)"
                           : "see numbers — gist not helping");
    return 0;
}
