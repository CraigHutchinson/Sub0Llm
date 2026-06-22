// mera_mixedn.cpp (Ch32 P3) — MIXED-N training: make variable-N MERA robust across lengths.
//
// variable-N MERA (mera_denoiser.hpp) ACCEPTS any N ≤ max_seq_len, but a model TRAINED at one fixed N
// has a mild train/use mismatch (the top block only ever saw one top-length). Mixed-N training samples
// a different valid N each step, so every level — including the top — is trained across the length
// range. This runner trains a MERA on a FIXED length and an identical MERA on a MIX of lengths, then
// evaluates BOTH at every length: mixed should be consistent everywhere; fixed should degrade off its
// training length. Build: cmake --build build-cuda --target ch32_mera_mixedn.

#include "sub0diff/eval/oov_cliff.hpp"
#include "sub0diff/eval/topic_drift.hpp"
#include "sub0diff/nn/mera_denoiser.hpp"
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
    for (const auto& t : texts) { auto v = tok.encode(t); ids.insert(ids.end(), v.begin(), v.end()); }
    return ids;
}

// Train sampling a length from `Ns` each step (one element ⇒ fixed-N). One reused loss-context per N.
template <class Model>
void train(Model& model, std::span<const std::int32_t> stream, int steps, std::int64_t B,
           std::span<const std::int64_t> Ns, float lr, std::uint64_t seed, const char* name) {
    model.to(sub0llm::Device::cuda());
    auto params = model.parameters();
    sub0llm::nn::Adam opt(params, lr);
    std::vector<dt::BatchedDiffusionLossContext> ctxs;
    for (auto n : Ns) ctxs.emplace_back(B, n);
    std::mt19937 rng(static_cast<std::uint32_t>(seed));
    std::uniform_int_distribution<std::size_t> pick(0, Ns.size() - 1);
    const auto t0 = std::chrono::steady_clock::now();
    for (int s = 0; s < steps; ++s) {
        const std::size_t i = pick(rng);
        const std::int64_t N = Ns[i];
        std::uniform_int_distribution<std::size_t> off(0, stream.size() - static_cast<std::size_t>(N));
        std::vector<std::size_t> offs(static_cast<std::size_t>(B));
        for (auto& o : offs) o = off(rng);
        for (auto* p : params)
            if (p->grad().numel() > 0)
                p->grad() = sub0llm::zeros(p->data().shape(), p->data().dtype(), p->data().device());
        auto res = dt::batched_diffusion_loss(model, stream, offs, rng, ctxs[i], 0.02f, 1.0f);
        res.loss.backward();
        (void)sub0llm::nn::clip_grad_norm(params, 5.0f);
        opt.step();
        if ((s + 1) % 1000 == 0)
            std::println("  [{}] step {:>5}  N={:>4}  nelbo={:.4f}  ({:.1f}s)", name, s + 1, N,
                         static_cast<double>(res.loss.data().to(sub0llm::Device::cpu()).item<float>()),
                         std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
    }
}

template <class Model>
double nll(const Model& model, std::span<const std::int32_t> eval_ids,
           std::span<const std::uint8_t> is_content, std::int64_t N, std::size_t windows) {
    std::mt19937 rng(123);
    const auto r = de::evaluate_oov_cliff(model, eval_ids, is_content, N, 0.5f, rng, windows);
    return (r.n_rare + r.n_common) ? (r.sum_ce_rare + r.sum_ce_common) / static_cast<double>(r.n_rare + r.n_common) : 0.0;
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
    const std::int64_t steps  = arg_i(argc, argv, "--steps", 3000);
    const std::int64_t D      = arg_i(argc, argv, "--embed_dim", 256);
    const std::int64_t w      = arg_i(argc, argv, "--window", 64);
    const std::int64_t c      = arg_i(argc, argv, "--coarsen", 4);
    const std::int64_t B      = arg_i(argc, argv, "--batch", 8);
    const std::int64_t stop_k = arg_i(argc, argv, "--stop_k", 100);
    const std::size_t windows = static_cast<std::size_t>(arg_i(argc, argv, "--eval_windows", 1500));
    const std::uint64_t seed  = static_cast<std::uint64_t>(arg_i(argc, argv, "--seed", 7));
    const std::vector<std::int64_t> Ns{128, 256, 512};       // the length range; max = 512
    const std::int64_t maxN = Ns.back();

    std::println("== Ch32 P3 — mixed-N training (variable-N MERA robustness) ==");
    std::println("lengths {{128,256,512}}, c={} w={} D={} steps={}", c, w, D, steps);
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
    if (static_cast<std::int64_t>(train_ids.size()) < maxN + B + 8)
        throw std::runtime_error("corpus too short for max length");

    // identical architecture (max_seq_len=512); only the training length DISTRIBUTION differs.
    std::array<std::int64_t, 1> fixedNs{maxN};
    std::println("\n-- training MERA-FIXED (N={} only) --", maxN);
    dn::MeraDenoiser mfix(Vr, D, 8, 4, c, w, maxN, 0, /*seed=*/seed);
    train(mfix, train_ids, static_cast<int>(steps), B, std::span<const std::int64_t>(fixedNs), 1e-3f, seed, "fixed");

    std::println("\n-- training MERA-MIXED (N in {{128,256,512}}) --");
    dn::MeraDenoiser mmix(Vr, D, 8, 4, c, w, maxN, 0, /*seed=*/seed);
    train(mmix, train_ids, static_cast<int>(steps), B, std::span<const std::int64_t>(Ns), 1e-3f, seed, "mixed");

    std::println("\n========= held-out masked NLL by eval length =========");
    std::println("  {:>6}   fixed-N   mixed-N", "N");
    for (auto N : Ns) {
        const double nf = nll(mfix, eval_ids, is_content, N, windows);
        const double nm = nll(mmix, eval_ids, is_content, N, windows);
        std::println("  {:>6}   {:.4f}    {:.4f}   ({:+.1f}% mixed vs fixed)",
                     N, nf, nm, nf > 0 ? (nm - nf) / nf * 100.0 : 0.0);
    }
    std::println("\n  expect: fixed-N best at N={} but worse at shorter N (top-level train/use mismatch);", maxN);
    std::println("  mixed-N consistent across all lengths (every level trained over the range).");
    return 0;
}
