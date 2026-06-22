// hier_ceiling.cpp (Ch32 P2 2e) — the CONTEXT-LENGTH ceiling: where flat OOMs, hier keeps running.
//
// The headline value of the coarse-to-fine hierarchy is compute/context (2E_RESULTS): flat attention
// is O(N²) memory, hier is O(w²) per window (flat in N). This runner times ONE model at ONE sequence
// length and reports s/step — or "OOM" if the device runs out. Each (model, N) is a SEPARATE process
// (a shell loop drives the sweep) so an OOM in one configuration cannot corrupt the CUDA context of
// the next: the cleanest way to demonstrate "flat dies at N, hier survives".
// Build: cmake --build build-cuda --target ch32_hier_ceiling.

#include "sub0diff/nn/denoiser.hpp"
#include "sub0diff/nn/hier_denoiser.hpp"
#include "sub0diff/nn/mera_denoiser.hpp"
#include "sub0diff/train/diffusion_loss.hpp"

#include "sub0llm/core/ops.hpp"
#include "sub0llm/nn/optimizer.hpp"
#include "sub0llm/tokenizer/bpe.hpp"

#include <chrono>
#include <cstdint>
#include <exception>
#include <fstream>
#include <format>
#include <print>
#include <random>
#include <span>
#include <string>
#include <vector>

using sub0llm::BPETokenizer;
namespace dn = sub0diff::nn;
namespace dt = sub0diff::train;

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

// Time `steps` train steps (fwd+bwd+opt) at sequence length N; returns mean s/step. Throws on OOM.
template <class Model>
double time_steps(Model& model, std::span<const std::int32_t> stream, std::int64_t B, std::int64_t N,
                  int warmup, int steps) {
    model.to(sub0llm::Device::cuda());
    auto params = model.parameters();
    sub0llm::nn::Adam opt(params, 1e-3f);
    dt::BatchedDiffusionLossContext ctx(B, N);
    std::mt19937 rng(1);
    std::uniform_int_distribution<std::size_t> off(0, stream.size() - static_cast<std::size_t>(N));
    auto one = [&] {
        std::vector<std::size_t> offs(static_cast<std::size_t>(B));
        for (auto& o : offs) o = off(rng);
        for (auto* p : params)
            if (p->grad().numel() > 0)
                p->grad() = sub0llm::zeros(p->data().shape(), p->data().dtype(), p->data().device());
        auto res = dt::batched_diffusion_loss(model, stream, offs, rng, ctx, 0.02f, 1.0f);
        res.loss.backward();
        opt.step();
        (void)res.loss.data().to(sub0llm::Device::cpu());   // force completion
    };
    for (int i = 0; i < warmup; ++i) one();
    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < steps; ++i) one();
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() / steps;
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
    const std::string which  = arg_s(argc, argv, "--model", "flat");   // flat | hier
    const std::int64_t N     = arg_i(argc, argv, "--seq_len", 512);
    const std::int64_t D     = arg_i(argc, argv, "--embed_dim", 128);
    const std::int64_t L     = arg_i(argc, argv, "--n_layers", 4);
    const std::int64_t w     = arg_i(argc, argv, "--window", 64);
    const std::int64_t c     = arg_i(argc, argv, "--coarsen", 8);
    const std::int64_t B     = arg_i(argc, argv, "--batch", 4);
    const std::int64_t plim  = arg_i(argc, argv, "--paragraphs", 4000);

    auto paras = read_paragraphs(corpus, plim);
    BPETokenizer tok = BPETokenizer::word_level(paras);
    const std::int64_t Vr = static_cast<std::int64_t>(tok.vocab_size());
    std::vector<std::int32_t> ids;
    for (const auto& p : paras) { auto v = tok.encode(p); ids.insert(ids.end(), v.begin(), v.end()); }
    // tile the stream until it comfortably exceeds N (so windows fit at large N)
    while (static_cast<std::int64_t>(ids.size()) < N + static_cast<std::int64_t>(B) + 8) {
        const std::size_t half = ids.size();
        ids.insert(ids.end(), ids.begin(), ids.begin() + static_cast<std::ptrdiff_t>(half));
    }

    try {
        double sps = 0.0;
        if (which == "mera") {
            dn::MeraDenoiser m(Vr, D, 8, 4, c, w, N, 0, /*seed=*/7);
            sps = time_steps(m, ids, B, N, /*warmup=*/3, /*steps=*/15);
        } else if (which == "hier") {
            dn::HierDenoiser m(Vr, D, 8, 4, L / 2, L - L / 2, c, w, 0, /*seed=*/7, /*mask_aware=*/true);
            sps = time_steps(m, ids, B, N, /*warmup=*/3, /*steps=*/15);
        } else {
            dn::Denoiser m(Vr, D, 8, 4, L, 0, /*seed=*/7);
            sps = time_steps(m, ids, B, N, /*warmup=*/3, /*steps=*/15);
        }
        const double toks_per_s = static_cast<double>(B * N) / sps;
        std::println("{:<5} N={:>6}  B={}  {:.4f} s/step  {:>9.0f} tok/s   OK",
                     which, N, B, sps, toks_per_s);
    } catch (const std::exception& e) {
        std::println("{:<5} N={:>6}  B={}  OOM/FAIL  ({})", which, N, B, e.what());
        return 2;
    }
    return 0;
}
