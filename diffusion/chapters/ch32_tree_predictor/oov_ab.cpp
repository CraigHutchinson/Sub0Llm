// oov_ab.cpp (Ch32 P1 1c) — the OOV A/B kill-test.
//
// Trains a BASELINE word Denoiser and a CodecDenoiser (char-composed embedding) on the SAME
// word-level corpus and budget, then measures the M1 OOV-cliff (NLL_rare / NLL_common) on both.
// The codec passes the kill-test iff its cliff falls toward ~1 while its in-vocab (common-bucket)
// NLL does not regress vs the baseline. See ch32 M1_RESULTS.md / P1_RESULTS.md.
//
// Build: cmake --build build-cuda --target ch32_oov_ab ; run with --device cuda.

#include "sub0diff/eval/oov_cliff.hpp"
#include "sub0diff/nn/char_codec.hpp"
#include "sub0diff/nn/codec_denoiser.hpp"
#include "sub0diff/nn/denoiser.hpp"
#include "sub0diff/train/diffusion_loss.hpp"

#include "sub0llm/core/ops.hpp"
#include "sub0llm/nn/optimizer.hpp"
#include "sub0llm/tokenizer/bpe.hpp"

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

// (Vm·max_len,) int32 byte-level spellings: word id → its UTF-8 bytes (0-255), padded with `pad`.
sub0llm::Tensor build_word_chars(const BPETokenizer& tok, std::int64_t Vm, std::int64_t max_len,
                                 std::int32_t pad) {
    sub0llm::Tensor wc({Vm * max_len}, sub0llm::DType::Int32);
    auto d = wc.data_as<std::int32_t>();
    for (std::int64_t id = 0; id < Vm; ++id) {
        const std::string_view s = (id < static_cast<std::int64_t>(tok.vocab_size()))
                                       ? tok.token_str(static_cast<BPETokenizer::TokenId>(id))
                                       : std::string_view{};
        for (std::int64_t j = 0; j < max_len; ++j)
            d[static_cast<std::size_t>(id * max_len + j)] =
                (j < static_cast<std::int64_t>(s.size()))
                    ? static_cast<std::int32_t>(static_cast<unsigned char>(s[static_cast<std::size_t>(j)]))
                    : pad;
    }
    return wc;
}

// Pretrain a CharComposer as a char autoencoder (compose→decode→reconstruct each word's spelling)
// so its composed vectors are meaningful BEFORE the LM objective biases them toward common words
// (1C_RESULTS.md follow-up 3). Trains composer + a throwaway decoder on the GPU; the composer is a
// reference into the CodecDenoiser, so its weights are updated in place.
void pretrain_composer(dn::CharComposer& comp, std::int64_t n_chars, std::int64_t D,
                       const sub0llm::Tensor& word_chars, std::int64_t Vm, std::int64_t max_len,
                       int steps, std::int64_t mb, std::uint64_t seed) {
    dn::CharDecoder dec(n_chars, D, 8, 4, /*depth=*/2, 0, seed, max_len);
    comp.to(sub0llm::Device::cuda());
    dec.to(sub0llm::Device::cuda());
    std::vector<sub0llm::autograd::Variable*> params = comp.parameters();
    auto dp = dec.parameters();
    params.insert(params.end(), dp.begin(), dp.end());
    sub0llm::nn::Adam opt(params, 1e-3f);
    const auto wc = word_chars.data_as<std::int32_t>();
    std::mt19937 rng(static_cast<std::uint32_t>(seed));
    std::uniform_int_distribution<std::int64_t> wid(0, Vm - 1);

    const auto t0 = std::chrono::steady_clock::now();
    for (int s = 0; s < steps; ++s) {
        for (auto* p : params)
            if (p->grad().numel() > 0)
                p->grad() = sub0llm::zeros(p->data().shape(), p->data().dtype(), p->data().device());
        sub0llm::autograd::Variable loss;
        for (std::int64_t j = 0; j < mb; ++j) {
            const std::int64_t id = wid(rng);
            sub0llm::Tensor ci({max_len}, sub0llm::DType::Int32);
            auto cid = ci.data_as<std::int32_t>();
            for (std::int64_t k = 0; k < max_len; ++k)
                cid[static_cast<std::size_t>(k)] = wc[static_cast<std::size_t>(id * max_len + k)];
            auto l = dn::char_recon_loss(comp, dec, ci);
            loss = (j == 0) ? l : sub0llm::autograd::add(loss, l);
        }
        loss = sub0llm::autograd::scale(loss, 1.0f / static_cast<float>(mb));
        loss.backward();
        (void)sub0llm::nn::clip_grad_norm(params, 5.0f);
        opt.step();
        if ((s + 1) % 200 == 0)
            std::println("  [pretrain] step {:>4}  recon_ce={:.4f}  ({:.1f}s)", s + 1,
                         static_cast<double>(loss.data().to(sub0llm::Device::cpu()).item<float>()),
                         std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
    }
}

template <class Model>
void train(Model& model, std::span<const std::int32_t> stream, int steps, std::int64_t B,
           std::int64_t T, float lr, std::uint64_t seed, const char* name,
           std::span<const float> tok_weight = {}) {
    model.to(sub0llm::Device::cuda());
    auto params = model.parameters();
    sub0llm::nn::Adam opt(params, lr);
    dt::BatchedDiffusionLossContext ctx(B, T);
    std::mt19937 rng(static_cast<std::uint32_t>(seed));
    const std::size_t n_pos = stream.size() - static_cast<std::size_t>(T) + 1;
    std::uniform_int_distribution<std::size_t> off_dist(0, n_pos - 1);

    const auto t0 = std::chrono::steady_clock::now();
    float last = 0.0f;
    for (int s = 0; s < steps; ++s) {
        std::vector<std::size_t> offsets(static_cast<std::size_t>(B));
        for (auto& o : offsets) o = off_dist(rng);
        for (auto* p : params)
            if (p->grad().numel() > 0)
                p->grad() = sub0llm::zeros(p->data().shape(), p->data().dtype(), p->data().device());
        auto res = dt::batched_diffusion_loss(model, stream, offsets, rng, ctx, 0.02f, 1.0f,
                                              /*shared_t=*/false, /*exact_count=*/false,
                                              /*is_word_start=*/{}, /*whole_word=*/false,
                                              /*seed_base=*/0, /*index0=*/0, /*contiguous=*/false,
                                              tok_weight);
        res.loss.backward();
        (void)sub0llm::nn::clip_grad_norm(params, 5.0f);
        opt.step();
        last = res.loss.data().to(sub0llm::Device::cpu()).item<float>();
        if ((s + 1) % 500 == 0)
            std::println("  [{}] step {:>5}  nelbo={:.4f}  ({:.1f}s)", name, s + 1,
                         static_cast<double>(last),
                         std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
    }
}

template <class Model>
de::OovCliffResult cliff(const Model& model, std::span<const std::int32_t> eval_ids,
                         std::span<const std::uint8_t> is_rare, std::int64_t T,
                         std::size_t windows) {
    std::mt19937 rng(123);
    return de::evaluate_oov_cliff(model, eval_ids, is_rare, T, 0.5f, rng, windows);
}

std::int64_t arg_i(int argc, char** argv, const std::string& key, std::int64_t def) {
    for (int i = 1; i + 1 < argc; ++i) if (key == argv[i]) return std::stoll(argv[i + 1]);
    return def;
}
std::string arg_s(int argc, char** argv, const std::string& key, const std::string& def) {
    for (int i = 1; i + 1 < argc; ++i) if (key == argv[i]) return argv[i + 1];
    return def;
}

}  // namespace

int main(int argc, char** argv) {
    const std::string corpus = arg_s(argc, argv, "--corpus", "data/tinystories_clean.txt");
    const std::int64_t plimit  = arg_i(argc, argv, "--paragraphs", 400);
    const std::int64_t steps   = arg_i(argc, argv, "--steps", 2000);
    const std::int64_t D       = arg_i(argc, argv, "--embed_dim", 256);
    const std::int64_t L       = arg_i(argc, argv, "--n_layers", 4);
    const std::int64_t T       = arg_i(argc, argv, "--seq_len", 64);
    const std::int64_t maxlen  = arg_i(argc, argv, "--max_len", 12);
    const std::int64_t B       = arg_i(argc, argv, "--batch", 16);
    const std::size_t  windows = static_cast<std::size_t>(arg_i(argc, argv, "--recall_windows", 4000));

    std::println("== Ch32 P1 1c — OOV A/B (baseline Denoiser vs CodecDenoiser) ==");
    auto paras = read_paragraphs(corpus, plimit);
    if (paras.size() < 20) throw std::runtime_error("need >=20 paragraphs");
    const std::size_t n_eval = std::max<std::size_t>(1, paras.size() / 20);   // 5% held out
    std::vector<std::string> train_p(paras.begin(), paras.end() - n_eval);
    std::vector<std::string> eval_p(paras.end() - n_eval, paras.end());

    std::print("building word tokenizer over {} paragraphs... ", train_p.size());
    BPETokenizer tok = BPETokenizer::word_level(train_p);
    const std::int64_t Vr = static_cast<std::int64_t>(tok.vocab_size());
    std::println("done — {} word vocab", Vr);

    const auto train_ids = encode_all(tok, train_p);
    const auto eval_ids  = encode_all(tok, eval_p);
    std::println("train {} tokens, eval {} tokens; D={} L={} T={} steps={} maxlen={}",
                 train_ids.size(), eval_ids.size(), D, L, T, steps, maxlen);

    // model_vocab = Vr + 1 (mask id); byte-level char vocab 0-255 + pad=256 → n_chars 257.
    const sub0llm::Tensor word_chars = build_word_chars(tok, Vr + 1, maxlen, /*pad=*/256);
    const auto is_rare = de::rare_type_mask(train_ids, Vr + 1, 0.5);

    // Rare-aware objective (1C_RESULTS.md follow-up #1). Build a per-token-id weight: rare types get
    // `rw`, common get 1. Default rw = "auto": the common:rare TOKEN-mass ratio, which equalises the
    // two buckets' total gradient mass — the most direct counter to the ~14:1 common dominance that
    // made the naive codec WORSEN the cliff. --rare_weight N overrides with a fixed multiplier.
    std::uint64_t n_common_tok = 0, n_rare_tok = 0;
    for (auto id : train_ids) (is_rare[static_cast<std::size_t>(id)] ? n_rare_tok : n_common_tok)++;
    const double auto_rw = n_rare_tok ? static_cast<double>(n_common_tok) / static_cast<double>(n_rare_tok) : 1.0;
    const std::int64_t rw_arg = arg_i(argc, argv, "--rare_weight", 0);   // 0 = auto
    const float rw = rw_arg > 0 ? static_cast<float>(rw_arg) : static_cast<float>(auto_rw);
    std::vector<float> tok_w(static_cast<std::size_t>(Vr + 1), 1.0f);
    for (std::size_t id = 0; id < tok_w.size(); ++id) if (is_rare[id]) tok_w[id] = rw;
    std::println("rare-aware objective: common {} / rare {} masked-eligible tokens (ratio {:.1f}); "
                 "rare_weight = {:.1f} {}", n_common_tok, n_rare_tok, auto_rw, rw,
                 rw_arg > 0 ? "(fixed)" : "(auto = balance mass)");

    // ── Follow-up (2)+(3): convex-blend (drop lookup for rare) + composer pretraining ──────────────
    // The 2×2 isolated the additive gate as the culprit. This branch tests the fix: a fixed convex
    // blend that REPLACES lookup with the char-composed vector for rare words, with the composer
    // PRETRAINED as a spelling autoencoder first. All arms rare-weighted (the better objective). The
    // question: does codec/convex+pretrain finally beat baseline/rw on the cliff?
    if (arg_i(argc, argv, "--convex", 0) > 0) {
        const int pre_steps = static_cast<int>(arg_i(argc, argv, "--pretrain", 600));
        const std::int64_t pre_mb = arg_i(argc, argv, "--pretrain_mb", 64);

        std::println("\n-- [convex experiment] BASELINE Denoiser, RARE-WEIGHTED (control) --");
        dn::Denoiser base_rw(Vr, D, 8, 4, L, 0, /*seed=*/7);
        train(base_rw, train_ids, static_cast<int>(steps), B, T, 1e-3f, 7, "base/rw", tok_w);
        const auto c_base_rw = cliff(base_rw, eval_ids, is_rare, T, windows);

        std::println("\n-- [convex experiment] CodecDenoiser ADDITIVE, RARE-WEIGHTED (control) --");
        dn::CodecDenoiser codec_add(Vr, D, 8, 4, L, 0, 257, maxlen, word_chars, /*seed=*/7);
        train(codec_add, train_ids, static_cast<int>(steps), B, T, 1e-3f, 7, "codec/add", tok_w);
        const auto c_codec_add = cliff(codec_add, eval_ids, is_rare, T, windows);

        std::println("\n-- [convex experiment] CodecDenoiser CONVEX+PRETRAIN, RARE-WEIGHTED (the test) --");
        dn::CodecDenoiser codec_cv(Vr, D, 8, 4, L, 0, 257, maxlen, word_chars, /*seed=*/7,
                                   dn::CodecDenoiser::Blend::ConvexFixed, is_rare);
        std::println("  pretraining composer ({} steps, mb {})...", pre_steps, pre_mb);
        pretrain_composer(codec_cv.composer(), 257, D, word_chars, Vr + 1, maxlen,
                          pre_steps, pre_mb, /*seed=*/9000);
        train(codec_cv, train_ids, static_cast<int>(steps), B, T, 1e-3f, 7, "codec/cv", tok_w);
        const auto c_codec_cv = cliff(codec_cv, eval_ids, is_rare, T, windows);

        auto row = [](const char* tag, const de::OovCliffResult& r) {
            std::println("  {:<22} NLL_common {:.3f}  NLL_rare {:.3f}  CLIFF {:.2f}x", tag,
                         r.nll_common(), r.nll_rare(), r.ratio());
        };
        std::println("\n===== OOV convex-blend + pretrain (rarest 50%; rare_weight {:.1f}) =====", rw);
        row("baseline/rw",            c_base_rw);
        row("codec/additive/rw",      c_codec_add);
        row("codec/convex+pre/rw",    c_codec_cv);
        const double cv_vs_base = c_base_rw.ratio() > 0
            ? (c_base_rw.ratio() - c_codec_cv.ratio()) / c_base_rw.ratio() * 100.0 : 0.0;
        const double cv_vs_add = c_codec_add.ratio() > 0
            ? (c_codec_add.ratio() - c_codec_cv.ratio()) / c_codec_add.ratio() * 100.0 : 0.0;
        std::println("\n  convex+pretrain vs additive codec: cliff {:.2f}x -> {:.2f}x ({:+.0f}%)",
                     c_codec_add.ratio(), c_codec_cv.ratio(), -cv_vs_add);
        std::println("  convex+pretrain vs baseline (both rare-weighted): {:.2f}x vs {:.2f}x ({:+.0f}% for codec)",
                     c_codec_cv.ratio(), c_base_rw.ratio(), cv_vs_base);
        std::println("  VERDICT: {}",
                     (c_codec_cv.ratio() < c_base_rw.ratio())
                         ? "codec convex+pretrain BEATS baseline on the cliff — char-composition salvageable"
                         : "codec still does not beat baseline — see numbers");
        return 0;
    }

    // 2×2: {baseline, codec} × {unweighted, rare-weighted}, identical data split + seeds.
    std::println("\n-- BASELINE Denoiser, UNWEIGHTED --");
    dn::Denoiser base_uw(Vr, D, 8, 4, L, 0, /*seed=*/7);
    train(base_uw, train_ids, static_cast<int>(steps), B, T, 1e-3f, 7, "base/uw");
    const auto c_base_uw = cliff(base_uw, eval_ids, is_rare, T, windows);

    std::println("\n-- BASELINE Denoiser, RARE-WEIGHTED --");
    dn::Denoiser base_rw(Vr, D, 8, 4, L, 0, /*seed=*/7);
    train(base_rw, train_ids, static_cast<int>(steps), B, T, 1e-3f, 7, "base/rw", tok_w);
    const auto c_base_rw = cliff(base_rw, eval_ids, is_rare, T, windows);

    std::println("\n-- CodecDenoiser, UNWEIGHTED --");
    dn::CodecDenoiser codec_uw(Vr, D, 8, 4, L, 0, /*n_chars=*/257, maxlen, word_chars, /*seed=*/7);
    train(codec_uw, train_ids, static_cast<int>(steps), B, T, 1e-3f, 7, "codec/uw");
    const auto c_codec_uw = cliff(codec_uw, eval_ids, is_rare, T, windows);

    std::println("\n-- CodecDenoiser, RARE-WEIGHTED --");
    dn::CodecDenoiser codec_rw(Vr, D, 8, 4, L, 0, /*n_chars=*/257, maxlen, word_chars, /*seed=*/7);
    train(codec_rw, train_ids, static_cast<int>(steps), B, T, 1e-3f, 7, "codec/rw", tok_w);
    const auto c_codec_rw = cliff(codec_rw, eval_ids, is_rare, T, windows);

    auto row = [](const char* tag, const de::OovCliffResult& r) {
        std::println("  {:<14} NLL_common {:.3f}  NLL_rare {:.3f}  CLIFF {:.2f}x", tag,
                     r.nll_common(), r.nll_rare(), r.ratio());
    };
    std::println("\n========== OOV 2x2 (rarest 50% of types; rare_weight {:.1f}) ==========", rw);
    row("baseline/uw",  c_base_uw);
    row("baseline/rw",  c_base_rw);
    row("codec/uw",     c_codec_uw);
    row("codec/rw",     c_codec_rw);

    // The key isolation: under the SAME rare-aware objective, does the codec beat the baseline?
    const double codec_vs_base_rw = c_base_rw.ratio() > 0
        ? (c_base_rw.ratio() - c_codec_rw.ratio()) / c_base_rw.ratio() * 100.0 : 0.0;
    const double rw_helps_codec = c_codec_uw.ratio() > 0
        ? (c_codec_uw.ratio() - c_codec_rw.ratio()) / c_codec_uw.ratio() * 100.0 : 0.0;
    std::println("\n  rare-weighting effect on codec cliff: {:.2f}x -> {:.2f}x ({:+.0f}%)",
                 c_codec_uw.ratio(), c_codec_rw.ratio(), -rw_helps_codec);
    std::println("  codec vs baseline UNDER rare-aware objective: cliff {:.2f}x vs {:.2f}x ({:+.0f}% for codec)",
                 c_codec_rw.ratio(), c_base_rw.ratio(), codec_vs_base_rw);
    std::println("  VERDICT: {}",
                 (c_codec_rw.ratio() < c_base_rw.ratio() && c_codec_rw.ratio() < c_codec_uw.ratio())
                     ? "codec helps under rare-aware objective (cliff below both controls)"
                     : "see numbers — codec still not winning");
    return 0;
}
