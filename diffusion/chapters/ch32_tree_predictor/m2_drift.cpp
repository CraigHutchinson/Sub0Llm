// m2_drift.cpp (Ch32 Phase 0, metric M2) — bank the TOPIC-DRIFT baseline.
//
// Trains a flat word-level Denoiser, GENERATES passages from it, and measures M2 (distinct-n +
// content-word persistence/drift) on the generations vs held-out CORPUS passages. The gap — model
// drifts/loops more than the corpus — is the quantity P2's gist conditioner must shrink (BUILD_PLAN
// §Phase 0 M2 / §Phase 2). A phrase-SHUFFLED corpus control validates that M2 actually responds to
// long-range structure (shuffling windows destroys persistence but not local n-gram statistics).
//
// Build: cmake --build build-cuda --target ch32_m2_drift ; run (trains on GPU, samples on CPU).

#include "sub0diff/eval/topic_drift.hpp"
#include "sub0diff/nn/denoiser.hpp"
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
    for (const auto& t : texts) {
        auto v = tok.encode(t);
        ids.insert(ids.end(), v.begin(), v.end());
    }
    return ids;
}

// Chunk a token stream into consecutive non-overlapping passages of length T.
std::vector<std::vector<std::int32_t>> chunk(std::span<const std::int32_t> ids, std::int64_t T) {
    std::vector<std::vector<std::int32_t>> out;
    const std::int64_t n = static_cast<std::int64_t>(ids.size()) / T;
    for (std::int64_t i = 0; i < n; ++i)
        out.emplace_back(ids.begin() + i * T, ids.begin() + (i + 1) * T);
    return out;
}

void m2_print(const char* tag, const de::TopicDriftResult& r) {
    std::println("  {:<16} recurrence {:.3f}   distinct-2/3/4 {:.3f}/{:.3f}/{:.3f}   persist near {:.3f} far {:.3f}",
                 tag, r.content_recurrence, r.distinct2, r.distinct3, r.distinct4,
                 r.persistence_near, r.persistence_far);
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
    const std::int64_t Bsz    = arg_i(argc, argv, "--batch", 16);
    const std::int64_t K      = arg_i(argc, argv, "--gen", 64);
    const std::int64_t stop_k = arg_i(argc, argv, "--stop_k", 100);
    const float temp          = static_cast<float>(arg_i(argc, argv, "--temp_x100", 90)) / 100.0f;
    const std::int64_t win    = arg_i(argc, argv, "--window", 16);

    std::println("== Ch32 Phase-0 M2 — topic-drift baseline (corpus vs flat model) ==");
    auto paras = read_paragraphs(corpus, plimit);
    if (paras.size() < 20) throw std::runtime_error("need >=20 paragraphs");
    const std::size_t n_eval = std::max<std::size_t>(4, paras.size() / 10);   // 10% held out
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

    // ── Train a flat Denoiser on GPU ──────────────────────────────────────────────────────────
    dn::Denoiser model(Vr, D, 8, 4, L, 0, /*seed=*/7);
    model.to(sub0llm::Device::cuda());
    auto params = model.parameters();
    sub0llm::nn::Adam opt(params, 1e-3f);
    dt::BatchedDiffusionLossContext ctx(Bsz, T);
    std::mt19937 rng(7);
    const std::size_t n_pos = train_ids.size() - static_cast<std::size_t>(T) + 1;
    std::uniform_int_distribution<std::size_t> off_dist(0, n_pos - 1);
    const auto t0 = std::chrono::steady_clock::now();
    for (int s = 0; s < steps; ++s) {
        std::vector<std::size_t> offsets(static_cast<std::size_t>(Bsz));
        for (auto& o : offsets) o = off_dist(rng);
        for (auto* p : params)
            if (p->grad().numel() > 0)
                p->grad() = sub0llm::zeros(p->data().shape(), p->data().dtype(), p->data().device());
        auto res = dt::batched_diffusion_loss(model, train_ids, offsets, rng, ctx, 0.02f, 1.0f);
        res.loss.backward();
        (void)sub0llm::nn::clip_grad_norm(params, 5.0f);
        opt.step();
        if ((s + 1) % 500 == 0)
            std::println("  [train] step {:>5}  nelbo={:.4f}  ({:.1f}s)", s + 1,
                         static_cast<double>(res.loss.data().to(sub0llm::Device::cpu()).item<float>()),
                         std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
    }

    // ── Generate K passages on CPU (the sampler reads host logits) ──────────────────────────────
    model.to(sub0llm::Device::cpu());
    dn::SamplerConfig cfg;
    cfg.temperature = temp;
    cfg.commit_order = dn::CommitOrder::Spread;   // ch32 4b decode default (less looping)
    std::println("\ngenerating {} passages (T={}, temp={:.2f}, spread)...", K, T, temp);
    std::vector<std::vector<std::int32_t>> gen;
    gen.reserve(static_cast<std::size_t>(K));
    std::mt19937 grng(2026);
    for (std::int64_t k = 0; k < K; ++k) {
        auto canvas = dn::make_canvas(model, T);
        dn::refine_canvas(model, canvas, cfg, grng);
        gen.push_back(std::move(canvas));
    }

    // ── Corpus passages = whole held-out PARAGRAPHS (coherent stories, entities recur) ───────────
    std::vector<std::vector<std::int32_t>> corpus_p;
    for (const auto& p : eval_p) {
        auto v = tok.encode(p);
        if (static_cast<std::int64_t>(v.size()) >= 2 * win) corpus_p.push_back(std::move(v));
    }

    // ── Control = UNIGRAM-resampled passages: each token i.i.d. from the corpus unigram
    //    distribution. SAME word frequencies, ZERO deliberate recurrence structure. This is the
    //    distribution-matched control: if corpus recurrence > unigram-chance, recurrence is a real
    //    coherence signal (entities reused above chance); if not, it is merely unigram frequency.
    //    (A window-REARRANGEMENT control is useless on TinyStories — it is too lexically homogeneous,
    //    so random windows reuse the same common content words as a real story. Measured.)
    std::vector<double> freq(static_cast<std::size_t>(Vr), 0.0);
    for (auto id : train_ids) if (id >= 0 && id < Vr) freq[static_cast<std::size_t>(id)] += 1.0;
    std::mt19937 srng(99);
    std::discrete_distribution<std::int32_t> uni(freq.begin(), freq.end());
    std::vector<std::vector<std::int32_t>> mixed;   // (unigram control)
    for (const auto& p : corpus_p) {
        std::vector<std::int32_t> q(p.size());
        for (auto& x : q) x = uni(srng);
        mixed.push_back(std::move(q));
    }

    auto span_of = [](const std::vector<std::vector<std::int32_t>>& v) {
        std::vector<std::span<const std::int32_t>> s;
        for (const auto& p : v) s.emplace_back(p);
        return s;
    };
    const auto rc = de::evaluate_topic_drift(span_of(corpus_p), is_content, win, 1, 2);
    const auto rs = de::evaluate_topic_drift(span_of(mixed),    is_content, win, 1, 2);
    const auto rg = de::evaluate_topic_drift(span_of(gen),      is_content, win, 1, 2);

    std::println("\n================ M2 topic-drift (window {}) ================", win);
    std::println("  recurrence = fraction of distinct content types reused ≥2× (entity persistence, ↑ coherent)");
    m2_print("corpus",         rc);
    m2_print("unigram-control", rs);
    m2_print("flat-model gen", rg);
    std::println("\n  control: UNIGRAM-resampled (freq-matched, no structure) should have LOWER recurrence than corpus");
    std::println("    recurrence corpus {:.3f} vs unigram {:.3f}   (recurrence is a real coherence signal, not just freq: {})",
                 rc.content_recurrence, rs.content_recurrence,
                 rs.content_recurrence < rc.content_recurrence ? "YES" : "NO — it's just unigram frequency");
    std::println("  the GAP P2 must close: content recurrence model {:.3f} vs corpus {:.3f}  ({:.1f}x lower); "
                 "distinct-3 model {:.3f} vs corpus {:.3f}",
                 rg.content_recurrence, rc.content_recurrence,
                 rg.content_recurrence > 0 ? rc.content_recurrence / rg.content_recurrence : 0.0,
                 rg.distinct3, rc.distinct3);
    return 0;
}
