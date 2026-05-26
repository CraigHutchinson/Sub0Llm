#include "sub0llm/nn/moe.hpp"
#include "sub0llm/nn/modern_gpt.hpp"
#include "sub0llm/nn/sampler.hpp"
#include "sub0llm/nn/optimizer.hpp"
#include "sub0llm/tokenizer/bpe.hpp"
#include "sub0llm/autograd/ops.hpp"
#include "sub0llm/core/tensor.hpp"
#include <cmath>
#include <format>
#include <iostream>
#include <random>
#include <vector>

using namespace sub0llm;
using namespace sub0llm::nn;
using namespace sub0llm::autograd;

// ── Corpus (Alice in Wonderland Ch.1, public domain) ─────────────────────────

static const std::string corpus_text =
    "Alice was beginning to get very tired of sitting by her sister on the bank, "
    "and of having nothing to do: once or twice she had peeped into the book her "
    "sister was reading, but it had no pictures or conversations in it, and what "
    "is the use of a book thought Alice without pictures or conversations. So she "
    "was considering in her own mind whether the pleasure of making a daisy-chain "
    "would be worth the trouble of getting up and picking the daisies when suddenly "
    "a White Rabbit with pink eyes ran close by her. There was nothing so very "
    "remarkable in that nor did Alice think it so very much out of the way to hear "
    "the Rabbit say to itself oh dear oh dear I shall be too late and Alice thought "
    "it over she ought to have wondered at this but at the time it all seemed quite natural.";

// ── Training helpers ──────────────────────────────────────────────────────────

static float train_step_moe(MoEGPT& model, Adam& adam,
                              const std::vector<int32_t>& tokens,
                              int64_t start, int64_t seq_len,
                              float aux_coeff)
{
    const int64_t n_pred = seq_len - 1;

    Tensor input({seq_len}, DType::Int32);
    Tensor targets({n_pred}, DType::Int32);
    {
        auto inp = input.data_as<int32_t>();
        auto tgt = targets.data_as<int32_t>();
        for (int64_t i = 0; i < seq_len; ++i)
            inp[static_cast<std::size_t>(i)] = tokens[static_cast<std::size_t>(start + i)];
        for (int64_t i = 0; i < n_pred; ++i)
            tgt[static_cast<std::size_t>(i)] = tokens[static_cast<std::size_t>(start + i + 1)];
    }

    adam.zero_grad();
    auto [logits, aux_loss] = model.forward_moe(input);
    auto ltrunc = narrow(logits, 0, n_pred);
    auto ce     = cross_entropy(ltrunc, targets);
    auto loss   = add(ce, scale(aux_loss, aux_coeff));
    loss.backward();
    (void)clip_grad_norm(model.parameters(), 1.0f);
    adam.step();
    return ce.data().data_as<float>()[0];
}

static float train_step_dense(ModernGPT& model, Adam& adam,
                                const std::vector<int32_t>& tokens,
                                int64_t start, int64_t seq_len)
{
    const int64_t n_pred = seq_len - 1;
    Tensor input({seq_len}, DType::Int32);
    Tensor targets({n_pred}, DType::Int32);
    {
        auto inp = input.data_as<int32_t>();
        auto tgt = targets.data_as<int32_t>();
        for (int64_t i = 0; i < seq_len; ++i)
            inp[static_cast<std::size_t>(i)] = tokens[static_cast<std::size_t>(start + i)];
        for (int64_t i = 0; i < n_pred; ++i)
            tgt[static_cast<std::size_t>(i)] = tokens[static_cast<std::size_t>(start + i + 1)];
    }
    adam.zero_grad();
    auto logits = model.forward(input);
    auto ltrunc = narrow(logits, 0, n_pred);
    auto loss   = cross_entropy(ltrunc, targets);
    loss.backward();
    (void)clip_grad_norm(model.parameters(), 1.0f);
    adam.step();
    return loss.data().data_as<float>()[0];
}

// ── §18.1 Architecture: parameter comparison ──────────────────────────────────

static void section_param_comparison() {
    std::cout << "\n=== §18.1  Parameters: MoEGPT vs Dense ModernGPT ===\n";

    const int64_t V = 98, D = 48;
    // Dense baseline: 2 layers, 1 FFN per layer
    // MoE: 2 layers, E experts per layer, top-k=2 active
    ModernGPT dense(V, D, 2, 1, 2);

    int64_t p_dense = 0;
    for (auto* v : dense.parameters()) p_dense += static_cast<int64_t>(v->data().numel());

    std::cout << std::format("  Dense  (2 layers, 1 FFN)  : {:>7} params\n", p_dense);

    for (int64_t E : {2LL, 4LL, 8LL}) {
        MoEGPT moe(V, D, 2, 1, 2, E, /*top_k=*/2);
        int64_t p_moe = 0;
        for (auto* v : moe.parameters()) p_moe += static_cast<int64_t>(v->data().numel());
        double ratio = static_cast<double>(p_moe) / static_cast<double>(p_dense) * 100.0;
        std::cout << std::format(
            "  MoE    (2 layers, E={:>1}, k=2)  : {:>7} params  ({:.0f}% of dense)\n",
            E, p_moe, ratio);
    }

    std::cout << "  Each added expert increases capacity but only 2 are\n"
                 "  active per token — compute stays roughly constant.\n";
}

// ── §18.2 Train MoEGPT ────────────────────────────────────────────────────────

struct TrainResult {
    BPETokenizer        tokenizer;
    MoEGPT              model;
    std::vector<int32_t> token_ids;
};

static TrainResult section_train() {
    std::cout << "\n=== §18.2  Training MoEGPT (E=4, top-k=2) ===\n";

    std::vector<std::string> corpus_vec = {corpus_text};
    auto tokenizer = BPETokenizer::train(corpus_vec, /*vocab_size=*/96);
    auto token_ids = tokenizer.encode(corpus_text);

    const int64_t V       = static_cast<int64_t>(tokenizer.vocab_size());
    const int64_t seq_len = 16;
    const int     steps   = 300;
    const float   aux_c   = 0.01f;

    MoEGPT model(V, 48, 2, 1, 2, /*n_experts=*/4, /*top_k=*/2, 0, /*seed=*/42);
    Adam adam(model.parameters(), 3e-3f);

    int64_t n_params = 0;
    for (auto* v : model.parameters()) n_params += static_cast<int64_t>(v->data().numel());
    std::cout << std::format("  vocab_size    : {}\n", tokenizer.vocab_size());
    std::cout << std::format("  corpus tokens : {}\n", token_ids.size());
    std::cout << std::format("  parameters    : {}\n", n_params);
    std::cout << std::format("  experts       : {}  top-k : {}\n", model.n_experts(), model.top_k());
    std::cout << std::format("  aux_coeff     : {}\n", aux_c);

    const int64_t n_tokens = static_cast<int64_t>(token_ids.size());
    if (n_tokens < seq_len + 2)
        throw std::runtime_error(
            std::format("corpus too short ({} tokens), need >= {}", n_tokens, seq_len + 2));

    std::mt19937 rng_train(1);
    std::uniform_int_distribution<int64_t> dist(0, n_tokens - seq_len - 2);

    float first_loss = -1.f, last_loss = -1.f;
    for (int step = 0; step < steps; ++step) {
        float lv = train_step_moe(model, adam, token_ids, dist(rng_train), seq_len, aux_c);
        if (step == 0) first_loss = lv;
        last_loss = lv;
        if (step == 0 || step % 100 == 0 || step == steps - 1)
            std::cout << std::format("  step {:>3}  ce_loss={:.4f}\n", step, lv);
    }
    std::cout << std::format("  Training: {:.4f} → {:.4f}\n", first_loss, last_loss);

    return {std::move(tokenizer), std::move(model), std::move(token_ids)};
}

// ── §18.3 MoE vs Dense comparison ────────────────────────────────────────────

static void section_compare(const BPETokenizer& tokenizer,
                              const std::vector<int32_t>& token_ids)
{
    std::cout << "\n=== §18.3  MoE vs Dense (300 steps, same capacity budget) ===\n";

    const int64_t V       = static_cast<int64_t>(tokenizer.vocab_size());
    const int64_t seq_len = 16;
    const int     steps   = 300;
    const int64_t n_tokens = static_cast<int64_t>(token_ids.size());
    if (n_tokens < seq_len + 2)
        throw std::runtime_error(std::format("corpus too short ({} tokens)", n_tokens));

    // Dense: 2 layers
    {
        ModernGPT model(V, 48, 2, 1, 2, 0, 0, /*seed=*/7);
        Adam adam(model.parameters(), 3e-3f);
        std::mt19937 rng(1);
        std::uniform_int_distribution<int64_t> dist(0, n_tokens - seq_len - 2);
        float last = 0.f;
        for (int s = 0; s < steps; ++s)
            last = train_step_dense(model, adam, token_ids, dist(rng), seq_len);
        int64_t p = 0;
        for (auto* v : model.parameters()) p += static_cast<int64_t>(v->data().numel());
        std::cout << std::format("  Dense  2-layer  params={:>6}  final_ce={:.4f}\n", p, last);
    }

    // MoE variants with same layers, different expert counts
    for (auto [E, k] : std::vector<std::pair<int64_t,int64_t>>{{2,1},{4,2},{8,2}}) {
        MoEGPT model(V, 48, 2, 1, 2, E, k, 0, /*seed=*/7);
        Adam adam(model.parameters(), 3e-3f);
        std::mt19937 rng(1);
        std::uniform_int_distribution<int64_t> dist(0, n_tokens - seq_len - 2);
        float last = 0.f;
        for (int s = 0; s < steps; ++s)
            last = train_step_moe(model, adam, token_ids, dist(rng), seq_len, 0.01f);
        int64_t p = 0;
        for (auto* v : model.parameters()) p += static_cast<int64_t>(v->data().numel());
        std::cout << std::format(
            "  MoE E={:>1} k={:>1}      params={:>6}  final_ce={:.4f}\n", E, k, p, last);
    }

    std::cout << "  MoE uses more params (wider) but keeps active compute\n"
                 "  at roughly k/E × dense FFN cost per token.\n";
}

// ── §18.4 Load balancing: aux loss effect ─────────────────────────────────────

static void section_load_balance(const BPETokenizer& tokenizer,
                                  const std::vector<int32_t>& token_ids)
{
    std::cout << "\n=== §18.4  Load Balancing: aux_coeff effect ===\n";

    const int64_t V       = static_cast<int64_t>(tokenizer.vocab_size());
    const int64_t seq_len = 16;
    const int     steps   = 200;
    const int64_t n_tokens = static_cast<int64_t>(token_ids.size());
    if (n_tokens < seq_len + 2)
        throw std::runtime_error(std::format("corpus too short ({} tokens)", n_tokens));

    for (float aux_c : {0.0f, 0.01f, 0.1f}) {
        MoEGPT model(V, 48, 2, 1, 2, 4, 2, 0, /*seed=*/7);
        Adam adam(model.parameters(), 3e-3f);
        std::mt19937 rng(1);
        std::uniform_int_distribution<int64_t> dist(0, n_tokens - seq_len - 2);

        float last_ce = 0.f;
        for (int s = 0; s < steps; ++s)
            last_ce = train_step_moe(model, adam, token_ids, dist(rng), seq_len, aux_c);

        // Measure expert utilization on a fixed batch
        Tensor probe({seq_len}, DType::Int32);
        {
            auto sp = probe.data_as<int32_t>();
            for (int64_t i = 0; i < seq_len; ++i)
                sp[static_cast<std::size_t>(i)] = token_ids[static_cast<std::size_t>(i)];
        }
        auto [logits, aux_val] = model.forward_moe(probe);
        float aux_v = aux_val.data().data_as<float>()[0];

        std::cout << std::format(
            "  aux_coeff={:.2f}  final_ce={:.4f}  aux_loss={:.4f}\n",
            aux_c, last_ce, aux_v);
    }

    std::cout << "  Higher aux_coeff forces more uniform expert utilisation\n"
                 "  at some cost to CE loss (regularisation trade-off).\n";
}

// ── §18.5 Conceptual summary ──────────────────────────────────────────────────

static void section_summary() {
    std::cout << "\n=== §18.5  Mixture-of-Experts: Key Ideas ===\n";
    std::cout <<
        "  Problem: scaling a dense model N× multiplies both params and compute.\n"
        "  MoE solution: scale params by adding experts; keep compute sparse.\n"
        "\n"
        "  Architecture:\n"
        "    • E expert FFNs replace the single FFN in each transformer block.\n"
        "    • A learned router selects the top-k experts for each token.\n"
        "    • Active compute ≈ k/E of a dense model with all E experts.\n"
        "\n"
        "  Load-balancing loss (Shazeer 2017 / Switch Transformer 2021):\n"
        "    L_aux = E × Σ_e f_e × p_e\n"
        "    f_e = fraction of tokens routed to expert e  (non-differentiable)\n"
        "    p_e = mean router probability for expert e   (differentiable)\n"
        "    Without this, all tokens collapse to one expert.\n"
        "\n"
        "  Real-world examples: Mixtral-8×7B (top-2 of 8), GPT-4 (rumoured).\n"
        "\n"
        "  Next chapter: speculative decoding — using a small draft model\n"
        "  to propose tokens verified by the large model in one forward pass.\n";
}

// ── main ──────────────────────────────────────────────────────────────────────

int main() {
    std::cout << "Chapter 18 — Mixture-of-Experts: Sparse Width Scaling\n";
    std::cout << std::string(60, '=') << '\n';

    section_param_comparison();

    auto [tokenizer, model, token_ids] = []{
        auto r = section_train();
        return std::tuple{std::move(r.tokenizer),
                          std::move(r.model),
                          std::move(r.token_ids)};
    }();

    section_compare(tokenizer, token_ids);
    section_load_balance(tokenizer, token_ids);
    section_summary();

    std::cout << "\nDone.\n";
    return 0;
}
