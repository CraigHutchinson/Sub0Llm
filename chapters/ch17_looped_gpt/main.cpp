#include "sub0llm/nn/looped_gpt.hpp"
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

// ── Training step helper ──────────────────────────────────────────────────────

static float train_step(LoopedGPT& model, Adam& adam,
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

// ── §17.1 Parameter comparison ────────────────────────────────────────────────

static void section_param_comparison() {
    std::cout << "\n=== §17.1  Parameter Count: LoopedGPT vs ModernGPT ===\n";

    const int64_t V = 98, D = 48;

    auto count_params = [](auto& m) {
        int64_t n = 0;
        for (auto* v : m.parameters())
            n += static_cast<int64_t>(v->data().numel());
        return n;
    };

    for (int64_t K : {1LL, 2LL, 4LL, 8LL}) {
        LoopedGPT looped(V, D, 2, 1, K);
        ModernGPT stacked(V, D, 2, 1, K);

        int64_t p_looped  = count_params(looped);
        int64_t p_stacked = count_params(stacked);
        double  ratio     = static_cast<double>(p_looped) /
                            static_cast<double>(p_stacked) * 100.0;

        std::cout << std::format(
            "  K={:>1}  looped={:>7}  stacked={:>7}  looped/stacked={:.1f}%\n",
            K, p_looped, p_stacked, ratio);
    }

    std::cout << "  LoopedGPT amortises the same weights K times — fewer\n"
                 "  parameters, same compute depth per forward pass.\n";
}

// ── §17.2 Train LoopedGPT and compare loop counts ────────────────────────────

struct TrainResult {
    BPETokenizer       tokenizer;
    LoopedGPT          model;
    std::vector<int32_t> token_ids;
};

static TrainResult section_train() {
    std::cout << "\n=== §17.2  Training LoopedGPT (K=4 loops) ===\n";

    std::vector<std::string> corpus_vec = {corpus_text};
    auto tokenizer = BPETokenizer::train(corpus_vec, /*vocab_size=*/96);
    auto token_ids = tokenizer.encode(corpus_text);

    std::cout << std::format("  vocab_size    : {}\n", tokenizer.vocab_size());
    std::cout << std::format("  corpus tokens : {}\n", token_ids.size());

    const int64_t V       = static_cast<int64_t>(tokenizer.vocab_size());
    const int64_t seq_len = 16;
    const int     steps   = 300;

    LoopedGPT model(V, 48, 2, 1, /*n_loops=*/4, 0, /*seed=*/42);
    Adam adam(model.parameters(), 3e-3f);

    {
        int64_t n_params = 0;
        for (auto* v : model.parameters())
            n_params += static_cast<int64_t>(v->data().numel());
        std::cout << std::format("  parameters    : {}\n", n_params);
        std::cout << std::format("  loops         : {}\n", model.n_loops());
    }

    const int64_t n_tokens = static_cast<int64_t>(token_ids.size());
    if (n_tokens < seq_len + 2)
        throw std::runtime_error(
            std::format("corpus too short ({} tokens), need >= {}", n_tokens, seq_len + 2));
    std::mt19937 rng_train(1);
    std::uniform_int_distribution<int64_t> dist(0, n_tokens - seq_len - 2);

    float first_loss = -1.f, last_loss = -1.f;
    for (int step = 0; step < steps; ++step) {
        const int64_t start = dist(rng_train);
        float lv = train_step(model, adam, token_ids, start, seq_len);
        if (step == 0) first_loss = lv;
        last_loss = lv;
        if (step == 0 || step % 100 == 0 || step == steps - 1)
            std::cout << std::format("  step {:>3}  loss={:.4f}\n", step, lv);
    }
    std::cout << std::format("  Training: {:.4f} → {:.4f}\n", first_loss, last_loss);

    return {std::move(tokenizer), std::move(model), std::move(token_ids)};
}

// ── §17.3 Loop-count comparison at training time ──────────────────────────────

static void section_loop_training_comparison(const BPETokenizer& tokenizer,
                                              const std::vector<int32_t>& token_ids)
{
    std::cout << "\n=== §17.3  Final Loss vs Loop Count (300 steps each) ===\n";

    const int64_t V       = static_cast<int64_t>(tokenizer.vocab_size());
    const int64_t seq_len = 16;
    const int     steps   = 300;

    const int64_t n_tokens = static_cast<int64_t>(token_ids.size());

    for (int64_t K : {1LL, 2LL, 4LL, 8LL}) {
        LoopedGPT model(V, 48, 2, 1, K, 0, /*seed=*/7);
        Adam adam(model.parameters(), 3e-3f);

        std::mt19937 rng(1);
        std::uniform_int_distribution<int64_t> dist(0, n_tokens - seq_len - 2);

        float last_loss = 0.f;
        for (int step = 0; step < steps; ++step) {
            last_loss = train_step(model, adam, token_ids, dist(rng), seq_len);
        }

        int64_t n_params = 0;
        for (auto* v : model.parameters())
            n_params += static_cast<int64_t>(v->data().numel());

        std::cout << std::format(
            "  K={:>1}  params={:>6}  final_loss={:.4f}\n",
            K, n_params, last_loss);
    }

    std::cout << "  With a fixed step budget, fewer loops train faster (K=1\n"
                 "  overfits most readily). More loops need more steps to\n"
                 "  propagate gradients through deeper unrolled iterations.\n";
}

// ── §17.4 Inference-time loop budget (forward_k) ─────────────────────────────

static void section_inference_budget(LoopedGPT& model,
                                      const BPETokenizer& tokenizer,
                                      const std::vector<int32_t>& token_ids)
{
    std::cout << "\n=== §17.4  Inference Loop Budget (forward_k) ===\n";

    std::vector<int32_t> prompt(token_ids.begin(), token_ids.begin() + 4);
    std::cout << std::format("  Prompt: \"{}\"\n", tokenizer.decode(prompt));

    SamplingConfig cfg;  // greedy

    for (int64_t K : {1LL, 2LL, 4LL, 8LL, 16LL}) {
        // generate() uses model.forward(), so we adapt: call forward_k manually
        // for a single generate step loop.
        std::vector<int32_t> context(prompt);
        const int64_t max_new = 15;
        std::mt19937 rng(0);

        for (int64_t step = 0; step < max_new; ++step) {
            Tensor ids({static_cast<int64_t>(context.size())}, DType::Int32);
            auto sp = ids.data_as<int32_t>();
            for (std::size_t i = 0; i < context.size(); ++i)
                sp[i] = context[i];

            Tensor logits_full = model.forward_k(ids, K).data();
            const int64_t T   = logits_full.shape(0);
            const int64_t V   = logits_full.shape(1);
            Tensor last({V}, DType::Float32);
            auto src = logits_full.data_as<float>();
            auto dst = last.data_as<float>();
            const std::size_t offset = static_cast<std::size_t>((T - 1) * V);
            for (std::size_t j = 0; j < static_cast<std::size_t>(V); ++j)
                dst[j] = src[offset + j];

            context.push_back(greedy_sample(last));
        }

        std::vector<int32_t> gen(context.begin() + static_cast<std::ptrdiff_t>(prompt.size()),
                                  context.end());
        std::cout << std::format("  K={:>2}: \"{}\"\n", K, tokenizer.decode(gen));
    }

    std::cout << "  Higher K = more refined hidden state = (potentially) better\n"
                 "  output without changing any weights.\n";
}

// ── §17.5 Conceptual comparison: loops vs thinking tokens ────────────────────

static void section_comparison() {
    std::cout << "\n=== §17.5  Loops vs Thinking Tokens (Ch16) ===\n";
    std::cout <<
        "  Ch16 (thinking tokens)  Ch17 (looped GPT)\n"
        "  ─────────────────────   ─────────────────────────────────\n"
        "  Budget = token count    Budget = loop count K\n"
        "  Thinking is visible     Thinking is internal (hidden state)\n"
        "  Uses full vocab space   Uses fixed computation path\n"
        "  Interpretable trace     Opaque iterative refinement\n"
        "  Larger KV cache         No extra KV cache\n"
        "  Supported by RLHF       RLHF trains loop count implicitly\n"
        "\n"
        "  Both approaches trade off extra compute for quality.\n"
        "  Thinking tokens expose the chain-of-thought; looped GPT\n"
        "  hides it inside repeated weight application.\n"
        "\n"
        "  Next chapter: Mixture-of-Experts — scaling width instead\n"
        "  of depth, routing tokens through sparse expert FFNs.\n";
}

// ── main ──────────────────────────────────────────────────────────────────────

int main() {
    std::cout << "Chapter 17 — LoopedGPT: Thinking as Computation Depth\n";
    std::cout << std::string(60, '=') << '\n';

    section_param_comparison();

    auto [tokenizer, model, token_ids] = []{
        auto r = section_train();
        return std::tuple{std::move(r.tokenizer),
                          std::move(r.model),
                          std::move(r.token_ids)};
    }();

    section_loop_training_comparison(tokenizer, token_ids);
    section_inference_budget(model, tokenizer, token_ids);
    section_comparison();

    std::cout << "\nDone.\n";
    return 0;
}
