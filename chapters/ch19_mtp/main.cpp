#include "sub0llm/nn/mtp.hpp"
#include "sub0llm/nn/modern_gpt.hpp"
#include "sub0llm/nn/optimizer.hpp"
#include "sub0llm/nn/sampler.hpp"
#include "sub0llm/tokenizer/bpe.hpp"
#include "sub0llm/autograd/ops.hpp"
#include "sub0llm/core/tensor.hpp"
#include <format>
#include <iostream>
#include <random>
#include <vector>

using namespace sub0llm;
using namespace sub0llm::nn;
using namespace sub0llm::autograd;

// ── Corpus ────────────────────────────────────────────────────────────────────

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

// ── §19.1 Architecture overview ───────────────────────────────────────────────

static void section_architecture() {
    std::cout << "\n=== §19.1  MTP Architecture ===\n";
    std::cout <<
        "  Standard GPT: one output head predicts token t+1 from position t.\n"
        "  MTP GPT: K extra heads each predict a future token:\n"
        "    head 0 (tied LM head) → token t+1  (standard next-token)\n"
        "    head 1                → token t+2  (2 steps ahead)\n"
        "    head k                → token t+k+1\n"
        "\n"
        "  Training loss:  L = CE(head_0) + α * Σ_{k=1}^{K} CE(head_k)\n"
        "  Benefits:\n"
        "    • Richer gradient signal — each position trains K+1 heads.\n"
        "    • Inference acceleration — K+1 tokens per forward pass.\n"
        "  Unlike speculative decoding there is no separate draft model:\n"
        "  the same weights produce all K+1 predictions simultaneously.\n";

    const int64_t V = 98, D = 48;
    for (int64_t K : {0LL, 1LL, 3LL}) {
        ModernGPT m(V, D, 2, 1, 2, 0, K);
        int64_t n = 0;
        for (auto* v : m.parameters()) n += static_cast<int64_t>(v->data().numel());
        std::cout << std::format(
            "  K={:>1} MTP heads  params={:>7}  (heads add {} Linear {:.0f}→{:.0f} each)\n",
            K, n, K, static_cast<double>(D), static_cast<double>(V));
    }
}

// ── training helpers ──────────────────────────────────────────────────────────

static float train_step_mtp(ModernGPT& model, Adam& adam,
                              const std::vector<int32_t>& tokens,
                              int64_t start, int64_t seq_len, float aux_w)
{
    Tensor ids({seq_len}, DType::Int32);
    {
        auto sp = ids.data_as<int32_t>();
        for (int64_t i = 0; i < seq_len; ++i)
            sp[static_cast<std::size_t>(i)] = tokens[static_cast<std::size_t>(start + i)];
    }
    adam.zero_grad();
    auto loss = mtp_train_loss(model, ids, seq_len, aux_w);
    loss.backward();
    (void)clip_grad_norm(model.parameters(), 1.0f);
    adam.step();
    return loss.data().data_as<float>()[0];
}

static float train_step_dense(ModernGPT& model, Adam& adam,
                                const std::vector<int32_t>& tokens,
                                int64_t start, int64_t seq_len)
{
    const int64_t n_pred = seq_len - 1;
    Tensor ids({seq_len}, DType::Int32);
    Tensor tgt({n_pred}, DType::Int32);
    {
        auto sp = ids.data_as<int32_t>();
        auto tp = tgt.data_as<int32_t>();
        for (int64_t i = 0; i < seq_len; ++i)
            sp[static_cast<std::size_t>(i)] = tokens[static_cast<std::size_t>(start + i)];
        for (int64_t i = 0; i < n_pred; ++i)
            tp[static_cast<std::size_t>(i)] = tokens[static_cast<std::size_t>(start + i + 1)];
    }
    adam.zero_grad();
    auto logits = model.forward(ids);
    auto loss = cross_entropy(narrow(logits, 0, n_pred), tgt);
    loss.backward();
    (void)clip_grad_norm(model.parameters(), 1.0f);
    adam.step();
    return loss.data().data_as<float>()[0];
}

// ── §19.2 Training comparison ─────────────────────────────────────────────────

struct TrainResult {
    BPETokenizer        tokenizer;
    ModernGPT           model;
    std::vector<int32_t> token_ids;
};

static TrainResult section_train(int64_t n_mtp, float aux_w) {
    std::vector<std::string> cv = {corpus_text};
    auto tokenizer = BPETokenizer::train(cv, 96);
    auto token_ids = tokenizer.encode(corpus_text);
    const int64_t V = static_cast<int64_t>(tokenizer.vocab_size());

    const int64_t seq_len = 20;  // >= 2 + n_mtp
    const int     steps   = 300;

    ModernGPT model(V, 48, 2, 1, 2, 0, n_mtp, /*seed=*/42);
    Adam adam(model.parameters(), 3e-3f);

    const int64_t n_tokens = static_cast<int64_t>(token_ids.size());
    if (n_tokens < seq_len + 2)
        throw std::runtime_error(std::format("corpus too short ({} tokens)", n_tokens));
    std::mt19937 rng(1);
    std::uniform_int_distribution<int64_t> dist(0, n_tokens - seq_len - 2);

    float first = -1.f, last = -1.f;
    for (int step = 0; step < steps; ++step) {
        float lv;
        if (n_mtp > 0)
            lv = train_step_mtp(model, adam, token_ids, dist(rng), seq_len, aux_w);
        else
            lv = train_step_dense(model, adam, token_ids, dist(rng), seq_len);
        if (step == 0) first = lv;
        last = lv;
        if (step == 0 || step % 100 == 0 || step == steps - 1)
            std::cout << std::format("    step {:>3}  loss={:.4f}\n", step, lv);
    }
    std::cout << std::format("    {} → {}\n", first, last);
    return {std::move(tokenizer), std::move(model), std::move(token_ids)};
}

static void section_training_comparison() {
    std::cout << "\n=== §19.2  Training: Dense vs MTP (300 steps each) ===\n";

    std::vector<std::string> cv = {corpus_text};
    auto tokenizer = BPETokenizer::train(cv, 96);
    auto token_ids = tokenizer.encode(corpus_text);
    const int64_t V = static_cast<int64_t>(tokenizer.vocab_size());
    const int64_t seq_len = 20;
    const int     steps   = 300;
    const int64_t n_tokens = static_cast<int64_t>(token_ids.size());
    if (n_tokens < seq_len + 2)
        throw std::runtime_error(std::format("corpus too short ({} tokens)", n_tokens));

    for (auto [K, label] : std::vector<std::pair<int64_t, std::string>>{
             {0, "dense (K=0)"}, {1, "MTP K=1"}, {3, "MTP K=3"}}) {
        ModernGPT model(V, 48, 2, 1, 2, 0, K, /*seed=*/42);
        Adam adam(model.parameters(), 3e-3f);
        std::mt19937 rng(1);
        std::uniform_int_distribution<int64_t> dist(0, n_tokens - seq_len - 2);

        float first = -1.f, last = -1.f;
        for (int s = 0; s < steps; ++s) {
            float lv;
            if (K > 0)
                lv = train_step_mtp(model, adam, token_ids, dist(rng), seq_len, 0.1f);
            else
                lv = train_step_dense(model, adam, token_ids, dist(rng), seq_len);
            if (s == 0) first = lv;
            last = lv;
        }
        int64_t n = 0;
        for (auto* v : model.parameters()) n += static_cast<int64_t>(v->data().numel());
        std::cout << std::format(
            "  {:>12}  params={:>6}  loss: {:.4f} → {:.4f}\n",
            label, n, first, last);
    }
    std::cout << "  MTP loss includes both main CE and auxiliary head CEs\n"
                 "  (weighted by aux_weight=0.1), so absolute values differ.\n"
                 "  MTP often converges faster due to richer gradient signal.\n";
}

// ── §19.3 MTP inference: tokens per forward pass ──────────────────────────────

static void section_inference(ModernGPT& model, const BPETokenizer& tokenizer,
                                const std::vector<int32_t>& token_ids)
{
    std::cout << "\n=== §19.3  MTP Inference: Tokens per Forward Pass ===\n";

    const int64_t K = model.n_mtp_heads();
    std::vector<int32_t> prompt(token_ids.begin(), token_ids.begin() + 4);
    const int64_t max_new = 20;
    SamplingConfig cfg;  // greedy

    std::cout << std::format("  Prompt    : \"{}\"\n", tokenizer.decode(prompt));
    std::cout << std::format("  MTP heads : K={}\n", K);
    std::cout << std::format("  Tokens/pass (MTP): up to K+1 = {}\n", K + 1);

    // MTP generation using all K+1 heads
    {
        std::mt19937 rng(0);
        auto result = mtp_generate_stats(model, prompt, max_new, cfg, rng);
        std::vector<int32_t> gen(result.tokens.begin() +
                                  static_cast<std::ptrdiff_t>(prompt.size()),
                                  result.tokens.end());
        std::cout << std::format(
            "  MTP gen   : \"{}\"\n"
            "              ({} tokens, {} forward passes, {:.1f} tokens/pass)\n",
            tokenizer.decode(gen),
            static_cast<int64_t>(gen.size()),
            result.n_forward_passes,
            result.tokens_per_pass);
    }

    // Standard generation (using generate() — 1 token/pass)
    {
        std::mt19937 rng(0);
        std::vector<int32_t> context(prompt);
        int64_t fwd_count = 0;
        for (int64_t s = 0; s < max_new; ++s) {
            Tensor ids({static_cast<int64_t>(context.size())}, DType::Int32);
            auto sp = ids.data_as<int32_t>();
            for (std::size_t i = 0; i < context.size(); ++i)
                sp[i] = context[i];
            Tensor logits = model.forward(ids).data();
            // Extract last row
            const int64_t T = logits.shape(0), V = logits.shape(1);
            Tensor last({V}, DType::Float32);
            auto src = logits.data_as<float>();
            auto dst = last.data_as<float>();
            for (std::size_t j = 0; j < static_cast<std::size_t>(V); ++j)
                dst[j] = src[static_cast<std::size_t>((T - 1) * V) + j];
            context.push_back(greedy_sample(last));
            ++fwd_count;
        }
        std::vector<int32_t> gen(context.begin() +
                                  static_cast<std::ptrdiff_t>(prompt.size()),
                                  context.end());
        std::cout << std::format(
            "  Dense gen : \"{}\"\n"
            "              ({} tokens, {} forward passes, {:.1f} tokens/pass)\n",
            tokenizer.decode(gen),
            static_cast<int64_t>(gen.size()),
            fwd_count,
            static_cast<double>(max_new) / static_cast<double>(fwd_count));
    }

    std::cout << std::format(
        "\n  Speedup factor (ideal): {:.1f}×  (K+1={} vs 1 token/pass)\n",
        static_cast<double>(K + 1), K + 1);
    std::cout << "  Note: actual speedup depends on hardware parallelism.\n"
                 "  On a batch of T tokens, the extra heads add only O(T×V)\n"
                 "  compute — negligible vs the O(T²×D) attention cost.\n";
}

// ── §19.4 Aux weight sensitivity ──────────────────────────────────────────────

static void section_aux_weight(const BPETokenizer& tokenizer,
                                const std::vector<int32_t>& token_ids)
{
    std::cout << "\n=== §19.4  Aux Weight Sensitivity ===\n";

    const int64_t V       = static_cast<int64_t>(tokenizer.vocab_size());
    const int64_t seq_len = 20;
    const int     steps   = 300;
    const int64_t n_tokens = static_cast<int64_t>(token_ids.size());
    if (n_tokens < seq_len + 2)
        throw std::runtime_error(std::format("corpus too short ({} tokens)", n_tokens));

    for (float aux : {0.01f, 0.1f, 0.5f, 1.0f}) {
        ModernGPT model(V, 48, 2, 1, 2, 0, /*K=*/3, /*seed=*/42);
        Adam adam(model.parameters(), 3e-3f);
        std::mt19937 rng(1);
        std::uniform_int_distribution<int64_t> dist(0, n_tokens - seq_len - 2);
        float last = 0.f;
        for (int s = 0; s < steps; ++s)
            last = train_step_mtp(model, adam, token_ids, dist(rng), seq_len, aux);
        std::cout << std::format("  aux_weight={:.2f}  final_loss={:.4f}\n", aux, last);
    }
    std::cout << "  Small aux_weight (0.01–0.1): main head dominates, MTP\n"
                 "  heads get auxiliary gradient signal (recommended).\n"
                 "  Large aux_weight: auxiliary heads compete with main head.\n";
}

// ── §19.5 DeepSeek MTP vs simple heads ───────────────────────────────────────

static void section_deepseek_comparison() {
    std::cout << "\n=== §19.5  DeepSeek MTP vs Simple Linear Heads ===\n";
    std::cout <<
        "  This implementation (ModernGPT MTP heads):\n"
        "    • Each head k is a Linear(D→V) projection from the same\n"
        "      final hidden state h.\n"
        "    • Simple, cheap: K×(D×V) extra parameters.\n"
        "    • No additional attention or FFN computation per head.\n"
        "\n"
        "  DeepSeek-V3 MTP modules:\n"
        "    • Each module k has: RMSNorm → Linear → TransformerBlock → Linear.\n"
        "    • Input: causal-masked combination of h and embedding(token_{t+k}).\n"
        "    • Much more expressive — can reason about intermediate steps.\n"
        "    • Higher parameter cost per head but better multi-token accuracy.\n"
        "\n"
        "  Both share the core benefit: K+1 tokens per forward pass at\n"
        "  inference, with no separate draft model (unlike speculative decoding).\n"
        "\n"
        "  When to use each:\n"
        "    Linear heads: cheap, sufficient for K=1–2, minimal overhead.\n"
        "    DeepSeek modules: higher K (4–8), better quality multi-token.\n";
}

// ── main ──────────────────────────────────────────────────────────────────────

int main() {
    std::cout << "Chapter 19 — Multi-Token Prediction: K+1 Tokens per Forward Pass\n";
    std::cout << std::string(60, '=') << '\n';

    section_architecture();

    // Train MTP model (K=3) for subsequent sections.
    std::cout << "\n--- Training MTP model (K=3, aux_weight=0.1, 300 steps) ---\n";
    auto [tokenizer, model, token_ids] = []{
        auto r = section_train(3, 0.1f);
        return std::tuple{std::move(r.tokenizer), std::move(r.model),
                          std::move(r.token_ids)};
    }();

    section_training_comparison();
    section_inference(model, tokenizer, token_ids);
    section_aux_weight(tokenizer, token_ids);
    section_deepseek_comparison();

    std::cout << "\nDone.\n";
    return 0;
}
