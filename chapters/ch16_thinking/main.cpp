#include "sub0llm/nn/thinking.hpp"
#include "sub0llm/nn/sampler.hpp"
#include "sub0llm/nn/modern_gpt.hpp"
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

static float train_step_lm(ModernGPT& model, Adam& adam,
                            const std::vector<int32_t>& tokens,
                            int64_t start, int64_t seq_len)
{
    const int64_t n_pred = seq_len - 1;

    Tensor input({seq_len}, DType::Int32);
    Tensor targets({n_pred}, DType::Int32);
    {
        auto inp_sp = input.data_as<int32_t>();
        auto tgt_sp = targets.data_as<int32_t>();
        for (int64_t i = 0; i < seq_len; ++i)
            inp_sp[static_cast<std::size_t>(i)] = tokens[static_cast<std::size_t>(start + i)];
        for (int64_t i = 0; i < n_pred; ++i)
            tgt_sp[static_cast<std::size_t>(i)] = tokens[static_cast<std::size_t>(start + i + 1)];
    }

    adam.zero_grad();
    auto logits       = model.forward(input);
    auto logits_trunc = narrow(logits, 0, n_pred);
    auto loss         = cross_entropy(logits_trunc, targets);
    loss.backward();
    (void)clip_grad_norm(model.parameters(), 1.0f);
    adam.step();
    return loss.data().data_as<float>()[0];
}

// ── §16.1 Special tokens ──────────────────────────────────────────────────────

static void section_special_tokens() {
    std::cout << "\n=== §16.1  Special Tokens ===\n";

    std::vector<std::string> corpus_vec = {corpus_text};
    auto tokenizer = BPETokenizer::train(corpus_vec, /*vocab_size=*/96);

    const std::size_t base_vocab = tokenizer.vocab_size();
    std::cout << std::format("  Base vocab size   : {}\n", base_vocab);

    int32_t think_bos = tokenizer.add_special_token("<think>");
    int32_t think_eos = tokenizer.add_special_token("</think>");

    std::cout << std::format("  <think>  token ID : {}\n", think_bos);
    std::cout << std::format("  </think> token ID : {}\n", think_eos);
    std::cout << std::format("  New vocab size    : {}\n", tokenizer.vocab_size());

    if (static_cast<std::size_t>(think_bos) >= base_vocab)
        std::cout << "  <think>  is beyond regular vocab (correct)\n";
    if (static_cast<std::size_t>(think_eos) >= base_vocab)
        std::cout << "  </think> is beyond regular vocab (correct)\n";
}

// ── §16.2 Train + generate_with_thinking demo ─────────────────────────────────

struct TrainResult {
    BPETokenizer tokenizer;
    ModernGPT model;
    std::vector<int32_t> token_ids;
};

static TrainResult section_train() {
    std::cout << "\n=== §16.2  Training + Chain-of-Thought Demo ===\n";

    std::vector<std::string> corpus_vec = {corpus_text};
    auto tokenizer = BPETokenizer::train(corpus_vec, /*vocab_size=*/96);
    tokenizer.add_special_token("<think>");
    tokenizer.add_special_token("</think>");

    auto token_ids = tokenizer.encode(corpus_text);
    std::cout << std::format("  vocab_size     : {}\n", tokenizer.vocab_size());
    std::cout << std::format("  corpus tokens  : {}\n", token_ids.size());

    const int64_t V       = static_cast<int64_t>(tokenizer.vocab_size());
    const int64_t seq_len = 16;
    const int     steps   = 200;

    ModernGPT model(V, 48, 2, 1, 3, 0, 0, /*seed=*/42);
    Adam adam(model.parameters(), 3e-3f);

    const int64_t n_tokens = static_cast<int64_t>(token_ids.size());
    std::mt19937 rng_train(1);
    std::uniform_int_distribution<int64_t> dist(0, n_tokens - seq_len - 2);

    float first_loss = -1.f, last_loss = -1.f;
    for (int step = 0; step < steps; ++step) {
        const int64_t start = dist(rng_train);
        float lv = train_step_lm(model, adam, token_ids, start, seq_len);
        if (step == 0) first_loss = lv;
        last_loss = lv;
        if (step == 0 || step % 50 == 0 || step == steps - 1)
            std::cout << std::format("  step {:>3}  loss={:.4f}\n", step, lv);
    }
    std::cout << std::format("  Training: {:.4f} -> {:.4f}\n", first_loss, last_loss);

    // Demo: generate_with_thinking
    int32_t bos_id = tokenizer.token_id("<think>");
    int32_t eos_id = tokenizer.token_id("</think>");

    ThinkingConfig cfg;
    cfg.think_bos_id     = bos_id;
    cfg.think_eos_id     = eos_id;
    cfg.max_think_tokens = 20;
    cfg.max_answer_tokens = 10;
    cfg.inner_cfg.mode        = SamplingMode::Temperature;
    cfg.inner_cfg.temperature = 1.2f;
    cfg.answer_cfg.mode        = SamplingMode::TopP;
    cfg.answer_cfg.top_p       = 0.9f;
    cfg.answer_cfg.temperature = 0.8f;

    std::vector<int32_t> prompt(token_ids.begin(), token_ids.begin() + 3);
    std::mt19937 rng_gen(42);

    std::cout << std::format("  Prompt ({} tokens): \"{}\"\n",
                              prompt.size(), tokenizer.decode(prompt));

    auto result = generate_with_thinking(model, prompt, cfg, rng_gen);

    std::cout << std::format("  Thinking tokens  : {} (complete={})\n",
                              result.thinking_tokens.size(),
                              result.thinking_complete ? "true" : "false");
    std::cout << std::format("  Thinking decoded : \"{}\"\n",
                              tokenizer.decode(result.thinking_tokens));
    std::cout << std::format("  Answer decoded   : \"{}\"\n",
                              tokenizer.decode(result.answer_tokens));

    return {std::move(tokenizer), std::move(model), std::move(token_ids)};
}

// ── §16.3 Thinking budget comparison ─────────────────────────────────────────

static void section_budget(ModernGPT& model,
                            const BPETokenizer& tokenizer,
                            const std::vector<int32_t>& token_ids)
{
    std::cout << "\n=== §16.3  Thinking Budget Comparison ===\n";

    int32_t bos_id = tokenizer.token_id("<think>");
    int32_t eos_id = tokenizer.token_id("</think>");

    std::vector<int32_t> prompt(token_ids.begin(), token_ids.begin() + 3);

    for (int64_t budget : {4LL, 16LL, 48LL}) {
        ThinkingConfig cfg;
        cfg.think_bos_id      = bos_id;
        cfg.think_eos_id      = eos_id;
        cfg.max_think_tokens  = budget;
        cfg.max_answer_tokens = 6;
        cfg.inner_cfg.mode    = SamplingMode::Greedy;
        cfg.answer_cfg.mode   = SamplingMode::Greedy;

        std::mt19937 rng(0);
        auto result = generate_with_thinking(model, prompt, cfg, rng);

        std::cout << std::format("  budget={:>2}  think_len={}  complete={}  answer=\"{}\"\n",
                                  budget,
                                  result.thinking_tokens.size(),
                                  result.thinking_complete ? "true " : "false",
                                  tokenizer.decode(result.answer_tokens));
    }

    std::cout << "  Note: with random/undertrained weights thinking content is\n"
                 "  not meaningful, but the budget mechanism is demonstrated.\n";
}

// ── §16.4 Self-consistency voting ─────────────────────────────────────────────

static void section_self_consistency(ModernGPT& model,
                                      const BPETokenizer& tokenizer,
                                      const std::vector<int32_t>& token_ids)
{
    std::cout << "\n=== §16.4  Self-Consistency Voting ===\n";

    int32_t bos_id = tokenizer.token_id("<think>");
    int32_t eos_id = tokenizer.token_id("</think>");

    std::vector<int32_t> prompt(token_ids.begin(), token_ids.begin() + 3);

    ThinkingConfig cfg;
    cfg.think_bos_id      = bos_id;
    cfg.think_eos_id      = eos_id;
    cfg.max_think_tokens  = 8;
    cfg.max_answer_tokens = 4;
    cfg.inner_cfg.mode        = SamplingMode::Temperature;
    cfg.inner_cfg.temperature = 0.8f;
    cfg.answer_cfg.mode       = SamplingMode::Greedy;

    // Show individual runs using the same advancing rng that think_self_consistency will use.
    std::cout << "  Individual first-answer tokens (7 runs):\n";
    std::mt19937 rng_show(99);
    for (int i = 0; i < 7; ++i) {
        auto result = generate_with_thinking(model, prompt, cfg, rng_show);
        int32_t first_tok = result.answer_tokens.empty() ? -1 : result.answer_tokens[0];
        std::string decoded = result.answer_tokens.empty() ? "(empty)" :
            tokenizer.decode(std::vector<int32_t>{first_tok});
        std::cout << std::format("    run {:>1}: token {:>3}  \"{}\"\n",
                                  i, first_tok, decoded);
    }

    // Voted result — identical seed so runs match what was displayed above.
    std::mt19937 rng_vote(99);
    int32_t voted = think_self_consistency(model, prompt, cfg, 7, rng_vote);
    std::cout << std::format("  Voted result: token {} \"{}\"\n",
                              voted,
                              tokenizer.decode(std::vector<int32_t>{voted}));
}

// ── §16.5 Architecture preview ────────────────────────────────────────────────

static void section_architecture_preview() {
    std::cout << "\n=== §16.5  Architecture Preview ===\n";
    std::cout <<
        "  Next chapter: LoopedGPT — the same transformer block run K times\n"
        "  per forward pass, expressing thinking as computation depth rather\n"
        "  than token generation. Each cycle refines the hidden state; K\n"
        "  becomes a reasoning budget analogous to max_think_tokens here.\n";
}

// ── main ──────────────────────────────────────────────────────────────────────

int main() {
    std::cout << "Chapter 16 — Thinking Tokens: Chain-of-Thought Generation\n";
    std::cout << std::string(60, '=') << '\n';

    section_special_tokens();

    auto [tokenizer, model, token_ids] = []{
        auto r = section_train();
        return std::tuple{std::move(r.tokenizer),
                          std::move(r.model),
                          std::move(r.token_ids)};
    }();

    section_budget(model, tokenizer, token_ids);
    section_self_consistency(model, tokenizer, token_ids);
    section_architecture_preview();

    std::cout << "\nDone.\n";
    return 0;
}
