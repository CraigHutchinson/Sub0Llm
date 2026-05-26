#include "sub0llm/nn/distillation.hpp"
#include "sub0llm/nn/modern_gpt.hpp"
#include "sub0llm/nn/optimizer.hpp"
#include "sub0llm/nn/sampler.hpp"
#include "sub0llm/tokenizer/bpe.hpp"
#include "sub0llm/autograd/ops.hpp"
#include "sub0llm/core/tensor.hpp"
#include <cmath>
#include <format>
#include <iostream>
#include <numeric>
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

// ── Training step helpers ─────────────────────────────────────────────────────

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
    auto logits      = model.forward(input);
    auto logits_trunc = narrow(logits, 0, n_pred);
    auto loss        = cross_entropy(logits_trunc, targets);
    loss.backward();
    (void)clip_grad_norm(model.parameters(), 1.0f);
    adam.step();
    return loss.data().data_as<float>()[0];
}

static float train_step_distill(ModernGPT& student, ModernGPT& teacher,
                                 Adam& adam,
                                 const std::vector<int32_t>& tokens,
                                 int64_t start, int64_t seq_len,
                                 float alpha, float temp)
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

    // Get teacher logits (inference only, no grad)
    Tensor teacher_logits_full = teacher.forward(input).data();
    // Narrow to first n_pred rows
    Variable teacher_logits_var(teacher_logits_full, false);
    Tensor teacher_logits = narrow(teacher_logits_var, 0, n_pred).data();

    adam.zero_grad();
    auto student_logits_full  = student.forward(input);
    auto student_logits_trunc = narrow(student_logits_full, 0, n_pred);

    auto loss = distillation_loss(student_logits_trunc, teacher_logits, targets, alpha, temp);
    loss.backward();
    (void)clip_grad_norm(student.parameters(), 1.0f);
    adam.step();
    return loss.data().data_as<float>()[0];
}

// ── §15.1 Corpus & Tokenizer ─────────────────────────────────────────────────

static std::pair<BPETokenizer, std::vector<int32_t>> section_corpus()
{
    std::cout << "\n=== §15.1  Corpus & Tokenizer ===\n";

    std::vector<std::string> corpus_vec = {corpus_text};
    auto tokenizer = BPETokenizer::train(corpus_vec, /*vocab_size=*/96);

    auto token_ids = tokenizer.encode(corpus_text);

    std::cout << std::format("  vocab_size    : {}\n", tokenizer.vocab_size());
    std::cout << std::format("  num_merges    : {}\n", tokenizer.num_merges());
    std::cout << std::format("  encoded_length: {}\n", token_ids.size());

    std::cout << "  first 20 token IDs: [";
    for (std::size_t i = 0; i < 20 && i < token_ids.size(); ++i) {
        if (i > 0) std::cout << ", ";
        std::cout << token_ids[i];
    }
    std::cout << "]\n";

    // Round-trip check
    const std::size_t first20_len = std::min<std::size_t>(20, token_ids.size());
    std::vector<int32_t> first20(token_ids.begin(),
                                  token_ids.begin() + static_cast<std::ptrdiff_t>(first20_len));
    std::string decoded = tokenizer.decode(first20);
    std::cout << std::format("  decoded first 20: \"{}\"\n", decoded);

    return {std::move(tokenizer), std::move(token_ids)};
}

// ── §15.2 Teacher Training ────────────────────────────────────────────────────

static ModernGPT section_teacher(const BPETokenizer& tokenizer,
                                  const std::vector<int32_t>& token_ids)
{
    std::cout << "\n=== §15.2  Teacher Training ===\n";

    const int64_t V       = static_cast<int64_t>(tokenizer.vocab_size());
    const int64_t seq_len = 16;
    const int     steps   = 300;

    ModernGPT teacher(V, 64, 2, 1, 4, 0, 0, /*seed=*/42);
    Adam adam(teacher.parameters(), 3e-3f);

    const int64_t n_tokens = static_cast<int64_t>(token_ids.size());
    if (n_tokens < seq_len + 2)
        throw std::runtime_error(std::format(
            "section_teacher: corpus too short ({} tokens), need >= {}", n_tokens, seq_len + 2));
    std::mt19937 rng(1);
    std::uniform_int_distribution<int64_t> dist(0, n_tokens - seq_len - 2);

    float first_loss = -1.f, last_loss = -1.f;

    for (int step = 0; step < steps; ++step) {
        const int64_t start = dist(rng);
        float loss_val = train_step_lm(teacher, adam, token_ids, start, seq_len);

        if (step == 0) first_loss = loss_val;
        last_loss = loss_val;

        if (step == 0 || step % 50 == 0 || step == steps - 1)
            std::cout << std::format("  step {:>3}  loss={:.4f}\n", step, loss_val);
    }

    std::cout << std::format("  Teacher loss: {:.4f} → {:.4f}  ({})\n",
                              first_loss, last_loss,
                              last_loss < first_loss * 0.8f ? "converging" : "learning");
    return teacher;
}

// ── §15.4 Knowledge Distillation ─────────────────────────────────────────────

static ModernGPT section_distillation(const BPETokenizer& tokenizer,
                                       const std::vector<int32_t>& token_ids,
                                       ModernGPT& teacher,
                                       float student_base_final_loss)
{
    std::cout << "\n=== §15.4  Knowledge Distillation ===\n";

    const int64_t V       = static_cast<int64_t>(tokenizer.vocab_size());
    const int64_t seq_len = 16;
    const int     steps   = 300;
    const float   alpha   = 0.5f;
    const float   temp    = 2.0f;

    ModernGPT student_dist(V, 32, 2, 1, 2, 0, 0, /*seed=*/7);
    Adam adam(student_dist.parameters(), 5e-3f);

    const int64_t n_tokens = static_cast<int64_t>(token_ids.size());
    if (n_tokens < seq_len + 2)
        throw std::runtime_error(std::format(
            "section_distillation: corpus too short ({} tokens), need >= {}", n_tokens, seq_len + 2));
    std::mt19937 rng(1);
    std::uniform_int_distribution<int64_t> dist_rng(0, n_tokens - seq_len - 2);

    float first_loss = -1.f, last_loss = -1.f;

    for (int step = 0; step < steps; ++step) {
        const int64_t start = dist_rng(rng);
        float loss_val = train_step_distill(student_dist, teacher, adam,
                                             token_ids, start, seq_len,
                                             alpha, temp);

        if (step == 0) first_loss = loss_val;
        last_loss = loss_val;

        if (step == 0 || step % 50 == 0 || step == steps - 1)
            std::cout << std::format("  step {:>3}  kd_loss={:.4f}\n", step, loss_val);
    }

    std::cout << std::format("  Student (distill) KD loss: {:.4f} → {:.4f}\n",
                              first_loss, last_loss);
    std::cout << std::format("  Note: KD loss = alpha*CE + (1-alpha)*T^2*soft_CE\n"
                              "        with alpha={:.1f}, T={:.1f} — T^2={:.0f}x scale factor\n"
                              "        makes KD loss larger than hard CE by design.\n",
                              alpha, temp, temp * temp);
    std::cout << std::format("  Student (base) final hard CE: {:.4f}  "
                              "Student (distill) trained on KD objective.\n",
                              student_base_final_loss);
    std::cout << "  Distillation transfers teacher soft-target knowledge to the student.\n";
    return student_dist;
}

// ── §15.5 Inference Comparison ────────────────────────────────────────────────

static void section_inference(const BPETokenizer& tokenizer,
                               const std::vector<int32_t>& token_ids,
                               ModernGPT& teacher,
                               ModernGPT& student_base,
                               ModernGPT& student_dist)
{
    std::cout << "\n=== §15.5  Inference Comparison ===\n";

    // Prompt: first 4 tokens
    std::vector<int32_t> prompt(token_ids.begin(), token_ids.begin() + 4);
    const int64_t max_new = 20;
    SamplingConfig cfg;  // defaults to Greedy

    std::cout << std::format("  Prompt ({} tokens): \"{}\"\n",
                              prompt.size(), tokenizer.decode(prompt));

    auto print_gen = [&](std::string_view label, ModernGPT& model) {
        std::mt19937 rng(0);
        auto out = generate(model, prompt, max_new, cfg, rng);
        std::vector<int32_t> gen_tokens(out.begin() + static_cast<std::ptrdiff_t>(prompt.size()), out.end());
        std::cout << std::format("  {:>14}: \"{}\"\n",
                                  label, tokenizer.decode(gen_tokens));
    };

    print_gen("teacher", teacher);
    print_gen("student_base", student_base);
    print_gen("student_dist", student_dist);

    std::cout << "  Note: all models overfit the small corpus rather than learning\n"
                 "  general language — distillation helps the student match teacher\n"
                 "  token distributions even on this toy dataset.\n";
}

// ── main ─────────────────────────────────────────────────────────────────────

int main() {
    std::cout << "Chapter 15 — Real Training and Knowledge Distillation\n";
    std::cout << std::string(60, '=') << '\n';

    auto [tokenizer, token_ids] = section_corpus();
    auto teacher     = section_teacher(tokenizer, token_ids);

    float base_final_loss = 0.0f;
    auto student_base = [&]() {
        // Capture final loss during student_base training
        const int64_t V       = static_cast<int64_t>(tokenizer.vocab_size());
        const int64_t seq_len = 16;
        const int     steps   = 300;

        std::cout << "\n=== §15.3  Student Baseline (hard labels only) ===\n";

        ModernGPT model(V, 32, 2, 1, 2, 0, 0, /*seed=*/7);
        Adam adam(model.parameters(), 5e-3f);

        const int64_t n_tokens = static_cast<int64_t>(token_ids.size());
        if (n_tokens < seq_len + 2)
            throw std::runtime_error(std::format(
                "student baseline: corpus too short ({} tokens), need >= {}", n_tokens, seq_len + 2));
        std::mt19937 rng(1);
        std::uniform_int_distribution<int64_t> dist(0, n_tokens - seq_len - 2);

        float first_loss = -1.f, last_loss = -1.f;
        for (int step = 0; step < steps; ++step) {
            const int64_t start = dist(rng);
            float loss_val = train_step_lm(model, adam, token_ids, start, seq_len);
            if (step == 0) first_loss = loss_val;
            last_loss = loss_val;
            if (step == 0 || step % 50 == 0 || step == steps - 1)
                std::cout << std::format("  step {:>3}  loss={:.4f}\n", step, loss_val);
        }
        std::cout << std::format("  Student (base) loss: {:.4f} → {:.4f}\n",
                                  first_loss, last_loss);
        base_final_loss = last_loss;
        return model;
    }();

    auto student_dist = section_distillation(tokenizer, token_ids, teacher, base_final_loss);
    section_inference(tokenizer, token_ids, teacher, student_base, student_dist);

    std::cout << "\nDone.\n";
    return 0;
}
