#include "sub0llm/nn/rlhf.hpp"
#include "sub0llm/nn/modern_gpt.hpp"
#include "sub0llm/nn/optimizer.hpp"
#include "sub0llm/autograd/ops.hpp"
#include "sub0llm/core/tensor.hpp"

#include <format>
#include <iostream>
#include <vector>
#include <cmath>

using namespace sub0llm;
using namespace sub0llm::nn;
using namespace sub0llm::autograd;

// ── helpers ───────────────────────────────────────────────────────────────────

static Tensor make_ids(std::initializer_list<int32_t> toks) {
    Tensor t({static_cast<int64_t>(toks.size())}, DType::Int32);
    auto sp = t.data_as<int32_t>();
    std::size_t i = 0;
    for (auto v : toks) sp[i++] = v;
    return t;
}

static int64_t count_params(const std::vector<Variable*>& params) {
    int64_t n = 0;
    for (auto* v : params) n += static_cast<int64_t>(v->data().numel());
    return n;
}

// ── §20.1  Architecture ───────────────────────────────────────────────────────

static void section_architecture() {
    std::cout << "\n=== §20.1  Reward Model Architecture ===\n";

    std::cout << "  Standard GPT:    Linear(D → vocab_size) output head (logits)\n";
    std::cout << "  RewardModel:     Linear(D → 1)          output head (scalar reward)\n";
    std::cout << "  Architecture:    Embedding → N×ModernTransformerBlock → RMSNorm → Linear(D,1)\n\n";

    const int64_t V = 20, D = 32;
    ModernGPT    gpt(V, D, 2, 1, 2, 0, 0, 42);
    RewardModel  rm (V, D, 2, 1, 2, 0,    42);

    int64_t n_gpt = count_params(const_cast<ModernGPT&>(gpt).parameters());
    int64_t n_rm  = count_params(rm.parameters());

    std::cout << std::format("  vocab_size={} embed_dim={} n_heads=2 n_kv_heads=1 n_layers=2\n",
                              V, D);
    std::cout << std::format("  GPT     params: {:6d}  (LM head = D×V = {}×{} = {})\n",
                              n_gpt, D, V, D * V);
    std::cout << std::format("  Reward  params: {:6d}  (reward head = D×1 = {})\n",
                              n_rm, D);
    // GPT weight-ties the LM head with tok_emb (0 extra params).
    // RewardModel has an independent reward_head Linear(D,1) = D+1 extra params.
    std::cout << "  GPT weight-ties LM head (0 extra params); "
                 "RewardModel adds Linear(D,1) = " << (n_rm - n_gpt) << " extra params.\n";

    std::cout << "\n  forward(ids) → (T, 1)  per-token scalar rewards\n";
    std::cout << "  score(ids)   → {1}     last-position scalar (sequence-level signal)\n";

    auto ids = make_ids({3, 7, 11, 2});
    auto rewards = rm.forward(ids);
    float r = rm.score(ids).data().data_as<float>()[0];
    std::cout << std::format("  Example: score([3,7,11,2]) = {:.4f}\n", r);
}

// ── §20.2  Preference Training ────────────────────────────────────────────────

static void section_preference_training() {
    std::cout << "\n=== §20.2  Preference Training (Bradley-Terry) ===\n";

    std::cout << "  L = -log σ(r_chosen - r_rejected)\n";
    std::cout << "  Trains reward model so that r_chosen > r_rejected.\n\n";

    // Toy preference setup: even tokens are "good", odd tokens are "bad"
    auto chosen_ids   = make_ids({0, 2, 4, 6, 8, 10, 12, 14});  // even
    auto rejected_ids = make_ids({1, 3, 5, 7, 9, 11, 13, 15});  // odd

    RewardModel rm(20, 32, 2, 1, 2, 0, 42);
    Adam adam(rm.parameters(), 3e-3f);

    float r0_c = rm.score(chosen_ids).data().data_as<float>()[0];
    float r0_r = rm.score(rejected_ids).data().data_as<float>()[0];
    std::cout << std::format("  Before training:  r_chosen={:+.4f}  r_rejected={:+.4f}  "
                              "margin={:+.4f}\n", r0_c, r0_r, r0_c - r0_r);

    for (int step = 0; step < 200; ++step) {
        adam.zero_grad();
        auto r_c = rm.score(chosen_ids);
        auto r_r = rm.score(rejected_ids);
        auto loss = reward_preference_loss(r_c, r_r);
        loss.backward();
        adam.step();
        if (step == 0 || (step + 1) % 50 == 0) {
            float lv = loss.data().data_as<float>()[0];
            float rc = r_c.data().data_as<float>()[0];
            float rr = r_r.data().data_as<float>()[0];
            std::cout << std::format("  step {:3d}  loss={:.4f}  r_chosen={:+.4f}  "
                                      "r_rejected={:+.4f}\n",
                                      step + 1, lv, rc, rr);
        }
    }

    float rf_c = rm.score(chosen_ids).data().data_as<float>()[0];
    float rf_r = rm.score(rejected_ids).data().data_as<float>()[0];
    std::cout << std::format("\n  After training:   r_chosen={:+.4f}  r_rejected={:+.4f}  "
                              "margin={:+.4f} {}\n",
                              rf_c, rf_r, rf_c - rf_r,
                              rf_c > rf_r ? "✓" : "✗");
}

// ── §20.3  REINFORCE Policy Gradient ─────────────────────────────────────────

static void section_reinforce() {
    std::cout << "\n=== §20.3  REINFORCE Policy Gradient ===\n";

    std::cout << "  L = reward × CE(logits, token_ids)\n";
    std::cout << "  ∇L = -reward × Σ_t ∇ log π(token_t | context_t)\n";
    std::cout << "  reward > 0 → policy encouraged to produce those tokens.\n\n";

    const int64_t V = 20;
    auto target_ids = make_ids({0, 2, 4, 6, 8, 10, 12, 14});  // even tokens

    // Train policy with a fixed positive reward = 1.0
    ModernGPT policy(V, 32, 2, 1, 2, 0, 0, 99);
    Adam adam(policy.parameters(), 3e-3f);

    float ce_first = -1.f, ce_last = -1.f;
    for (int step = 0; step < 100; ++step) {
        adam.zero_grad();
        auto logits = policy.forward(target_ids);
        // CE alone: to show that reward>0 + CE is equivalent to supervised finetuning
        auto loss = reinforce_loss(logits, target_ids, 1.0f);
        loss.backward();
        adam.step();
        // Track raw CE for interpretability
        float ce = cross_entropy(policy.forward(target_ids), target_ids)
                       .data().data_as<float>()[0];
        if (step == 0) ce_first = ce;
        ce_last = ce;
        if (step == 0 || (step + 1) % 25 == 0)
            std::cout << std::format("  step {:3d}  CE={:.4f}\n", step + 1, ce);
    }
    std::cout << std::format("  CE: {:.4f} → {:.4f}  (policy fitting target tokens)\n",
                              ce_first, ce_last);

    std::cout << "\n  Negative reward discourages tokens:\n";
    ModernGPT policy2(V, 32, 2, 1, 2, 0, 0, 77);
    Adam adam2(policy2.parameters(), 3e-3f);
    float ce0 = cross_entropy(policy2.forward(target_ids), target_ids)
                    .data().data_as<float>()[0];
    for (int step = 0; step < 50; ++step) {
        adam2.zero_grad();
        auto logits = policy2.forward(target_ids);
        auto loss = reinforce_loss(logits, target_ids, -1.0f);
        loss.backward();
        adam2.step();
    }
    float ce1 = cross_entropy(policy2.forward(target_ids), target_ids)
                    .data().data_as<float>()[0];
    std::cout << std::format("  CE with reward=-1.0: {:.4f} → {:.4f}  "
                              "(policy pushing away from target tokens)\n",
                              ce0, ce1);
}

// ── §20.4  KL Penalty ────────────────────────────────────────────────────────

static void section_kl_penalty() {
    std::cout << "\n=== §20.4  KL Divergence Penalty ===\n";

    std::cout << "  KL(π_new ‖ π_ref) = (1/T) Σ_t Σ_v π_new(v|t) log(π_new/π_ref)\n";
    std::cout << "  β × KL penalises deviation from the reference (frozen) policy.\n\n";

    const int64_t V = 20;
    auto ids = make_ids({0, 2, 4, 6, 8, 10, 12, 14});

    ModernGPT policy(V, 32, 2, 1, 2, 0, 0, 55);
    // Reference = initial policy (frozen snapshot)
    Tensor ref_logits = policy.forward(ids).data();  // (T, V) constant

    float kl0 = kl_penalty(policy.forward(ids), ref_logits)
                    .data().data_as<float>()[0];
    std::cout << std::format("  KL before any training: {:.6f}\n\n", kl0);

    // Without KL penalty: pure REINFORCE drives policy away from reference
    ModernGPT pol_no_kl(V, 32, 2, 1, 2, 0, 0, 55);
    Adam adam_no_kl(pol_no_kl.parameters(), 3e-3f);
    for (int step = 0; step < 80; ++step) {
        adam_no_kl.zero_grad();
        auto logits = pol_no_kl.forward(ids);
        auto loss = reinforce_loss(logits, ids, 1.0f);
        loss.backward();
        adam_no_kl.step();
    }
    float kl_no_kl = kl_penalty(pol_no_kl.forward(ids), ref_logits)
                         .data().data_as<float>()[0];
    std::cout << std::format("  β=0.0 (no KL penalty):  KL after 80 steps = {:.4f}  "
                              "(policy drifts freely)\n", kl_no_kl);

    // With KL penalty (β=0.5): REINFORCE + KL constraint
    ModernGPT pol_kl(V, 32, 2, 1, 2, 0, 0, 55);
    Adam adam_kl(pol_kl.parameters(), 3e-3f);
    for (int step = 0; step < 80; ++step) {
        adam_kl.zero_grad();
        auto logits = pol_kl.forward(ids);
        auto l_rf  = reinforce_loss(logits, ids, 1.0f);
        auto l_kl  = kl_penalty(logits, ref_logits);
        auto loss  = add(l_rf, scale(l_kl, 0.5f));
        loss.backward();
        adam_kl.step();
    }
    float kl_with_kl = kl_penalty(pol_kl.forward(ids), ref_logits)
                           .data().data_as<float>()[0];
    std::cout << std::format("  β=0.5 (with KL penalty): KL after 80 steps = {:.4f}  "
                              "(policy stays closer to reference)\n", kl_with_kl);

    std::cout << std::format("\n  KL reduction factor: {:.1f}×  "
                              "(β=0.5 constrains drift vs β=0.0)\n",
                              kl_no_kl / std::max(kl_with_kl, 1e-6f));
}

// ── §20.5  Full RLHF Loop ─────────────────────────────────────────────────────

static void section_full_rlhf() {
    std::cout << "\n=== §20.5  Full RLHF Loop ===\n";

    std::cout << "  Phase 1: Train reward model on preference pairs.\n";
    std::cout << "  Phase 2: Optimise policy: L = REINFORCE(reward) + β × KL(π_new ‖ π_ref)\n\n";

    const int64_t V = 20;
    auto chosen_ids   = make_ids({0, 2, 4, 6, 8, 10, 12, 14});
    auto rejected_ids = make_ids({1, 3, 5, 7, 9, 11, 13, 15});

    // ── Phase 1: Reward model ──────────────────────────────────────────────────
    RewardModel rm(V, 32, 2, 1, 2, 0, 42);
    Adam adam_rm(rm.parameters(), 3e-3f);

    std::cout << "  --- Phase 1: Reward Model Training (200 steps) ---\n";
    for (int step = 0; step < 200; ++step) {
        adam_rm.zero_grad();
        auto r_c = rm.score(chosen_ids);
        auto r_r = rm.score(rejected_ids);
        auto loss = reward_preference_loss(r_c, r_r);
        loss.backward();
        adam_rm.step();
    }
    float rm_c = rm.score(chosen_ids).data().data_as<float>()[0];
    float rm_r = rm.score(rejected_ids).data().data_as<float>()[0];
    std::cout << std::format("  r_chosen={:+.3f}  r_rejected={:+.3f}  "
                              "margin={:+.3f} {}\n\n",
                              rm_c, rm_r, rm_c - rm_r, rm_c > rm_r ? "✓" : "✗");

    // ── Phase 2: Policy optimisation ──────────────────────────────────────────
    ModernGPT policy(V, 32, 2, 1, 2, 0, 0, 77);
    // Snapshot of initial policy = reference (frozen)
    Tensor ref_logits = policy.forward(chosen_ids).data();

    Adam adam_pol(policy.parameters(), 1e-3f);
    const float beta = 0.1f;

    std::cout << "  --- Phase 2: Policy Optimisation (β=0.1, 150 steps) ---\n";

    float ce_before = cross_entropy(policy.forward(chosen_ids), chosen_ids)
                          .data().data_as<float>()[0];

    for (int step = 0; step < 150; ++step) {
        adam_pol.zero_grad();

        // Get reward from frozen reward model
        float reward = rm.score(chosen_ids).data().data_as<float>()[0];

        // Compute policy loss
        auto logits   = policy.forward(chosen_ids);
        auto l_rf     = reinforce_loss(logits, chosen_ids, reward);
        auto l_kl     = kl_penalty(logits, ref_logits);
        auto loss     = add(l_rf, scale(l_kl, beta));
        loss.backward();
        adam_pol.step();

        if (step == 0 || (step + 1) % 50 == 0) {
            float reward_v = rm.score(chosen_ids).data().data_as<float>()[0];
            float kl_v     = kl_penalty(policy.forward(chosen_ids), ref_logits)
                                 .data().data_as<float>()[0];
            float ce_v     = cross_entropy(policy.forward(chosen_ids), chosen_ids)
                                 .data().data_as<float>()[0];
            std::cout << std::format("  step {:3d}  reward={:+.3f}  KL={:.4f}  "
                                      "CE(chosen)={:.4f}\n",
                                      step + 1, reward_v, kl_v, ce_v);
        }
    }

    float ce_after = cross_entropy(policy.forward(chosen_ids), chosen_ids)
                         .data().data_as<float>()[0];
    float kl_final = kl_penalty(policy.forward(chosen_ids), ref_logits)
                         .data().data_as<float>()[0];

    std::cout << std::format("\n  CE on chosen sequence:  {:.4f} → {:.4f}  "
                              "(policy fits chosen tokens)\n",
                              ce_before, ce_after);
    std::cout << std::format("  Final KL from reference: {:.4f}  "
                              "(β={:.1f} limits policy drift)\n",
                              kl_final, beta);

    std::cout << "\n  Key ideas:\n";
    std::cout << "    1. Reward model is trained once on preference pairs (offline).\n";
    std::cout << "    2. Policy is updated to maximise reward while staying near reference.\n";
    std::cout << "    3. KL penalty prevents reward hacking / distribution collapse.\n";
    std::cout << "    4. β controls the tradeoff: small β → more reward, large β → safer.\n";
    std::cout << "    5. PPO (Schulman et al., 2017) adds clipped surrogate for stability.\n";
    std::cout << "       This implementation uses KL-penalised REINFORCE — simpler and\n";
    std::cout << "       sufficient to demonstrate the core RLHF mechanism.\n";
}

// ── main ──────────────────────────────────────────────────────────────────────

int main() {
    std::cout << "Chapter 20 — RLHF: Reward Modeling and Policy Optimization\n";
    std::cout << "============================================================\n";

    section_architecture();
    section_preference_training();
    section_reinforce();
    section_kl_penalty();
    section_full_rlhf();

    std::cout << "\nDone.\n";
}
