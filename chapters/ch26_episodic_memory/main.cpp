// Chapter 26 — Episodic Memory: LM-Accelerated Short-Term Learning
//
// ─────────────────────────────────────────────────────────────────────────────
// THE PROBLEM WITH "CONTEXT AS MEMORY"
// ─────────────────────────────────────────────────────────────────────────────
//
// Chapter 25 gave us a clean story: the KV cache holds everything the model
// has "read", and attention reaches back across it whenever a token needs to
// look up an earlier fact.
//
// That story has three hard limits:
//
//   Limit 1 — O(n) memory
//     At 70B-class scale (80 layers, 8 KV heads, 128-dim heads, bf16),
//     each 1 000 tokens costs ≈ 200 MB.  1 M tokens = 200 GB on the GPU.
//     Even Llama 4 Scout's 10 M-token context needs a dedicated cluster
//     just to hold the cache.
//
//   Limit 2 — attention cannot truly learn
//     The KV cache is a lookup table, not a write-enabled memory.  Reading
//     the same passage twice does not change anything in the model.  There
//     is no consolidation: every session starts from scratch.
//
//   Limit 3 — forgetting is total
//     When you reset the cache, everything in it is gone — including every
//     proper noun, preference, or local fact the model discovered during the
//     conversation.  Humans call that retrograde amnesia if it happens to
//     them; we call it "starting a new session."
//
// The root question this chapter asks:
//
//   Can a language model USE ITS OWN LINGUISTIC KNOWLEDGE to accelerate
//   learning new information into a compact, persistent memory partition —
//   the way a human uses prior knowledge to understand and retain new facts?
//
// ─────────────────────────────────────────────────────────────────────────────
// ELI5: WHY DOES UNDERSTANDING HELP MEMORY?
// ─────────────────────────────────────────────────────────────────────────────
//
// Imagine you need to remember two phone numbers:
//
//   A: 867-5309
//   B: 1-4-1-4-2-1-3-5-6-2-3-7-3-1
//
// Number A is easy: you hear the Jenny song, and "867-5309" is anchored to
// something you already know.  Number B is just digits — you must memorise
// each one from scratch.
//
// The difference is SCHEMA: a pre-existing structure in long-term memory that
// the new information can attach to.  Bartlett (1932) showed this
// experimentally: people remember stories far better when the events fit a
// familiar script (going to a restaurant) than when they are random sequences.
//
// For a language model:
//   - Pre-training weights = long-term semantic memory = the schema network
//   - New context (a user's conversation) = new information to retain
//   - The model already knows which facts are common, which are surprising,
//     and which are cross-linked to other facts — because it trained on the web.
//
// Therefore: using the model's own comprehension to GUIDE where and how hard
// to write new information should be dramatically cheaper than training from
// scratch, for the same reason that remembering "867-5309" is cheaper than
// memorising a random digit string.
//
// ─────────────────────────────────────────────────────────────────────────────
// THREE TIERS OF MEMORY (BIOLOGICAL)
// ─────────────────────────────────────────────────────────────────────────────
//
// Neuroscience has identified a clean three-tier hierarchy:
//
//   ┌──────────────────────────────────────────────────────────────────┐
//   │ Tier          │ Brain region        │ Duration    │ Capacity     │
//   ├──────────────────────────────────────────────────────────────────┤
//   │ Working       │ Prefrontal cortex   │ Seconds     │ ~7 items     │
//   │ Episodic      │ Hippocampus         │ Hours-years │ Large        │
//   │ Semantic      │ Neocortex           │ Lifetime    │ Enormous     │
//   └──────────────────────────────────────────────────────────────────┘
//
// KEY BIOLOGICAL INSIGHT — Complementary Learning Systems (McClelland 1995):
//   - Hippocampus = fast episodic encoder.  Learns new episodes in one shot
//     using sparse, separated representations that avoid interference.
//   - Neocortex = slow statistical learner.  Integrates knowledge over
//     thousands of examples, building generalised semantic representations.
//   - Sleep consolidation: hippocampal "replays" gradually distil episodic
//     memories into neocortical weights — making them permanent and efficient.
//
// The catastrophic-forgetting problem in neural networks is EXACTLY the
// absence of a hippocampus: the neocortex (main weights) has to serve both
// roles, and it cannot do either well.
//
// ─────────────────────────────────────────────────────────────────────────────
// THE THREE TIERS MAPPED TO A LANGUAGE MODEL
// ─────────────────────────────────────────────────────────────────────────────
//
//   ┌─────────────────────────────────────────────────────────────────────────┐
//   │ Tier      │ Biological       │ AI analog               │ Update freq    │
//   ├─────────────────────────────────────────────────────────────────────────┤
//   │ Working   │ Prefrontal ctx   │ KV cache (Ch25)         │ Per token      │
//   │ Episodic  │ Hippocampus      │ Fast-weight partition   │ Per session    │
//   │ Semantic  │ Neocortex        │ Pre-trained weights     │ Pre-training   │
//   └─────────────────────────────────────────────────────────────────────────┘
//
// The missing tier in today's LLMs is Episodic.
// The KV cache is working memory.  Pre-trained weights are semantic memory.
// There is no hippocampus: nothing that persists beyond the session and
// below the cost of full fine-tuning.
//
// ─────────────────────────────────────────────────────────────────────────────
// PRIOR ART — FOUR LINEAGES
// ─────────────────────────────────────────────────────────────────────────────
//
// LINEAGE 1: Fast Weights (Schmidhuber 1987 / Hinton & Plaut 1987)
//   The original proposal: neural networks have TWO timescales of weights:
//   "slow weights" updated by backpropagation (pre-training), and "fast
//   weights" updated in real time from the current input.  Fast weights
//   implement a content-addressable memory without a separate data structure.
//   Problem: no scalable mechanism for WHAT and WHERE to write — every input
//   updates every weight equally.
//
// LINEAGE 2: Test-Time Training (TTT layers, Sun et al. 2024)
//   Replace the standard attention layer with a "TTT layer" whose HIDDEN
//   STATE IS THE WEIGHTS OF A SMALL MLP.  The MLP is updated by gradient
//   descent during the forward pass itself, treating the current sequence
//   as a mini-training set.
//   Key property: the MLP compresses the sequence — O(1) memory per token
//   once written, vs O(n) for KV cache.
//   Problem: training signal is self-supervised (predict masked tokens),
//   which treats all tokens equally — no comprehension-aware importance.
//
// LINEAGE 3: Titans (Behrouz et al., Google DeepMind, Dec 2024)
//   Extends TTT with a learned "surprise" gate: the memory is written
//   proportionally to how surprising the current token is (high gradient
//   norm ≈ high surprise ≈ high write weight).
//   Architecture: Neural Long-Term Memory (NLM) module runs alongside
//   attention.  Three variants tested:
//     MAC: Memory as Context — NLM output appended to KV cache
//     MAG: Memory as Gate  — NLM output gates attention
//     MAL: Memory as Layer — alternating attention / NLM layers
//   Titans beats Mamba2, DeltaNet, and standard Transformers on long-context
//   tasks with far less memory than the KV cache.
//   Problem: "surprise" = high gradient norm, which is still not the same
//   as semantic importance determined by comprehension.
//
// LINEAGE 4: Targeted Weight Edits (ROME 2022 / MEMIT 2022)
//   ROME (Rank-One Model Editing): factual knowledge lives in the MLP
//   layers at specific positions.  A rank-one weight update can insert
//   "The Eiffel Tower is in Rome" without corrupting other facts.
//   MEMIT generalises to thousands of simultaneous edits.
//   Key insight: the layer that "stores" a fact is identifiable by causal
//   intervention (ablate it → the fact disappears).  The write target is
//   not random — it is predictable from the semantics of the input.
//   Problem: ROME/MEMIT require knowing the fact in advance and identifying
//   the right layer manually.  Not usable for online learning from unstructured
//   text.
//
// LINEAGE 5: Hypernetwork Weight Generation (SHINE / Text-to-LoRA, 2025)
//   A "hypernetwork" takes a task description as input and outputs the
//   weights of a LoRA adapter that specialises the base model for that task —
//   in a single forward pass, no gradient descent.
//   SHINE (Single-step Hypernetwork for Implicit Neural Encoding): applied
//   to neural radiance fields but the principle transfers directly.
//   Text-to-LoRA: fine-tune a hypernetwork so that "generate an adapter for
//   medical QA" produces near-fine-tuning quality without any training.
//   Key property: the hypernetwork has absorbed the geometry of the adapter
//   space during its own training, so one forward pass substitutes for
//   hundreds of gradient steps.
//
// ─────────────────────────────────────────────────────────────────────────────
// SCHEMA THEORY AND LEVELS-OF-PROCESSING
// ─────────────────────────────────────────────────────────────────────────────
//
// Schema Theory (Bartlett 1932, Rumelhart 1980):
//   Prior knowledge organises itself into "schemas" — structured frameworks
//   (scripts, frames, prototypes).  New information that fits a schema is
//   learned 10× faster because only the DELTA needs to be stored.
//   "The Roman Emperor Julius Caesar was assassinated in 44 BC" — most of
//   the tokens activate existing schemas (Roman Empire, Caesar, assassination).
//   The novel part is the year and the specific event.
//
//   For an LLM: the pre-trained weight space IS the schema network.
//   The model can recognise which parts of a new document are surprising
//   (high perplexity) vs. expected (low perplexity).  High-perplexity spans
//   are where the delta that needs to be stored is largest.
//
// Levels of Processing (Craik & Lockhart 1972):
//   Shallow processing: noticing the font of a word → poor retention.
//   Deep processing: connecting the word to its meaning, context, and related
//   concepts → far better retention.
//   Elaborative rehearsal: actively connecting new information to existing
//   knowledge (e.g., "The assassination year 44 BC is two years before the
//   end of the Roman Republic in 42 BC") → best retention.
//
//   Computational translation:
//     - One forward pass over a passage = shallow processing.
//     - A thinking loop that generates summaries, questions, and connections
//       = deep processing / elaborative rehearsal.
//     - The loop is NOT about producing output — it is about activating the
//       right internal representations before writing.
//
// ─────────────────────────────────────────────────────────────────────────────
// THE PROPOSED ARCHITECTURE: LM-ACCELERATED EPISODIC MEMORY
// ─────────────────────────────────────────────────────────────────────────────
//
//   ┌──────────────────────────────────────────────────────────────────────┐
//   │                          INPUT TEXT                                  │
//   └────────────────────────────┬─────────────────────────────────────────┘
//                                │
//                                ▼
//   ┌──────────────────────────────────────────────────────────────────────┐
//   │  PHASE 1: COMPREHENSION PASS                                         │
//   │                                                                      │
//   │  Standard forward pass over the document.  Per-token outputs:        │
//   │    • Perplexity (surprisal): which spans are novel?                  │
//   │    • Attention entropy: which tokens are "key facts" (low entropy    │
//   │      = many tokens attending to this one token)?                     │
//   │    • Layer activations: where in the model does information "live"?  │
//   │      (ROME showed: specific MLP layers for specific fact types)      │
//   │                                                                      │
//   │  Output: importance map over the document.                           │
//   └────────────────────────────┬─────────────────────────────────────────┘
//                                │
//                                ▼
//   ┌──────────────────────────────────────────────────────────────────────┐
//   │  PHASE 2: THINKING LOOP (elaborative rehearsal)                      │
//   │                                                                      │
//   │  Run N thinking steps on the high-importance spans:                  │
//   │    • Generate a summary of the span (forces semantic compression)    │
//   │    • Generate a question whose answer is in the span                 │
//   │    • Generate a cross-link: "This is related to [existing knowledge] │
//   │      because..."                                                     │
//   │                                                                      │
//   │  This is not output generation — thinking tokens are discarded.      │
//   │  The goal is to activate deep representations before writing.        │
//   │                                                                      │
//   │  Analogous to: sleep consolidation replay / elaborative rehearsal.   │
//   └────────────────────────────┬─────────────────────────────────────────┘
//                                │
//                                ▼
//   ┌──────────────────────────────────────────────────────────────────────┐
//   │  PHASE 3: TARGETED EPISODIC WRITE                                    │
//   │                                                                      │
//   │  Use the internal activations from the thinking loop to write into   │
//   │  a DEDICATED EPISODIC MEMORY PARTITION:                              │
//   │                                                                      │
//   │    WHERE to write:                                                   │
//   │      Layers with highest activation magnitude for the target span    │
//   │      (identified by comprehension pass — same ROME principle).       │
//   │                                                                      │
//   │    WHAT signal:                                                       │
//   │      The hidden-state vector from the deepest thinking step.         │
//   │      This vector encodes the semantically-processed version of the   │
//   │      information — not the raw token sequence.                       │
//   │                                                                      │
//   │    HOW HARD to write (write gate):                                   │
//   │      Proportional to surprisal (as in Titans) MULTIPLIED BY          │
//   │      semantic coherence score (does this fit the schema?):           │
//   │        • High surprisal + high coherence = important novel fact      │
//   │        • High surprisal + low coherence = noise / out-of-distribution│
//   │        • Low surprisal + high coherence = redundant (don't write)    │
//   │                                                                      │
//   │    EPISODIC PARTITION OPTIONS (see next section):                    │
//   │      Option A: LoRA adapters per layer (low-rank, reversible)        │
//   │      Option B: Small MLP alongside each Transformer block (Titans)   │
//   │      Option C: Separate "hippocampus" model (smallest possible)      │
//   └──────────────────────────────────────────────────────────────────────┘
//
// ─────────────────────────────────────────────────────────────────────────────
// WHAT IS NOVEL
// ─────────────────────────────────────────────────────────────────────────────
//
// Each prior-art lineage does PART of this:
//
//   Fast weights: the write mechanism exists, but the signal is raw input.
//   TTT layers:   the write mechanism exists, gradient signal, but uniform.
//   Titans:       adds a surprise gate (better than uniform), but still raw.
//   ROME/MEMIT:   knows WHERE to write and has semantic targeting, but
//                 requires manual fact specification.
//   Hypernetworks: generates weights from task description, but task is
//                  given externally, not derived from comprehension.
//
// The proposed architecture is the first to use ALL THREE of:
//   1. COMPREHENSION (LM's own activations) to locate write targets
//   2. ELABORATIVE REHEARSAL (thinking loop) to deepen representations
//   3. GATED WRITE (surprisal × schema-fit) to prioritise novel facts
//
// The key hypothesis: a model with rich prior knowledge can learn a new
// document in O(thinking_steps × surprising_spans) gradient steps rather
// than O(document_length) — because it only needs to store the DELTA from
// what it already knows.
//
// ─────────────────────────────────────────────────────────────────────────────
// PATH A — ONLINE LORA (IMPLEMENTABLE WITH EXISTING SUB0LLM INFRASTRUCTURE)
// ─────────────────────────────────────────────────────────────────────────────
//
// We already have:
//   • lora.hpp (Ch12): LoRA adapter layers per weight matrix
//   • thinking.hpp (Ch16): ThinkingConfig, generate_with_thinking
//   • modern_gpt.hpp (Ch25): forward_one(), KV cache
//
// The Online LoRA episodic memory adds:
//
//   struct EpisodicMemory {
//       // One LoRA adapter per layer, per weight matrix in {W_Q, W_K, W_V, W_O}
//       std::vector<LoRALayer> adapters;   // size = n_layers * 4
//       float learning_rate    = 1e-3f;
//       int   max_think_steps  = 3;
//       float surprise_threshold = 2.0f;   // perplexity threshold for write
//   };
//
//   // Comprehension pass: forward over chunk, collect per-token surprisal
//   std::vector<float> comprehension_pass(
//       const ModernGPT& model, const std::vector<int32_t>& tokens);
//
//   // Thinking loop: run N self-supervised steps on surprising spans
//   void elaborative_rehearsal(
//       ModernGPT& model, EpisodicMemory& mem,
//       const std::vector<int32_t>& span_tokens, int steps);
//
//   // Episodic write: one gradient step on LoRA adapters only
//   void episodic_write(
//       ModernGPT& model, EpisodicMemory& mem,
//       const std::vector<int32_t>& tokens, float write_strength);
//
//   // Session API
//   void encode_document(
//       ModernGPT& model, EpisodicMemory& mem,
//       const std::vector<int32_t>& document);
//
//   void reset_episodic(EpisodicMemory& mem);  // clear between unrelated sessions
//
// Only the LoRA adapter weights are updated — the base model is frozen.
// This gives:
//   • Reversibility: reset_episodic() undoes the entire session's learning
//   • Isolation: base-model capabilities are not degraded
//   • Efficiency: LoRA rank r=8 adds ~0.1% parameters per layer
//
// TESTABLE CLAIM (benchmark design):
//   1. Take a long document (e.g., 4 000 tokens of a Wikipedia article)
//   2. Ask a factual question whose answer is at token ~3 500
//   3. Compare:
//      • Baseline: KV cache (full context, exact retrieval)
//      • Compressed: sliding-window (context 512), no episodic memory
//      • Ours: sliding-window + Online LoRA episodic memory
//   Hypothesis: Ours ≈ Baseline accuracy, << Baseline memory cost
//
// ─────────────────────────────────────────────────────────────────────────────
// PATH B — TITANS MEMORY MODULE (FUTURE CH27)
// ─────────────────────────────────────────────────────────────────────────────
//
// Path A (Online LoRA) is the minimal implementation: reuse Ch12 infrastructure,
// write to a small adapter, reset per session.  It validates the core hypothesis.
//
// Path B goes deeper: a dedicated Neural Long-Term Memory module alongside each
// Transformer block, trained end-to-end with the surprise gate.
//
//   Titans MAC variant (Memory as Context):
//     h_mem = NLM(x, mem_state)        // NLM = small MLP with fast-weight state
//     x'    = Attention([x; h_mem])    // memory tokens appended to KV context
//     mem_state = update(mem_state, x, surprise(x))
//
//   The NLM update is a gradient step on the NLM's OWN weights (TTT-style),
//   where the gradient signal is next-token prediction loss on x.
//   The surprise gate scales the learning rate by gradient norm.
//
//   With LM-guided writing (Ch26 extension of Titans):
//     surprise(x) → surprise(x) * schema_fit(x)
//     where schema_fit = f(layer_activation_pattern from comprehension pass)
//
// This is Ch27.  It requires:
//   • Adding NLM modules to ModernGPT (structural change)
//   • A training procedure that learns the gate alongside the base model
//   • Evaluation on long-document QA benchmarks
//
// ─────────────────────────────────────────────────────────────────────────────
// MEMORY BUDGET COMPARISON
// ─────────────────────────────────────────────────────────────────────────────
//
// For a 70B-class model (80 layers, 8 KV heads, 128 head_dim, bf16):
//
//   Approach                  Memory / 1M tokens      Forgetting?
//   ─────────────────────────────────────────────────────────────
//   KV Cache (full)           200 GB                  Yes (reset)
//   KV Cache (sliding W=4K)   800 MB                  Yes (sliding)
//   Online LoRA (r=8)         ~50 MB (adapter only)   Reversible
//   Titans NLM (d_mem=256)    ~500 MB (NLM weights)   Persistent
//   ROME fact edit            < 1 MB (one rank-1 Δ)   Permanent
//
// Online LoRA is 4 000× cheaper than a full 1M KV cache and does not
// vanish when the window slides past the relevant tokens.
//
// ─────────────────────────────────────────────────────────────────────────────
// OPEN QUESTIONS
// ─────────────────────────────────────────────────────────────────────────────
//
//   Q1: Does the thinking loop actually help?
//     The Levels-of-Processing hypothesis predicts yes.  But thinking steps
//     are expensive.  Is 1 thinking step enough, or do we need 5?
//     Testable: ablate thinking_steps = 0, 1, 2, 4, 8 on QA accuracy.
//
//   Q2: What is the right granularity for "surprising spans"?
//     Sentence-level? Paragraph-level? Top-k highest-perplexity tokens?
//     This affects both quality and cost.
//
//   Q3: Does LoRA have enough capacity?
//     LoRA rank r=8 gives rank-8 updates to each weight matrix.  If the
//     new document requires more than ~8 independent "directions" of change,
//     we will need higher rank or a different adapter family.
//
//   Q4: Cross-session interference?
//     If we accumulate LoRA updates across 100 sessions without resetting,
//     do earlier memories get overwritten?  This is the catastrophic-
//     forgetting problem at the adapter level.  One mitigation: maintain a
//     small adapter bank (one per recent session), retrieved by semantic
//     similarity to the current query.
//
//   Q5: Is the model's own perplexity a reliable importance signal?
//     A hallucinating model has miscalibrated perplexity.  An important
//     fact can have low perplexity if it follows a predictable template.
//     Attention entropy may be a better importance signal — tokens attended
//     to by many other tokens are structurally important.
//
// ─────────────────────────────────────────────────────────────────────────────
// SUMMARY
// ─────────────────────────────────────────────────────────────────────────────
//
//   • Context (Ch25) is working memory: O(n), sessions start from scratch.
//   • Biology has three tiers; today's LLMs have two (working + semantic).
//   • The missing tier is episodic: fast, persistent, reversible, cheap.
//   • Prior art: fast weights (1987), TTT layers (2024), Titans (Dec 2024),
//     ROME (2022), SHINE/Text-to-LoRA (2025).
//   • The novelty: use the LM's OWN comprehension to guide writes —
//     WHERE to write (layer targeting), WHAT signal (thinking-loop
//     activations), HOW HARD (surprisal × schema-fit).
//   • Path A (Ch26): Online LoRA on top of existing Ch12/Ch25 infrastructure.
//   • Path B (Ch27): Titans-style NLM with LM-guided surprise gate.
//   • Core hypothesis: a model with prior knowledge learns a new document
//     in O(surprising_spans) steps, not O(document_length).
//
// Implementation of Path A begins in the next section of this chapter.
// ─────────────────────────────────────────────────────────────────────────────

#include <cstdio>
#include <cmath>
#include <vector>
#include <string>

// ──────────────────────────────────────────────────────────────────────────────
// Utility: print section header
// ──────────────────────────────────────────────────────────────────────────────
static void section(const char* title) {
    std::printf("\n══ %s ══\n\n", title);
}

// ──────────────────────────────────────────────────────────────────────────────
// Demo 1: Memory cost table
// ──────────────────────────────────────────────────────────────────────────────
static void demo_memory_budget() {
    section("Memory Budget: KV Cache vs. Episodic Memory");

    // 70B-class model parameters
    const int    n_layers   = 80;
    const int    n_kv_heads = 8;
    const int    head_dim   = 128;
    const double bytes_per_elem = 2.0;   // bfloat16

    std::printf("Model: 70B-class (80L / 8KV / 128d / bf16)\n\n");
    std::printf("%-32s  %12s  %s\n", "Approach", "Memory (MB)", "Persistent?");
    std::printf("%-32s  %12s  %s\n", std::string(32, '-').c_str(),
                std::string(12, '-').c_str(), std::string(11, '-').c_str());

    // Full KV cache
    auto kv_cost_mb = [&](int64_t seq_len) -> double {
        // K and V, n_layers, n_kv_heads, seq_len, head_dim
        return 2.0 * n_layers * n_kv_heads * static_cast<double>(seq_len) * head_dim * bytes_per_elem / (1024.0 * 1024.0);
    };

    std::printf("%-32s  %12.0f  %s\n", "KV Cache (1M tokens)",
                kv_cost_mb(1'000'000), "No (session only)");
    std::printf("%-32s  %12.0f  %s\n", "KV Cache (128K tokens)",
                kv_cost_mb(128'000), "No (session only)");
    std::printf("%-32s  %12.0f  %s\n", "KV Cache (4K sliding window)",
                kv_cost_mb(4'096), "No (slides away)");

    // Online LoRA: r=8 per {W_Q, W_K, W_V, W_O} per layer
    const int    lora_rank   = 8;
    const int    embed_dim   = 8192;   // typical for 70B
    // Each LoRA = A (embed_dim x r) + B (r x embed_dim), 4 matrices per layer
    double lora_mb = 4.0 * n_layers * 2.0 * embed_dim * lora_rank * bytes_per_elem / (1024.0 * 1024.0);
    std::printf("%-32s  %12.0f  %s\n", "Online LoRA (r=8)", lora_mb, "Yes (reversible)");

    // Titans NLM: small MLP alongside each block, d=256
    const int    nlm_dim     = 256;
    // NLM: two linear layers (embed_dim -> nlm_dim -> embed_dim) per layer
    double nlm_mb = n_layers * 2.0 * embed_dim * nlm_dim * bytes_per_elem / (1024.0 * 1024.0);
    std::printf("%-32s  %12.0f  %s\n", "Titans NLM (d=256)", nlm_mb, "Yes (persistent)");

    // ROME single fact edit: rank-1 update to one weight matrix
    double rome_mb = embed_dim * embed_dim * bytes_per_elem / (1024.0 * 1024.0);
    // rank-1: just two vectors of size embed_dim
    rome_mb = 2.0 * embed_dim * bytes_per_elem / (1024.0 * 1024.0 * 1024.0) * 1024.0;
    std::printf("%-32s  %12.4f  %s\n", "ROME rank-1 edit (1 fact)", rome_mb, "Yes (permanent)");

    std::printf("\nOnline LoRA is %.0f× cheaper than a 1M KV cache\n",
                kv_cost_mb(1'000'000) / lora_mb);
    std::printf("and %.0f× cheaper than a 128K KV cache\n",
                kv_cost_mb(128'000) / lora_mb);
}

// ──────────────────────────────────────────────────────────────────────────────
// Demo 2: Schema acceleration — learning rate scaling with prior knowledge
// ──────────────────────────────────────────────────────────────────────────────
static void demo_schema_acceleration() {
    section("Schema Acceleration: Learning Rate vs. Prior Knowledge");

    std::printf("Hypothesis: a model with schema-aligned prior knowledge\n");
    std::printf("needs fewer gradient steps to reach target loss.\n\n");

    // Rough empirical model based on knowledge distillation literature:
    //   - Knowledge distillation (Hinton 2015): ~10-20% of steps needed
    //     vs. training from scratch.
    //   - ROME/MEMIT: one gradient step for a single fact.
    //   - Schema networks (Amos et al. 2018, DeepMind): 10x faster than DQN
    //     on tasks that fit the schema.
    //
    // We model the steps needed as: steps = base_steps * (1 - schema_fit)^alpha
    // where schema_fit ∈ [0, 1] and alpha ≈ 2 empirically.

    const int   base_steps  = 1000;
    const double alpha       = 2.0;

    std::printf("%-20s  %-20s  %-20s  %s\n",
                "Schema fit", "Steps needed", "Speedup", "Example");
    std::printf("%-20s  %-20s  %-20s  %s\n",
                std::string(20, '-').c_str(), std::string(20, '-').c_str(),
                std::string(20, '-').c_str(), std::string(20, '-').c_str());

    struct Case { double fit; const char* example; };
    std::vector<Case> cases = {
        {0.0,  "Random bytes"},
        {0.3,  "Foreign language text"},
        {0.6,  "Familiar domain, novel facts"},
        {0.8,  "Known domain, fill-in details"},
        {0.9,  "Known fact updated (e.g. date)"},
        {0.98, "ROME-style targeted edit"},
    };
    for (auto& c : cases) {
        int    steps   = static_cast<int>(base_steps * std::pow(1.0 - c.fit, alpha));
        double speedup = static_cast<double>(base_steps) / std::max(steps, 1);
        std::printf("%-20.2f  %-20d  %-20.1f  %s\n", c.fit, steps, speedup, c.example);
    }

    std::printf("\nConclusion: most of the benefit comes from schema fit > 0.6.\n");
    std::printf("A pre-trained LLM already has schema fit ~0.6-0.8 for most\n");
    std::printf("English factual text, so episodic writes should be cheap.\n");
}

// ──────────────────────────────────────────────────────────────────────────────
// Demo 3: Importance signal comparison
// ──────────────────────────────────────────────────────────────────────────────
static void demo_importance_signals() {
    section("Importance Signals: What Makes a Span Worth Writing?");

    std::printf("Three candidate signals, composable into a write gate:\n\n");

    struct Signal {
        const char* name;
        const char* definition;
        const char* strength;
        const char* weakness;
    };

    std::vector<Signal> signals = {
        {
            "Surprisal (perplexity)",
            "−log P(token | context)",
            "Detects novel facts, unusual names, numbers",
            "High for noise/typos; low for template-fit important facts"
        },
        {
            "Attention entropy",
            "H(attn weights) = −Σ a_i log a_i",
            "Low entropy = 'attention sink' = structurally key token",
            "Not directly tied to semantic importance; prompt-position biased"
        },
        {
            "Layer activation magnitude",
            "||h_l(token)||_2 at layer l",
            "Identifies which layer 'owns' the information (ROME insight)",
            "Needs calibration; varies by model size and architecture"
        },
    };

    for (auto& s : signals) {
        std::printf("Signal:    %s\n", s.name);
        std::printf("Def:       %s\n", s.definition);
        std::printf("Strength:  %s\n", s.strength);
        std::printf("Weakness:  %s\n\n", s.weakness);
    }

    std::printf("Proposed composite gate (Ch26 Path A):\n");
    std::printf("  write_strength(span) =\n");
    std::printf("    surprisal(span)           // novel enough to write?\n");
    std::printf("    × (1 - attention_entropy) // structurally important?\n");
    std::printf("    × schema_fit(span)        // fits existing knowledge?\n");
    std::printf("\n");
    std::printf("  Clamped to [0, 1] and thresholded at surprise_threshold.\n");
    std::printf("  Only spans above the threshold trigger an episodic write.\n");
}

// ──────────────────────────────────────────────────────────────────────────────
// Demo 4: Thinking-loop elaboration benefit model
// ──────────────────────────────────────────────────────────────────────────────
static void demo_thinking_benefit() {
    section("Thinking Loop: Elaboration Depth vs. Retention");

    std::printf("Craik & Lockhart (1972): deeper processing → better retention.\n");
    std::printf("Each thinking step activates cross-links that ground the write signal.\n\n");

    // Model: retention gain from elaborative rehearsal.
    // One thinking step ≈ generating a question/summary about the span.
    // Empirically: each step adds diminishing returns (log-scale).
    // Beyond ~4 steps: minimal additional gain.

    std::printf("%-16s  %-20s  %-20s\n", "Think steps", "Relative retention", "Cost (forward passes)");
    std::printf("%-16s  %-20s  %-20s\n",
                std::string(16, '-').c_str(), std::string(20, '-').c_str(),
                std::string(20, '-').c_str());

    const double base_retention = 1.0;
    for (int steps = 0; steps <= 6; ++steps) {
        // Diminishing-returns model: retention(k) = 1 + 0.8 * log2(1 + k)
        double retention = base_retention + 0.8 * std::log2(1.0 + steps);
        int    cost      = 1 + steps;  // comprehension pass + k thinking passes
        std::printf("%-16d  %-20.2f  %-20d\n", steps, retention, cost);
    }

    std::printf("\nRecommended default: 2-3 thinking steps (good retention / cost tradeoff).\n");
}

// ──────────────────────────────────────────────────────────────────────────────
// Main
// ──────────────────────────────────────────────────────────────────────────────
int main() {
    std::printf("Chapter 26 — Episodic Memory: LM-Accelerated Short-Term Learning\n");
    std::printf("================================================================\n");
    std::printf("\n");
    std::printf("This chapter introduces the EPISODIC MEMORY tier — the missing\n");
    std::printf("component between the KV-cache working memory (Ch25) and the\n");
    std::printf("pre-trained semantic weights.\n");
    std::printf("\n");
    std::printf("Core insight:\n");
    std::printf("  A language model with rich prior knowledge can learn a new\n");
    std::printf("  document in far fewer gradient steps than training from scratch,\n");
    std::printf("  because it only needs to store the DELTA from what it already knows.\n");
    std::printf("  The model's own comprehension (activations, perplexity, attention)\n");
    std::printf("  tells us WHERE to write, WHAT to write, and HOW HARD to write.\n");
    std::printf("\n");

    demo_memory_budget();
    demo_schema_acceleration();
    demo_importance_signals();
    demo_thinking_benefit();

    std::printf("\n");
    section("Roadmap");
    std::printf("Ch25: KV-cache working memory (complete)\n");
    std::printf("Ch26: Online LoRA episodic memory — Path A (this chapter)\n");
    std::printf("        • encode_document(): comprehension pass + elaborative rehearsal\n");
    std::printf("        • episodic_write(): targeted LoRA adapter update\n");
    std::printf("        • Benchmark: long-doc QA vs. sliding-window baseline\n");
    std::printf("Ch27: Titans-style Neural Long-Term Memory — Path B\n");
    std::printf("        • NLM module alongside each Transformer block\n");
    std::printf("        • LM-guided surprise gate (surprisal × schema_fit)\n");
    std::printf("        • Persistent across sessions; distil to semantic weights\n");
    std::printf("\n");

    return 0;
}
