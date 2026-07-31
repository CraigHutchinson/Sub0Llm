// sub0/tutorspike.hpp -- the MASTERY SURFACE: per-document training-progress state (docs/TUTOR.md).
//
// SPIKE. Read-only by design at this stage: the surface is built, checkpointed and rendered, and it
// changes NOTHING about what training does. That ordering is deliberate and is TUTOR.md's own -- a
// feedback controller whose state you cannot visualise is very hard to debug, so the instrument has to
// be trustworthy before any controller exists. It is also useful on its own: it answers the corpus
// COVERAGE question docs/PLATEAU_DESIGN.md 4a raises and which nothing in this project measures.
//
// WHAT IT RECORDS, AND WHY EACH FIELD IS THERE
//
// Per document: visits, tokens trained, accumulated APPLIED LEARNING, the latest NELBO, and the velocity
// derived from them. The one that is easy to get wrong is applied learning, and getting it wrong breaks
// the whole scheme, so it is worth restating here (TUTOR.md "the normalization trap"):
//
//     velocity_i  =  -delta NELBO_i  /  delta (sum of effective_lr * tokens)_i
//
// The denominator must be applied learning, NOT visits and NOT steps. Otherwise a down-weighted entry
// receives less learning, its NELBO stops moving, it LOOKS mastered, and it is down-weighted further --
// a self-reinforcing false-mastery loop in which an entry is driven to zero weight purely by having been
// given a low weight, with nothing in the signal revealing it. That is a controller confounding its own
// actuator with its measurement. Normalising by applied learning makes the reading invariant to the
// actuator, which is the property that makes the loop safe to close later.
//
// FLOAT, NOT DOUBLE, AND THAT IS NOT A COMPROMISE
//
// The per-window loss this is fed from is float-limited: the backends accumulate the batch scalar in
// float32, and the per-window readout agrees with it to ~6e-9 relative (measured -- see
// docs/TUTOR_SPIKE.md stage 0). Storing losses in double would record digits the measurement does not
// have, at twice the size of the one structure that actually becomes large at scale (~1.6 GB at
// fineweb's ~40M documents, where aggregation or sampling becomes necessary regardless).
//
// VELOCITY IS ONLY MEANINGFUL ABOVE THE DRIFT FLOOR
//
// An entry's NELBO is recorded as of its last visit. Between visits the model moves underneath it,
// pulled by every other document trained since, so a measured delta confounds this entry's own learning
// with drift from everything else -- and at a long gap the drift term dominates and can even be
// positive. DriftProbe measures that floor directly: a set of documents that are NEVER trained, scored
// at the same cadence, whose delta is therefore interference ALONE with no own-gradient term. A velocity
// smaller than the floor is not a reading. See docs/TUTOR_SPIKE.md for the full argument.
//
// THE DRIFT TERM IS NOT ONLY NOISE -- IT IS THE CONFLICT/REINFORCEMENT SIGNAL
//
// The between-visit change is contamination when you want this entry's own learning, but taken on its
// own it is a DIRECT measurement of how the rest of the corpus acts on this document. An entry is not
// trained between its own visits, by definition, so whatever moved it came from everything else:
//
//   * moved DOWN more than the corpus-wide floor -> other documents taught this one. Reinforcement,
//     redundancy, shared structure. It is being learned for free, and may deserve LESS training.
//   * moved UP  -> conflict. Something else is competing for the same capacity and is displacing it.
//     That is where training more is actually indicated, and it is invisible to any level-based rule.
//
// This is the same measurement read two ways: the corpus-wide MEAN magnitude is the noise floor, and
// each entry's DEVIATION from it is that entry's relationship to the rest of the training set. The
// second reading may be the more valuable product of the whole exercise, because nothing else in this
// project can see corpus structure at all.
//
// Separating the two needs one extra reading, and it is the one already planned: a re-score immediately
// after an entry's own update (docs/TUTOR_SPIKE.md "the back-to-back reading"). With it the split is
// complete and each half is separately observable --
//
//     own learning   =  nelbo_post(visit k)      -  nelbo(visit k)          [its own gradient]
//     transfer       =  nelbo(visit k+1)         -  nelbo_post(visit k)     [everything else's]
//
// -- because the loss recorded at a visit is the training forward's, taken BEFORE that step's update.
// Without the post reading the two terms stay summed and neither can be recovered.

#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace sub0::tutor {

// --- population manifest ----------------------------------------------------------------------
// Which population each document came from, as written by tools/sub0llm-tutorspike.cpp. The join key is
// the document ORDINAL: the configurator records document starts in the order it reads them, so document
// i in the corpus file is document i in corpus.tok. `total_docs` exists so that assumption can be
// CHECKED rather than trusted -- see doc_count_matches below.
struct Manifest {
    std::vector<std::string>   populations;   // population names, indexed by population id
    std::vector<std::uint8_t>  doc_pop;       // [total_docs] population id per document ordinal
    std::uint64_t              seed = 0;

    bool empty() const { return doc_pop.empty(); }
    std::size_t total_docs() const { return doc_pop.size(); }

    // The configurator's doc_starts carries ONE trailing phantom entry: scan_doc_boundaries pushes a
    // start for the corpus's final EOS, equal to the token count, which no window can ever resolve to.
    // So a correctly joined corpus has exactly total_docs + 1 starts. Anything else means documents were
    // dropped, merged or reordered between splice and tokenization, and every label after the first
    // discrepancy would be attached to the wrong document -- loud is the only acceptable behaviour.
    bool doc_count_matches(std::size_t doc_starts_size) const {
        return doc_starts_size == doc_pop.size() + 1;
    }
};

// Parse a manifest written by sub0llm-tutorspike. Returns false (and leaves `out` empty) on any
// malformed input. Defined in src/tutorspike.cpp -- simdjson types must not cross the DLL boundary, the
// same constraint registry.hpp documents for read_config_json.
[[nodiscard]] bool read_manifest(const std::string& path, Manifest& out);

// --- event stream -----------------------------------------------------------------------------
// One record per velocity or transfer update. The ledger keeps only an entry's CURRENT state, which
// cannot support the analysis the 10-epoch run showed is actually needed: velocity depends on TWO axes
// (the entry's own applied learning AND the model's global maturity) and the ledger collapses the
// second, so any cross-population comparison drawn from it is confounded. It also cannot show a
// quantity's history -- Entry::transfer being a running mean since step 0 made late readings a lifetime
// average that could not say what the model was doing now, and that defect was invisible until the
// numbers were compared across three reads.
//
// Emitting events instead makes every normalisation a post-hoc analysis decision rather than something
// baked into a 44-byte struct and impossible to revisit. Written at the eval cadence from a reserved
// buffer, so the per-step path still allocates nothing.
struct Event {
    std::uint32_t doc;
    std::uint32_t step;
    std::uint32_t visits;
    std::uint8_t  kind;            // 0 = velocity update, 1 = transfer update
    std::uint8_t  pop;             // population id, filled by the writer from the manifest
    std::int32_t  win_len;         // the window's trained length, an axis in its own right
    float         own_applied;     // this entry's accumulated applied learning
    float         global_applied;  // the corpus's -- the second axis the ledger cannot express
    float         nelbo;
    float         value;           // velocity (kind 0) or this interval's transfer (kind 1)
};

// --- the surface ------------------------------------------------------------------------------
// One document's state. 44 bytes; see the float rationale in the file header. The transfer fields are
// the larger half of the point -- see "the drift term is not only noise" above.
struct Entry {
    std::uint32_t visits       = 0;     // times a window from this document has been trained
    std::uint32_t tokens       = 0;     // trained positions accumulated (a document, not a corpus, count)
    float         applied      = 0.f;   // sum of effective_lr * tokens -- the velocity DENOMINATOR
    float         nelbo        = 0.f;   // most recent per-window loss (the training forward, PRE-update)
    float         nelbo_mark   = 0.f;   // nelbo at the last velocity update
    // Applied learning accumulated SINCE the last velocity update, accumulated forward and reset at each
    // mark -- deliberately not "applied at the last mark" with a subtraction at use. The subtraction is
    // a catastrophic cancellation waiting to happen: `applied` grows without bound (~1e6 by the end of
    // this run) while the threshold stays a few units, so the difference of two large nearly-equal
    // floats loses precision in proportion to how long the run has been going. The velocity denominator
    // would quietly degrade over a long run -- worst exactly where the readings matter most.
    float         applied_since = 0.f;
    float         velocity     = 0.f;   // -d(nelbo) / d(applied), 0 until two marks exist

    // Post-update reading from the back-to-back re-score (the re-score runs at a cadence, so most
    // visits do not have one). Its presence is what makes the own-learning / transfer split recoverable
    // at the NEXT visit -- so "is there one" must be a FLAG, not a magic value. 0.0 is a legitimate
    // loss: cross_entropy returns exactly 0 for a fully loss-masked window, which the per-window readout
    // faithfully reports, so a 0-sentinel would silently classify a real reading as an absent one.
    float         nelbo_post   = 0.f;
    std::uint8_t  flags        = 0;     // bit 0: nelbo_post is valid
    // Signed between-visit change per unit of GLOBAL applied learning, averaged over this entry's
    // measured intervals. NEGATIVE means the entry improved while it was not being trained
    // (reinforcement from the rest of the corpus); POSITIVE means it degraded (conflict). Normalised by
    // global rather than own applied learning because the quantity being attributed is everything
    // ELSE's training, not this entry's.
    float         transfer     = 0.f;
    std::uint32_t transfer_n   = 0;     // intervals folded into `transfer` (0 = no reading yet)
    float         global_mark  = 0.f;   // global applied learning as of this entry's last visit

    static constexpr std::uint8_t kHasPost = 1u;
    bool seen() const { return visits > 0; }
    bool has_post() const { return (flags & kHasPost) != 0; }
    bool has_transfer() const { return transfer_n > 0; }
};

// Velocity is recomputed only once an entry has accumulated this much NEW applied learning since its
// last mark. Recomputing on every visit would divide by a denominator so small that float cancellation
// in the numerator dominates the result -- the reading would be mostly noise, and would look like a
// wildly oscillating velocity rather than an absent one. Scale-derived rather than a fixed constant
// (this project's standing rule): it is a multiple of what a SINGLE typical visit applies, so it tracks
// the learning rate and the window width instead of being calibrated for one corpus.
inline constexpr float VELOCITY_MARK_MULTIPLE = 4.0f;

// --- cadences, as constants rather than CLI knobs ---------------------------------------------
// None of these is a decision a user makes per run (this project's "only add the surface actually
// consumed" rule), and each has a derivation rather than a preference behind it.

// Reserve every Nth document as a never-trained drift probe. At the spike's 36000 documents this is
// ~280 probes: enough that the mean absolute delta is a stable floor, few enough that excluding them
// costs ~0.8% of the corpus, and few enough that scoring them on the CPU at every eval stays close to
// the cost of the existing 128-window validation eval. The manifest interleaves
// populations in runs of 64, and the trainer LOGS the resulting per-population probe counts so a stride
// that accidentally samples one population is visible rather than silently biasing the floor.
inline constexpr std::size_t TUTOR_PROBE_STRIDE = 128;

// How often the post-update re-score runs. EVERY step -- and the original reasoning for 20 was wrong in
// a way only the recorded data revealed.
//
// The cadence was chosen against a cost of "a forward is ~1/3 of a forward+backward, so every step is
// ~33%". That would be true if the re-score covered the whole batch. It cannot: it is bounded by the
// unchunked [n*T, VOCAB] logits buffer to ~22 windows out of a ~448-window step (see
// GpuTrainer::rescore), so the real cost is 22/448 * 1/3 ~= 1.6% per step, not 33%.
//
// Paying 20x less than budgeted bought 20x less data, and it was the binding constraint on the ONE
// measurement the spike added: the first run produced 69471 velocity events but only 1149 transfer
// events -- against a theoretical ceiling of (1110/20) * 22 = 1210. Transfer was not sparse because
// documents rarely revisit; it was sparse because almost no visit carried a post reading. At every step
// the ceiling rises ~20x, which is the difference between a signal with error bars and one without.
inline constexpr long TUTOR_RESCORE_EVERY = 1;

// The per-document ledger. Flat and pre-sized: every update runs inside the training step, so there is
// no allocation on the hot path (AGENTS.md 1). At corpus scale this is the structure that has to become
// sparse or aggregated; at spike scale a flat array is right and honest.
class Surface {
public:
    void reset(std::size_t n_docs, float typical_visit_applied) {
        entries_.assign(n_docs, Entry{});
        global_applied_ = 0.0;
        // The mark threshold derives from what one visit typically applies, so it scales with LR and
        // window width. A caller that cannot estimate it passes 0 and gets every-visit updates.
        mark_threshold_ = VELOCITY_MARK_MULTIPLE * typical_visit_applied;
    }

    std::size_t size() const { return entries_.size(); }
    bool  empty() const { return entries_.empty(); }
    const Entry& at(std::size_t doc) const { return entries_[doc]; }
    std::span<const Entry> entries() const { return entries_; }

    // Total applied learning delivered to the WHOLE corpus so far. The transfer term is normalised by
    // this, because what it attributes is everything else's training, not the entry's own.
    double global_applied() const { return global_applied_; }
    void   add_global_applied(double v) { global_applied_ += v; }
    // The step every subsequent record() is attributed to. Set once per step by the trainer rather than
    // threaded through record()'s arguments, which would put a value that is constant across the whole
    // batch into a per-window parameter.
    void   set_step(long s) { step_ = static_cast<std::uint32_t>(s); }

    // Event capture (see Event). Reserved once and never grown, so the per-step path cannot allocate --
    // which means the buffer CAN fill. Overflow is COUNTED, never silently discarded: a diagnostic
    // stream with an unrecorded hole is worse than no stream, because the hole is invisible in the
    // analysis and looks like an absence of events rather than an absence of recording. The trainer
    // reports dropped_events() and the count rides the snapshot.
    void reserve_events(std::size_t n) { events_.clear(); events_.reserve(n); cap_ = n; dropped_ = 0; }
    std::vector<Event>& events() { return events_; }
    std::uint64_t dropped_events() const { return dropped_; }

    // Record one trained window. `lr` is the EFFECTIVE learning rate this step ran at and `tokens` the
    // window's trained length, so their product is the applied learning this visit delivered -- the
    // quantity velocity is normalised by. `nelbo` is the TRAINING forward's loss, i.e. taken BEFORE this
    // step's update; that ordering is what makes the transfer split below well-defined.
    // Allocation-free; safe to call once per window per step.
    void record(std::size_t doc, float nelbo, float lr, int tokens) {
        if (doc >= entries_.size()) return;            // a window outside the labelled corpus
        Entry& e = entries_[doc];

        // TRANSFER, measured before this visit's own numbers overwrite the previous ones. The entry was
        // not trained between its last visit and now, so any change since its post-update reading came
        // from the rest of the corpus. Requires a post reading to subtract from -- without one, the
        // interval's change still contains the previous visit's own learning and is not attributable.
        if (e.visits > 0 && e.has_post()) {
            const double d_global = global_applied_ - static_cast<double>(e.global_mark);
            if (d_global > 0.0) {
                const float per_unit = static_cast<float>(
                    (static_cast<double>(nelbo) - static_cast<double>(e.nelbo_post)) / d_global);
                // Running mean rather than the latest reading: a single interval is one noisy sample of
                // a relationship, and the quantity of interest is whether this entry is SYSTEMATICALLY
                // reinforced or conflicted by the rest of the corpus.
                ++e.transfer_n;
                e.transfer += (per_unit - e.transfer) / static_cast<float>(e.transfer_n);
                emit(Event{ static_cast<std::uint32_t>(doc), step_, e.visits, /*kind=*/1, /*pop=*/0,
                            tokens, e.applied, static_cast<float>(global_applied_), nelbo, per_unit });
            }
        }

        ++e.visits;
        e.tokens        += static_cast<std::uint32_t>(tokens);
        e.applied       += lr * static_cast<float>(tokens);
        e.applied_since += lr * static_cast<float>(tokens);
        e.nelbo    = nelbo;
        e.flags &= static_cast<std::uint8_t>(~Entry::kHasPost);   // consumed; set again by a re-score
        e.global_mark = static_cast<float>(global_applied_);
        if (e.visits == 1) {                            // first sighting: establish the baseline only
            e.nelbo_mark    = nelbo;
            e.applied_since = 0.f;
            return;
        }
        const float d_applied = e.applied_since;
        if (d_applied <= 0.f || d_applied < mark_threshold_) return;   // too small to divide by yet
        // Sign convention: FALLING nelbo is POSITIVE velocity (learning is happening). A negative
        // velocity therefore means the entry is regressing -- being forgotten -- which TUTOR.md's table
        // wants weighted UP, and which a level-based rule only ever catches by accident.
        e.velocity      = -(e.nelbo - e.nelbo_mark) / d_applied;
        e.nelbo_mark    = e.nelbo;
        e.applied_since = 0.f;
        emit(Event{ static_cast<std::uint32_t>(doc), step_, e.visits, /*kind=*/0, /*pop=*/0,
                    tokens, e.applied, static_cast<float>(global_applied_), e.nelbo, e.velocity });
    }

    // The back-to-back reading: this entry's loss re-scored immediately after the step that trained it.
    // Runs at a cadence, not every step (a forward is ~1/3 of a forward+backward, so every-step would
    // cost ~33%; every 20th step costs ~1.7%). Two things come out of it -- own learning for THIS visit,
    // available immediately, and the baseline the NEXT visit measures transfer against.
    void record_post(std::size_t doc, float nelbo_post) {
        if (doc >= entries_.size()) return;
        Entry& e = entries_[doc];
        if (e.visits == 0) return;                      // never trained: nothing to be "post" of
        e.nelbo_post = nelbo_post;
        e.flags |= Entry::kHasPost;
    }

    // Coverage: the fraction of documents visited at least once. Poisson sampling leaves ~1/e of the
    // corpus untouched after one epoch's worth of steps, so an unvisited entry's zero velocity means
    // "never asked", not "nothing to learn" -- reading this alongside the surface is what keeps the two
    // apart. It is also PLATEAU_DESIGN.md 4a's coverage number, on the training side.
    double coverage() const {
        const std::size_t denom = eligible_ ? eligible_ : entries_.size();
        if (denom == 0) return 0.0;
        std::size_t seen = 0;
        for (const Entry& e : entries_) if (e.seen()) ++seen;
        return static_cast<double>(seen) / static_cast<double>(denom);
    }

    // How many documents training can actually reach. NOT the document count: the validation split and
    // the drift probes are permanently out of reach, so dividing by the total makes 100% unattainable
    // and prints a shortfall that looks like a coverage failure. On this corpus that gap is 8.9% -- and
    // it coincidentally matched the old sampler's genuine Poisson shortfall almost exactly, which is
    // how a metric with the wrong denominator hides the very regression it is there to catch.
    void set_eligible(std::size_t n) { eligible_ = n; }
    std::size_t eligible() const { return eligible_ ? eligible_ : entries_.size(); }

    // Exact persistence. The surface is FEEDBACK STATE: TUTOR.md is explicit that if it is not restored
    // exactly, matched-arm A/Bs quietly stop being matched. Written as a SIDECAR next to the checkpoint
    // rather than inside it, deliberately -- adding a variable-length section to the fixed-size .ckpt
    // struct is the highest-blast-radius change in this codebase (AGENTS.md 3), and a spike has no
    // business stranding in-progress production runs to store state nothing in the mainline reads.
    [[nodiscard]] bool save(const std::string& path) const;
    [[nodiscard]] bool load(const std::string& path);

    // Snapshot for the live heat map: a self-contained JSON document the viewer polls. Written whole and
    // renamed into place by the implementation, so a reader never observes a half-written file.
    // Identity of the run that produced a recording. Written into every artefact, because a directory
    // of numbers with no record of what produced them is not a diagnostic -- it is a puzzle. Six months
    // on, "which seed / batch / corpus / code was this?" is unanswerable from the numbers themselves,
    // and answering it wrongly is worse than not having them. Deliberately a small POD of the axes that
    // change between arms, not a config dump.
    struct RunInfo {
        std::string  label;          // model directory, the human handle for the run
        std::string  manifest;       // which splice these ordinals refer to
        std::uint64_t manifest_seed = 0;
        unsigned     seed = 0;       // the arm axis a replicate varies
        int          batch = 0;
        float        peak_lr = 0.f;
        int          seq_len = 0;
        std::size_t  windows_per_epoch = 0;
        std::size_t  eligible_docs = 0;
    };

    [[nodiscard]] bool write_snapshot(const std::string& path, const Manifest& man, const RunInfo& run,
                                      long step, double drift_floor,
                                      std::span<const std::uint8_t> reachable) const;

    // Append the buffered events to a CSV, stamping each with its document's population. CSV rather
    // than JSON because this is the one output that is genuinely large and is read by analysis tools,
    // not by the viewer.
    [[nodiscard]] bool append_events(const std::string& path, const Manifest& man);
    // The run's identity as a sidecar JSON, written once. Kept separate from the event CSV so the CSV
    // stays a clean flat table, and separate from the snapshot so it survives even if a run dies before
    // its first eval.
    [[nodiscard]] bool write_run_info(const std::string& path, const RunInfo& run) const;

private:
    void emit(const Event& ev) {
        if (events_.size() < cap_) events_.push_back(ev);
        else ++dropped_;
    }

    std::vector<Entry> entries_;
    std::vector<Event> events_;
    std::size_t        cap_ = 0;
    std::uint64_t      dropped_ = 0;
    std::size_t        eligible_ = 0;
    std::uint32_t      step_ = 0;
    float              mark_threshold_ = 0.f;
    double             global_applied_ = 0.0;   // double: this one accumulates over the WHOLE run
};

// --- epoch permutation ------------------------------------------------------------------------
// An epoch is a SHUFFLED PERMUTATION of every window tiling the corpus, drawn without replacement --
// not independent random sampling. The distinction is not cosmetic; it removes three separate defects
// the 10-epoch run exposed, all of them properties of sampling WITH replacement:
//
//   * COVERAGE. Independent draws give Poisson coverage, so after ten epochs 8.9% of documents had
//     still never been sampled and read as zero velocity -- indistinguishable, in the surface, from
//     "nothing left to learn". A permutation visits every window exactly once per epoch, so coverage is
//     100% by construction and "never asked" stops being a state the surface can be in.
//   * VARIANCE. Under independent draws a document's visit count is Poisson-distributed around its
//     expectation, so two identical documents accumulate materially different applied learning purely
//     by luck. That variance lands directly in every per-document reading. A permutation makes visits
//     per epoch exact.
//   * THE PROBE REJECTION LOOP disappears. Probe documents are simply omitted when the plan is built,
//     rather than drawn and redrawn (11386 redraws in the previous run), so exclusion costs nothing and
//     is exact rather than statistical.
//
// The unit is the WINDOW, not the document. Tiling means every token is trained exactly once per epoch,
// and a long document therefore contributes proportionally more windows -- which is correct, not a
// confound: it has more content to learn. The confound the previous run hit was that visit counts were
// random AND length-driven at once; here the length term is exact and the random term is gone, and
// velocity's applied-learning denominator already handles length by design.
//
// Memory: one entry per window, ~12 bytes. At this corpus (~49k windows) that is ~0.6 MB. At fineweb
// scale (~180M windows) it is ~2 GB, so a production version would need a streaming/blocked shuffle
// rather than a materialised permutation -- noted, not solved, exactly as the ledger's own scaling is.
class EpochPlan {
public:
    struct Slot { std::uint64_t start; std::int32_t len; std::uint32_t doc; };

    // Build the tiling. `doc_starts` is the corpus document index (ascending, with the trailing phantom
    // boundary), `train_tok` the trainable prefix, `T` the window width. Documents for which `skip`
    // returns true are omitted entirely -- that is how drift probes are excluded.
    template <typename SkipFn>
    void build(std::span<const std::uint64_t> doc_starts, std::size_t train_tok, int T, SkipFn skip) {
        slots_.clear();
        cursor_ = 0;
        drawn_  = 0;
        if (doc_starts.size() < 2 || T < 2) return;
        const std::size_t Tsz = static_cast<std::size_t>(T);
        for (std::size_t d = 0; d + 1 < doc_starts.size(); ++d) {
            if (skip(d)) continue;
            const std::size_t beg = doc_starts[d];
            std::size_t end = doc_starts[d + 1];
            if (end > train_tok) end = train_tok;
            if (end <= beg + 1) continue;                 // no input/target pair to learn from
            // Tile [beg, end) in strides of T. The last window of a document is SHORT rather than
            // overlapping backwards: an overlapping tail would train those tokens twice per epoch and
            // quietly break the "every token exactly once" property this class exists to provide.
            // Every input position in [beg, end-1) is covered exactly once, with ONE documented
            // exception: a remainder of a SINGLE token is dropped, because a 1-token window has no
            // context to predict from and the eval path requires ctx >= 2. That costs at most one input
            // per document, and only when (doc_len - 1) mod T == 1. The alternative -- shifting the last
            // window back to full width -- would instead RE-train T-1 tokens, a much larger departure
            // from "exactly once" than dropping one.
            for (std::size_t s = beg; s + 1 < end; s += Tsz) {
                const std::size_t len = std::min(Tsz, end - s - 1);
                if (len < 2) break;                        // see above: a 1-token remainder is dropped
                slots_.push_back(Slot{ s, static_cast<std::int32_t>(len),
                                       static_cast<std::uint32_t>(d) });
            }
        }
    }

    bool empty() const { return slots_.empty(); }
    std::size_t size() const { return slots_.size(); }
    // Distinct documents the plan reaches -- the honest denominator for coverage (see
    // Surface::set_eligible). Documents in the validation tail or reserved as probes appear in no slot.
    // Visit each distinct document the plan reaches. Exists so consumers derive reachability FROM the
    // plan rather than re-implementing "val split or probe" independently -- the coverage denominator
    // bug came from exactly that kind of second, hand-maintained copy.
    template <typename Fn>
    void for_each_document(Fn fn) const {
        std::vector<std::uint32_t> seen;
        seen.reserve(slots_.size());
        for (const Slot& s : slots_) seen.push_back(s.doc);
        std::sort(seen.begin(), seen.end());
        seen.erase(std::unique(seen.begin(), seen.end()), seen.end());
        for (std::uint32_t d : seen) fn(d);
    }

    // Delegates, so there is exactly ONE implementation of "which documents does the plan reach". Two
    // copies of that question is how the coverage denominator came to disagree with reality. Not cheap
    // (it sorts the slot list) -- call it once and cache, which the trainer does.
    std::size_t documents() const {
        std::size_t n = 0;
        for_each_document([&](std::uint32_t) { ++n; });
        return n;
    }

    // Next window in the current permutation, reshuffling at the epoch boundary. The caller's RNG is
    // used so the whole schedule stays reproducible from the run seed, exactly like the sampler it
    // replaces.
    template <typename Rng>
    const Slot& next(Rng& rng) {
        // Shuffle at the START of every epoch, INCLUDING the first. Reshuffling only on wrap would leave
        // epoch 0 running in corpus order -- and since the splice interleaves populations in runs of 64,
        // that would train them in near-blocks for a whole epoch. A test pins this.
        if (cursor_ == 0 || cursor_ >= slots_.size()) {
            std::shuffle(slots_.begin(), slots_.end(), rng);
            cursor_ = 0;
        }
        ++drawn_;
        return slots_[cursor_++];
    }

    // COMPLETED epochs, derived from the draw count rather than counted at the wrap: after consuming
    // exactly size() slots one epoch IS complete, but no wrap has happened yet, so a wrap-counter reads
    // one too low precisely at the boundary a caller is most likely to inspect.
    std::uint64_t epoch() const { return slots_.empty() ? 0 : drawn_ / slots_.size(); }

private:
    std::vector<Slot> slots_;
    std::size_t       cursor_ = 0;
    std::uint64_t     drawn_  = 0;
};

// --- drift probe ------------------------------------------------------------------------------
// Documents that are NEVER trained, re-scored at the eval cadence. Their NELBO moves only because the
// model moved under them, so their mean absolute delta per unit of GLOBAL applied learning is the floor
// below which a trained entry's velocity carries no information about that entry.
//
// This is the same instrument PLATEAU_DESIGN.md 4a asks for as a held-out probe set, and it answers its
// [Q10] the strict way: probes are excluded from training by construction here, not merely unlikely to
// be drawn.
class DriftProbe {
public:
    // Reserve every `stride`-th document as a probe. Deterministic and stride-based rather than random
    // so a resumed run reconstructs the identical set from one integer, and so probes stay spread across
    // the corpus (and therefore across populations, which the manifest interleaves in runs of 64).
    void reset(std::size_t n_docs, std::size_t stride) {
        docs_.clear();
        last_.clear();
        stride_ = stride ? stride : 0;
        if (!stride_) return;
        for (std::size_t d = 0; d < n_docs; d += stride_) docs_.push_back(d);
        last_.assign(docs_.size(), 0.f);
    }

    bool active() const { return !docs_.empty(); }
    std::span<const std::size_t> docs() const { return docs_; }

    // Shrink the tracked set to `n` readings. The caller resolves a window plan over the probe documents
    // and some are not gradeable (a document too short to host even a 2-token window), so the number of
    // SCORES is <= the number of probe documents. observe() requires the two to agree exactly -- a
    // mismatched length there would silently pair each probe with the wrong document's previous value,
    // producing a drift floor made entirely of between-document differences.
    //
    // is_probe() is deliberately NOT narrowed to match: a document dropped from the readings is still
    // excluded from training, which keeps the exclusion rule a pure function of the stride and therefore
    // identical across resumes.
    void trim_to(std::size_t n) {
        if (n >= docs_.size()) return;
        docs_.resize(n);
        last_.assign(n, 0.f);
        primed_ = false;
    }

    // True when document `doc` is a probe and must therefore be EXCLUDED from training. The exclusion is
    // what makes the measurement mean anything: a probe that is trained even occasionally contributes an
    // own-gradient term and stops being a pure drift reading.
    bool is_probe(std::size_t doc) const {
        return stride_ != 0 && (doc % stride_) == 0;
    }

    // Fold in a fresh round of probe scores, returning the mean absolute NELBO change per unit of global
    // applied learning since the previous round -- the drift floor. Returns 0 on the first round, which
    // establishes the baseline and has nothing to compare against.
    double observe(std::span<const float> nelbos, double applied_since_last) {
        if (nelbos.size() != last_.size() || nelbos.empty()) return 0.0;
        if (!primed_) {
            std::copy(nelbos.begin(), nelbos.end(), last_.begin());
            primed_ = true;
            return 0.0;
        }
        // Non-finite scores are SKIPPED rather than folded in. nelbo_cpu_each writes NaN for a window
        // it cannot grade, and a single NaN propagates through the sum to make the whole floor NaN --
        // which then compares false against every velocity, silently disabling the one guard that says
        // which readings are real.
        double total = 0.0;
        std::size_t n = 0;
        for (std::size_t i = 0; i < nelbos.size(); ++i) {
            if (!std::isfinite(nelbos[i])) continue;
            if (std::isfinite(last_[i])) { total += std::fabs(static_cast<double>(nelbos[i]) -
                                                              static_cast<double>(last_[i])); ++n; }
            last_[i] = nelbos[i];
        }
        if (n == 0) return 0.0;
        const double mean_abs = total / static_cast<double>(n);
        // Per unit of applied learning, to be directly comparable with Entry::velocity, which is
        // normalised the same way. Absolute value because drift has no preferred sign -- what is wanted
        // is its MAGNITUDE as a noise floor, not a mean that would cancel to nothing.
        return applied_since_last > 0.0 ? mean_abs / applied_since_last : 0.0;
    }

private:
    std::vector<std::size_t> docs_;
    std::vector<float>       last_;
    std::size_t              stride_  = 0;
    bool                     primed_  = false;
};

}  // namespace sub0::tutor
