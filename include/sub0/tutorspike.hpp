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

// --- the surface ------------------------------------------------------------------------------
// One document's state. 44 bytes; see the float rationale in the file header. The transfer fields are
// the larger half of the point -- see "the drift term is not only noise" above.
struct Entry {
    std::uint32_t visits       = 0;     // times a window from this document has been trained
    std::uint32_t tokens       = 0;     // trained positions accumulated (a document, not a corpus, count)
    float         applied      = 0.f;   // sum of effective_lr * tokens -- the velocity DENOMINATOR
    float         nelbo        = 0.f;   // most recent per-window loss (the training forward, PRE-update)
    float         nelbo_mark   = 0.f;   // nelbo at the last velocity update
    float         applied_mark = 0.f;   // applied at the last velocity update
    float         velocity     = 0.f;   // -d(nelbo) / d(applied), 0 until two marks exist

    // Post-update reading from the back-to-back re-score, and 0 when this entry has not been re-scored
    // (the re-score runs at a cadence, so most visits do not have one). Its presence is what makes the
    // own-learning / transfer split recoverable at the NEXT visit.
    float         nelbo_post   = 0.f;
    // Signed between-visit change per unit of GLOBAL applied learning, averaged over this entry's
    // measured intervals. NEGATIVE means the entry improved while it was not being trained
    // (reinforcement from the rest of the corpus); POSITIVE means it degraded (conflict). Normalised by
    // global rather than own applied learning because the quantity being attributed is everything
    // ELSE's training, not this entry's.
    float         transfer     = 0.f;
    std::uint32_t transfer_n   = 0;     // intervals folded into `transfer` (0 = no reading yet)
    float         global_mark  = 0.f;   // global applied learning as of this entry's last visit

    bool seen() const { return visits > 0; }
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

// How often the post-update re-score runs. A forward is roughly a third of a forward+backward, so every
// step would cost ~33%; at every 20th step it is ~1.7%, and the transfer term only needs SOME visits to
// carry a post reading, not all of them.
inline constexpr long TUTOR_RESCORE_EVERY = 20;

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
        if (e.visits > 0 && e.nelbo_post != 0.f) {
            const double d_global = global_applied_ - static_cast<double>(e.global_mark);
            if (d_global > 0.0) {
                const float per_unit = static_cast<float>(
                    (static_cast<double>(nelbo) - static_cast<double>(e.nelbo_post)) / d_global);
                // Running mean rather than the latest reading: a single interval is one noisy sample of
                // a relationship, and the quantity of interest is whether this entry is SYSTEMATICALLY
                // reinforced or conflicted by the rest of the corpus.
                ++e.transfer_n;
                e.transfer += (per_unit - e.transfer) / static_cast<float>(e.transfer_n);
            }
        }

        ++e.visits;
        e.tokens  += static_cast<std::uint32_t>(tokens);
        e.applied += lr * static_cast<float>(tokens);
        e.nelbo    = nelbo;
        e.nelbo_post = 0.f;                             // consumed; set again only if a re-score runs
        e.global_mark = static_cast<float>(global_applied_);
        if (e.visits == 1) {                            // first sighting: establish the baseline only
            e.nelbo_mark   = nelbo;
            e.applied_mark = e.applied;
            return;
        }
        const float d_applied = e.applied - e.applied_mark;
        if (d_applied <= 0.f || d_applied < mark_threshold_) return;   // too small to divide by yet
        // Sign convention: FALLING nelbo is POSITIVE velocity (learning is happening). A negative
        // velocity therefore means the entry is regressing -- being forgotten -- which TUTOR.md's table
        // wants weighted UP, and which a level-based rule only ever catches by accident.
        e.velocity     = -(e.nelbo - e.nelbo_mark) / d_applied;
        e.nelbo_mark   = e.nelbo;
        e.applied_mark = e.applied;
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
    }

    // Coverage: the fraction of documents visited at least once. Poisson sampling leaves ~1/e of the
    // corpus untouched after one epoch's worth of steps, so an unvisited entry's zero velocity means
    // "never asked", not "nothing to learn" -- reading this alongside the surface is what keeps the two
    // apart. It is also PLATEAU_DESIGN.md 4a's coverage number, on the training side.
    double coverage() const {
        if (entries_.empty()) return 0.0;
        std::size_t seen = 0;
        for (const Entry& e : entries_) if (e.seen()) ++seen;
        return static_cast<double>(seen) / static_cast<double>(entries_.size());
    }

    // Exact persistence. The surface is FEEDBACK STATE: TUTOR.md is explicit that if it is not restored
    // exactly, matched-arm A/Bs quietly stop being matched. Written as a SIDECAR next to the checkpoint
    // rather than inside it, deliberately -- adding a variable-length section to the fixed-size .ckpt
    // struct is the highest-blast-radius change in this codebase (AGENTS.md 3), and a spike has no
    // business stranding in-progress production runs to store state nothing in the mainline reads.
    [[nodiscard]] bool save(const std::string& path) const;
    [[nodiscard]] bool load(const std::string& path);

    // Snapshot for the live heat map: a self-contained JSON document the viewer polls. Written whole and
    // renamed into place by the implementation, so a reader never observes a half-written file.
    [[nodiscard]] bool write_snapshot(const std::string& path, const Manifest& man,
                                      long step, double drift_floor) const;

private:
    std::vector<Entry> entries_;
    float              mark_threshold_ = 0.f;
    double             global_applied_ = 0.0;   // double: this one accumulates over the WHOLE run
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
        double total = 0.0;
        for (std::size_t i = 0; i < nelbos.size(); ++i) {
            total += std::fabs(static_cast<double>(nelbos[i]) - static_cast<double>(last_[i]));
            last_[i] = nelbos[i];
        }
        const double mean_abs = total / static_cast<double>(nelbos.size());
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
