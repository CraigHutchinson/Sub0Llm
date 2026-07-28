// evalcache.hpp -- per-model cache of RAW evaluation measurements (metrics.txt, a sibling of
// state.json/report.txt in each model directory).
//
// Split deliberately from registry.hpp's state.json: that file is repeatedly read-modify-written by
// the ACTIVE TRAINING LOOP on every checkpoint tick, so folding eval-tool output into the same
// struct/file risks it being silently clobbered by a later training resume.
//
// Holds only what genuinely requires RUNNING the model (a forward pass over held-out data, or a
// generation sweep) -- train/val NELBO, and autotemp's real-text anchors + full temperature/
// repetition grid. Deliberately does NOT store DERIVED values (bits/byte, tokens/param, autotemp's
// interpolated crossing temperatures, the underfit diagnosis) -- those are pure computation over
// these raw numbers, recomputed fresh every time by whatever build of `sub0llm models --metrics` is
// currently running, so a new derived metric/diagnosis added later never needs to re-run the model.
// `ce_ppl` (autotemp's real-text cross-entropy perplexity anchor) is likewise not stored: it is
// exactly `exp(val_nelbo)`, so it's a one-line derivation at read time, not a fourth number to keep
// in sync with `val_nelbo` by hand.
//
// Callers own the "preserve fields the other tool wrote" merge: `report` and `autotemp` each
// populate only their own subset of an `EvalMetrics` read from any existing cache before writing it
// back (see `compute_raw_metrics()` in train_stage.cpp) -- this header is a plain, unconditional
// serialize/deserialize, mirroring registry.hpp's own read_state/write_state split.
#pragma once

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace sub0::evalcache {

// Parallel arrays, one entry per autotemp grid point (temperature -> generation stats at that temp).
struct AutotempGrid {
    std::vector<float>  temp;
    std::vector<double> gen_ppl, rep4;
};

struct EvalMetrics {
    long long   steps = -1;      // the model's step count when measured (vs state.json's -> staleness)
    std::string measured_at;     // ISO timestamp (registry::now_iso())
    // report-owned raw numbers.
    double train_nelbo = -1, val_nelbo = -1, bytes_per_tok = 0;
    // autotemp-owned raw numbers: the real-text anchors + the full sweep (not the interpolated
    // crossing temperatures -- those are derived, see the file header comment above).
    double       target_ppl = -1, target_rep = -1;
    AutotempGrid grid;
};

inline void write_metrics(const std::filesystem::path& dir, const EvalMetrics& m) {
    std::error_code ec; std::filesystem::create_directories(dir, ec);
    std::ofstream os(dir / "metrics.txt", std::ios::trunc);
    if (!os) return;
    os << "steps="         << m.steps         << "\n"
       << "measured_at="   << m.measured_at   << "\n"
       << "train_nelbo="   << m.train_nelbo   << "\n"
       << "val_nelbo="     << m.val_nelbo     << "\n"
       << "bytes_per_tok=" << m.bytes_per_tok << "\n"
       << "target_ppl="    << m.target_ppl    << "\n"
       << "target_rep="    << m.target_rep    << "\n";
    for (std::size_t i = 0; i < m.grid.temp.size(); ++i)
        os << "grid\t" << m.grid.temp[i] << "\t" << m.grid.gen_ppl[i] << "\t" << m.grid.rep4[i] << "\n";
}

inline bool read_metrics(const std::filesystem::path& dir, EvalMetrics& m) {
    std::ifstream is(dir / "metrics.txt");
    if (!is) return false;
    m = EvalMetrics{};   // start from field defaults, matching registry::read_state's convention
    for (std::string line; std::getline(is, line);) {
        if (line.rfind("grid\t", 0) == 0) {
            const auto t1 = line.find('\t', 5);
            const auto t2 = (t1 == std::string::npos) ? std::string::npos : line.find('\t', t1 + 1);
            if (t1 == std::string::npos || t2 == std::string::npos) continue;
            m.grid.temp.push_back(std::strtof(line.substr(5, t1 - 5).c_str(), nullptr));
            m.grid.gen_ppl.push_back(std::strtod(line.substr(t1 + 1, t2 - t1 - 1).c_str(), nullptr));
            m.grid.rep4.push_back(std::strtod(line.substr(t2 + 1).c_str(), nullptr));
            continue;
        }
        const auto eq = line.find('=');
        if (eq == std::string::npos) continue;
        const std::string k = line.substr(0, eq), v = line.substr(eq + 1);
        if      (k == "steps")         m.steps = std::strtoll(v.c_str(), nullptr, 10);
        else if (k == "measured_at")   m.measured_at = v;
        else if (k == "train_nelbo")   m.train_nelbo = std::strtod(v.c_str(), nullptr);
        else if (k == "val_nelbo")     m.val_nelbo = std::strtod(v.c_str(), nullptr);
        else if (k == "bytes_per_tok") m.bytes_per_tok = std::strtod(v.c_str(), nullptr);
        else if (k == "target_ppl")    m.target_ppl = std::strtod(v.c_str(), nullptr);
        else if (k == "target_rep")    m.target_rep = std::strtod(v.c_str(), nullptr);
    }
    return true;
}

}  // namespace sub0::evalcache
