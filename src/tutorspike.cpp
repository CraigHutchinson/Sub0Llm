// src/tutorspike.cpp -- out-of-line half of the mastery surface (include/sub0/tutorspike.hpp).
//
// Three things live here rather than in the header, each for its own reason:
//   * the manifest parser, because simdjson types must not cross the DLL boundary (the same constraint
//     registry.hpp documents for read_config_json);
//   * the sidecar save/load, because it is file I/O with a versioned binary header;
//   * the snapshot writer, because the live heat map's schema is a contract with tools/tutor_heatmap.html
//     and belongs next to nothing that runs per step.
//
// Nothing here is on a hot path: save/snapshot run at a cadence, the parser runs once at startup.

#include "sub0/tutorspike.hpp"

#include "sub0/log.hpp"

#include <simdjson.h>

#include <cstdio>
#include <filesystem>
#include <fstream>

namespace sub0::tutor {

namespace {

// Sidecar format. Versioned for the same reason every other binary format here is: a silently
// mis-parsed feedback state is worse than a refused one, because a matched-arm A/B would keep running
// and quietly stop being matched (TUTOR.md's reproducibility risk).
constexpr std::uint32_t kSurfaceMagic   = 0x53555230u;   // "0RUS" little-endian
constexpr std::uint32_t kSurfaceVersion = 2u;   // v2 records sizeof(Entry); see load()

template <typename T>
void wr(std::ostream& os, const T& v) {
    os.write(reinterpret_cast<const char*>(&v), sizeof(T));
}
template <typename T>
bool rd(std::istream& is, T& v) {
    return static_cast<bool>(is.read(reinterpret_cast<char*>(&v), sizeof(T)));
}

}  // namespace

bool read_manifest(const std::string& path, Manifest& out) {
    out = Manifest{};
    simdjson::ondemand::parser parser;
    simdjson::padded_string json;
    if (simdjson::padded_string::load(path).get(json) != simdjson::SUCCESS) {
        sub0::log::error("tutor: cannot read manifest '{}'", path);
        return false;
    }
    simdjson::ondemand::document doc;
    if (parser.iterate(json).get(doc) != simdjson::SUCCESS) {
        sub0::log::error("tutor: manifest '{}' is not valid JSON", path);
        return false;
    }

    // Single forward pass, handler per key -- this project's JSON convention (no DOM tree). Key order in
    // the file therefore matters to on-demand iteration, and sub0llm-tutorspike writes them in this
    // order; anything missing leaves its default and is caught by the validation below rather than here.
    std::uint64_t total_docs = 0;
    std::vector<std::pair<int, int>> runs;
    simdjson::ondemand::object obj;
    if (doc.get_object().get(obj) != simdjson::SUCCESS) return false;
    for (auto field : obj) {
        std::string_view key;
        if (field.unescaped_key().get(key) != simdjson::SUCCESS) return false;
        if (key == "seed") {
            (void)field.value().get_uint64().get(out.seed);
        } else if (key == "total_docs") {
            (void)field.value().get_uint64().get(total_docs);
        } else if (key == "populations") {
            simdjson::ondemand::array arr;
            if (field.value().get_array().get(arr) != simdjson::SUCCESS) return false;
            for (auto v : arr) {
                std::string_view name;
                if (v.get_string().get(name) != simdjson::SUCCESS) return false;
                out.populations.emplace_back(name);
            }
        } else if (key == "runs") {
            simdjson::ondemand::array arr;
            if (field.value().get_array().get(arr) != simdjson::SUCCESS) return false;
            for (auto v : arr) {
                simdjson::ondemand::array pair;
                if (v.get_array().get(pair) != simdjson::SUCCESS) return false;
                std::int64_t vals[2] = { -1, -1 };
                int n = 0;
                for (auto x : pair) {
                    if (n < 2 && x.get_int64().get(vals[n]) != simdjson::SUCCESS) return false;
                    ++n;
                }
                if (n != 2 || vals[0] < 0 || vals[1] < 0) return false;
                runs.emplace_back(static_cast<int>(vals[0]), static_cast<int>(vals[1]));
            }
        }
    }

    if (out.populations.empty() || total_docs == 0 || runs.empty()) {
        sub0::log::error("tutor: manifest '{}' is missing populations/total_docs/runs", path);
        return false;
    }
    // Expand the run-lengths into the per-ordinal label array the surface indexes directly. Validated
    // rather than trusted: the runs must cover exactly total_docs documents and name only declared
    // populations, or the labels would silently slide relative to the corpus.
    out.doc_pop.reserve(static_cast<std::size_t>(total_docs));
    for (const auto& [pop, count] : runs) {
        if (pop < 0 || static_cast<std::size_t>(pop) >= out.populations.size()) {
            sub0::log::error("tutor: manifest '{}' names population {} but declares only {}",
                             path, pop, out.populations.size());
            out = Manifest{};
            return false;
        }
        out.doc_pop.insert(out.doc_pop.end(), static_cast<std::size_t>(count),
                           static_cast<std::uint8_t>(pop));
    }
    if (out.doc_pop.size() != static_cast<std::size_t>(total_docs)) {
        sub0::log::error("tutor: manifest '{}' runs cover {} documents but total_docs says {}",
                         path, out.doc_pop.size(), total_docs);
        out = Manifest{};
        return false;
    }
    return true;
}

bool Surface::save(const std::string& path) const {
    std::ofstream os(path, std::ios::binary);
    if (!os) return false;
    wr(os, kSurfaceMagic);
    wr(os, kSurfaceVersion);
    // sizeof(Entry) rides the header because the payload is the raw struct array. A field added to Entry
    // without a version bump would otherwise reinterpret every entry at the wrong stride and load
    // SILENTLY -- producing a plausible surface built from misaligned bytes. The version alone does not
    // protect against this, because the mistake that causes it is precisely forgetting to bump it.
    const std::uint32_t entry_size = static_cast<std::uint32_t>(sizeof(Entry));
    wr(os, entry_size);
    const std::uint64_t n = entries_.size();
    wr(os, n);
    wr(os, mark_threshold_);
    wr(os, global_applied_);
    os.write(reinterpret_cast<const char*>(entries_.data()),
             static_cast<std::streamsize>(entries_.size() * sizeof(Entry)));
    return static_cast<bool>(os);
}

bool Surface::load(const std::string& path) {
    std::ifstream is(path, std::ios::binary);
    if (!is) return false;
    std::uint32_t magic = 0, version = 0, entry_size = 0;
    std::uint64_t n = 0;
    if (!rd(is, magic) || !rd(is, version) || !rd(is, entry_size) || !rd(is, n)) return false;
    // Magic/version FIRST: on a foreign or truncated file every later field is garbage, and reporting a
    // confident "wrong entry size" for what is actually "not a surface file" sends the reader after the
    // wrong problem.
    if (magic != kSurfaceMagic || version != kSurfaceVersion) {
        sub0::log::error("tutor: surface '{}' has magic/version {:#x}/{} (expected {:#x}/{}) -- refusing "
                         "to load rather than resume a matched arm against mismatched feedback state",
                         path, magic, version, kSurfaceMagic, kSurfaceVersion);
        return false;
    }
    if (entry_size != sizeof(Entry)) {
        sub0::log::error("tutor: surface '{}' has {}-byte entries, this build has {} -- refusing to "
                         "load. Entry's layout changed without a version bump; the data is not lost, "
                         "but it cannot be read at this stride.",
                         path, entry_size, sizeof(Entry));
        return false;
    }
    // TRANSACTIONAL: read into scratch and commit only on success. Assigning into entries_ first would
    // leave a failed load having already RESIZED the surface -- and the caller's contract is "false means
    // start fresh", so it would carry on with a ledger sized to the stale file rather than to the corpus.
    // Every subsequent record() would then index a differently-shaped array.
    float  mark = 0.f;
    double global = 0.0;
    if (!rd(is, mark) || !rd(is, global)) return false;
    std::vector<Entry> scratch(static_cast<std::size_t>(n));
    is.read(reinterpret_cast<char*>(scratch.data()),
            static_cast<std::streamsize>(scratch.size() * sizeof(Entry)));
    if (!is) return false;
    entries_        = std::move(scratch);
    mark_threshold_ = mark;
    global_applied_ = global;
    return true;
}

bool Surface::append_events(const std::string& path, const Manifest& man) {
    if (events_.empty()) return true;
    // The header is decided from the FILE alone -- no caller-held "first write" flag. A flag gets this
    // wrong in both directions and did: the caller cleared it after the first eval, but the first eval
    // had no events yet, so this returned early and the flag was consumed without a header ever being
    // written (the whole 70621-row stream from the first run has none). The mirror failure is a resumed
    // run injecting a SECOND header partway down. The file's own emptiness answers both.
    std::error_code ec;
    const bool empty_file = !std::filesystem::exists(path, ec) || std::filesystem::file_size(path, ec) == 0;
    std::ofstream os(path, std::ios::app);
    if (!os) return false;
    if (empty_file)
        os << "kind,step,doc,pop,visits,win_len,own_applied,global_applied,nelbo,value\n";
    for (Event& e : events_) {
        // Population is stamped here rather than at emit time: record() runs per window in the training
        // step and has no business reaching into the manifest, and the join is a pure array lookup.
        e.pop = (e.doc < man.doc_pop.size()) ? man.doc_pop[e.doc] : 0u;
        os << static_cast<int>(e.kind) << ',' << e.step << ',' << e.doc << ','
           << static_cast<int>(e.pop) << ',' << e.visits << ',' << e.win_len << ','
           << e.own_applied << ',' << e.global_applied << ',' << e.nelbo << ',' << e.value << '\n';
    }
    events_.clear();
    return static_cast<bool>(os);
}

bool Surface::write_run_info(const std::string& path, const RunInfo& r) const {
    std::ofstream os(path);
    if (!os) return false;
    // The cadence constants are written out alongside the run's own axes deliberately: they are
    // compile-time in this build, so a recording made under different ones is not comparable, and
    // nothing else in the directory would say which were in force.
    os << "{\n"
       << "  \"schema\": 1,\n"
       << "  \"label\": \"" << r.label << "\",\n"
       << "  \"manifest\": \"" << r.manifest << "\",\n"
       << "  \"manifest_seed\": " << r.manifest_seed << ",\n"
       << "  \"seed\": " << r.seed << ",\n"
       << "  \"batch\": " << r.batch << ",\n"
       << "  \"peak_lr\": " << r.peak_lr << ",\n"
       << "  \"seq_len\": " << r.seq_len << ",\n"
       << "  \"windows_per_epoch\": " << r.windows_per_epoch << ",\n"
       << "  \"eligible_docs\": " << r.eligible_docs << ",\n"
       << "  \"probe_stride\": " << TUTOR_PROBE_STRIDE << ",\n"
       << "  \"rescore_every\": " << TUTOR_RESCORE_EVERY << ",\n"
       << "  \"velocity_mark_multiple\": " << VELOCITY_MARK_MULTIPLE << ",\n"
       << "  \"entry_bytes\": " << sizeof(Entry) << "\n"
       << "}\n";
    return static_cast<bool>(os);
}

bool Surface::write_snapshot(const std::string& path, const Manifest& man, const RunInfo& run, long step,
                             double drift_floor, std::span<const std::uint8_t> reachable) const {
    // Write whole, then rename into place. The heat map polls this file on a timer, so a reader must
    // never observe a partial document -- rename is atomic on both platforms this project targets.
    const std::string tmp = path + ".tmp";
    {
        std::ofstream os(tmp);
        if (!os) return false;
        os << "{\n  \"step\": " << step
           << ",\n  \"coverage\": " << coverage()
           << ",\n  \"global_applied\": " << global_applied_
           << ",\n  \"drift_floor\": " << drift_floor
           << ",\n  \"populations\": [";
        for (std::size_t i = 0; i < man.populations.size(); ++i)
            os << (i ? ", " : "") << '"' << man.populations[i] << '"';
        os << "],\n";
        // Column-major: one array per field rather than an array of objects. At 36000 documents the
        // object-per-entry form is ~4x the bytes and materially slower to parse in the viewer, and this
        // file is rewritten every snapshot for the whole run.
        os << "  \"doc_pop\": [";
        for (std::size_t i = 0; i < man.doc_pop.size(); ++i)
            os << (i ? "," : "") << static_cast<int>(man.doc_pop[i]);
        os << "],\n  \"visits\": [";
        for (std::size_t i = 0; i < entries_.size(); ++i)
            os << (i ? "," : "") << entries_[i].visits;
        os << "],\n  \"nelbo\": [";
        for (std::size_t i = 0; i < entries_.size(); ++i)
            os << (i ? "," : "") << entries_[i].nelbo;
        os << "],\n  \"velocity\": [";
        for (std::size_t i = 0; i < entries_.size(); ++i)
            os << (i ? "," : "") << entries_[i].velocity;
        os << "],\n  \"transfer\": [";
        for (std::size_t i = 0; i < entries_.size(); ++i)
            os << (i ? "," : "") << (entries_[i].has_transfer() ? entries_[i].transfer : 0.f);
        os << "],\n  \"applied\": [";
        for (std::size_t i = 0; i < entries_.size(); ++i)
            os << (i ? "," : "") << entries_[i].applied;
        // Reachability is exported so the viewer can distinguish "training never reaches this document"
        // (validation split, drift probe) from "not visited yet". Both are visits==0, and rendering them
        // identically is how a permanently-unreachable 9% of the corpus reads as a coverage failure.
        os << "],\n  \"reachable\": [";
        for (std::size_t i = 0; i < entries_.size(); ++i)
            os << (i ? "," : "") << (i < reachable.size() ? static_cast<int>(reachable[i]) : 1);
        os << "]\n}\n";
        if (!os) return false;
    }
    std::error_code ec;
    std::filesystem::rename(tmp, path, ec);
    if (ec) {   // Windows refuses rename-onto-existing in some configurations; fall back explicitly.
        std::filesystem::remove(path, ec);
        std::filesystem::rename(tmp, path, ec);
    }
    return !ec;
}

}  // namespace sub0::tutor
