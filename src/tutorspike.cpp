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
constexpr std::uint32_t kSurfaceVersion = 1u;

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
    std::uint32_t magic = 0, version = 0;
    std::uint64_t n = 0;
    if (!rd(is, magic) || !rd(is, version) || !rd(is, n)) return false;
    if (magic != kSurfaceMagic || version != kSurfaceVersion) {
        sub0::log::error("tutor: surface '{}' has magic/version {:#x}/{} (expected {:#x}/{}) -- refusing "
                         "to load rather than resume a matched arm against mismatched feedback state",
                         path, magic, version, kSurfaceMagic, kSurfaceVersion);
        return false;
    }
    if (!rd(is, mark_threshold_) || !rd(is, global_applied_)) return false;
    entries_.assign(static_cast<std::size_t>(n), Entry{});
    is.read(reinterpret_cast<char*>(entries_.data()),
            static_cast<std::streamsize>(entries_.size() * sizeof(Entry)));
    return static_cast<bool>(is);
}

bool Surface::append_events(const std::string& path, const Manifest& man, bool write_header) {
    if (events_.empty()) return true;
    std::ofstream os(path, std::ios::app);
    if (!os) return false;
    if (write_header)
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

bool Surface::write_snapshot(const std::string& path, const Manifest& man, long step,
                             double drift_floor) const {
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
