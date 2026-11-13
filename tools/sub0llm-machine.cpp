// sub0llm-machine -- `sub0llm machine` (issue #78). `--facts` (increment 1) prints this machine's hardware
// facts and writes them as JSON. `--roofs` (increments 2-3) also measures the roof sets it names inside a time
// budget and writes everything as one JSON document: `mem` is the memory roofs (bandwidth by placement,
// working-set sweep, latency), `compute` the compute roofs (arithmetic peak by kernel, placement and thread count,
// with the effective clock). The budget is divided between the selected sets.
// Header-only against include/sub0/measure/{machine_facts,memory_roofs,compute_roofs}.hpp: no engine, no generated
// config, no third-party dependency.
//
//   sub0llm-machine --facts [--out FILE] [--root DIR]
//   sub0llm-machine --roofs mem|compute|mem,compute|all [--budget SECONDS] [--out FILE] [--root DIR]
//
// The summary goes to stdout. The JSON goes to --out, or by default to out/machine/<host>-<YYYYMMDD>.json
// under the working directory. --root points the reader at a fabricated /proc and /sys tree (tests).

#include "sub0/measure/compute_roofs.hpp"
#include "sub0/measure/machine_facts.hpp"
#include "sub0/measure/memory_roofs.hpp"

#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <optional>
#include <string>
#include <string_view>
#include <system_error>

namespace {

void usage(const char* argv0) {
    std::fprintf(stderr, "usage: %s --facts [--out FILE] [--root DIR]\n", argv0);
    std::fprintf(stderr, "       %s --roofs mem|compute|mem,compute|all [--budget SECONDS] [--out FILE] [--root DIR]\n",
                 argv0);
}

// The roof sets a --roofs value selects, as {memory, compute}. Empty when the value names an unknown set.
struct RoofSets {
    bool memory = false;
    bool compute = false;
};

std::optional<RoofSets> parseRoofSets(std::string_view value) {
    RoofSets sets;
    if (value == "all") return RoofSets{true, true};
    while (!value.empty()) {
        const std::size_t comma = value.find(',');
        const std::string_view item = value.substr(0, comma);
        if (item == "mem") sets.memory = true;
        else if (item == "compute") sets.compute = true;
        else return std::nullopt;
        if (comma == std::string_view::npos) break;
        value.remove_prefix(comma + 1);
    }
    if (!sets.memory && !sets.compute) return std::nullopt;
    return sets;
}

// Hostnames are used in a file name: keep letters, digits, '.', '-' and '_'; anything else becomes '_'.
std::string safeName(std::string s) {
    if (s.empty()) return "unknown-host";
    for (char& c : s) {
        const auto u = static_cast<unsigned char>(c);
        if (!(std::isalnum(u) || c == '-' || c == '.' || c == '_')) c = '_';
    }
    return s;
}

std::string todayYYYYMMDD() {
    const std::time_t now = std::time(nullptr);
    std::tm tm{};
#if defined(_WIN32)
    localtime_s(&tm, &now);
#else
    localtime_r(&now, &tm);
#endif
    char buf[64];
    std::snprintf(buf, sizeof buf, "%04d%02d%02d", tm.tm_year + 1900, tm.tm_mon + 1, tm.tm_mday);
    return buf;
}

}  // namespace

int main(int argc, char** argv) {
    bool facts = false;
    RoofSets roofSets;
    double budgetSeconds = 30.0;
    std::filesystem::path outPath;
    std::filesystem::path root = "/";
    for (int i = 1; i < argc; ++i) {
        const std::string_view a = argv[i];
        if (a == "--facts") facts = true;
        else if (a == "--roofs" && i + 1 < argc) {
            const std::optional<RoofSets> sets = parseRoofSets(argv[++i]);
            if (!sets) {
                std::fprintf(stderr, "sub0llm-machine: --roofs takes mem, compute, mem,compute or all\n");
                return 2;
            }
            roofSets = *sets;
        } else if (a == "--budget" && i + 1 < argc) {
            char* end = nullptr;
            budgetSeconds = std::strtod(argv[++i], &end);
            if (end == argv[i] || *end != '\0' || !sub0::measure::validRoofBudget(budgetSeconds)) {
                std::fprintf(stderr, "sub0llm-machine: --budget needs a finite number in (0, 86400] seconds\n");
                return 2;
            }
        } else if (a == "--out" && i + 1 < argc) outPath = argv[++i];
        else if (a == "--root" && i + 1 < argc) root = argv[++i];
        else {
            usage(argv[0]);
            return 2;
        }
    }
    const bool roofs = roofSets.memory || roofSets.compute;
    if (!facts && !roofs) {
        usage(argv[0]);
        return 2;
    }
    if (roofs && root != "/") {
        std::fprintf(stderr, "sub0llm-machine: --roofs measures this machine; --root applies to --facts only\n");
        return 2;
    }

    const sub0::measure::MachineFacts m = sub0::measure::readMachineFacts(root);
    std::string summary = sub0::measure::toSummary(m);
    std::string json;
    if (roofs) {
        // The budget is split evenly between the selected sets.
        const double share = budgetSeconds / (roofSets.memory && roofSets.compute ? 2.0 : 1.0);
        std::optional<sub0::measure::MemoryRoofs> memory;
        std::optional<sub0::measure::ComputeRoofs> compute;
        if (roofSets.memory) {
            sub0::measure::RoofConfig cfg;
            cfg.budgetSeconds = share;
            memory = sub0::measure::measureMemoryRoofs(m, cfg);
            summary += "\n" + sub0::measure::toSummary(*memory);
        }
        if (roofSets.compute) {
            sub0::measure::ComputeConfig cfg;
            cfg.budgetSeconds = share;
            compute = sub0::measure::measureComputeRoofs(m, cfg);
            summary += "\n" + sub0::measure::toSummary(*compute);
        }
        json = sub0::measure::toRoofsJson(m, memory ? &*memory : nullptr, compute ? &*compute : nullptr);
    } else {
        json = sub0::measure::toJson(m);
    }
    std::fputs(summary.c_str(), stdout);

    if (outPath.empty()) {
        const std::string host = safeName(m.hostname.value.value_or(""));
        outPath = std::filesystem::path("out") / "machine" / (host + "-" + todayYYYYMMDD() + ".json");
    }
    std::error_code ec;
    if (outPath.has_parent_path()) std::filesystem::create_directories(outPath.parent_path(), ec);

    std::ofstream os(outPath, std::ios::binary);
    if (!os) {
        std::fprintf(stderr, "sub0llm-machine: cannot write %s\n", outPath.string().c_str());
        return 1;
    }
    os << json << '\n';
    if (!os) {
        std::fprintf(stderr, "sub0llm-machine: write failed for %s\n", outPath.string().c_str());
        return 1;
    }
    std::printf("wrote %s\n", outPath.string().c_str());
    return 0;
}
