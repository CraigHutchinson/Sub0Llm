// sub0llm-machine -- `sub0llm machine` (issue #78). `--facts` (increment 1) prints this machine's hardware
// facts and writes them as JSON. `--roofs mem` (increment 2) also measures the memory roofs (bandwidth by
// placement, working-set sweep, latency) inside a time budget and writes both as one JSON document.
// Header-only against include/sub0/measure/{machine_facts,memory_roofs}.hpp: no engine, no generated config,
// no third-party dependency.
//
//   sub0llm-machine --facts [--out FILE] [--root DIR]
//   sub0llm-machine --roofs mem [--budget SECONDS] [--out FILE] [--root DIR]
//
// The summary goes to stdout. The JSON goes to --out, or by default to out/machine/<host>-<YYYYMMDD>.json
// under the working directory. --root points the reader at a fabricated /proc and /sys tree (tests).

#include "sub0/measure/machine_facts.hpp"
#include "sub0/measure/memory_roofs.hpp"

#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <string>
#include <string_view>
#include <system_error>

namespace {

void usage(const char* argv0) {
    std::fprintf(stderr, "usage: %s --facts [--out FILE] [--root DIR]\n", argv0);
    std::fprintf(stderr, "       %s --roofs mem [--budget SECONDS] [--out FILE] [--root DIR]\n", argv0);
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
    bool roofsMem = false;
    double budgetSeconds = 30.0;
    std::filesystem::path outPath;
    std::filesystem::path root = "/";
    for (int i = 1; i < argc; ++i) {
        const std::string_view a = argv[i];
        if (a == "--facts") facts = true;
        else if (a == "--roofs" && i + 1 < argc && std::string_view(argv[i + 1]) == "mem") {
            roofsMem = true;
            ++i;
        } else if (a == "--budget" && i + 1 < argc) {
            char* end = nullptr;
            budgetSeconds = std::strtod(argv[++i], &end);
            if (end == argv[i] || *end != '\0' || !(budgetSeconds > 0.0)) {
                std::fprintf(stderr, "sub0llm-machine: --budget needs a positive number of seconds\n");
                return 2;
            }
        } else if (a == "--out" && i + 1 < argc) outPath = argv[++i];
        else if (a == "--root" && i + 1 < argc) root = argv[++i];
        else {
            usage(argv[0]);
            return 2;
        }
    }
    if (!facts && !roofsMem) {
        usage(argv[0]);
        return 2;
    }
    if (roofsMem && root != "/") {
        std::fprintf(stderr, "sub0llm-machine: --roofs measures this machine; --root applies to --facts only\n");
        return 2;
    }

    const sub0::measure::MachineFacts m = sub0::measure::readMachineFacts(root);
    std::string summary = sub0::measure::toSummary(m);
    std::string json;
    if (roofsMem) {
        sub0::measure::RoofConfig cfg;
        cfg.budgetSeconds = budgetSeconds;
        const sub0::measure::MemoryRoofs roofs = sub0::measure::measureMemoryRoofs(m, cfg);
        json = sub0::measure::toJson(m, roofs);
        summary += "\n" + sub0::measure::toSummary(roofs);
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
