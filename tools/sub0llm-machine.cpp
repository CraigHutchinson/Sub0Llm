// sub0llm-machine -- `sub0llm machine --facts` (issue #78, increment 1): prints this machine's hardware
// facts and writes them as JSON. Header-only against include/sub0/measure/machine_facts.hpp: no engine,
// no generated config, no third-party dependency, so it builds in any configuration in seconds.
//
//   sub0llm-machine --facts [--out FILE] [--root DIR]
//
// The summary goes to stdout. The JSON goes to --out, or by default to out/machine/<host>-<YYYYMMDD>.json
// under the working directory. --root points the reader at a fabricated /proc and /sys tree (tests).

#include "sub0/measure/machine_facts.hpp"

#include <cctype>
#include <cstdio>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <string>
#include <string_view>
#include <system_error>

namespace {

void usage(const char* argv0) {
    std::fprintf(stderr, "usage: %s --facts [--out FILE] [--root DIR]\n", argv0);
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
    std::filesystem::path outPath;
    std::filesystem::path root = "/";
    for (int i = 1; i < argc; ++i) {
        const std::string_view a = argv[i];
        if (a == "--facts") facts = true;
        else if (a == "--out" && i + 1 < argc) outPath = argv[++i];
        else if (a == "--root" && i + 1 < argc) root = argv[++i];
        else {
            usage(argv[0]);
            return 2;
        }
    }
    if (!facts) {
        usage(argv[0]);
        return 2;
    }

    const sub0::measure::MachineFacts m = sub0::measure::readMachineFacts(root);
    const std::string summary = sub0::measure::toSummary(m);
    const std::string json = sub0::measure::toJson(m);
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
