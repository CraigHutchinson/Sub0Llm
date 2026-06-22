#pragma once

// cli.hpp — minimal argument validation for the ad-hoc-parsed tool binaries (viz_gen/viz_train/
// viz_server). The config-driven trainers go through sub0diff::config::resolve(), which already rejects
// unknown flags; this gives the simpler utilities the SAME "an unknown argument is an error" guarantee
// (a silently-ignored typo'd flag is how a run quietly uses the wrong settings).

#include <format>
#include <initializer_list>
#include <stdexcept>
#include <string_view>

namespace sub0diff::cli {

// Throw std::runtime_error on the first `--flag` not present in `known`. Only `--`-prefixed tokens are
// validated (values like `8080` or positional args are skipped), so a tool lists exactly the flags it
// accepts and any other `--x` is an error.
inline void require_known(int argc, char** argv, std::initializer_list<std::string_view> known) {
    for (int i = 1; i < argc; ++i) {
        const std::string_view a = argv[i];
        if (a.size() < 2 || a[0] != '-' || a[1] != '-') continue;
        bool ok = false;
        for (const auto k : known)
            if (a == k) { ok = true; break; }
        if (!ok) throw std::runtime_error(std::format("unknown argument: {}", a));
    }
}

}  // namespace sub0diff::cli
