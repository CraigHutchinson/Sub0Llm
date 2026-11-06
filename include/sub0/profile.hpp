// sub0/profile.hpp -- architecture profile files for sub0llm-configure (--profile).
//
// A profile is a small text file of configurator options, so the list of presets is data that can be
// added to and iterated on without rebuilding the configurator. Pure parsing + directory discovery:
// no knowledge of the configurator's variables (it owns the key -> variable mapping and rejects keys it
// does not know), so this is unit-testable on its own. Header-only, std-only.
//
// File format (`<name>.profile`):
//     # comment                      (also allowed after a value, introduced by " #")
//     name  = qwen3.8-27b            (optional; defaults to the file name without ".profile")
//     alias = qwen3.8-dense, other   (optional; comma-separated extra names)
//     dmodel = 5120                  (any other key = a sub0llm-configure long option WITHOUT the "--")
//     rope-theta = 1e7
// An option given explicitly on the command line always wins over the profile's value.

#pragma once

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

namespace sub0::profile {

struct Profile {
    std::string name;                                            // primary name ("" until resolved)
    std::vector<std::string> aliases;
    std::string source;                                          // file path, or "<stdin>"
    std::vector<std::pair<std::string, std::string>> values;     // option (no dashes) -> value, file order
};

struct ParseResult {
    bool ok = false;
    Profile profile;
    std::string error;       // "<source>:<line>: ..." when !ok
};

namespace detail {
inline std::string lower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return s;
}
inline std::string trim(std::string_view v) {
    std::size_t b = 0, e = v.size();
    while (b < e && std::isspace(static_cast<unsigned char>(v[b]))) ++b;
    while (e > b && std::isspace(static_cast<unsigned char>(v[e - 1]))) --e;
    return std::string(v.substr(b, e - b));
}
}  // namespace detail

[[nodiscard]] inline ParseResult parse(std::string_view text, std::string_view source) {
    ParseResult r;
    r.profile.source = std::string(source);
    std::istringstream in{std::string(text)};
    std::string raw;
    int line_no = 0;
    auto fail = [&](const std::string& why) {
        r.ok = false;
        r.error = std::string(source) + ":" + std::to_string(line_no) + ": " + why;
        return r;
    };
    while (std::getline(in, raw)) {
        ++line_no;
        std::string line = raw;
        if (const auto hash = line.find('#'); hash != std::string::npos &&
            (hash == 0 || std::isspace(static_cast<unsigned char>(line[hash - 1])))) line.erase(hash);
        line = detail::trim(line);
        if (line.empty()) continue;
        const auto eq = line.find('=');
        if (eq == std::string::npos) return fail("expected 'key = value', got '" + line + "'");
        const std::string key = detail::lower(detail::trim(std::string_view(line).substr(0, eq)));
        const std::string val = detail::trim(std::string_view(line).substr(eq + 1));
        if (key.empty() || !std::all_of(key.begin(), key.end(), [](unsigned char c) {
                return std::islower(c) || std::isdigit(c) || c == '-'; }))
            return fail("invalid option name '" + key + "' (use the option's long name without the leading --)");
        if (val.empty()) return fail("'" + key + "' has no value");
        if (key == "name") {
            if (!r.profile.name.empty()) return fail("'name' given twice");
            r.profile.name = val;
        } else if (key == "alias") {
            std::istringstream parts(val);
            std::string a;
            while (std::getline(parts, a, ',')) {
                a = detail::trim(a);
                if (!a.empty()) r.profile.aliases.push_back(a);
            }
        } else {
            for (const auto& kv : r.profile.values)
                if (kv.first == key) return fail("'" + key + "' set twice");
            r.profile.values.emplace_back(key, val);
        }
    }
    r.ok = true;
    return r;
}

// True when `name` is this profile's name or one of its aliases (case-insensitive).
[[nodiscard]] inline bool matches(const Profile& p, std::string_view name) {
    const std::string n = detail::lower(std::string(name));
    if (detail::lower(p.name) == n) return true;
    return std::any_of(p.aliases.begin(), p.aliases.end(), [&](const std::string& a) { return detail::lower(a) == n; });
}

// A --profile argument that names a file rather than a profile: contains a path separator or ends in ".profile".
[[nodiscard]] inline bool looks_like_path(std::string_view arg) {
    return arg.find('/') != std::string_view::npos || arg.find('\\') != std::string_view::npos ||
           (arg.size() > 8 && arg.substr(arg.size() - 8) == ".profile");
}

struct Discovered {
    std::vector<Profile> profiles;     // sorted by name
    std::vector<std::string> errors;   // unparseable files, reported but never fatal to the listing
};

// Every `*.profile` directly inside `dir`. A missing directory is not an error (empty result).
[[nodiscard]] inline Discovered discover(const std::filesystem::path& dir) {
    Discovered d;
    std::error_code ec;
    if (!std::filesystem::is_directory(dir, ec)) return d;
    for (const auto& entry : std::filesystem::directory_iterator(dir, ec)) {
        if (!entry.is_regular_file() || entry.path().extension() != ".profile") continue;
        std::ifstream f(entry.path());
        std::ostringstream buf;
        buf << f.rdbuf();
        ParseResult r = parse(buf.str(), entry.path().string());
        if (!r.ok) { d.errors.push_back(r.error); continue; }
        if (r.profile.name.empty()) r.profile.name = entry.path().stem().string();
        d.profiles.push_back(std::move(r.profile));
    }
    std::sort(d.profiles.begin(), d.profiles.end(), [](const Profile& a, const Profile& b) { return a.name < b.name; });
    return d;
}

}  // namespace sub0::profile
