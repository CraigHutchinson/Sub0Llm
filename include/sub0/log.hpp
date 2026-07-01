// sub0/log.hpp — a tiny process-wide logging wrapper for the whole project (frontend + engine + tools).
// Header-only + std-only, so it lives in the sub0_frontend layer and is engine-free testable.
//
// Two output kinds:
//   * LEVELED diagnostics -- log::error / warn / info / debug -- prefixed with "[level] ", filtered by a
//     runtime threshold, and routed to stderr (error/warn) or stdout (info/debug).
//   * RAW program output  -- log::line -- no prefix, never filtered (formatted progress/tables etc.).
// BOTH additionally tee to an optional file sink (log::set_file), e.g. a model's train.log, so a long
// or background run keeps a complete record next to its artifacts. Thread-safe (a mutex guards the sink
// + the interleaving), so the data-parallel paths can log without garbling lines.

#pragma once

#include <cstdio>
#include <format>
#include <fstream>
#include <mutex>
#include <string>
#include <string_view>
#include <utility>

namespace sub0::log {

enum class Level { Error = 0, Warn = 1, Info = 2, Debug = 3 };

inline const char* tag(Level l) {
    switch (l) {
        case Level::Error: return "error";
        case Level::Warn:  return "warn";
        case Level::Info:  return "info";
        default:           return "debug";
    }
}

// The process-wide logger state. A function-local static so header-only use has one instance and no
// static-init-order surprises.
struct Logger {
    Level         level = Level::Info;   // leveled messages strictly above this are dropped
    std::ofstream file;                  // optional tee sink (see set_file)
    std::mutex    mu;                    // guards `file` + serializes interleaved writes
};
inline Logger& logger() { static Logger g; return g; }

inline void  set_level(Level l) { logger().level = l; }
inline Level get_level()        { return logger().level; }

// Open (or replace) the file sink. Everything logged after this also lands in `path`. Append by default
// so a resumed run continues the same file. Returns false if the file could not be opened.
inline bool set_file(const std::string& path, bool append = true) {
    Logger& g = logger();
    std::scoped_lock lk(g.mu);
    g.file.close();
    g.file.open(path, append ? std::ios::app : std::ios::trunc);
    return g.file.is_open();
}
inline void close_file() { Logger& g = logger(); std::scoped_lock lk(g.mu); g.file.close(); }

// --- core sinks (take an already-formatted message) ------------------------
inline void emit(Level l, std::string_view msg) {
    Logger& g = logger();
    if (static_cast<int>(l) > static_cast<int>(g.level)) return;   // below the threshold -> drop
    std::scoped_lock lk(g.mu);
    std::FILE* con = (l <= Level::Warn) ? stderr : stdout;         // diagnostics: err/warn -> stderr
    std::fprintf(con, "[%s] %.*s\n", tag(l), static_cast<int>(msg.size()), msg.data());
    std::fflush(con);
    if (g.file.is_open()) { g.file << '[' << tag(l) << "] " << msg << '\n'; g.file.flush(); }
}
inline void emit_line(std::string_view msg) {                      // raw program output (no prefix, always on)
    Logger& g = logger();
    std::scoped_lock lk(g.mu);
    std::fprintf(stdout, "%.*s\n", static_cast<int>(msg.size()), msg.data());
    std::fflush(stdout);
    if (g.file.is_open()) { g.file << msg << '\n'; g.file.flush(); }
}

// --- formatted front-ends --------------------------------------------------
template <class... A> void error(std::format_string<A...> f, A&&... a) { emit(Level::Error, std::format(f, std::forward<A>(a)...)); }
template <class... A> void warn (std::format_string<A...> f, A&&... a) { emit(Level::Warn,  std::format(f, std::forward<A>(a)...)); }
template <class... A> void info (std::format_string<A...> f, A&&... a) { emit(Level::Info,  std::format(f, std::forward<A>(a)...)); }
template <class... A> void debug(std::format_string<A...> f, A&&... a) { emit(Level::Debug, std::format(f, std::forward<A>(a)...)); }
template <class... A> void line (std::format_string<A...> f, A&&... a) { emit_line(std::format(f, std::forward<A>(a)...)); }

}  // namespace sub0::log
