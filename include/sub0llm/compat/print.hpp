#pragma once
// std::print / std::println compat shim for GCC 13 / Clang 17.
// Drop this header once we can assume GCC 14+ or Clang 18+.
#if __has_include(<print>)
#  include <print>
#else
#  include <cstdio>
#  include <format>
#  include <string>

namespace std {

template<typename... Args>
void print(std::format_string<Args...> fmt, Args&&... args) {
    const std::string s = std::format(fmt, std::forward<Args>(args)...);
    std::fputs(s.c_str(), stdout);
}

template<typename... Args>
void println(std::format_string<Args...> fmt, Args&&... args) {
    const std::string s = std::format(fmt, std::forward<Args>(args)...);
    std::puts(s.c_str());
}

inline void println() { std::putchar('\n'); }

} // namespace std
#endif
