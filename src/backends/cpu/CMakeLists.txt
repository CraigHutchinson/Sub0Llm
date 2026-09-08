# CPU backend source manifest. Keep backend-owned source selection beside the backend.
#
# NOT `PARENT_SCOPE`. cmake/Backends.cmake pulls this in with `include()`, which executes in the
# CALLER'S scope rather than creating a child one -- so `set(... PARENT_SCOPE)` here writes above the
# top-level directory scope, where nothing reads it, and leaves SUB0_BACKEND_SOURCES EMPTY at the
# `add_library(sub0_core ...)` that consumes it. The symptom is not a configure error: the configure
# banner prints "(engine: )" and the build gets all the way to a link failure on every sub0:: entry
# point the backend defines. Fixed by a plain `set()`, which is what an included file needs. If this
# ever becomes an `add_subdirectory()` instead, it will need PARENT_SCOPE back -- so the include site
# and this line are a pair, and moving one means moving the other.
set(SUB0_BACKEND_SOURCES
    "${CMAKE_CURRENT_LIST_DIR}/backend.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/api.cpp")
