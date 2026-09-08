# CPU backend source manifest. Keep backend-owned source selection beside the backend.
# NOTE: included via include(), not add_subdirectory() -- there is no child scope to pop out
# of, so PARENT_SCOPE here would silently set an unused variable and leave this one empty in
# the caller (cmake/Backends.cmake, itself include()'d from the top-level CMakeLists.txt).
set(SUB0_BACKEND_SOURCES
    "${CMAKE_CURRENT_LIST_DIR}/backend.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/api.cpp")
