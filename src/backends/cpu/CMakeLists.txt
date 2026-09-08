# CPU backend source manifest. Keep backend-owned source selection beside the backend.
set(SUB0_BACKEND_SOURCES
    "${CMAKE_CURRENT_LIST_DIR}/backend.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/api.cpp"
    PARENT_SCOPE)
