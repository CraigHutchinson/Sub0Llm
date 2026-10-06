include_guard(GLOBAL)
include(FetchContent)
# Shared by the optional engine expert cache (--moe-io-mode cache) and the standalone storage tests.
# MemPage comes first and from Sub0Llm itself, so one MemPage target serves both libraries.
include("${CMAKE_CURRENT_LIST_DIR}/MemPage.cmake")
# 0511d55: SizeClassedTable (exact-width slots per expert size), fill_chunk_bytes (chunked row fills) and
# fill_alignment (aligned-window fills for uncached reads), all used by the expert cache. Not yet published: until it is, build with FETCHCONTENT_SOURCE_DIR_SUB0TIEREDCACHE at
# exactly this revision, plus FETCHCONTENT_FULLY_DISCONNECTED=ON.
set(SUB0_STORAGE_SUB0TIEREDCACHE_REVISION "0511d55")
set(SUB0TIEREDCACHE_BUILD_TESTING OFF CACHE BOOL "" FORCE)
set(SUB0TIEREDCACHE_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
FetchContent_Declare(Sub0TieredCache
  GIT_REPOSITORY https://github.com/CraigHutchinson/Sub0TieredCache.git
  GIT_TAG ${SUB0_STORAGE_SUB0TIEREDCACHE_REVISION})
FetchContent_MakeAvailable(Sub0TieredCache)
