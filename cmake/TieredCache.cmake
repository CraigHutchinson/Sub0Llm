include_guard(GLOBAL)
include(FetchContent)
# Shared by the optional engine expert cache (--moe-io-mode cache) and the standalone storage tests.
# MemPage comes first and from Sub0Llm itself, so one MemPage target serves both libraries.
include("${CMAKE_CURRENT_LIST_DIR}/MemPage.cmake")
# 0511d55 added SizeClassedTable (exact-width slots per expert size), fill_chunk_bytes (chunked row fills) and
# fill_alignment (aligned-window fills for uncached reads), all used by the expert cache; 1fa4700 is that
# plus TieredCache's own MemPage pin moved to the revision in MemPage.cmake. Published. For offline work,
# set FETCHCONTENT_SOURCE_DIR_SUB0TIEREDCACHE to a checkout at exactly this revision.
set(SUB0_STORAGE_SUB0TIEREDCACHE_REVISION "1fa4700d70e1468eade2d8f402d4e7d25ef0a964")
set(SUB0TIEREDCACHE_BUILD_TESTING OFF CACHE BOOL "" FORCE)
set(SUB0TIEREDCACHE_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
FetchContent_Declare(Sub0TieredCache
  GIT_REPOSITORY https://github.com/CraigHutchinson/Sub0TieredCache.git
  GIT_TAG ${SUB0_STORAGE_SUB0TIEREDCACHE_REVISION})
FetchContent_MakeAvailable(Sub0TieredCache)
