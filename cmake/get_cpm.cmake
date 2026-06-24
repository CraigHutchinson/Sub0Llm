# cmake/get_cpm.cmake — download CPM.cmake on first configure if not cached.
# Pin the version here; bump deliberately when updating.
set(CPM_DOWNLOAD_VERSION 0.40.2)
set(CPM_DOWNLOAD_LOCATION "${CMAKE_BINARY_DIR}/cmake/CPM_${CPM_DOWNLOAD_VERSION}.cmake")

if(NOT EXISTS "${CPM_DOWNLOAD_LOCATION}")
  message(STATUS "Downloading CPM.cmake v${CPM_DOWNLOAD_VERSION}...")
  file(DOWNLOAD
    "https://github.com/cpm-cmake/CPM.cmake/releases/download/v${CPM_DOWNLOAD_VERSION}/CPM.cmake"
    "${CPM_DOWNLOAD_LOCATION}"
    TIMEOUT 60
  )
endif()

include("${CPM_DOWNLOAD_LOCATION}")
