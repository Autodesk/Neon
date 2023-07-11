#Toggle NVTX ranges. To enable use "-DNEON_USE_NVTX=ON"

set(NEON_USE_NVTX "ON" CACHE BOOL "Use NVTX Ranges")

if (${NEON_USE_NVTX})
	message(STATUS "NVTX Ranges is enabled")
else ()
	message(STATUS "NVTX Ranges is disabled")
endif ()
