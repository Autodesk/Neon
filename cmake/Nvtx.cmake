#Toggle NVTX ranges. Enabled by default on linux. To enable use "-DNEON_USE_NVTX=ON"
if (WIN32)
	set(NEON_USE_NVTX "OFF" CACHE BOOL "Use NVTX Ranges")
else ()
	set(NEON_USE_NVTX "ON" CACHE BOOL "Use NVTX Ranges")
endif ()

if (${NEON_USE_NVTX})
	message(STATUS "NVTX Ranges is enabled")
else ()
	message(STATUS "NVTX Ranges is disabled")
endif ()
