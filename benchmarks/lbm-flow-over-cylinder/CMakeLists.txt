cmake_minimum_required(VERSION 3.19 FATAL_ERROR)

SET(APP "lbm-flow-over-cylinder")

file(GLOB_RECURSE SrcFiles src/*.*)

add_executable(${APP} ${SrcFiles})

target_link_libraries(${APP}
		PUBLIC libNeonDomain
		PUBLIC libNeonSkeleton)

set_target_properties(${APP} PROPERTIES
		CUDA_SEPARABLE_COMPILATION ON
		CUDA_RESOLVE_DEVICE_SYMBOLS ON)

target_compile_options(${APP} INTERFACE
		$<$<COMPILE_LANGUAGE:CXX>:${NeonCXXFlags}>
		$<$<COMPILE_LANGUAGE:CUDA>:${NeonCUDAFlags}>
		)
