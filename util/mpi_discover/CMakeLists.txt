cmake_minimum_required(VERSION 3.19 FATAL_ERROR)

SET(APP "utils-mpi-discover")
# Find MPI
find_package(MPI REQUIRED)
# Find NCCL
find_library(NCCL_LIBRARIES nccl)
find_path(NCCL_INCLUDE_DIRS nccl.h)

file(GLOB_RECURSE SrcFiles src/*.*)

add_executable(${APP} ${SrcFiles})

target_link_libraries(${APP}
		PUBLIC libNeonDomain
		PUBLIC MPI::MPI_CXX
		PUBLIC ${NCCL_LIBRARIES})



# Include MPI headers
target_include_directories(${APP} PRIVATE ${MPI_INCLUDE_PATH} ${NCCL_INCLUDE_DIRS})

set_target_properties(${APP} PROPERTIES
		CUDA_SEPARABLE_COMPILATION ON
		CUDA_RESOLVE_DEVICE_SYMBOLS ON)

target_compile_options(${APP} INTERFACE
		$<$<COMPILE_LANGUAGE:CXX>:${NeonCXXFlags}>
		$<$<COMPILE_LANGUAGE:CUDA>:${NeonCUDAFlags}>
		)

#set_target_properties(${APP} PROPERTIES FOLDER "libNeonDomain")
#source_group(TREE ${CMAKE_CURRENT_LIST_DIR} PREFIX "gUt_containers" FILES ${SrcFiles})

#add_test(NAME ${APP} COMMAND ${APP})