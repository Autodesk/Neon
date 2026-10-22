# Carbon benchmark programs.
#
# Larger / more representative workloads than the toy examples — primary purpose
# is to exercise the public API end-to-end on a realistic scientific kernel and
# to surface MLUPS / GLUPS / wall-time numbers that downstream perf work can
# track. Anything functional / verification-shaped lives under tests/ instead.
#
# Backend dispatch (Cuda vs Sycl + cuda:/hip:) follows the exact same pattern as
# examples/CMakeLists.txt: under Cuda the .cu file is compiled with nvcc; under
# Sycl it is configure_file'd to a .cpp twin in the build tree and routed
# through add_sycl_to_target. Keep that single helper in this directory rather
# than reusing the examples one so future benchmark-only options (e.g. PGO,
# nvbench integration, MPS toggles) can land here without touching the example
# helper.

function(_carbon_benchmark_add target source_cu)
  if(CARBON_BACKEND STREQUAL "Cuda")
    add_executable(${target} ${source_cu})
    target_compile_options(${target} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:${CARBON_NVCC_DEVICE_FLAGS}>)
    set_target_properties(${target} PROPERTIES
      BUILD_RPATH "${CARBON_RUNTIME_SEARCH_DIRS}"
      CUDA_STANDARD 20
      CUDA_STANDARD_REQUIRED ON)
  elseif(CARBON_BACKEND STREQUAL "Hip")
    add_executable(${target} ${source_cu})
    set_source_files_properties(${source_cu} PROPERTIES LANGUAGE HIP)
    if(CARBON_HIP_NVCC)
      target_compile_options(${target} PRIVATE $<$<COMPILE_LANGUAGE:HIP>:${CARBON_NVCC_DEVICE_FLAGS}>)
    endif()
    set_target_properties(${target} PROPERTIES
      BUILD_RPATH "${CARBON_RUNTIME_SEARCH_DIRS}"
      HIP_STANDARD 20
      HIP_STANDARD_REQUIRED ON)
  elseif(CARBON_BACKEND STREQUAL "Sycl")
    set(_src "${CMAKE_CURRENT_BINARY_DIR}/${target}.cpp")
    configure_file("${CMAKE_CURRENT_SOURCE_DIR}/${source_cu}" "${_src}" COPYONLY)
    add_executable(${target} "${_src}")
    target_compile_features(${target} PRIVATE cxx_std_20)
    set_target_properties(${target} PROPERTIES
      BUILD_RPATH "${CARBON_RUNTIME_SEARCH_DIRS};${CMAKE_BINARY_DIR}")
    add_sycl_to_target(TARGET ${target} SOURCES "${_src}")
    if(CARBON_SYCL_TARGETS MATCHES "cuda")
      target_compile_options(${target} PRIVATE --cuda-path=${CARBON_SYCL_CUDA_PATH})
      target_link_directories(${target} PRIVATE ${CARBON_SYCL_CUDA_PATH}/lib64)
    endif()
  endif()
endfunction()

add_subdirectory(common)
add_subdirectory(lbm_lid_driven_cavity)
add_subdirectory(halo_scalability)
add_subdirectory(reduce_throughput)
add_subdirectory(skeleton_overlap)
add_subdirectory(block_sparse)
add_subdirectory(jacobi_nvidia)
