# SYCL registration of a .cu source as an executable.
#
# acpp / clang sniff the .cu extension regardless of CMake's LANGUAGE CXX setting and turn
# on offload modes that need a libomptarget-nvptx-sm_XX.bc bitcode library we don't ship.
# The source is therefore copied to a .cpp twin in the build tree (configure_file COPYONLY
# re-runs whenever the .cu changes) and that twin is what acpp compiles. Nothing in a
# dual-backend .cu is CUDA-specific (no <<<>>>, no __global__); the extension only routes
# it through nvcc on the CUDA build.
#
# The twin lives in the binary dir, so quoted includes of headers that sit next to the
# original source ("test_domain.h", "wind_tunnel_common.h", ...) would not resolve; the
# source directory is added to the target's private include path to keep them working.
#
# Usage: carbon_add_sycl_mirrored_executable(<target> <source.cu>)
#   <source.cu> is relative to CMAKE_CURRENT_SOURCE_DIR of the caller.
function(carbon_add_sycl_mirrored_executable target source_cu)
  set(_src "${CMAKE_CURRENT_BINARY_DIR}/${target}.cpp")
  configure_file("${CMAKE_CURRENT_SOURCE_DIR}/${source_cu}" "${_src}" COPYONLY)
  get_filename_component(_src_dir "${CMAKE_CURRENT_SOURCE_DIR}/${source_cu}" DIRECTORY)
  add_executable(${target} "${_src}")
  target_include_directories(${target} PRIVATE "${_src_dir}")
  target_compile_features(${target} PRIVATE cxx_std_20)
  add_sycl_to_target(TARGET ${target} SOURCES "${_src}")
  if(CARBON_SYCL_TARGETS MATCHES "cuda")
    target_compile_options(${target} PRIVATE --cuda-path=${CARBON_SYCL_CUDA_PATH})
    target_link_directories(${target} PRIVATE ${CARBON_SYCL_CUDA_PATH}/lib64)
  endif()
endfunction()
