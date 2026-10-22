# Backend-aware example registration.
#
# Under CARBON_BACKEND=Cuda the .cu file is compiled with nvcc as a CUDA TU (the
# original behaviour). Under CARBON_BACKEND=Sycl the .cu file is copied to a .cpp twin in
# the build tree and routed through add_sycl_to_target — the same trick tests/CMakeLists
# uses to dodge clang's .cu extension sniffing for libomptarget-nvptx-sm_80.bc.
#
# The .cpp init_and_dense example is plain C++ and does not include any backend-specific
# detail header that pulls sycl::queue, so under Sycl it still goes through
# add_sycl_to_target only because the public carbon headers transitively include
# detail/backend/sycl/* under that backend.
function(_carbon_example_add target source_cu)
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
    set_target_properties(${target} PROPERTIES BUILD_RPATH "${CARBON_RUNTIME_SEARCH_DIRS};${CMAKE_BINARY_DIR}")
    add_sycl_to_target(TARGET ${target} SOURCES "${_src}")
    if(CARBON_SYCL_TARGETS MATCHES "cuda")
      target_compile_options(${target} PRIVATE --cuda-path=${CARBON_SYCL_CUDA_PATH})
      target_link_directories(${target} PRIVATE ${CARBON_SYCL_CUDA_PATH}/lib64)
    endif()
  endif()
endfunction()

if(CARBON_BUILD_VIEWER)
  add_library(carbon_viewer STATIC viewer/carbon_viewer.cpp)
  add_library(carbon::viewer ALIAS carbon_viewer)
  target_link_libraries(carbon_viewer PUBLIC carbon::carbon carbon::imgui)
  target_include_directories(carbon_viewer PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")
  set_target_properties(carbon_viewer PROPERTIES POSITION_INDEPENDENT_CODE ON)

  set(_carbon_viewer_desktop OFF)
  if(CARBON_VIEWER_DESKTOP)
    set(_carbon_viewer_desktop ON)
    target_compile_definitions(carbon_viewer PUBLIC CARBON_HAVE_VIEWER_DESKTOP=1)
  endif()

  set(_carbon_viewer_web OFF)
  if(CARBON_VIEWER_WEB AND TARGET carbon::imgui_ws)
    set(_carbon_viewer_web ON)
    target_link_libraries(carbon_viewer PRIVATE carbon::imgui_ws)
    target_compile_definitions(carbon_viewer PUBLIC CARBON_HAVE_VIEWER_WEB=1)
  endif()

  function(_carbon_example_link_viewer target)
    target_link_libraries(${target} PRIVATE carbon_viewer)
    target_compile_definitions(${target} PRIVATE CARBON_HAVE_VIEWER=1)
    if(_carbon_viewer_desktop)
      target_compile_definitions(${target} PRIVATE CARBON_HAVE_VIEWER_DESKTOP=1)
    endif()
    if(_carbon_viewer_web)
      target_compile_definitions(${target} PRIVATE CARBON_HAVE_VIEWER_WEB=1)
    endif()
  endfunction()
else()
  function(_carbon_example_link_viewer target)
  endfunction()
endif()

# init_and_dense is plain C++; under Sycl it still needs the SYCL include path because
# carbon's public headers pull in <sycl/sycl.hpp> through detail/backend/sycl/*.
add_executable(carbon_example_init_and_dense init_and_dense.cpp)
target_link_libraries(carbon_example_init_and_dense PRIVATE carbon::carbon)
set_target_properties(carbon_example_init_and_dense PROPERTIES
  BUILD_RPATH "${CARBON_RUNTIME_SEARCH_DIRS}")
if(CARBON_BACKEND STREQUAL "Sycl")
  add_sycl_to_target(TARGET carbon_example_init_and_dense SOURCES init_and_dense.cpp)
  if(CARBON_SYCL_TARGETS MATCHES "cuda")
    target_compile_options(carbon_example_init_and_dense PRIVATE --cuda-path=${CARBON_SYCL_CUDA_PATH})
    target_link_directories(carbon_example_init_and_dense PRIVATE ${CARBON_SYCL_CUDA_PATH}/lib64)
  endif()
endif()

# The canonical carbon Jacobi: one source file covering every grid family
# (Dense / DenseDisg / Block), every backend (CUDA / SYCL), every transport
# (single-GPU / multi-GPU same-node / cross-rank NCCL/RCCL/oneCCL), and every
# Skeleton::OverlapMode (none / standard / extended / two-way-extended)
# selected at runtime via --grid / --overlap CLI flags.
_carbon_example_add(carbon_example_jacobi_unified jacobi_unified.cu)
target_link_libraries(carbon_example_jacobi_unified PRIVATE carbon::carbon)

# Non-Jacobi solver. Same one-source-file pattern as @c jacobi_unified : one
# CG-Poisson implementation across every grid family / backend / transport,
# with @c --grid , @c --overlap , and @c --dot CLI flags.
_carbon_example_add(carbon_example_cg_poisson_unified cg_poisson_unified.cu)
target_link_libraries(carbon_example_cg_poisson_unified
  PRIVATE carbon::carbon carbon::third_party::clipp)

_carbon_example_add(carbon_example_salt_pepper_filter salt_pepper_filter.cu)
target_link_libraries(carbon_example_salt_pepper_filter PRIVATE carbon::carbon)
# Bake the absolute path of examples/data into the binary so running with no args picks
# up the bundled astronaut.pgm regardless of cwd. The example only ever feeds this path
# to std::ifstream / std::filesystem -- no shell, no exec.
target_compile_definitions(carbon_example_salt_pepper_filter
  PRIVATE "CARBON_EXAMPLE_DATA_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/data\"")

# End-to-end tutorial for carbon::CheckpointWriter / CheckpointReader on the Dense
# grid family. Walks through save -> inspect -> reload -> verify with both
# sync_from_device and sync_to_device on; runs identically single-process and under
# mpiexec -n N when carbon was built with CARBON_ENABLE_MPI=ON.
_carbon_example_add(carbon_example_checkpoint_dense checkpoint_dense.cu)
target_link_libraries(carbon_example_checkpoint_dense PRIVATE carbon::carbon)

# Sphere SDF field, checkpoint/restart, downsampled XY slice, PGM export (single-rank
# image path; MPI multi-rank skips PGM with a log hint).
_carbon_example_add(carbon_example_checkpoint_sphere_image checkpoint_sphere_image.cu)
target_link_libraries(carbon_example_checkpoint_sphere_image PRIVATE carbon::carbon)

# Parallel density splatting: sphere SDF field, perspective camera, multi-GPU combine, PGM.
_carbon_example_add(carbon_example_render_density render_density.cu)
target_link_libraries(carbon_example_render_density PRIVATE carbon::carbon)
_carbon_example_link_viewer(carbon_example_render_density)
_carbon_example_add(carbon_example_render_density_mgrid render_density_mgrid.cu)
target_link_libraries(carbon_example_render_density_mgrid PRIVATE carbon::carbon)
_carbon_example_link_viewer(carbon_example_render_density_mgrid)
_carbon_example_add(carbon_example_render_ray_march render_ray_march.cu)
target_link_libraries(carbon_example_render_ray_march PRIVATE carbon::carbon)
_carbon_example_link_viewer(carbon_example_render_ray_march)
_carbon_example_add(carbon_example_render_wind_tunnel render_wind_tunnel.cu)
target_link_libraries(carbon_example_render_wind_tunnel PRIVATE carbon::carbon)
_carbon_example_link_viewer(carbon_example_render_wind_tunnel)
# Reuse the self-contained carbon::bench::lbm headers (cell_type.h / d3q27.h /
# lbm_kernels.h) so the wind-tunnel example runs a real D3Q27 + KBC LBM solve.
target_include_directories(carbon_example_render_wind_tunnel
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../benchmarks/lbm_lid_driven_cavity/src)

_carbon_example_add(carbon_example_render_wind_tunnel_live render_wind_tunnel_live.cu)
target_link_libraries(carbon_example_render_wind_tunnel_live PRIVATE carbon::carbon)
_carbon_example_link_viewer(carbon_example_render_wind_tunnel_live)
target_include_directories(carbon_example_render_wind_tunnel_live
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../benchmarks/lbm_lid_driven_cavity/src)

# Multi-resolution (mGrid) worked example: cross-level restriction / prolongation, a reduction,
# and an intra-level halo'd smoothing step on carbon::MultiResBlock — every operation through the
# kernel layer (make_kernel / make_reduce_kernel / level_view / make_halo_kernel), each stage
# self-verified against a host reference (exact for the linear seed field).
_carbon_example_add(carbon_example_mgrid_multires mgrid_multires.cu)
target_link_libraries(carbon_example_mgrid_multires PRIVATE carbon::carbon)