# Register a unit test: source file test_<name>.cpp, executable test_<name>, CTest name carbon_<name>.
#
# Under CARBON_BACKEND=Sycl the test sees carbon's public headers, which transitively pull
# in detail/backend/sycl/{stream,event,gpu}.h and therefore <sycl/sycl.hpp>. The SYCL
# include path is only on the command line for TUs registered via add_sycl_to_target, so
# every test (CPU-only included) goes through that helper under SYCL — same way carbon's
# own .cpp sources are tagged in the top-level CMakeLists.
# Tests state their invariants with <cassert>. CMake defines NDEBUG for Release and
# RelWithDebInfo, which turns every assert() into a no-op — and Release is what the
# documented workflow and both CI lanes configure. Measured 2026-07: 305 assert() call
# sites across 11 test files, 7 of which have no other failure mechanism, so those
# binaries reported success regardless of what they computed. Undefine NDEBUG for test
# TUs only (the library keeps its Release semantics) so the assertions are live.
function(carbon_test_enable_assertions target)
  target_compile_options(${target} PRIVATE -UNDEBUG)
endfunction()

function(carbon_add_unit_test name)
  set(target "test_${name}")
  add_executable(${target} ${target}.cpp)
  target_compile_features(${target} PRIVATE cxx_std_23)
  target_link_libraries(${target} PRIVATE carbon::carbon)
  carbon_test_enable_assertions(${target})
  set_target_properties(${target} PROPERTIES BUILD_RPATH "${CMAKE_BINARY_DIR}")
  if(CARBON_BACKEND STREQUAL "Sycl")
    add_sycl_to_target(TARGET ${target} SOURCES ${target}.cpp)
    if(CARBON_SYCL_TARGETS MATCHES "cuda")
      target_compile_options(${target} PRIVATE --cuda-path=${CARBON_SYCL_CUDA_PATH})
      target_link_directories(${target} PRIVATE ${CARBON_SYCL_CUDA_PATH}/lib64)
    endif()
  endif()
  add_test(NAME carbon_${name} COMMAND ${target})
  # Most registrations here are CPU-only and may run in parallel. A few .cpp
  # tests still touch CUDA (see RESOURCE_LOCK grouping after the call sites).
  #
  # Label semantics (see the label override blocks further down):
  #   cpu             - verified to pass with zero visible devices AND to do real work
  #                     there, i.e. it does not reach a device-count self-skip. This is
  #                     what the CPU-only CI lane selects on.
  #   gpu             - needs at least one device to execute its assertions.
  #   needs_multi_device - core coverage needs >= 2 devices; self-skips below that.
  # No label is a substring of another: ctest -L takes a regex, so a "needs_multi_gpu"
  # label would also be selected by -L gpu and dropped by -LE gpu.
  # Host .cpp tests default to cpu and are demoted individually below.
  set_tests_properties(carbon_${name} PROPERTIES LABELS "cpu")
endfunction()

# Backend-aware compute unit test: source file test_<name>.cu.
#
# Under CARBON_BACKEND=Cuda the .cu file is compiled with nvcc as a CUDA TU (the existing
# behaviour). Under CARBON_BACKEND=Sycl the same .cu file is compiled with acpp as a SYCL
# TU via add_sycl_to_target — acpp accepts the .cu suffix and treats it the same as a
# .cpp SYCL TU. Tests that pass under both backends are registered here unconditionally;
# tests that still need backend-specific glue (cudaXxx APIs, CUB, etc. hand-rolled in the
# test body) opt into CUDA-only with carbon_add_cuda_only_unit_test below.
function(carbon_add_cuda_unit_test name)
  set(target "test_${name}")
  if(CARBON_BACKEND STREQUAL "Cuda")
    add_executable(${target} ${target}.cu)
    target_link_libraries(${target} PRIVATE carbon::carbon)
    target_compile_options(${target} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:${CARBON_NVCC_DEVICE_FLAGS}>)
    set_target_properties(${target} PROPERTIES
      BUILD_RPATH "${CMAKE_BINARY_DIR}"
      CUDA_STANDARD 20
      CUDA_STANDARD_REQUIRED ON)
  elseif(CARBON_BACKEND STREQUAL "Hip")
    # The .cu source is compiled as HIP device code. On AMD (amdclang++/hipcc) extended
    # device lambdas are on by default; on the NVIDIA platform nvcc backs the HIP language
    # and needs --extended-lambda (added below).
    add_executable(${target} ${target}.cu)
    set_source_files_properties(${target}.cu PROPERTIES LANGUAGE HIP)
    target_link_libraries(${target} PRIVATE carbon::carbon)
    if(CARBON_HIP_NVCC)
      target_compile_options(${target} PRIVATE $<$<COMPILE_LANGUAGE:HIP>:${CARBON_NVCC_DEVICE_FLAGS}>)
    endif()
    set_target_properties(${target} PROPERTIES
      BUILD_RPATH "${CMAKE_BINARY_DIR}"
      HIP_STANDARD 20
      HIP_STANDARD_REQUIRED ON)
  elseif(CARBON_BACKEND STREQUAL "Sycl")
    # acpp / clang sniff the .cu extension regardless of CMake's LANGUAGE CXX setting and
    # turn on offload modes that need a libomptarget-nvptx-sm_80.bc bitcode library we
    # don't ship. The simplest workaround is to copy the source to a .cpp twin in the
    # build tree before compilation; configure_file with COPYONLY does that and re-runs
    # whenever the .cu changes (so an edit doesn't need a fresh configure). Nothing in
    # the test source itself is .cu-specific (no <<<>>>, no __global__) — the .cu
    # extension is only there to make CMake route through nvcc on the CUDA build.
    set(_src "${CMAKE_CURRENT_BINARY_DIR}/${target}.cpp")
    configure_file("${CMAKE_CURRENT_SOURCE_DIR}/${target}.cu" "${_src}" COPYONLY)
    add_executable(${target} "${_src}")
    target_link_libraries(${target} PRIVATE carbon::carbon)
    target_compile_features(${target} PRIVATE cxx_std_20)
    set_target_properties(${target} PROPERTIES BUILD_RPATH "${CMAKE_BINARY_DIR}")
    add_sycl_to_target(TARGET ${target} SOURCES "${_src}")
    if(CARBON_SYCL_TARGETS MATCHES "cuda")
      target_compile_options(${target} PRIVATE --cuda-path=${CARBON_SYCL_CUDA_PATH})
      target_link_directories(${target} PRIVATE ${CARBON_SYCL_CUDA_PATH}/lib64)
    endif()
  endif()
  add_test(NAME carbon_${name} COMMAND ${target})
  # Compute tests in this list grab every visible CUDA device through
  # carbon::get_all_devices() and run real launches, peer copies, and
  # halo exchanges on them. The historical "tiny FP drift" flakiness was
  # root-caused (2026-06) to a real correctness bug — the multi-GPU halo
  # exchange issued cudaMemcpyPeerAsync on the destination GPU's stream
  # with no cross-GPU fence against the source GPU's prior compute — and
  # fixed at the source by the producer fence now in carbon::Halo /
  # HaloDisg / HaloBlock (StreamGroup::wait_for_all). With that fix the
  # multi-process skeleton stress is clean (0/160 vs the prior ~98% fail).
  # RESOURCE_LOCK is retained as defense-in-depth against the *resource*
  # side of -j contention these tests still share — aggregate device
  # memory when every test grabs all GPUs at once, plus NCCL collective
  # interference between MPI tests — not as the fix for the halo race.
  #
  # Every test registered here needs at least one device, so none carries the
  # cpu label: with zero devices they either fail outright or reach a self-skip
  # and assert nothing. The CPU-only CI lane must not select them.
  set_tests_properties(carbon_${name} PROPERTIES
    RESOURCE_LOCK carbon_gpu
    LABELS "gpu")
  if(TARGET ${target})
    carbon_test_enable_assertions(${target})
  endif()
endfunction()

# Tests that have not yet been adapted for SYCL (still call cudaXxx APIs, use <<<>>>
# launch syntax in the test body, depend on CUB, etc.) register through this helper so
# they only build under CARBON_BACKEND=Cuda. The intent is for this list to shrink as
# tests get migrated to backend-neutral synchronisation primitives (Stream::wait,
# carbon::Kernel, etc.).
function(carbon_add_cuda_only_unit_test name)
  if(NOT CARBON_BACKEND STREQUAL "Cuda")
    return()
  endif()
  carbon_add_cuda_unit_test(${name})
endfunction()

carbon_add_unit_test(dynamic_load)
carbon_add_unit_test(logging)
carbon_add_unit_test(gpu)
carbon_add_unit_test(gpu_group_all_devices)
carbon_add_unit_test(data_view)
carbon_add_unit_test(dense)
carbon_add_unit_test(dense_disg)
carbon_add_unit_test(block_static)
carbon_add_unit_test(mgrid_static)
carbon_add_unit_test(memory_side)
# Host branch of detail::atomic_add / atomic_min under real thread contention. Pure host
# code (std::thread, no device), so it stays on the cpu lane.
carbon_add_unit_test(atomic_host)
carbon_add_unit_test(stencil)
carbon_add_unit_test(system_default_stream)
carbon_add_unit_test(transfer)
carbon_add_unit_test(halo_coalesce)
# Environment gate for the needs_multi_device class — see tests/test_multi_gpu_gate.cpp.
# Labelled cpu because it only reads the device count (never launches), so it runs on
# every lane and reports honestly on each.
carbon_add_unit_test(multi_gpu_gate)
set_tests_properties(carbon_multi_gpu_gate PROPERTIES LABELS "cpu;needs_multi_device")

# Host .cpp tests that enumerate devices or allocate GPU memory share the same
# ctest RESOURCE_LOCK as carbon_add_cuda_unit_test so ctest -j never doubles up
# on aggregate GPU memory. (The halo cross-GPU race that once surfaced here as a
# flaky carbon_skeleton parallel-sibling-vs-ref mismatch is now fixed at the
# source — see the RESOURCE_LOCK rationale on carbon_add_cuda_unit_test above.)
#
# These also drop the default cpu label: verified 2026-07 by running the suite under
# CUDA_VISIBLE_DEVICES="", each either fails outright (gpu, gpu_group_all_devices) or
# reaches a device-count self-skip and asserts nothing (dense, mgrid_static,
# system_default_stream). Leaving them on the CPU lane is what made that lane red.
set_tests_properties(
  carbon_gpu
  carbon_gpu_group_all_devices
  carbon_dense
  carbon_mgrid_static
  carbon_system_default_stream
  PROPERTIES
    RESOURCE_LOCK carbon_gpu
    LABELS "gpu")
# Tests adapted for both backends (no direct cudaXxx calls in the test body, only
# carbon-managed primitives — Stream::wait, Transfer, Kernel, ReductionResult — so the
# SYCL backend's launch + reduction paths are exercised by the same source). Built
# under both Cuda and Sycl backends.
carbon_add_cuda_unit_test(kernel_laplacian_single_gpu)
carbon_add_cuda_unit_test(reduce_sum)
carbon_add_cuda_unit_test(reduce_scalar_ring)
carbon_add_cuda_unit_test(skeleton_scalar_pattern)
carbon_add_cuda_unit_test(skeleton_scalar_pattern_multi_gpu)
carbon_add_cuda_unit_test(reduce_atomic)
carbon_add_cuda_unit_test(reduce_fused_stencil)
carbon_add_cuda_unit_test(render_density)
carbon_add_cuda_unit_test(render_surfel)
carbon_add_cuda_unit_test(render_ray_march)
carbon_add_cuda_unit_test(render_ray_march_sparse)
carbon_add_cuda_unit_test(block_locator)
carbon_add_cuda_unit_test(mgrid_locator)
carbon_add_cuda_unit_test(render_sample)
carbon_add_cuda_unit_test(reduce_ops)
carbon_add_cuda_unit_test(reduce_sum_multi_gpu)
carbon_add_cuda_unit_test(reduce_sum_host_multi_gpu)
carbon_add_cuda_unit_test(kernel_polymorphic)
carbon_add_cuda_unit_test(field_fill_host)
carbon_add_cuda_unit_test(field_lazy_host)
carbon_add_cuda_unit_test(data_view_split)
carbon_add_cuda_unit_test(kernel_host_parity)
carbon_add_cuda_unit_test(halo_single_node)
carbon_add_cuda_unit_test(dense_disg_multi_gpu)
carbon_add_cuda_unit_test(dense_disg_kernel)
carbon_add_cuda_unit_test(dense_disg_reduce)
carbon_add_cuda_unit_test(dense_disg_host)
carbon_add_cuda_unit_test(halo_disg_periodic_z)
carbon_add_cuda_unit_test(dense_disg_overlapped)
carbon_add_cuda_unit_test(dense_disg_skeleton_extended)
# --- CUDA-only tests (rationale tracked in P2.γ of docs/audit_2026_05_third_pass.md
# and P2.β of docs/audit_2026_05_fourth_pass.md). Each entry's per-line comment
# records why it isn't on the cross-backend gate. The 2026-05-18 P2.β closure
# (System::async_memset wrapper + Block-family test ports) promoted nine tests
# off this gate; the only remaining entry is block_cuda, which is genuinely
# backend-specific by design.
carbon_add_cuda_unit_test(skeleton_block_extended)
carbon_add_cuda_unit_test(halo_multicomponent)
carbon_add_cuda_unit_test(lattice_halo_disg)
carbon_add_cuda_unit_test(lattice_halo_aggregated)
carbon_add_cuda_unit_test(lattice_halo_dense_auto)
carbon_add_cuda_unit_test(lattice_halo_disg_auto)
# block_cuda: smoke test legitimately exercising raw cudaMalloc / cudaMemset /
# cudaDeviceSynchronize / cudaGetErrorString to validate the CUDA-runtime
# boundary directly. Genuinely backend-specific by design.
carbon_add_cuda_only_unit_test(block_cuda)
# mgrid_cross_level: single-GPU MultiResBlock cross-level data-plane test. Uses a hand-rolled
# <<<>>> kernel (PR1 defers KernelLoader integration for mGrid), so it is genuinely
# backend-specific by design — same rationale as block_cuda.
carbon_add_cuda_only_unit_test(mgrid_cross_level)
# mgrid_multi_gpu: intra-process multi-GPU MultiResBlock co-resident cross-level test. Builds
# the depth-2 tower across every visible GPU and verifies a parent brick and all of its
# children stay on one GPU (cross-level reads are pure local dereferences). Raw <<<>>> kernel
# + per-device cudaMalloc/cudaMemcpy for counters — CUDA-only, same rationale as block_cuda.
carbon_add_cuda_only_unit_test(mgrid_multi_gpu)
# Descendant-weighted coarsest z-cut under imbalanced refinement (2-GPU load balance + co-residency).
carbon_add_cuda_only_unit_test(mgrid_weighted_cuts)
# mgrid_kernel: the mGrid *kernel layer* — MultiResBlock::make_kernel + MultiResField::level_view
# bound through the ordinary KernelLoader, with NO hand-rolled <<<>>> kernel. Fills each level
# and verifies cross-level parent/child reads on the co-resident layout across every visible GPU.
carbon_add_cuda_unit_test(mgrid_kernel)
# mgrid_halo: intra-level halo across every visible GPU. mGrid's cross-level path is
# transfer-free (co-residency), so the only data movement is the same-level neighbour exchange;
# this verifies it at every level of the projected co-resident cut via
# MultiResField::make_halo_kernel + byte-level ghost-content checks (test_halo_block Invariant 1
# per level). Needs >= 2 GPUs (skips otherwise).
carbon_add_cuda_unit_test(mgrid_halo)
# mgrid_skeleton: multi-level Skeleton integration. The Skeleton is grid-agnostic; this proves a
# V-cycle-shaped pipeline (fill -> restrict -> stencil-smooth -> prolong) rides it correctly once
# cross-level reads are declared via MultiResField::bind_cross_level (the structural assertions
# check the fill->restrict / restrict->prolong dependency edges those tokens produce) and that the
# per-level auto-halo is inserted before the stencil consumer.
carbon_add_cuda_unit_test(mgrid_skeleton)
# Cross-tower OverlapMode parity on the mGrid V-cycle pipeline (Standard/Extended/TwoWayExtended).
carbon_add_cuda_unit_test(mgrid_skeleton_overlap)
# OpenMP host path: level_view::host_partitions() wiring (test_mgrid_host) and full
# cross-level make_host_kernel / make_host_reduce_kernel parity (test_mgrid_host_kernel).
carbon_add_cuda_unit_test(mgrid_host)
carbon_add_cuda_unit_test(mgrid_host_kernel)
carbon_add_cuda_unit_test(block_kernel)
carbon_add_cuda_unit_test(block_multi_gpu)
# Promoted 2026-05-18 (P2.β fourth-pass closure): per-brick D2H readback now
# routes through System::async_copy on an unmanaged sub-view of the per-GPU
# buffer, and the per-GPU zero-init uses System::async_memset. Body now has
# no direct CUDA-runtime calls and runs under both backends.
carbon_add_cuda_unit_test(halo_block)
carbon_add_cuda_unit_test(halo_block_host)
carbon_add_cuda_unit_test(block_data_view)
carbon_add_cuda_unit_test(block_overlapped_stencil)
carbon_add_cuda_unit_test(block_reduce)
# Promoted 2026-05-13 (P2.γ partial closure of audit third pass): the
# test body uses no direct CUDA APIs and rides on the standard carbon
# API exclusively.
carbon_add_cuda_unit_test(block_host_mirror)
# Anchors the P2.α contract change (audit third pass): Block's per-GPU
# vector accessors throw std::out_of_range on OOB index instead of
# silently returning a static empty sentinel. Body uses no backend APIs.
carbon_add_cuda_unit_test(block_oob_accessor)
# Closes P1.δ #1 (audit third pass): every other DenseDisg factory had a
# test, but make_universal_reduce_kernel was uncovered. Verifies kind tag
# + both launch paths agree on the integer sum.
carbon_add_cuda_unit_test(dense_disg_universal_reduce)
carbon_add_cuda_unit_test(block_host_parity)
carbon_add_cuda_unit_test(block_universal)
carbon_add_cuda_unit_test(skeleton_block)
carbon_add_cuda_unit_test(halo_host)
# Host-side HaloDisg::launch_on_host on multi-GPU Open layout (P2.γ fourth-pass gap).
carbon_add_cuda_unit_test(halo_disg_host)
carbon_add_cuda_unit_test(overlapped_stencil)
carbon_add_cuda_unit_test(skeleton)
carbon_add_cuda_unit_test(skeleton_initial_halo)
carbon_add_cuda_unit_test(skeleton_cuda_graph)
carbon_add_cuda_unit_test(skeleton_cuda_graph_multi)
carbon_add_cuda_unit_test(skeleton_reduce)
carbon_add_cuda_unit_test(skeleton_overlapped)
carbon_add_cuda_unit_test(skeleton_anchor)
carbon_add_cuda_unit_test(halo_periodic_z)
carbon_add_cuda_unit_test(halo_periodic_z_lattice)
carbon_add_cuda_unit_test(native_io)
# DenseDisg sibling of native_io. P1.epsilon (fourth pass) closure: write_native /
# read_native are now templated on Grid so DenseDisg fields participate in the
# Cartesian native-binary file format. The test exercises a multi-slab single-process
# round-trip and a Dense -> DenseDisg cross-family read into the same global_extent.
carbon_add_cuda_unit_test(native_io_disg)
carbon_add_cuda_unit_test(checkpoint_dense)
carbon_add_cuda_unit_test(hdf5_io)
carbon_add_cuda_unit_test(nanovdb_io)
carbon_add_cuda_unit_test(vtk_io)
# Migrated off raw cudaXxx APIs to backend-neutral carbon primitives (carbon::Buffer<T> +
# Transfer<T> for raw device buffers, Stream::wait() for stream sync, EventGroup::record /
# StreamGroup::wait_for for cross-stream ordering); now compile + pass under both backends.
carbon_add_cuda_unit_test(dense_kernel)
carbon_add_cuda_unit_test(partition_read)
carbon_add_cuda_unit_test(transfer_cuda)
carbon_add_cuda_unit_test(integration)
carbon_add_cuda_unit_test(partition_halo)
carbon_add_cuda_unit_test(transfer_peer)
carbon_add_cuda_unit_test(data_view_split_jacobi)
# int32 Jacobi sibling of data_view_split_jacobi + test_skeleton: overlap + Skeleton
# ping-pong paths vs a Standard single-stream reference, bit-exact integer compare.
carbon_add_cuda_unit_test(jacobi_integer)
# (carbon_add_cuda_only_unit_test entries all live near the top of this file, next to the
# rationale for each one; nothing below this point is CUDA-gated.)

# --- needs_multi_device ---------------------------------------------------------------
# Tests whose core coverage requires >= 2 visible devices. Each one opens with a
# `device_count() < 2` (or gpu_group().size() < 2) guard and returns success below that,
# so on a single-GPU host they contribute a pass without exercising the cross-device path
# they were written for. Labelling them makes the affected set enumerable
# (`ctest -L needs_multi_device`) and lets carbon_multi_gpu_gate speak for the whole class
# under CARBON_TEST_STRICT.
#
# The last three are partial: they do run their single-GPU assertions and skip only the
# multi-GPU combine / bit-equality section. They are labelled all the same, because that
# section is the reason a multi-GPU runner exists.
set(_carbon_multi_gpu_tests
  transfer_peer
  halo_single_node
  halo_multicomponent
  halo_host
  halo_disg_host
  halo_block
  halo_block_host
  reduce_sum_multi_gpu
  reduce_sum_host_multi_gpu
  jacobi_integer
  dense_disg_multi_gpu
  lattice_halo_disg
  lattice_halo_aggregated
  lattice_halo_dense_auto
  lattice_halo_disg_auto
  block_multi_gpu
  block_data_view
  block_overlapped_stencil
  skeleton_block
  skeleton_block_extended
  skeleton_initial_halo
  skeleton_cuda_graph_multi
  skeleton_scalar_pattern_multi_gpu
  mgrid_halo
  render_density
  render_surfel
  render_ray_march
  render_ray_march_sparse)
if(CARBON_BACKEND STREQUAL "Cuda")
  # Registered through carbon_add_cuda_only_unit_test, so the CTest entry only exists here.
  list(APPEND _carbon_multi_gpu_tests mgrid_weighted_cuts)
endif()
foreach(_carbon_mg_test IN LISTS _carbon_multi_gpu_tests)
  set_property(TEST carbon_${_carbon_mg_test} APPEND PROPERTY LABELS "needs_multi_device")
endforeach()
unset(_carbon_mg_test)

if(CARBON_ENABLE_MPI AND MPI_FOUND)
  # MPI distributed test fixture. Source file is test_<name>.cu (kept .cu because the
  # CUDA build feeds it to nvcc and several of these tests use CUB / cudaXxx APIs in
  # their bodies). Under CARBON_BACKEND=Sycl the same source is mirrored to a .cpp twin
  # in the build tree (configure_file COPYONLY) and tagged via add_sycl_to_target so
  # acpp picks it up — same trick as carbon_add_cuda_unit_test above. Tests whose
  # bodies depend on cuda runtime APIs are still gated to the CUDA backend by the
  # if(CARBON_BACKEND STREQUAL "Cuda") guard inside the function; SYCL-clean ones run
  # under both.
  function(carbon_add_mpi_cuda_test name num_procs)
    set(target "test_${name}")
    if(CARBON_BACKEND STREQUAL "Cuda")
      add_executable(${target} ${target}.cu)
      target_link_libraries(${target} PRIVATE carbon::carbon)
      target_compile_options(${target} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:${CARBON_NVCC_DEVICE_FLAGS}>)
      set_target_properties(${target} PROPERTIES
        BUILD_RPATH "${CARBON_RUNTIME_SEARCH_DIRS};${CMAKE_BINARY_DIR}"
        CUDA_STANDARD 20
        CUDA_STANDARD_REQUIRED ON)
    elseif(CARBON_BACKEND STREQUAL "Hip")
      add_executable(${target} ${target}.cu)
      set_source_files_properties(${target}.cu PROPERTIES LANGUAGE HIP)
      target_link_libraries(${target} PRIVATE carbon::carbon)
      if(CARBON_HIP_NVCC)
        target_compile_options(${target} PRIVATE $<$<COMPILE_LANGUAGE:HIP>:${CARBON_NVCC_DEVICE_FLAGS}>)
      endif()
      set_target_properties(${target} PROPERTIES
        BUILD_RPATH "${CARBON_RUNTIME_SEARCH_DIRS};${CMAKE_BINARY_DIR}"
        HIP_STANDARD 20
        HIP_STANDARD_REQUIRED ON)
    elseif(CARBON_BACKEND STREQUAL "Sycl")
      set(_src "${CMAKE_CURRENT_BINARY_DIR}/${target}.cpp")
      configure_file("${CMAKE_CURRENT_SOURCE_DIR}/${target}.cu" "${_src}" COPYONLY)
      add_executable(${target} "${_src}")
      target_link_libraries(${target} PRIVATE carbon::carbon)
      target_compile_features(${target} PRIVATE cxx_std_20)
      set_target_properties(${target} PROPERTIES
        BUILD_RPATH "${CARBON_RUNTIME_SEARCH_DIRS};${CMAKE_BINARY_DIR}")
      add_sycl_to_target(TARGET ${target} SOURCES "${_src}")
      if(CARBON_SYCL_TARGETS MATCHES "cuda")
        target_compile_options(${target} PRIVATE --cuda-path=${CARBON_SYCL_CUDA_PATH})
        target_link_directories(${target} PRIVATE ${CARBON_SYCL_CUDA_PATH}/lib64)
      endif()
    endif()
    if(TARGET ${target})
      carbon_test_enable_assertions(${target})
    endif()
    if(NOT MPIEXEC_EXECUTABLE)
      message(WARNING "CARBON_ENABLE_MPI is ON but MPIEXEC_EXECUTABLE not set; "
                      "${target} will not run as a CTest entry.")
    else()
      add_test(NAME carbon_${name}
        COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${num_procs} ${MPIEXEC_PREFLAGS}
                $<TARGET_FILE:${target}> ${MPIEXEC_POSTFLAGS})
      # distributed for the MPI lane; gpu because every one of these launches on a
      # device, so the CPU-only lane must not select them.
      set_tests_properties(carbon_${name} PROPERTIES LABELS "distributed;gpu")
    endif()
  endfunction()

  carbon_add_mpi_cuda_test(dist_transfer 2)
  carbon_add_mpi_cuda_test(dense_dist 2)
  carbon_add_mpi_cuda_test(halo_dist 2)
  carbon_add_mpi_cuda_test(halo_dist_periodic_z 2)
  carbon_add_mpi_cuda_test(halo_block_dist 2)
  # Host-side cross-rank HaloBlock::launch_on_host (MPI_Isend/Irecv via run_host).
  carbon_add_mpi_cuda_test(halo_block_host_dist 2)
  carbon_add_mpi_cuda_test(block_neighbour_dist 2)
  # Sphere-predicate Block cross-rank stencil parity: six-direction neighbour
  # probe on slab boundaries + Jacobi sum|u|^2 vs host reference under MPI.
  carbon_add_mpi_cuda_test(block_dist_stencil_parity 2)
  carbon_add_mpi_cuda_test(native_io_dist 2)
  # DenseDisg sibling of native_io_dist. P1.epsilon (fourth pass) closure: per-rank
  # shard round-trip mirroring native_io_dist for the disaggregated layout.
  carbon_add_mpi_cuda_test(native_io_dist_disg 2)
  # Closes P1.δ #2 (audit third pass): no MPI test exercised the VTK / HDF5 /
  # nanoVDB writers' per-rank or collective shape. test_io_dist round-trips a
  # scalar f32 + 3-component i32 field through every format whose flag is on
  # at compile time and asserts owned-cell parity + halo-untouched per rank;
  # the HDF5 section also locks the serial-HDF5+MPI throw-with-clean-error
  # contract when CARBON_HAVE_PHDF5 is off.
  carbon_add_mpi_cuda_test(io_dist 2)
  carbon_add_mpi_cuda_test(dist_reduce 2)
  # Closes P1.δ #2 (audit third pass): cross-rank Min / Max / Prod through the
  # NCCL/RCCL allreduce path -- locks reduce::op_kind<Op>::value's mapping to
  # ncclMin / ncclMax / ncclProd. Also verifies the host (OpenMP +
  # MPI_Allreduce) path on a non-Sum op for parity.
  carbon_add_mpi_cuda_test(dist_reduce_ops 2)
  carbon_add_mpi_cuda_test(block_dist_reduce 2)
  # Closes P1.γ (audit fourth pass): cross-rank Min / Max / Prod through the
  # NCCL/RCCL allreduce path on Block, plus MPI_Min on the block host
  # launcher. Sibling of dist_reduce_ops; locks reduce::op_kind<Op>::value
  # routing for the block dispatch.
  carbon_add_mpi_cuda_test(block_dist_reduce_ops 2)
  carbon_add_mpi_cuda_test(dense_disg_dist 2)
  carbon_add_mpi_cuda_test(dense_disg_dist_reduce 2)
  # Closes P1.γ (audit fourth pass): same as block_dist_reduce_ops but for
  # DenseDisg — locks the DisgPolicy / UniversalReduceKernelImpl reduce path
  # on Min / Max / Prod cross-rank, plus MPI_Min on the disg host launcher.
  carbon_add_mpi_cuda_test(dense_disg_dist_reduce_ops 2)
  # Closes P1.δ #5 (audit third pass): no MPI test for Skeleton's OverlapMode
  # rewrite path. Anchors cross-rank parity for {Standard, Extended,
  # TwoWayExtended} vs OverlapMode::None on a halo + Jacobi pipeline.
  carbon_add_mpi_cuda_test(skeleton_dist_overlap 2)
  # Closes the DenseDisg half of P1.δ (audit fourth pass): same OCC-MPI
  # parity matrix as skeleton_dist_overlap, but driven through
  # DenseDisg::Field::make_halo_kernel + DenseDisg::make_kernel so apply_occ_
  # rewrites land on HaloDisg's tokens shape.
  carbon_add_mpi_cuda_test(skeleton_dist_overlap_disg 2)
  # Closes the Block half of P1.δ (audit fourth pass) — fully resolves the item.
  # Same OCC-MPI parity matrix as the Dense / DenseDisg siblings, but driven
  # through Block::Field::make_halo_kernel (HaloBlock) + Kernel::compute<Block>.
  # Bit-equality holds under one-rank-per-GPU because the cross-rank halo path
  # is DistTransfer-only (no intra-process cudaMemcpyPeerAsync race that
  # motivates the kTol tolerance in the rank-local test_skeleton_block).
  carbon_add_mpi_cuda_test(skeleton_dist_overlap_block 2)
  carbon_add_mpi_cuda_test(halo_disg_dist 2)
  # Closes the user-requested "checkpoint a set of fields on Dense" feature; covers
  # cross-rank atomic commit (rank-0 manifest write fenced by two MPI_Barriers via
  # CheckpointWriter::commit), per-rank shard layout (.bin.rankN), step / time
  # round-trip, and the global_extent-mismatch reader-ctor throw.
  carbon_add_mpi_cuda_test(checkpoint_dense_dist 2)
  # mgrid_dist: cross-rank (one-rank-per-GPU) MultiResBlock co-resident cross-level test.
  # Projects the coarsest z-cut down every level so a parent brick + all descendants share a
  # rank; verifies every owned cross-level read resolves locally (no halo / DistTransfer / NCCL
  # on the cross-level path), matching encode(level, global). Raw <<<>>> kernel — CUDA-only.
  carbon_add_mpi_cuda_test(mgrid_dist 2)
  # mgrid_halo_dist: cross-rank (one-rank-per-GPU) intra-level halo. Co-resident sibling of
  # mgrid_halo (intra-process): proves the DistTransfer companion of each level's HaloBlock
  # delivers the neighbour rank's data into this rank's ghost bricks at the projected cut, on
  # every level. Kernel-side verification (block::Partition::neighbour +z/-z vs encode(level,
  # global)) since a rank cannot peek the neighbour's buffer.
  carbon_add_mpi_cuda_test(mgrid_halo_dist 2)
  # mgrid_skeleton_dist: cross-rank (one-rank-per-GPU) multi-level Skeleton. Distributed sibling of
  # mgrid_skeleton: proves the Skeleton's AUTO-INSERTED per-level halo carries the cross-rank
  # DistTransfer companion (the same-level neighbour exchange it wires before a stencil read
  # actually delivers the neighbour rank's bricks into this rank's ghosts, on every level).
  # Kernel-side verification (block::Partition::neighbour +z/-z vs encode(level, global)).
  carbon_add_mpi_cuda_test(mgrid_skeleton_dist 2)
endif()

if(CARBON_ENABLE_PYTHON)
  find_package(Python3 COMPONENTS Interpreter REQUIRED)

  # Every Warp test needs the vendored fork (`warp/`) and the binding package (`py/`) on
  # PYTHONPATH. Without them `import warp` raises ImportError, each test takes its own
  # skipTest branch, and ctest reports a pass: before 2026-08 the whole suite "passed" in
  # 0.65s having executed nothing, because ${Python3_EXECUTABLE} resolves to a venv that
  # has no warp installed. Setting it here (rather than relying on the caller's
  # environment, or on `pip install -e warp/`) makes `ctest` alone sufficient and pins the
  # tests to the *vendored* fork instead of whatever warp happens to be installed.
  # PYTHONPATH is replaced, not prepended, deliberately: these tests must run against the
  # fork whose external-type support they exercise. Path separator is POSIX-only, matching
  # the rest of this layer (the tests hardcode `linux_${machine}` library paths).
  set(CARBON_WARP_PYTHONPATH "${CMAKE_SOURCE_DIR}/warp:${CMAKE_SOURCE_DIR}/py")

  # labels: `warp` selects this layer as a group (`ctest -L warp`); `gpu` / `cpu` follow the
  # semantics documented above. Every one of these self-skips without a device, so only the
  # pure-ctypes layout check earns the cpu label.
  function(carbon_add_warp_test name)
    cmake_parse_arguments(PARSE_ARGV 1 arg "CPU" "" "")
    set(test_name "carbon_warp_${name}")
    add_test(NAME ${test_name}
      COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test_warp_${name}.py)
    if(arg_CPU)
      set(lane "cpu")
    else()
      set(lane "gpu")
    endif()
    set_tests_properties(${test_name} PROPERTIES
      ENVIRONMENT "PYTHONPATH=${CARBON_WARP_PYTHONPATH}"
      LABELS "warp;${lane}")
  endfunction()

  # Environment gate for the whole warp layer — see tests/test_warp_gate.py. Speaks for the
  # coverage of every `warp`-labelled test the way carbon_multi_gpu_gate speaks for
  # needs_multi_device. cpu lane: it only inspects importability and the device count.
  carbon_add_warp_test(gate CPU)
  carbon_add_warp_test(pod_layout CPU)

  carbon_add_warp_test(smoke)
  carbon_add_warp_test(external_types)
  carbon_add_warp_test(block_external_types)
  carbon_add_warp_test(axpy)
  carbon_add_warp_test(container_run)
  carbon_add_warp_test(block_container_run)
  carbon_add_warp_test(skeleton_sequence)
  carbon_add_warp_test(mgrid_container_run)
  carbon_add_warp_test(jacobi)
  carbon_add_warp_test(skeleton_token_graph)
  carbon_add_warp_test(mgrid_cross_level)
  carbon_add_warp_test(multi_component)
  carbon_add_warp_test(dtypes)
  carbon_add_warp_test(kernel_decorator)
  carbon_add_warp_test(kernel_factory)
  carbon_add_warp_test(skeleton_factory)
  carbon_add_warp_test(neighbour)
  carbon_add_warp_test(skeleton_halo)
  carbon_add_warp_test(skeleton_occ)
  carbon_add_warp_test(field_halo_update)
  # Registered 2026-08: these nine existed but were never added to ctest, so they ran only
  # under scripts/py-test-cuda12.sh (which globs tests/test_warp_*.py through pytest) and
  # never in a `ctest` run. mgrid_field_ops is the layer's only multi-component mgrid
  # coverage, so the omission mattered.
  carbon_add_warp_test(dense_readback)
  carbon_add_warp_test(dense_spanless)
  carbon_add_warp_test(dense_u8)
  carbon_add_warp_test(field_fill)
  carbon_add_warp_test(mgrid_field_ops)
  carbon_add_warp_test(mgrid_multires)
  carbon_add_warp_test(mgrid_spanless)
  carbon_add_warp_test(mgrid_sparse)
  carbon_add_warp_test(neon_aliases)

  # Spike scaffolding for the two-layer binding plan: locks the nested-registered-type-return
  # capability that both layers depend on, so a Warp fork bump cannot regress it silently.
  # Retires with warp_probe.h once the real ValidatedIdx / fat-index types land.
  carbon_add_warp_test(probe_nested_return)
endif()
