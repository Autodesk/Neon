cmake_minimum_required(VERSION 3.20)

project(carbon
  VERSION 0.1.0
  DESCRIPTION "Multi-GPU grid computation library (Neon-inspired)"
  LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Default to hidden visibility; export only CARBON_API symbols.
set(CMAKE_CXX_VISIBILITY_PRESET hidden)
set(CMAKE_VISIBILITY_INLINES_HIDDEN ON)

option(CARBON_BUILD_TESTS "Build unit tests" ON)
option(CARBON_BUILD_EXAMPLES "Build example programs" ON)
option(CARBON_BUILD_BENCHMARKS "Build benchmark programs" ON)
option(CARBON_BUILD_VIEWER "Build the interactive viewer for render examples" OFF)
option(CARBON_VIEWER_DESKTOP "Build the GLFW+OpenGL desktop backend of the viewer (needs OpenGL/GLFW dev)" ON)
option(CARBON_VIEWER_WEB "Add a browser (imgui-ws / WebSocket) backend to the viewer" ON)
option(CARBON_ENABLE_MPI "Enable MPI + NCCL distributed transport" OFF)
option(CARBON_ENABLE_HDF5 "Enable HDF5 read/write of carbon::Dense fields" OFF)
option(CARBON_ENABLE_NANOVDB "Enable nanoVDB read/write of carbon::Dense fields" OFF)
option(CARBON_ENABLE_PYTHON "Build libcarbonpy Warp bindings (CUDA backend only)" OFF)
option(CARBON_WERROR "Treat compiler warnings as errors when building libcarbon (CI gate)" OFF)
set(CARBON_SANITIZE "" CACHE STRING
  "Host sanitizers to build carbon and its tests with: '' (off), address, undefined, \
address+undefined, or thread.")
set_property(CACHE CARBON_SANITIZE PROPERTY STRINGS "" address undefined address+undefined thread)

# --------------------------------------------------------------------------------------------
# Compute backend selection
# --------------------------------------------------------------------------------------------
#
# Carbon supports three compute backends (CUDA, HIP, SYCL). The active backend is fixed at
# build time. All backends share the same public API; what differs is which detail/ headers
# get pulled in (CUDA and HIP share detail/backend/cudalike/* through the gpu_* aliases in
# detail/backend/cuda_like.h; SYCL uses detail/backend/sycl/*), which compile-time defines
# carbon.so is built with, and which toolchain compiles the .cu files.
#
# CARBON_BACKEND is a cache string (not a boolean) so future values slot in cleanly.
# Exactly one CARBON_BACKEND_<NAME>=1 compile definition is propagated to consumers and
# normalised by include/carbon/detail/backend/select.h.
#
# .cu file extension is intentionally preserved for every backend. Under Cuda the CMake CUDA
# language is enabled; under Hip the CMake HIP language is enabled (targeting AMD via hipcc or
# NVIDIA via the CUDA-backed HIP shim); under Sycl, AdaptiveCpp's add_sycl_to_target() drives
# compilation (acpp can compile .cu as SYCL TUs).
set(CARBON_BACKEND "Cuda" CACHE STRING
  "Compute backend for carbon: Cuda, Sycl (AdaptiveCpp), or Hip (ROCm/native HIP).")
set_property(CACHE CARBON_BACKEND PROPERTY STRINGS Cuda Sycl Hip)

# nvcc flags shared by every device TU (libcarbon, tests, examples, benchmarks).
#
# -Werror cross-execution-space-call promotes nvcc warning #20011 — "calling a __host__
# function from a __host__ __device__ function is not allowed" — to a hard error. Left as a
# warning it is a silent-wrong-answer generator: the TU still compiles and links, but the
# device path of the offending call is simply never emitted, so a kernel that reaches an
# accidentally host-only helper leaves its output untouched and the run reports zeros with no
# error anywhere. A missing CARBON_HD on one factory function in the render sampler did
# exactly that (2026-08); see docs/CHANGELOG.md.
set(CARBON_NVCC_DEVICE_FLAGS --extended-lambda -Werror cross-execution-space-call)

find_package(OpenMP REQUIRED)

if(CARBON_BACKEND STREQUAL "Cuda")
  find_package(CUDAToolkit REQUIRED)
  find_program(CARBON_NVCC_EXECUTABLE nvcc REQUIRED
    HINTS "${CUDAToolkit_BIN_DIR}" "${CUDAToolkit_ROOT}/bin" ENV CUDA_PATH ENV CUDA_HOME
    PATH_SUFFIXES bin
    DOC "CUDA compiler (nvcc)")
  set(CMAKE_CUDA_COMPILER "${CARBON_NVCC_EXECUTABLE}")
  enable_language(CUDA)
  # Host code uses C++23; nvcc commonly supports up to C++20 for .cu compilation.
  set(CMAKE_CUDA_STANDARD 20)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
elseif(CARBON_BACKEND STREQUAL "Sycl")
  # Per-rank GPU targets to instruct AdaptiveCpp's code generator. Defaults cover the A100
  # (sm_80) plus the OpenMP host fallback so a CPU-only debug run works on any machine.
  # Override at configure time, e.g.:
  #   cmake -DCARBON_BACKEND=Sycl -DCARBON_SYCL_TARGETS="cuda:sm_90;omp" ...
  set(CARBON_SYCL_TARGETS "cuda:sm_80;omp" CACHE STRING
    "AdaptiveCpp targets (passed as ACPP_TARGETS); e.g. cuda:sm_80;omp or cuda:sm_90.")
  # AdaptiveCpp keys off the ACPP_TARGETS cache variable (and the older HIPSYCL_TARGETS
  # fallback). FORCE so CARBON_SYCL_TARGETS is the single source of truth even when an
  # earlier configure left a stale ACPP_TARGETS in the cache.
  set(ACPP_TARGETS "${CARBON_SYCL_TARGETS}" CACHE STRING "AdaptiveCpp ACPP_TARGETS" FORCE)

  # AdaptiveCpp's CUDA backend needs a host CUDA installation for libdevice + headers; it
  # doesn't ship its own. Honour CUDA_PATH / CUDA_HOME (same env as nvcc) and forward the
  # path to acpp via --cuda-path. CMake-side find_package(CUDAToolkit) is intentionally NOT
  # called: we don't link against libcudart.so directly, AdaptiveCpp does that internally.
  if(CARBON_SYCL_TARGETS MATCHES "cuda")
    if(NOT DEFINED CARBON_SYCL_CUDA_PATH OR CARBON_SYCL_CUDA_PATH STREQUAL "")
      if(DEFINED ENV{CUDA_PATH} AND NOT "$ENV{CUDA_PATH}" STREQUAL "")
        set(_carbon_sycl_cuda_default "$ENV{CUDA_PATH}")
      else()
        set(_carbon_sycl_cuda_default "/usr/local/cuda")
      endif()
      set(CARBON_SYCL_CUDA_PATH "${_carbon_sycl_cuda_default}" CACHE PATH
        "CUDA toolkit path forwarded to acpp as --cuda-path")
    endif()
    if(NOT EXISTS "${CARBON_SYCL_CUDA_PATH}/include/cuda_runtime.h")
      message(FATAL_ERROR
        "CARBON_BACKEND=Sycl with cuda target requires a CUDA toolkit on disk for libdevice "
        "and headers. Could not find cuda_runtime.h under '${CARBON_SYCL_CUDA_PATH}'. Set "
        "-DCARBON_SYCL_CUDA_PATH=/path/to/cuda or export CUDA_PATH and re-run.")
    endif()
  endif()

  # AdaptiveCpp's HIP backend needs a ROCm install for HIP headers + amdgcn device libs;
  # it doesn't ship its own. Honour ROCM_PATH / ROCM_HOME (same env as hipcc) and forward
  # the path to acpp via --acpp-rocm-path, mirroring the CUDA knob above.
  if(CARBON_SYCL_TARGETS MATCHES "hip")
    if(NOT DEFINED CARBON_SYCL_ROCM_PATH OR CARBON_SYCL_ROCM_PATH STREQUAL "")
      if(DEFINED ENV{ROCM_PATH} AND NOT "$ENV{ROCM_PATH}" STREQUAL "")
        set(_carbon_sycl_rocm_default "$ENV{ROCM_PATH}")
      elseif(DEFINED ENV{ROCM_HOME} AND NOT "$ENV{ROCM_HOME}" STREQUAL "")
        set(_carbon_sycl_rocm_default "$ENV{ROCM_HOME}")
      else()
        set(_carbon_sycl_rocm_default "/opt/rocm")
      endif()
      set(CARBON_SYCL_ROCM_PATH "${_carbon_sycl_rocm_default}" CACHE PATH
        "ROCm install path forwarded to acpp as --acpp-rocm-path")
    endif()
    if(NOT EXISTS "${CARBON_SYCL_ROCM_PATH}/include/hip/hip_runtime.h")
      message(FATAL_ERROR
        "CARBON_BACKEND=Sycl with hip target requires a ROCm install on disk for HIP "
        "headers and amdgcn device libraries. Could not find hip/hip_runtime.h under "
        "'${CARBON_SYCL_ROCM_PATH}/include'. Set -DCARBON_SYCL_ROCM_PATH=/path/to/rocm "
        "or export ROCM_PATH and re-run.")
    endif()
  endif()

  find_package(AdaptiveCpp CONFIG REQUIRED)
  if(NOT TARGET AdaptiveCpp::acpp-rt)
    message(FATAL_ERROR
      "find_package(AdaptiveCpp) succeeded but did not provide the expected target "
      "AdaptiveCpp::acpp-rt. Check that AdaptiveCpp >= 24.x is installed and that "
      "AdaptiveCpp_DIR points at <prefix>/lib/cmake/AdaptiveCpp.")
  endif()

  if(CARBON_ENABLE_MPI
     AND NOT CARBON_SYCL_TARGETS MATCHES "cuda"
     AND NOT CARBON_SYCL_TARGETS MATCHES "hip")
    message(FATAL_ERROR
      "CARBON_ENABLE_MPI=ON under CARBON_BACKEND=Sycl currently requires either a 'cuda:' "
      "(NCCL transport) or 'hip:' (RCCL transport) target in CARBON_SYCL_TARGETS (got "
      "'${CARBON_SYCL_TARGETS}'). Carbon's distributed transport calls NCCL/RCCL on the "
      "underlying cudaMalloc'd / hipMalloc'd USM pointers extracted via "
      "sycl::interop_handle::get_native_queue<sycl::backend::cuda|hip>; that interop only "
      "exists when AdaptiveCpp's CUDA or HIP backend is in CARBON_SYCL_TARGETS. Host-MPI "
      "(OMP / Apple-CPU) is a planned follow-up; until then, either add 'cuda:sm_XX' / "
      "'hip:gfxXXX' to CARBON_SYCL_TARGETS or re-configure with CARBON_ENABLE_MPI=OFF.")
  endif()

  message(STATUS "carbon: SYCL backend selected (AdaptiveCpp). ACPP_TARGETS=${ACPP_TARGETS}")
elseif(CARBON_BACKEND STREQUAL "Hip")
  # Native HIP (ROCm) backend. The .cu device TUs (src/hip_stub.cu + every .cu test /
  # example) are compiled by the CMake HIP language (amdclang++/hipcc); the host .cpp
  # sources stay on the C++ compiler and link the HIP runtime. This is the AMD analogue
  # of the Cuda branch above.
  #
  # ROCm path discovery mirrors the SYCL hip: knob (ROCM_PATH / ROCM_HOME, else /opt/rocm)
  # so the same environment drives both the SYCL+hip and the native HIP builds.
  if(NOT DEFINED CARBON_ROCM_PATH OR CARBON_ROCM_PATH STREQUAL "")
    if(DEFINED ENV{ROCM_PATH} AND NOT "$ENV{ROCM_PATH}" STREQUAL "")
      set(_carbon_rocm_default "$ENV{ROCM_PATH}")
    elseif(DEFINED ENV{ROCM_HOME} AND NOT "$ENV{ROCM_HOME}" STREQUAL "")
      set(_carbon_rocm_default "$ENV{ROCM_HOME}")
    else()
      set(_carbon_rocm_default "/opt/rocm")
    endif()
    set(CARBON_ROCM_PATH "${_carbon_rocm_default}" CACHE PATH
      "ROCm install path (HIP runtime, hipcc, hipcub headers).")
  endif()
  if(NOT EXISTS "${CARBON_ROCM_PATH}/include/hip/hip_runtime.h")
    message(FATAL_ERROR
      "CARBON_BACKEND=Hip requires a ROCm install on disk. Could not find "
      "hip/hip_runtime.h under '${CARBON_ROCM_PATH}/include'. Set -DCARBON_ROCM_PATH=/path/"
      "to/rocm or export ROCM_PATH and re-run.")
  endif()
  # find_package(hip) / find_package(hipcub) live under $ROCM_PATH/lib/cmake.
  list(APPEND CMAKE_PREFIX_PATH "${CARBON_ROCM_PATH}")
  # GPU platform. HIP runs on AMD GPUs (the default) *and* on NVIDIA GPUs: with
  # -DCMAKE_HIP_PLATFORM=nvidia (CMake >= 3.28) the HIP language is compiled by nvcc,
  # <hip/hip_runtime.h> forwards every hipXxx call to the matching cudaXxx, hipcub maps to
  # cub, and HIP_ARCHITECTURES are CUDA arches (e.g. 80). This lets the exact same HIP
  # source tree be exercised at runtime on an NVIDIA card. We honour an explicit
  # -DCMAKE_HIP_PLATFORM; otherwise CMake computes it via `hipconfig --platform`.
  set(CARBON_HIP_NVCC OFF)
  if(CMAKE_HIP_PLATFORM STREQUAL "nvidia")
    set(CARBON_HIP_NVCC ON)
  endif()
  # GPU architecture(s) for the device code generator. AMD: gfx90a (MI200), gfx942 (MI300),
  # gfx1100 (RDNA3). NVIDIA: a CUDA compute capability (80 for A100). Override with
  # -DCARBON_HIP_ARCH=... or -DCMAKE_HIP_ARCHITECTURES=...
  set(CARBON_HIP_ARCH "gfx90a" CACHE STRING
    "GPU architecture(s) for the HIP device compiler (CMAKE_HIP_ARCHITECTURES).")
  if(NOT DEFINED CMAKE_HIP_ARCHITECTURES)
    if(CARBON_HIP_NVCC)
      set(CMAKE_HIP_ARCHITECTURES "80")
    else()
      set(CMAKE_HIP_ARCHITECTURES "${CARBON_HIP_ARCH}")
    endif()
  endif()
  enable_language(HIP)
  set(CMAKE_HIP_STANDARD 20)
  set(CMAKE_HIP_STANDARD_REQUIRED ON)
  # enable_language may have resolved the platform from hipconfig; re-read it.
  if(CMAKE_HIP_PLATFORM STREQUAL "nvidia")
    set(CARBON_HIP_NVCC ON)
  endif()
  if(CARBON_HIP_NVCC)
    # nvcc backs the HIP language; the runtime is CUDA + cub (both via CUDA::cudart's
    # toolkit). We deliberately skip find_package(hip) / find_package(hipcub): ROCm's
    # hip-config-nvidia.cmake creates hip::device / hip::host / hip::amdhip64 with bare
    # add_library(IMPORTED) (no `if(NOT TARGET ...)` guard), so it is not idempotent and
    # aborts the second time it is pulled in (hipcub → rocprim → hip). On the NVIDIA
    # platform we don't use those targets anyway — the HIP + hipcub headers are forwarded
    # as plain -I and the runtime symbols come from CUDA::cudart (wired in the source
    # block below).
    find_package(CUDAToolkit REQUIRED)
  else()
    find_package(hip REQUIRED)
    # hipcub provides the BlockReduce / DeviceReduce primitives the reduce kernels need
    # (rocPRIM-backed). Linked PUBLIC so consumer .cu TUs that instantiate carbon's reduce
    # kernels see the headers too.
    find_package(hipcub REQUIRED)
  endif()
  message(STATUS "carbon: HIP backend selected (ROCm at ${CARBON_ROCM_PATH}, "
    "platform=${CMAKE_HIP_PLATFORM}). CMAKE_HIP_ARCHITECTURES=${CMAKE_HIP_ARCHITECTURES}")
else()
  message(FATAL_ERROR
    "Unknown CARBON_BACKEND='${CARBON_BACKEND}'. Valid values: Cuda, Sycl, Hip.")
endif()

include(GNUInstallDirs)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(CarbonSpdlog)
include(CarbonRapidjson)

set(CARBON_VERSION_STRING "${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR}")

# Optional: stamp the build with `git describe` so carbon::info() can return
# a unique identifier for bug reports (commit hash + dirty flag). Empty
# string when the working tree is not a git checkout — e.g. unpacked
# tarball — which keeps the configure step quiet on consumer builds.
set(CARBON_GIT_DESCRIBE "")
find_package(Git QUIET)
if(Git_FOUND AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/.git")
  execute_process(
    COMMAND "${GIT_EXECUTABLE}" describe --tags --always --dirty
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    OUTPUT_VARIABLE _carbon_git_describe
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
    RESULT_VARIABLE _carbon_git_describe_rc)
  if(_carbon_git_describe_rc EQUAL 0)
    set(CARBON_GIT_DESCRIBE "${_carbon_git_describe}")
  endif()
endif()

if(CARBON_GIT_DESCRIBE STREQUAL "")
  set(CARBON_VERSION_FULL "${CARBON_VERSION_STRING}")
else()
  set(CARBON_VERSION_FULL "${CARBON_VERSION_STRING} (${CARBON_GIT_DESCRIBE})")
endif()

configure_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/include/carbon/version.h.in"
  "${CMAKE_CURRENT_BINARY_DIR}/include/carbon/version.h"
  @ONLY)

add_library(carbon SHARED
  src/carbon.cpp
  src/checkpoint.cpp
  src/data_view.cpp
  src/dense.cpp
  src/dense_disg.cpp
  src/dense_launch_shape.cpp
  src/dist_reduce.cpp
  src/dist_render.cpp
  src/io.cpp
  src/io_vtk.cpp
  src/logging.cpp
  src/memory_side.cpp
  src/report.cpp
  src/skeleton.cpp
  src/stencil.cpp
  src/system.cpp)

target_link_libraries(carbon PUBLIC spdlog::spdlog rapidjson OpenMP::OpenMP_CXX)

if(CARBON_BACKEND STREQUAL "Cuda")
  target_sources(carbon PRIVATE
    src/cuda_stub.cu
    src/gpu.cpp)
  target_link_libraries(carbon PUBLIC CUDA::cudart)
  target_compile_definitions(carbon PUBLIC CARBON_HAVE_NVTX=1)
  # NVTX3 is header-only. CUDA 12.8+ dropped the legacy compiled CUDA::nvToolsExt
  # target; prefer the header-only CUDA::nvtx3 interface target when present and
  # fall back to the classic target only on toolkits old enough to still ship it.
  if(TARGET CUDA::nvtx3)
    target_link_libraries(carbon PRIVATE CUDA::nvtx3)
  else()
    target_link_libraries(carbon PRIVATE CUDA::nvToolsExt)
  endif()
  # OpenMP::OpenMP_CXX only injects its host flag under COMPILE_LANGUAGE:CXX, so
  # `.cu` TUs (CUDA language) that instantiate carbon's templated per-GPU launch
  # loop or host-kernel `#pragma omp` paths silently drop the pragmas and run
  # single-threaded. Forward the host OpenMP flag through nvcc for CUDA
  # compilation; PUBLIC so consumer `.cu` targets (tests / benchmarks / examples)
  # inherit it and the per-GPU kernel launches actually fan out across threads.
  target_compile_options(carbon PUBLIC
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=${OpenMP_CXX_FLAGS}>)
  set(CARBON_RUNTIME_SEARCH_DIRS "${CUDAToolkit_LIBRARY_DIR}")
elseif(CARBON_BACKEND STREQUAL "Hip")
  target_sources(carbon PRIVATE
    src/hip_stub.cu
    src/gpu_hip.cpp)
  # The .cu stub is HIP device code; tell CMake to feed it to the HIP compiler (the .cu
  # suffix would otherwise be treated as CUDA). The .cpp host source stays on the C++
  # compiler. Consumer .cu TUs do the same via set_source_files_properties in tests/.
  set_source_files_properties(src/hip_stub.cu PROPERTIES LANGUAGE HIP)
  if(CARBON_HIP_NVCC)
    # HIP-on-NVIDIA (nvcc backs the HIP language). The HIP runtime is a thin shim over
    # CUDA, so host .cpp TUs that include <hip/hip_runtime.h> pull in <cuda_runtime.h> and
    # need (a) the __HIP_PLATFORM_NVIDIA__ define, (b) the HIP + CUDA include dirs, and
    # (c) libcudart for the runtime symbols. ROCm's hip-config-nvidia.cmake historically
    # ships an *unpopulated* hip::host target (ROCm#3748), so we wire these up explicitly
    # rather than rely on the imported target.
    #
    # PUBLIC is deliberate, not a leak: a downstream consumer's own .cu TUs are compiled as
    # HIP under nvcc and include carbon's public headers, which transitively pull
    # <hip/hip_runtime.h>. Those headers branch on __HIP_PLATFORM_NVIDIA__ to forward
    # hipXxx -> cudaXxx, so the define MUST reach consumer TUs or they would compile against
    # the AMD platform and fail to link. The matching SYSTEM PUBLIC include dir and
    # PUBLIC CUDA::cudart link below propagate for the same reason; carbonConfig.cmake.in
    # re-finds CUDAToolkit so an installed HIP-on-NVIDIA package stays consumable.
    target_compile_definitions(carbon PUBLIC __HIP_PLATFORM_NVIDIA__=1)
    # HIP + hipcub headers live under $ROCM/include; cub + cuda_runtime come from the CUDA
    # toolkit (CUDA::cudart's interface include). Both forwarded as plain -I (SYSTEM keeps
    # -Wall/-Wpedantic quiet inside them).
    target_include_directories(carbon SYSTEM PUBLIC "${CARBON_ROCM_PATH}/include")
    target_link_libraries(carbon PUBLIC CUDA::cudart)
    target_compile_definitions(carbon PUBLIC CARBON_HAVE_NVTX=1)
    # See the Cuda branch: prefer header-only CUDA::nvtx3 (CUDA 12.8+ dropped
    # the compiled CUDA::nvToolsExt target), fall back on older toolkits.
    if(TARGET CUDA::nvtx3)
      target_link_libraries(carbon PRIVATE CUDA::nvtx3)
    else()
      target_link_libraries(carbon PRIVATE CUDA::nvToolsExt)
    endif()
    # nvcc needs --extended-lambda for carbon's device lambdas (same as the Cuda backend);
    # scope it to HIP-language TUs so host .cpp stays on the plain C++ compiler.
    target_compile_options(carbon PRIVATE $<$<COMPILE_LANGUAGE:HIP>:${CARBON_NVCC_DEVICE_FLAGS}>)
    set(CARBON_RUNTIME_SEARCH_DIRS "${CUDAToolkit_LIBRARY_DIR}")
  else()
    # Native HIP on AMD. Link the *host* HIP runtime (libamdhip64), the AMD analogue of
    # CUDA::cudart: it exposes the runtime API (hipMalloc / hipStream* / ...) and the
    # runtime headers to host .cpp TUs without forcing device compilation. We deliberately
    # do NOT link hip::device / hip::hipcub here — those imported targets add `-x hip`
    # (+ --offload-arch) as *unconditional* INTERFACE_COMPILE_OPTIONS (no
    # $<COMPILE_LANGUAGE:HIP> guard), which the C++ compiler driving carbon's host .cpp
    # sources rejects with "language hip not recognized". Device compilation of the .cu
    # TUs comes from CMake's HIP language + CMAKE_HIP_ARCHITECTURES instead.
    target_link_libraries(carbon PUBLIC hip::host)
    set(CARBON_RUNTIME_SEARCH_DIRS "${CARBON_ROCM_PATH}/lib")
    # The HIP-compiled TUs (hip_stub.cu + every consumer .cu that instantiates carbon's
    # reduce kernels) need the hipcub / rocPRIM headers. Forward only the *include dirs*
    # from the hipcub interface target (plain -I, no compile options) so host .cpp TUs stay
    # unpoisoned. SYSTEM keeps -Wall/-Wpedantic from firing inside the ROCm headers.
    target_include_directories(carbon SYSTEM PUBLIC
      $<TARGET_PROPERTY:hip::hipcub,INTERFACE_INCLUDE_DIRECTORIES>)
    if(TARGET roc::rocprim)
      target_include_directories(carbon SYSTEM PUBLIC
        $<TARGET_PROPERTY:roc::rocprim,INTERFACE_INCLUDE_DIRECTORIES>)
    endif()
  endif()
elseif(CARBON_BACKEND STREQUAL "Sycl")
  target_sources(carbon PRIVATE
    src/sycl_stub.cpp
    src/gpu_sycl.cpp)
  # AdaptiveCpp's add_sycl_to_target tags every TU listed in SOURCES with the SYCL
  # compile/link flags. Apply it to all .cu / .cpp files participating in the carbon
  # target; this is the AdaptiveCpp-recommended way to get cross-TU SYCL with the
  # omp+cuda targets active simultaneously.
  add_sycl_to_target(TARGET carbon SOURCES
    src/carbon.cpp
    src/checkpoint.cpp
    src/data_view.cpp
    src/dense.cpp
    src/dense_disg.cpp
    src/dist_reduce.cpp
    src/io.cpp
    src/io_vtk.cpp
    src/logging.cpp
    src/memory_side.cpp
    src/report.cpp
    src/skeleton.cpp
    src/stencil.cpp
    src/system.cpp
    src/sycl_stub.cpp
    src/gpu_sycl.cpp)
  if(CARBON_SYCL_TARGETS MATCHES "cuda")
    # acpp picks up --cuda-path either via the flag or via CUDA_PATH env. Forward it as a
    # compile option so every TU sees it without the user having to export the env var.
    target_compile_options(carbon PRIVATE --cuda-path=${CARBON_SYCL_CUDA_PATH})
    # AdaptiveCpp's CUDA backend implicitly links against libcudart at the final shared
    # library link, but it does not add the toolkit's lib64 to the link search path on
    # its own. Forward CARBON_SYCL_CUDA_PATH/lib64 to the linker so libcarbon.so links
    # cleanly without requiring the user to export LIBRARY_PATH=$CUDA_PATH/lib64. We use
    # link_directories rather than target_link_options(... -L...) so consumer targets
    # also pick up the path when they link against carbon's IMPORTED target.
    target_link_directories(carbon PUBLIC ${CARBON_SYCL_CUDA_PATH}/lib64)
    find_package(CUDAToolkit REQUIRED)
    target_compile_definitions(carbon PUBLIC CARBON_HAVE_NVTX=1)
    # See the Cuda branch: prefer header-only CUDA::nvtx3 (CUDA 12.8+ dropped
    # the compiled CUDA::nvToolsExt target), fall back on older toolkits.
    if(TARGET CUDA::nvtx3)
      target_link_libraries(carbon PRIVATE CUDA::nvtx3)
    else()
      target_link_libraries(carbon PRIVATE CUDA::nvToolsExt)
    endif()
  endif()
  if(CARBON_SYCL_TARGETS MATCHES "hip")
    # Mirror the CUDA path forwarding: acpp's --acpp-rocm-path picks up everything (HIP
    # headers via -isystem, ROCm device libs via --rocm-device-lib-path, --rocm-path for
    # clang's HIP driver). Without the explicit flag acpp falls back to /opt/rocm which
    # is empty on a no-sudo dev box. PUBLIC because consumers (tests, examples) compile
    # SYCL TUs against the same hip: target and need the same flag — unlike the CUDA
    # path, where acpp can find /usr/local/cuda by itself, ROCm's default-search /opt/rocm
    # is rarely populated so the flag has to be propagated explicitly.
    target_compile_options(carbon PUBLIC --acpp-rocm-path=${CARBON_SYCL_ROCM_PATH})
    target_link_directories(carbon PUBLIC ${CARBON_SYCL_ROCM_PATH}/lib)
  endif()
  set(CARBON_RUNTIME_SEARCH_DIRS "")
endif()

if(CARBON_ENABLE_MPI)
  find_package(MPI REQUIRED)

  # Pick the collective backend. Default = NCCL (CUDA path: covers
  # CARBON_BACKEND=Cuda and CARBON_BACKEND=Sycl + cuda: target). Switch to RCCL
  # whenever CARBON_BACKEND=Sycl and CARBON_SYCL_TARGETS contains a hip: entry —
  # AdaptiveCpp's HIP backend allocates USM as hipMalloc'd pointers, so RCCL is
  # the right collective for AMD targets. RCCL is ABI-compatible with NCCL (same
  # ncclSend/Recv/CommInitRank/...), so the source delta is just (i) which header
  # gets included and (ii) which sycl::backend tag the interop_handle yields.
  set(_carbon_use_rccl OFF)
  if(CARBON_BACKEND STREQUAL "Sycl"
     AND CARBON_SYCL_TARGETS MATCHES "hip"
     AND NOT CARBON_SYCL_TARGETS MATCHES "cuda")
    set(_carbon_use_rccl ON)
  endif()
  # Native HIP backend always uses RCCL for the cross-rank collective.
  if(CARBON_BACKEND STREQUAL "Hip")
    set(_carbon_use_rccl ON)
  endif()

  if(_carbon_use_rccl)
    find_package(RCCL REQUIRED)
    # HIP headers (<hip/hip_runtime.h>, <hip/hip_fp16.h>) are reached via acpp's
    # --acpp-rocm-path forwarding (-isystem $ROCM_PATH/include) configured above. The
    # HIP runtime lib (libamdhip64.so) lives under $ROCM_PATH/lib (already added to the
    # link directories) or in AdaptiveCpp's pixi env on its RPATH.
    message(STATUS "carbon: distributed transport = RCCL ${RCCL_VERSION} (HIP backend)")
  else()
    find_package(NCCL REQUIRED)
    message(STATUS "carbon: distributed transport = NCCL ${NCCL_VERSION} (CUDA backend)")
  endif()

  target_sources(carbon PRIVATE
    src/distributed.cpp
    src/communicator.cpp
    src/dist_transfer.cpp)

  # Under the SYCL backend the three MPI sources include carbon's public headers, which
  # transitively pull in <sycl/sycl.hpp>; src/dist_transfer.cpp also calls
  # sycl::interop_handle::get_native_queue<sycl::backend::{cuda,hip}>(queue) to feed
  # NCCL/RCCL the underlying vendor stream. Tag them with add_sycl_to_target so they get
  # the SYCL include path + AdaptiveCpp's backend-interop glue.
  if(CARBON_BACKEND STREQUAL "Sycl")
    add_sycl_to_target(TARGET carbon SOURCES
      src/distributed.cpp
      src/communicator.cpp
      src/dist_transfer.cpp)

    # The four MPI sources (distributed, communicator, dist_transfer, dist_reduce)
    # include `gpu_runtime_compat.h` which pulls in <nccl.h> / <rccl/rccl.h>, which
    # transitively pulls in <cuda_runtime.h> / <hip/hip_runtime.h>. NCCL::nccl exposes
    # its include dir (/usr/include under apt's libnccl-dev) via
    # INTERFACE_INCLUDE_DIRECTORIES, but cmake treats /usr/include as a
    # CMAKE_PLATFORM_IMPLICIT_INCLUDE_DIRECTORIES entry on Linux and drops it from
    # the compile line. With the conda-shipped clang-19 that carbon uses under
    # CARBON_BACKEND=Sycl, /usr/include is NOT in the actual compiler search path
    # (conda clang has its own sysroot), so the header isn't found.
    #
    # Re-introduce the include dirs via per-source -I flags (narrow to the four MPI
    # files so the globally-breaking stdlib.h / omp.h interleaving -isystem
    # /usr/include would trigger on other TUs is avoided). -I (not -isystem) avoids
    # reordering vs. conda's sysroot and matches the -I cmake already emits for the
    # normal include dirs.
    set(_carbon_mpi_srcs
      src/distributed.cpp
      src/communicator.cpp
      src/dist_transfer.cpp
      src/dist_reduce.cpp)
    if(_carbon_use_rccl)
      set(_carbon_mpi_extra_includes
        "${RCCL_INCLUDE_DIR}"
        "${CARBON_SYCL_ROCM_PATH}/include")
    else()
      # Under CARBON_BACKEND=Sycl we don't call find_package(CUDAToolkit) (AdaptiveCpp
      # handles libcudart linkage internally), so use CARBON_SYCL_CUDA_PATH to locate
      # the CUDA runtime headers that <nccl.h> transitively depends on.
      set(_carbon_mpi_extra_includes
        "${NCCL_INCLUDE_DIR}"
        "${CARBON_SYCL_CUDA_PATH}/include")
    endif()
    foreach(_carbon_mpi_inc IN LISTS _carbon_mpi_extra_includes)
      if(_carbon_mpi_inc)
        set_property(SOURCE ${_carbon_mpi_srcs}
          APPEND PROPERTY COMPILE_OPTIONS "-I${_carbon_mpi_inc}")
      endif()
    endforeach()
  endif()

  # MPI/NCCL/RCCL stay PRIVATE: nothing in any public header includes <mpi.h>, <nccl.h>,
  # or <rccl/rccl.h> (Distributed and Communicator are pImpl, dist_transfer.inl forwards
  # to a byte-typed CARBON_API helper in src/dist_transfer.cpp), so consumer TUs only
  # need to link carbon. CARBON_HAVE_MPI_NCCL stays PUBLIC because the umbrella header
  # branches on it to conditionally expose <carbon/distributed.h> and
  # <carbon/dist_transfer.h>; the spelling is kept stable across NCCL/RCCL transports
  # since the public ABI is identical (RCCL is NCCL-compatible at the API level).
  target_link_libraries(carbon PRIVATE MPI::MPI_CXX)
  target_compile_definitions(carbon PUBLIC CARBON_HAVE_MPI_NCCL)

  if(_carbon_use_rccl)
    target_link_libraries(carbon PRIVATE RCCL::rccl)
    target_compile_definitions(carbon PRIVATE CARBON_USE_RCCL)
    # The AMD HIP runtime headers gate the AMD-side declarations on
    # __HIP_PLATFORM_AMD__. Without it the headers select the NVIDIA-on-HIP code path
    # which would (re)require nvcc-style preprocessing inside an AdaptiveCpp SYCL TU.
    target_compile_definitions(carbon PRIVATE __HIP_PLATFORM_AMD__=1)
    set(_carbon_collective_lib_dir "${RCCL_LIBRARY_DIR}")
  else()
    target_link_libraries(carbon PRIVATE NCCL::nccl)
    set(_carbon_collective_lib_dir "${NCCL_LIBRARY_DIR}")
  endif()

  if(_carbon_collective_lib_dir)
    list(APPEND CARBON_RUNTIME_SEARCH_DIRS "${_carbon_collective_lib_dir}")
    # NCCL/RCCL is PRIVATE-linked into libcarbon.so but consumers (test exes, examples)
    # have their linker chase libcarbon.so's DT_NEEDED to validate symbols. The
    # AdaptiveCpp+conda toolchain's bundled x86_64-conda-linux-gnu-ld defaults to
    # --no-allow-shlib-undefined and additionally won't satisfy weak NCCL/RCCL symbols
    # (both libs export ncclSend/ncclRecv as weak) without explicit
    # --allow-shlib-undefined. That flag tells the linker "trust DT_NEEDED + RPATH for
    # these at runtime" which is exactly what we want — libcarbon.so already has the
    # collective libs as DT_NEEDED and their libdirs on its RPATH. -L is also added so
    # the linker can locate the .so for inspection.
    target_link_directories(carbon INTERFACE "${_carbon_collective_lib_dir}")
    target_link_options(carbon INTERFACE "LINKER:--allow-shlib-undefined")
    # libcarbon.so itself also benefits from --allow-shlib-undefined when the collective
    # is RCCL: librccl.so has transitive DT_NEEDED to librocm_smi64.so.7 /
    # librocprofiler-register.so.0 which are not always present on a CPU-only / NVIDIA
    # box. The flag tells the linker to trust DT_NEEDED + RPATH at runtime instead of
    # failing the libcarbon.so link. Harmless under NCCL (libnccl.so's DT_NEEDED is just
    # libcudart, already on the link line).
    target_link_options(carbon PRIVATE "LINKER:--allow-shlib-undefined")
  endif()
  if(MPI_CXX_LIBRARIES)
    foreach(_mpi_lib IN LISTS MPI_CXX_LIBRARIES)
      get_filename_component(_mpi_dir "${_mpi_lib}" DIRECTORY)
      if(_mpi_dir)
        list(APPEND CARBON_RUNTIME_SEARCH_DIRS "${_mpi_dir}")
        target_link_directories(carbon INTERFACE "${_mpi_dir}")
      endif()
    endforeach()
    list(REMOVE_DUPLICATES CARBON_RUNTIME_SEARCH_DIRS)
  endif()
endif()

if(CARBON_ENABLE_HDF5)
  # find_package(HDF5) honours HDF5_ROOT / HDF5_DIR, and (when we ask for the C COMPONENT)
  # exposes HDF5::HDF5 as a usable IMPORTED target plus HDF5_IS_PARALLEL telling us whether
  # the discovered build of HDF5 was compiled against MPI.
  find_package(HDF5 REQUIRED COMPONENTS C)

  target_sources(carbon PRIVATE src/io_hdf5.cpp)
  if(TARGET HDF5::HDF5)
    target_link_libraries(carbon PRIVATE HDF5::HDF5)
  else()
    target_include_directories(carbon PRIVATE ${HDF5_INCLUDE_DIRS})
    target_link_libraries(carbon PRIVATE ${HDF5_LIBRARIES})
  endif()
  target_compile_definitions(carbon PUBLIC CARBON_HAVE_HDF5)

  if(CARBON_ENABLE_MPI AND HDF5_IS_PARALLEL)
    target_compile_definitions(carbon PUBLIC CARBON_HAVE_PHDF5)
  elseif(CARBON_ENABLE_MPI AND NOT HDF5_IS_PARALLEL)
    message(WARNING
      "CARBON_ENABLE_HDF5=ON but the discovered HDF5 was not built with MPI support; "
      "carbon::io::write_hdf5 / read_hdf5 will throw at runtime when used with a "
      "distributed Dense. Install parallel HDF5 (libhdf5-mpich-dev or equivalent) for "
      "collective I/O across ranks.")
  endif()

  if(HDF5_LIBRARIES)
    foreach(_h5_lib IN LISTS HDF5_LIBRARIES)
      get_filename_component(_h5_dir "${_h5_lib}" DIRECTORY)
      if(_h5_dir)
        list(APPEND CARBON_RUNTIME_SEARCH_DIRS "${_h5_dir}")
      endif()
    endforeach()
    list(REMOVE_DUPLICATES CARBON_RUNTIME_SEARCH_DIRS)
  endif()
endif()

if(CARBON_ENABLE_NANOVDB)
  # nanoVDB ships inside the OpenVDB repository as a header-only subdirectory
  # (openvdb/nanovdb/nanovdb/). Three discovery paths, in order of preference:
  #   1. CARBON_NANOVDB_INCLUDE_DIR (if the user already has nanoVDB on disk).
  #   2. find_package(nanovdb CONFIG) (newer OpenVDB installs ship a config target).
  #   3. FetchContent of the OpenVDB repo, pinned to a release tag, populated only
  #      (no add_subdirectory — OpenVDB's top-level CMake pulls TBB/Boost/etc. and
  #      we only need the header tree).
  set(_carbon_nanovdb_include_dir "")

  if(CARBON_NANOVDB_INCLUDE_DIR AND EXISTS "${CARBON_NANOVDB_INCLUDE_DIR}/nanovdb/NanoVDB.h")
    set(_carbon_nanovdb_include_dir "${CARBON_NANOVDB_INCLUDE_DIR}")
    message(STATUS "carbon: using user-supplied nanoVDB headers at ${_carbon_nanovdb_include_dir}")
  else()
    find_package(nanovdb CONFIG QUIET)
    if(TARGET nanovdb)
      message(STATUS "carbon: using installed nanovdb CMake package")
    else()
      include(FetchContent)
      set(CARBON_OPENVDB_GIT_TAG "v11.0.0" CACHE STRING
        "OpenVDB tag to fetch when CARBON_ENABLE_NANOVDB=ON and no nanoVDB is preinstalled")
      message(STATUS "carbon: fetching OpenVDB ${CARBON_OPENVDB_GIT_TAG} for nanoVDB headers")
      FetchContent_Declare(carbon_openvdb
        GIT_REPOSITORY https://github.com/AcademySoftwareFoundation/openvdb.git
        GIT_TAG        ${CARBON_OPENVDB_GIT_TAG}
        GIT_SHALLOW    TRUE
        GIT_SUBMODULES ""  # do not recurse: nanovdb itself has no submodules we need
        )
      # Populate without configuring: we don't want OpenVDB's top-level CMakeLists to
      # run because it expects TBB, Boost, blosc, etc. — we just need the source tree
      # so we can hand-pick the nanovdb/nanovdb include directory. CMake 3.30 deprecates
      # the bare FetchContent_Populate path (CMP0169); the alternative would be
      # FetchContent_MakeAvailable with SOURCE_SUBDIR pointing at a CMakeLists-free
      # folder, but it's brittle to the OpenVDB tree layout — pin the policy to OLD
      # locally so this stays a one-liner that survives layout changes.
      if(POLICY CMP0169)
        cmake_policy(PUSH)
        cmake_policy(SET CMP0169 OLD)
      endif()
      FetchContent_GetProperties(carbon_openvdb)
      if(NOT carbon_openvdb_POPULATED)
        FetchContent_Populate(carbon_openvdb)
      endif()
      if(POLICY CMP0169)
        cmake_policy(POP)
      endif()
      if(NOT EXISTS "${carbon_openvdb_SOURCE_DIR}/nanovdb/nanovdb/NanoVDB.h")
        message(FATAL_ERROR
          "carbon: FetchContent populated OpenVDB at ${carbon_openvdb_SOURCE_DIR} but "
          "nanovdb/nanovdb/NanoVDB.h is missing — does ${CARBON_OPENVDB_GIT_TAG} ship "
          "nanoVDB? Try a newer tag via -DCARBON_OPENVDB_GIT_TAG=<tag>.")
      endif()
      set(_carbon_nanovdb_include_dir "${carbon_openvdb_SOURCE_DIR}/nanovdb")
    endif()
  endif()

  target_sources(carbon PRIVATE src/io_nanovdb.cpp)
  if(TARGET nanovdb)
    target_link_libraries(carbon PRIVATE nanovdb)
  else()
    target_include_directories(carbon PRIVATE "${_carbon_nanovdb_include_dir}")
  endif()
  target_compile_definitions(carbon PUBLIC CARBON_HAVE_NANOVDB)
endif()

# So tests and dlopen can resolve libcudart (and libnccl, libmpi when enabled) from the build tree.
set_target_properties(carbon PROPERTIES
  BUILD_RPATH "${CARBON_RUNTIME_SEARCH_DIRS}")

add_library(carbon::carbon ALIAS carbon)

target_include_directories(carbon
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)

target_compile_definitions(carbon
  PRIVATE
    carbon_EXPORTS
  PUBLIC
    $<$<BOOL:${BUILD_SHARED_LIBS}>:CARBON_BUILD_SHARED>
    $<$<STREQUAL:${CARBON_BACKEND},Cuda>:CARBON_BACKEND_CUDA=1>
    $<$<STREQUAL:${CARBON_BACKEND},Sycl>:CARBON_BACKEND_SYCL=1>
    $<$<STREQUAL:${CARBON_BACKEND},Hip>:CARBON_BACKEND_HIP=1>)

if(CARBON_BACKEND STREQUAL "Cuda")
  target_compile_options(carbon PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:${CARBON_NVCC_DEVICE_FLAGS}>)
endif()

set_target_properties(carbon PROPERTIES
  OUTPUT_NAME carbon
  SOVERSION ${PROJECT_VERSION_MAJOR}
  VERSION ${PROJECT_VERSION})

if(MSVC)
  target_compile_definitions(carbon PRIVATE _CRT_SECURE_NO_WARNINGS)
  if(CARBON_WERROR)
    target_compile_options(carbon PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/WX>)
  endif()
else()
  target_compile_options(carbon PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wpedantic>)
  # Opt-in CI gate: the libcarbon compile (library .cpp + the headers they pull
  # in) is warning-clean under -Wall -Wextra -Wpedantic, so promote warnings to
  # errors when requested to keep it that way. Scoped to the library's host (CXX)
  # compile only — nvcc (CUDA) emits cross-execution-space diagnostics that are
  # not host-warning-clean across toolchain versions, so we do not gate device
  # TUs here; the format-check target plus this host gate cover the day-to-day
  # regressions. Flip on with -DCARBON_WERROR=ON in CI.
  if(CARBON_WERROR)
    target_compile_options(carbon PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-Werror>)
  endif()
endif()

# Host sanitizers. Scoped to the CXX language: neither nvcc nor acpp accepts -fsanitize on the
# device pass, and the failure modes these catch (buffer lifetimes, the RAII deleter paths, the
# I/O readers, host-side races in the OpenMP launchers) live in host TUs. Device-side memory
# errors are compute-sanitizer's department, which needs no build flag.
#
# PUBLIC rather than PRIVATE because the flags must also reach every test and example: ASan and
# TSan change the ABI of the runtime they interpose on and must be present at link time, so a
# sanitized libcarbon linked into an unsanitized executable does not work.
if(CARBON_SANITIZE)
  if(CARBON_SANITIZE STREQUAL "address+undefined")
    set(_carbon_san "address,undefined")
  elseif(CARBON_SANITIZE MATCHES "^(address|undefined|thread)$")
    set(_carbon_san "${CARBON_SANITIZE}")
  else()
    message(FATAL_ERROR
      "CARBON_SANITIZE='${CARBON_SANITIZE}' is not recognised; expected one of: "
      "address, undefined, address+undefined, thread (or empty to disable).")
  endif()
  if(MSVC)
    message(FATAL_ERROR "CARBON_SANITIZE is only wired for GCC / Clang host compilers.")
  endif()
  # -fno-omit-frame-pointer and -g are what make the reports name a carbon frame instead of an
  # address; without them an ASan trace through an -O3 build is largely unusable.
  target_compile_options(carbon PUBLIC
    $<$<COMPILE_LANGUAGE:CXX>:-fsanitize=${_carbon_san};-fno-omit-frame-pointer;-g>)
  target_link_options(carbon PUBLIC -fsanitize=${_carbon_san})
  message(STATUS "carbon: host sanitizers enabled (-fsanitize=${_carbon_san})")
endif()

if(CARBON_ENABLE_PYTHON)
  if(NOT CARBON_BACKEND STREQUAL "Cuda")
    message(FATAL_ERROR "CARBON_ENABLE_PYTHON requires CARBON_BACKEND=Cuda")
  endif()
  add_subdirectory(libcarbonpy)
endif()

if(CARBON_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()

# Examples are dual-backend after phase 4+: every example replaced its raw cuda*
# call sites with carbon::Stream::wait() and is registered through the same
# add_executable / add_sycl_to_target dance the tests use.
if(CARBON_BUILD_EXAMPLES)
  add_subdirectory(third_party)
  if(CARBON_BUILD_VIEWER)
    if(NOT CARBON_VIEWER_DESKTOP AND NOT CARBON_VIEWER_WEB)
      message(FATAL_ERROR
        "CARBON_BUILD_VIEWER=ON requires at least one backend: enable "
        "CARBON_VIEWER_DESKTOP (GLFW+OpenGL) and/or CARBON_VIEWER_WEB (imgui-ws).")
    endif()
    # CarbonViewer builds the ImGui core (always) plus, when CARBON_VIEWER_DESKTOP
    # is ON, the GLFW+OpenGL desktop backend. A web-only build (desktop OFF) needs
    # neither OpenGL nor GLFW, so headless hosts can serve the browser viewer.
    include(CarbonViewer)
    if(CARBON_VIEWER_WEB)
      include(CarbonViewerWs)
    endif()
  endif()
  add_subdirectory(examples)
endif()

# Benchmarks share the same dual-backend (Cuda/Sycl) toolchain plumbing as
# examples. Larger, focused workloads aimed at headline performance numbers
# (MLUPS / GLUPS / wall time) rather than functional coverage. Pull in the
# bundled clipp header from third_party/ if examples didn't already.
if(CARBON_BUILD_BENCHMARKS)
  if(NOT CARBON_BUILD_EXAMPLES)
    add_subdirectory(third_party)
  endif()
  add_subdirectory(benchmarks)
endif()

# Convenience targets for clang-format. Only registered when clang-format is
# discoverable; otherwise configure stays quiet. Driven by the repo
# .clang-format / .clang-format-ignore files.
find_program(CARBON_CLANG_FORMAT_EXECUTABLE
  NAMES clang-format
  DOC "clang-format used by the `format` / `format-check` targets")

# Convenience target for Doxygen-based API docs. Only registered when
# `doxygen` is on PATH; otherwise the configure stays quiet.
find_package(Doxygen QUIET)
if(DOXYGEN_FOUND)
  set(CARBON_DOXYGEN_OUTPUT_DIR "${CMAKE_BINARY_DIR}/docs")
  configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/Doxyfile.in"
    "${CMAKE_CURRENT_BINARY_DIR}/Doxyfile"
    @ONLY)
  add_custom_target(docs
    COMMAND ${CMAKE_COMMAND} -E make_directory "${CARBON_DOXYGEN_OUTPUT_DIR}"
    COMMAND ${DOXYGEN_EXECUTABLE} "${CMAKE_CURRENT_BINARY_DIR}/Doxyfile"
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    COMMENT "Rendering carbon API docs to ${CARBON_DOXYGEN_OUTPUT_DIR}/html/index.html"
    VERBATIM)
endif()

if(CARBON_CLANG_FORMAT_EXECUTABLE)
  file(GLOB_RECURSE CARBON_FORMAT_SOURCES CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/include/*.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/include/*.hpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/include/*.inl"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cu"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cuh"
    "${CMAKE_CURRENT_SOURCE_DIR}/tests/*.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/tests/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/tests/*.cu"
    "${CMAKE_CURRENT_SOURCE_DIR}/tests/*.cuh"
    "${CMAKE_CURRENT_SOURCE_DIR}/examples/*.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/examples/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/examples/*.cu"
    "${CMAKE_CURRENT_SOURCE_DIR}/examples/*.cuh"
    "${CMAKE_CURRENT_SOURCE_DIR}/benchmarks/*.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/benchmarks/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/benchmarks/*.cu"
    "${CMAKE_CURRENT_SOURCE_DIR}/benchmarks/*.cuh")

  add_custom_target(format
    COMMAND "${CARBON_CLANG_FORMAT_EXECUTABLE}" -i ${CARBON_FORMAT_SOURCES}
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    COMMENT "Running clang-format in-place over carbon sources"
    VERBATIM)

  add_custom_target(format-check
    COMMAND "${CARBON_CLANG_FORMAT_EXECUTABLE}" --dry-run --Werror ${CARBON_FORMAT_SOURCES}
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    COMMENT "Checking carbon sources against .clang-format (non-zero on drift)"
    VERBATIM)

  # CI guard: run format-check as a CTest entry so `ctest` fails on style drift
  # alongside the unit tests, instead of relying on developers to remember
  # `cmake --build build --target format-check`.
  if(CARBON_BUILD_TESTS)
    add_test(NAME carbon_format_check
      COMMAND "${CARBON_CLANG_FORMAT_EXECUTABLE}" --dry-run --Werror ${CARBON_FORMAT_SOURCES}
      WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")
    # cpu: needs no device, so the CPU-only CI lane (ctest -L cpu) picks it up.
    # See the label semantics documented in tests/CMakeLists.txt.
    set_tests_properties(carbon_format_check PROPERTIES LABELS "cpu")
  endif()
endif()

# --------------------------------------------------------------------------------------------
# Install + carbonConfig.cmake package
# --------------------------------------------------------------------------------------------
#
# Lays down the carbon library, headers, the bundled rapidjson headers, and a CMake package
# config so external projects can do:
#
#   find_package(carbon REQUIRED)
#   target_link_libraries(my_app PRIVATE carbon::carbon)
#
# Transitive deps that are PUBLIC-linked into carbon (OpenMP, CUDAToolkit, spdlog) are pulled
# in by carbonConfig.cmake via find_dependency(); MPI/NCCL stay PRIVATE so consumers don't need
# them on their search path even when carbon was built with CARBON_ENABLE_MPI=ON.

install(TARGETS carbon rapidjson
  EXPORT carbonTargets
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

install(DIRECTORY include/carbon
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  FILES_MATCHING PATTERN "*.h" PATTERN "*.inl")

install(FILES "${CMAKE_CURRENT_BINARY_DIR}/include/carbon/version.h"
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/carbon)

# Bundled rapidjson headers (header-only). Goes under the same include prefix, so the
# rapidjson INTERFACE target's $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}> resolves
# correctly for consumers.
if(CARBON_RAPIDJSON_SOURCE_DIR)
  install(DIRECTORY "${CARBON_RAPIDJSON_SOURCE_DIR}/include/rapidjson"
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp")
endif()

install(EXPORT carbonTargets
  FILE carbonTargets.cmake
  NAMESPACE carbon::
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/carbon)

include(CMakePackageConfigHelpers)

configure_package_config_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/cmake/carbonConfig.cmake.in"
  "${CMAKE_CURRENT_BINARY_DIR}/carbonConfig.cmake"
  INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/carbon)

write_basic_package_version_file(
  "${CMAKE_CURRENT_BINARY_DIR}/carbonConfigVersion.cmake"
  VERSION ${PROJECT_VERSION}
  COMPATIBILITY SameMinorVersion)

install(FILES
  "${CMAKE_CURRENT_BINARY_DIR}/carbonConfig.cmake"
  "${CMAKE_CURRENT_BINARY_DIR}/carbonConfigVersion.cmake"
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/carbon)
