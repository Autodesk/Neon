#!/usr/bin/env bash
#
# scripts/docker-build-all.sh
#
# Build every carbon configuration in its own Docker container. If a matching
# local GPU is reachable from the host, run that configuration's test suite
# inside the container; otherwise stop after the build + format-check (the
# whole CMake/compile/link path is exercised either way).
#
# The seven configurations covered match the validation matrix in AGENTS.md:
#
#   ┌─────────────┬──────────────┬──────────────┬─────────────────────────────┐
#   │ name        │ image        │ runtime GPU  │ extra cmake flags           │
#   ├─────────────┼──────────────┼──────────────┼─────────────────────────────┤
#   │ cuda        │ carbon-cuda  │ NVIDIA       │ (defaults)                  │
#   │ cuda-mpi    │ carbon-cuda  │ NVIDIA       │ -DCARBON_ENABLE_MPI=ON      │
#   │ sycl        │ carbon-sycl  │ NVIDIA       │ -DCARBON_BACKEND=Sycl ...   │
#   │ sycl-mpi    │ carbon-sycl  │ NVIDIA       │ ... + ENABLE_MPI=ON         │
#   │ sycl-rccl   │ carbon-sycl- │ AMD          │ Sycl hip:<arch> + MPI       │
#   │             │ rccl         │              │                             │
#   │ hip         │ carbon-hip   │ AMD          │ -DCARBON_BACKEND=Hip ...    │
#   │ hip-mpi     │ carbon-hip   │ AMD          │ ... + ENABLE_MPI=ON (RCCL)  │
#   └─────────────┴──────────────┴──────────────┴─────────────────────────────┘
#
# Usage:
#   scripts/docker-build-all.sh [--only NAME[,NAME...]] [--no-cache]
#                               [--sm-arch SM] [--gfx-arch GFX] [--werror]
#                               [--skip-build-images]
#
#   --werror   Gate the libcarbon host compile warning-clean (-DCARBON_WERROR=ON)
#              on the `cuda` config. Used by the gpu-ci workflow; off by default
#              for local runs so a stray host-gcc warning never blocks the matrix.
#
# Examples:
#   # Build everything; run tests for whatever GPU is present.
#   scripts/docker-build-all.sh
#
#   # Just rebuild the SYCL+RCCL container (no NVIDIA/CUDA work).
#   scripts/docker-build-all.sh --only sycl-rccl --gfx-arch gfx90a
#
#   # NVIDIA RTX 4090 host (sm_89), skip image rebuilds (use cached).
#   scripts/docker-build-all.sh --sm-arch sm_89 --skip-build-images

set -uo pipefail

# ----------------------------------------------------------------------------
# Layout
# ----------------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
DOCKER_DIR="${REPO_ROOT}/docker"
LOG_DIR="${REPO_ROOT}/docker-logs"
mkdir -p "${LOG_DIR}"

CUDA_IMAGE="carbon-cuda:local"
SYCL_IMAGE="carbon-sycl:local"
SYCL_RCCL_IMAGE="carbon-sycl-rccl:local"
HIP_IMAGE="carbon-hip:local"
HIP_NVIDIA_IMAGE="carbon-hip-nvidia:local"

# Sensible defaults; override via --sm-arch / --gfx-arch.
SM_ARCH="${SM_ARCH:-sm_80}"
GFX_ARCH="${GFX_ARCH:-gfx908}"
# Numeric CUDA arch (e.g. 80) derived from SM_ARCH (sm_80) for CMAKE_HIP_ARCHITECTURES
# under the HIP-on-NVIDIA platform, which interprets arches as CUDA compute capabilities.
SM_NUM="${SM_ARCH#sm_}"
DOCKER_NO_CACHE=""
SKIP_BUILD_IMAGES=0
ONLY_FILTER=""
# When 1, append -DCARBON_WERROR=ON to the native-compiler configs (cuda / hip /
# hip-nvidia) so the libcarbon host compile is gated warning-clean on the real
# container toolchain. Off by default so a local matrix run is never blocked by a
# stray warning from a newer/older host gcc; CI opts in via --werror.
WERROR=0

# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------

usage() {
  sed -n '2,30p' "$0"
}

while [ $# -gt 0 ]; do
  case "$1" in
    --only)              ONLY_FILTER="$2"; shift 2 ;;
    --no-cache)          DOCKER_NO_CACHE="--no-cache"; shift ;;
    --sm-arch)           SM_ARCH="$2"; shift 2 ;;
    --gfx-arch)          GFX_ARCH="$2"; shift 2 ;;
    --werror)            WERROR=1; shift ;;
    --skip-build-images) SKIP_BUILD_IMAGES=1; shift ;;
    -h|--help)           usage; exit 0 ;;
    *) echo "unknown arg: $1" >&2; usage; exit 2 ;;
  esac
done

# ----------------------------------------------------------------------------
# Pre-flight
# ----------------------------------------------------------------------------

command -v docker >/dev/null 2>&1 || {
  echo "error: docker not found on PATH" >&2; exit 1; }

# Assembled once; appended to the `cuda` config's flags. Scoped to `cuda` because
# CARBON_WERROR only gates the libcarbon host (CXX) compile, which is g++ in the
# CUDA container (Ubuntu 22.04 / gcc 11) — the toolchain the warning-clean
# baseline was verified against. Empty unless --werror was passed.
WERROR_FLAG=""
if [ "${WERROR}" = "1" ]; then
  WERROR_FLAG="-DCARBON_WERROR=ON"
fi

# ----------------------------------------------------------------------------
# GPU detection (host side)
#
# NVIDIA: `nvidia-smi -L` lists devices; presence implies the nvidia-container
#         runtime hooks are usually present too, but we still gate the actual
#         `--gpus` flag on whether the runtime accepts it.
# AMD:    `/dev/kfd` is the AMDGPU compute device; presence is the canonical
#         test ROCm uses internally.
# ----------------------------------------------------------------------------

HAS_NVIDIA=0
HAS_AMD=0

if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
  HAS_NVIDIA=1
fi
if [ -r /dev/kfd ] && [ -d /dev/dri ]; then
  HAS_AMD=1
fi

# Probe whether docker actually exposes NVIDIA GPUs (nvidia-container-toolkit
# is installed and the daemon is configured for it). We don't strictly need
# this info up-front — the per-config `docker run` will fail with --gpus all
# if it isn't available — but a clean banner up top is worth one cheap test.
NVIDIA_DOCKER_OK=0
if [ "${HAS_NVIDIA}" = "1" ]; then
  if docker run --rm --gpus all --entrypoint /bin/true \
       nvidia/cuda:12.6.3-base-ubuntu22.04 >/dev/null 2>&1; then
    NVIDIA_DOCKER_OK=1
  fi
fi

echo "================================================================"
echo " carbon — docker build matrix"
echo "================================================================"
echo "  repo root         : ${REPO_ROOT}"
echo "  log dir           : ${LOG_DIR}"
echo "  NVIDIA on host    : ${HAS_NVIDIA}  (docker --gpus all OK: ${NVIDIA_DOCKER_OK})"
echo "  AMD on host       : ${HAS_AMD}"
echo "  NVIDIA SM arch    : ${SM_ARCH}"
echo "  AMD GFX arch      : ${GFX_ARCH}"
echo "  only filter       : ${ONLY_FILTER:-<all>}"
echo "================================================================"
echo

# ----------------------------------------------------------------------------
# Build the four base images (cuda, sycl, sycl-rccl, hip), filtered by --only.
# ----------------------------------------------------------------------------

build_image() {
  local image="$1"; local dockerfile="$2"; local stage="$3"
  echo "[image] building ${image} from ${dockerfile}"
  docker build ${DOCKER_NO_CACHE} \
    -f "${dockerfile}" \
    -t "${image}" \
    "${REPO_ROOT}" 2>&1 | tee "${LOG_DIR}/image-${stage}.log"
  local rc=${PIPESTATUS[0]}
  if [ "${rc}" != "0" ]; then
    echo "[image] FAILED to build ${image} (exit ${rc})" >&2
    return "${rc}"
  fi
}

want() {
  # Returns 0 iff $1 should run given the --only filter.
  local name="$1"
  if [ -z "${ONLY_FILTER}" ]; then return 0; fi
  case ",${ONLY_FILTER}," in
    *",${name},"*) return 0 ;;
    *) return 1 ;;
  esac
}

if [ "${SKIP_BUILD_IMAGES}" = "0" ]; then
  declare -A IMAGE_BUILD_RC=()
  if want cuda || want cuda-mpi; then
    build_image "${CUDA_IMAGE}"      "${DOCKER_DIR}/Dockerfile.cuda"      "cuda"
    IMAGE_BUILD_RC[cuda]=$?
  fi
  if want sycl || want sycl-mpi; then
    build_image "${SYCL_IMAGE}"      "${DOCKER_DIR}/Dockerfile.sycl"      "sycl"
    IMAGE_BUILD_RC[sycl]=$?
  fi
  if want sycl-rccl; then
    build_image "${SYCL_RCCL_IMAGE}" "${DOCKER_DIR}/Dockerfile.sycl-rccl" "sycl-rccl"
    IMAGE_BUILD_RC[sycl-rccl]=$?
  fi
  if want hip || want hip-mpi; then
    build_image "${HIP_IMAGE}"       "${DOCKER_DIR}/Dockerfile.hip"       "hip"
    IMAGE_BUILD_RC[hip]=$?
  fi
  if want hip-nvidia; then
    build_image "${HIP_NVIDIA_IMAGE}" "${DOCKER_DIR}/Dockerfile.hip-nvidia" "hip-nvidia"
    IMAGE_BUILD_RC[hip-nvidia]=$?
  fi
fi

# ----------------------------------------------------------------------------
# Per-configuration runner.
#
# Each config bind-mounts the source tree READ-ONLY (so a stray `rm -rf` inside
# the container can't trash your checkout), uses an in-container build dir
# inside the ephemeral container fs, and writes results to a per-config log.
# Test artifacts are mirrored into ${LOG_DIR}/<name>/ via a second mount.
# ----------------------------------------------------------------------------

declare -A CFG_RC=()
declare -A CFG_RAN_TESTS=()

run_config() {
  local name="$1"
  local image="$2"
  local gpu_kind="$3"                   # nvidia | amd | none
  local cmake_flags="$4"
  local format_check="$5"               # 1 to also run carbon_format_check
  shift 5

  if ! want "${name}"; then
    echo "[run] ${name} — skipped (filtered out)"
    return 0
  fi

  echo
  echo "================================================================"
  echo " [run] ${name}"
  echo "   image      : ${image}"
  echo "   gpu kind   : ${gpu_kind}"
  echo "   cmake flags: ${cmake_flags}"
  echo "================================================================"

  # GPU passthrough.
  local docker_gpu_args=()
  local can_run_tests=0
  case "${gpu_kind}" in
    nvidia)
      if [ "${NVIDIA_DOCKER_OK}" = "1" ]; then
        docker_gpu_args+=(--gpus all)
        can_run_tests=1
      fi
      ;;
    amd)
      if [ "${HAS_AMD}" = "1" ]; then
        docker_gpu_args+=(
          --device=/dev/kfd
          --device=/dev/dri
          --group-add video
          --security-opt seccomp=unconfined
          --ipc=host
        )
        can_run_tests=1
      fi
      ;;
    none) ;;
  esac

  CFG_RAN_TESTS[${name}]="${can_run_tests}"

  local cfg_log_dir="${LOG_DIR}/${name}"
  mkdir -p "${cfg_log_dir}"

  # Body of the container script. CMake flags arrive via the CMAKE_FLAGS env
  # var (which the inner bash word-splits) — values containing `;` (e.g.
  # CARBON_SYCL_TARGETS=cuda:sm_80;omp) are wrapped in inner double quotes by
  # the caller so the inner shell keeps them as a single token.
  local format_check_cmd=""
  if [ "${format_check}" = "1" ]; then
    format_check_cmd='cmake --build . --target format-check'
  fi

  local container_script
  container_script=$(cat <<'INNER_EOF'
set -uo pipefail
set -x
cd /work
mkdir -p build && cd build
# shellcheck disable=SC2086  # intentional word-split of CMAKE_FLAGS
eval "cmake -S /src -B . -G Ninja -DCMAKE_BUILD_TYPE=Release ${CMAKE_FLAGS}"
cmake --build . -j"$(nproc)"
INNER_EOF
)
  container_script+=$'\n'"${format_check_cmd}"
  container_script+=$'\n'"if [ \"${can_run_tests}\" = \"1\" ]; then"
  container_script+=$'\n'"  ctest --output-on-failure"
  container_script+=$'\n'"else"
  container_script+=$'\n'"  echo '(no matching GPU on host — skipping ctest)'"
  container_script+=$'\n'"fi"

  docker run --rm -t \
    "${docker_gpu_args[@]}" \
    -v "${REPO_ROOT}:/src:ro" \
    -v "${cfg_log_dir}:/work" \
    -e CARBON_CONFIG="${name}" \
    -e CMAKE_FLAGS="${cmake_flags}" \
    -e CARBON_TEST_STRICT="${CARBON_TEST_STRICT:-}" \
    "${image}" \
    /bin/bash -lc "${container_script}" 2>&1 | tee "${LOG_DIR}/run-${name}.log"

  local rc=${PIPESTATUS[0]}
  CFG_RC[${name}]="${rc}"

  if [ "${rc}" = "0" ]; then
    echo "[run] ${name} — OK"
  else
    echo "[run] ${name} — FAILED (exit ${rc})" >&2
  fi
  return "${rc}"
}

# ----------------------------------------------------------------------------
# Configurations
#
# Notes on the SYCL CMake flag string:
#   - We pass CARBON_SYCL_TARGETS as a single semicolon-separated string;
#     CMake list semantics handle that via the cache-variable parser.
#   - For RCCL we pass CARBON_SYCL_ROCM_PATH=/opt/rocm so AdaptiveCpp finds
#     the system ROCm install (hipcc, device libs, lld) shipped by the
#     rocm/dev base image.
#   - CMAKE_CXX_COMPILER is wired through CARBON_SYCL_CXX (set in the SYCL
#     Dockerfiles) so cmake finds the AdaptiveCpp-shipped clang-19.
# ----------------------------------------------------------------------------

# For SYCL configs the targets string holds an embedded `;`; we wrap that one
# value in inner double quotes so the in-container `eval` parses the whole
# `-DCARBON_SYCL_TARGETS=...` flag as one argv entry. ${VAR} references
# expand on the host where useful (SM_ARCH/GFX_ARCH); ${CARBON_SYCL_CXX} and
# ${AdaptiveCpp_DIR} live in the container env so they're written with a
# preserved `$` (the host shell sees them in the single-quoted strings below).

run_config "cuda"      "${CUDA_IMAGE}"       "nvidia" \
  "${WERROR_FLAG}" \
  "1"

run_config "cuda-mpi"  "${CUDA_IMAGE}"       "nvidia" \
  '-DCARBON_ENABLE_MPI=ON' \
  "0"

run_config "sycl"      "${SYCL_IMAGE}"       "nvidia" \
  "-DCARBON_BACKEND=Sycl -DCARBON_SYCL_TARGETS=\"cuda:${SM_ARCH};omp\""' -DCMAKE_CXX_COMPILER=${CARBON_SYCL_CXX} -DAdaptiveCpp_DIR=${AdaptiveCpp_DIR}' \
  "0"

#   The `-Wl,-rpath-link,/usr/lib/x86_64-linux-gnu` on the SYCL+MPI configs is the
#   container mirror of the host-side recipe in AGENTS.md: conda-shipped clang-19
#   links via its own `x86_64-conda-linux-gnu-ld`, which does not search
#   `/usr/lib/x86_64-linux-gnu` by default. Without this flag, FindMPI's C++
#   compile probe can't resolve OpenMPI's transitive deps (`libopen-pal.so.40`,
#   `libopen-rte.so.40`, `libhwloc.so.15`) and cmake reports `MPI_CXX_WORKS`
#   missing. Both EXE and SHARED linker flags need it — tests link as EXE,
#   libcarbon as SHARED.
#
#   The `-Wl,-rpath,${ACPP_ENV}/lib` (runtime rpath, not rpath-link) is added ahead
#   of MPI's link dir in cmake's RPATH computation so the runtime loader picks
#   conda's newer libstdc++.so.6 (GLIBCXX_3.4.32, needed by libacpp-rt.so and
#   libboost_fiber.so) before the distro libstdc++ that ships with
#   `/usr/lib/x86_64-linux-gnu`. Without it, cmake's automatic RPATH addition
#   (derived from MPI_CXX_LIBRARIES et al.) lands `/usr/lib/x86_64-linux-gnu`
#   before the conda lib and test binaries fail at load time with
#   `GLIBCXX_3.4.32 not found`.
run_config "sycl-mpi"  "${SYCL_IMAGE}"       "nvidia" \
  "-DCARBON_BACKEND=Sycl -DCARBON_SYCL_TARGETS=\"cuda:${SM_ARCH};omp\""' -DCARBON_ENABLE_MPI=ON -DCMAKE_CXX_COMPILER=${CARBON_SYCL_CXX} -DAdaptiveCpp_DIR=${AdaptiveCpp_DIR} "-DCMAKE_EXE_LINKER_FLAGS=-Wl,-rpath-link,/usr/lib/x86_64-linux-gnu -Wl,-rpath,${ACPP_ENV}/lib" "-DCMAKE_SHARED_LINKER_FLAGS=-Wl,-rpath-link,/usr/lib/x86_64-linux-gnu -Wl,-rpath,${ACPP_ENV}/lib"' \
  "0"

run_config "sycl-rccl" "${SYCL_RCCL_IMAGE}"  "amd" \
  "-DCARBON_BACKEND=Sycl -DCARBON_SYCL_TARGETS=\"hip:${GFX_ARCH}\""' -DCARBON_ENABLE_MPI=ON -DCARBON_SYCL_ROCM_PATH=/opt/rocm -DCMAKE_CXX_COMPILER=${CARBON_SYCL_CXX} -DAdaptiveCpp_DIR=${AdaptiveCpp_DIR} "-DCMAKE_EXE_LINKER_FLAGS=-Wl,-rpath-link,/usr/lib/x86_64-linux-gnu -Wl,-rpath,${ACPP_ENV}/lib" "-DCMAKE_SHARED_LINKER_FLAGS=-Wl,-rpath-link,/usr/lib/x86_64-linux-gnu -Wl,-rpath,${ACPP_ENV}/lib"' \
  "0"

# Native HIP. The .cu device TUs are compiled by ROCm's HIP language compiler
# (CMAKE_HIP_COMPILER = amdclang++, found via /opt/rocm/bin on PATH); host .cpp
# stays on the distro g++. CARBON_HIP_ARCH drives CMAKE_HIP_ARCHITECTURES.
# `hip` is the AMD analogue of the `cuda` config (and runs format-check); the
# `hip-mpi` sibling adds the RCCL transport (auto-selected by carbon's CMake for
# the native HIP backend).
run_config "hip"       "${HIP_IMAGE}"        "amd" \
  "-DCARBON_BACKEND=Hip -DCARBON_HIP_ARCH=${GFX_ARCH} -DCARBON_ROCM_PATH=/opt/rocm" \
  "1"

run_config "hip-mpi"   "${HIP_IMAGE}"        "amd" \
  "-DCARBON_BACKEND=Hip -DCARBON_HIP_ARCH=${GFX_ARCH} -DCARBON_ROCM_PATH=/opt/rocm -DCARBON_ENABLE_MPI=ON" \
  "0"

# HIP-on-NVIDIA: the native HIP backend compiled by nvcc (CMAKE_HIP_PLATFORM=nvidia),
# runnable on an NVIDIA GPU. This is the config that gives the HIP source tree real
# runtime coverage on an NVIDIA host (ctest runs when --gpus all is available). MPI is
# left off here — the distributed collective auto-selects RCCL for the HIP backend, which
# is wrong on an NVIDIA box; HIP-on-NVIDIA + NCCL is a tracked follow-up.
run_config "hip-nvidia" "${HIP_NVIDIA_IMAGE}" "nvidia" \
  "-DCARBON_BACKEND=Hip -DCMAKE_HIP_PLATFORM=nvidia -DCMAKE_HIP_ARCHITECTURES=${SM_NUM} -DCARBON_ROCM_PATH=/opt/rocm" \
  "0"

# ----------------------------------------------------------------------------
# Summary
# ----------------------------------------------------------------------------

echo
echo "================================================================"
echo " summary"
echo "================================================================"
printf " %-12s  %-7s  %-9s  %s\n" "config" "build" "tests-ran" "log"
printf " %-12s  %-7s  %-9s  %s\n" "------" "-----" "---------" "---"

overall_rc=0
for name in cuda cuda-mpi sycl sycl-mpi sycl-rccl hip hip-mpi hip-nvidia; do
  rc="${CFG_RC[$name]:-skipped}"
  ran="${CFG_RAN_TESTS[$name]:-?}"
  if [ "${rc}" = "0" ]; then
    status="ok"
  elif [ "${rc}" = "skipped" ]; then
    status="skip"
  else
    status="FAIL(${rc})"
    overall_rc=1
  fi
  printf " %-12s  %-7s  %-9s  %s\n" "${name}" "${status}" "${ran}" "${LOG_DIR}/run-${name}.log"
done
echo "================================================================"

exit "${overall_rc}"
