#!/usr/bin/env bash
#
# benchmarks/lbm_lid_driven_cavity/scripts/run_local_single_gpu_n256.sh
#
# Run the carbon LBM lid-driven cavity benchmark on the local machine, on
# a single GPU, at N=256, across every valid (lattice, collision)
# configuration the binary supports — and render a bar chart comparing
# MLUPS per configuration.
#
# Configurations covered (3 runs total):
#   - q19 / bgk   — D3Q19 with BGK SRT collision (the bandwidth-bound baseline).
#   - q27 / bgk   — D3Q27 with BGK SRT (more populations per cell -> lower MLUPS,
#                   higher GB/s).
#   - q27 / kbc   — D3Q27 with KBC entropic stabilisation (compute-bound, lowest
#                   MLUPS but unconditionally stable at high Re).
# (q19+kbc is intentionally excluded — KBC needs the full D3Q27 lattice.)
#
# Output:
#   reports/single_gpu_n256_local/
#     run-000-q19-bgk-n256-g1.json
#     run-001-q27-bgk-n256-g1.json
#     run-002-q27-kbc-n256-g1.json
#     mlups_by_config.png        <- the requested plot
#
# Usage:
#   ./benchmarks/lbm_lid_driven_cavity/scripts/run_local_single_gpu_n256.sh
#
# Knobs (env vars):
#   ITERS=600     total iterations per run
#   WARMUP=200    warmup iterations excluded from timing
#   OUT_DIR=...   output directory (default reports/single_gpu_n256_local)
#   BUILD_DIR=... CMake build dir (default ./build)
#   JOBS=8        parallelism for the (auto) cmake build step
#
# The script (re)builds the benchmark binary if it is missing in the
# selected build directory; pass `BUILD_DIR=build-release ...` to point
# at an alternate tree (e.g. a Release-mode build).

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${repo_root}"

ITERS="${ITERS:-600}"
WARMUP="${WARMUP:-200}"
BUILD_DIR="${BUILD_DIR:-build}"
OUT_DIR="${OUT_DIR:-reports/single_gpu_n256_local}"
JOBS="${JOBS:-8}"

bench="${BUILD_DIR}/benchmarks/lbm_lid_driven_cavity/carbon_benchmark_lbm_lid_driven_cavity"

if [[ ! -x "${bench}" ]]; then
  echo "==> binary not found at ${bench}; building it now" >&2
  if [[ ! -d "${BUILD_DIR}" ]]; then
    echo "error: ${BUILD_DIR}/ does not exist — configure cmake first:" >&2
    echo "  cmake -S . -B ${BUILD_DIR} -DCARBON_BACKEND=Cuda -DCMAKE_BUILD_TYPE=Release" >&2
    exit 1
  fi
  cmake --build "${BUILD_DIR}" --target carbon_benchmark_lbm_lid_driven_cavity -j "${JOBS}"
fi

mkdir -p "${OUT_DIR}"

echo "==> running 3 single-GPU N=256 configurations into ${OUT_DIR}/"
python3 "${repo_root}/benchmarks/lbm_lid_driven_cavity/scripts/sweep_and_plot.py" \
  --bench "${bench}" \
  --mode sweep+plot \
  --output-dir "${OUT_DIR}" \
  --sizes 256 \
  --gpus 1 \
  --configs q19:bgk q27:bgk q27:kbc \
  --iters "${ITERS}" \
  --warmup "${WARMUP}" \
  --plot-x config \
  --plot-y mlups \
  --plot-out "${OUT_DIR}/mlups_by_config.png"

echo
echo "==> done"
echo "    reports: ${OUT_DIR}/run-*.json"
echo "    plot:    ${OUT_DIR}/mlups_by_config.png"
