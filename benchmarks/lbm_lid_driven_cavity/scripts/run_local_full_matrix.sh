#!/usr/bin/env bash
#
# benchmarks/lbm_lid_driven_cavity/scripts/run_local_full_matrix.sh
#
# Run the carbon LBM lid-driven cavity benchmark across the full
# matrix of locally-available system configurations:
#
#   backend  ∈ { cuda, sycl }       — whichever build trees expose a built binary
#   gpus     ∈ { 1, $TOTAL }        — single GPU and "all GPUs in the system"
#                                     ($TOTAL = nvidia-smi device count)
#   config   ∈ { q19/bgk, q27/bgk, q27/kbc }
#   N        ∈ { 256 }              — fixed at the steady-state point
#
# Total runs = backends * gpu_counts * configs (= up to 12 on a 2-GPU host
# with both backends built). Output goes to one shared dir; the resulting
# bar chart shows MLUPS per (lattice, collision) config with one bar per
# (backend, gpu_count) sub-series.
#
# Output:
#   reports/full_matrix_local/
#     run-NNN-<backend>-<lattice>-<collision>-nN-gG.json   (one per cell)
#     mlups_full_matrix.png                                (the requested plot)
#
# Backend discovery:
#   - cuda binary is searched at  build/    , build-mpi/    , build-cuda/
#   - sycl binary is searched at  build-sycl/, build-sycl-mpi/
#   The first existing & executable match per backend wins. Only backends
#   whose binary is found are run; missing backends are skipped (a single-
#   backend host therefore gets a single-column matrix). Override paths via
#   the env vars CUDA_BENCH= / SYCL_BENCH= when the auto-search misses
#   your tree.
#
# Knobs (env vars, all optional):
#   ITERS=600        total iterations per run
#   WARMUP=200       warmup iterations excluded from timing
#   N=256            grid edge length
#   GPUS_LIST="1 2"  override the gpu-count axis directly (defaults to
#                    "1 ${TOTAL}", deduplicated; on a 1-GPU host that is
#                    just "1")
#   OUT_DIR=...      output directory (default reports/full_matrix_local)
#   CUDA_BENCH=...   path to the CUDA benchmark binary (overrides search)
#   SYCL_BENCH=...   path to the SYCL benchmark binary (overrides search)
#
# Usage:
#   ./benchmarks/lbm_lid_driven_cavity/scripts/run_local_full_matrix.sh
#
# Tip: build both backends ahead of time so this script doesn't have to:
#   cmake -S . -B build      -DCARBON_BACKEND=Cuda -DCMAKE_BUILD_TYPE=Release \
#                            -DCARBON_BUILD_BENCHMARKS=ON
#   cmake -S . -B build-sycl -DCARBON_BACKEND=Sycl -DCMAKE_BUILD_TYPE=Release \
#                            -DCARBON_BUILD_BENCHMARKS=ON \
#                            -DCARBON_SYCL_TARGETS="cuda:sm_80;omp"
#   cmake --build build      --target carbon_benchmark_lbm_lid_driven_cavity -j
#   cmake --build build-sycl --target carbon_benchmark_lbm_lid_driven_cavity -j

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${repo_root}"

ITERS="${ITERS:-600}"
WARMUP="${WARMUP:-200}"
N="${N:-256}"
OUT_DIR="${OUT_DIR:-reports/full_matrix_local}"

bench_relpath="benchmarks/lbm_lid_driven_cavity/carbon_benchmark_lbm_lid_driven_cavity"

# --- discover binaries -------------------------------------------------

discover_bin() {
  # discover_bin BACKEND CANDIDATE_DIR... -> echoes the first existing
  # path under <CANDIDATE>/<bench_relpath>; returns 1 if none found.
  local backend="$1"; shift
  for d in "$@"; do
    local cand="${d}/${bench_relpath}"
    if [[ -x "${cand}" ]]; then
      echo "${cand}"
      return 0
    fi
  done
  return 1
}

cuda_bin="${CUDA_BENCH:-}"
if [[ -z "${cuda_bin}" ]]; then
  cuda_bin="$(discover_bin cuda build build-mpi build-cuda || true)"
fi
sycl_bin="${SYCL_BENCH:-}"
if [[ -z "${sycl_bin}" ]]; then
  sycl_bin="$(discover_bin sycl build-sycl build-sycl-mpi || true)"
fi

backend_args=()
if [[ -n "${cuda_bin}" ]]; then
  echo "==> cuda backend: ${cuda_bin}"
  backend_args+=("cuda=${cuda_bin}")
fi
if [[ -n "${sycl_bin}" ]]; then
  echo "==> sycl backend: ${sycl_bin}"
  backend_args+=("sycl=${sycl_bin}")
fi

if [[ "${#backend_args[@]}" -eq 0 ]]; then
  cat >&2 <<EOF
error: no carbon_benchmark_lbm_lid_driven_cavity binary found in any of the
       known build trees. Build at least one backend, or pass an explicit
       binary path via CUDA_BENCH=... / SYCL_BENCH=... .

       Searched (relative to ${repo_root}):
         CUDA: build/ build-mpi/ build-cuda/
         SYCL: build-sycl/ build-sycl-mpi/

       To build CUDA:
         cmake -S . -B build      -DCARBON_BACKEND=Cuda \\
                                 -DCMAKE_BUILD_TYPE=Release \\
                                 -DCARBON_BUILD_BENCHMARKS=ON
         cmake --build build      --target carbon_benchmark_lbm_lid_driven_cavity -j

       To build SYCL (cuda:sm_80 + omp targets):
         cmake -S . -B build-sycl -DCARBON_BACKEND=Sycl \\
                                 -DCMAKE_BUILD_TYPE=Release \\
                                 -DCARBON_BUILD_BENCHMARKS=ON \\
                                 -DCARBON_SYCL_TARGETS="cuda:sm_80;omp"
         cmake --build build-sycl --target carbon_benchmark_lbm_lid_driven_cavity -j
EOF
  exit 1
fi

# --- discover GPU count ------------------------------------------------

if [[ -n "${GPUS_LIST:-}" ]]; then
  gpus_list="${GPUS_LIST}"
else
  total_gpus=0
  if command -v nvidia-smi >/dev/null 2>&1; then
    total_gpus="$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')"
  fi
  if [[ -z "${total_gpus}" || "${total_gpus}" -le 0 ]]; then
    echo "warning: could not determine GPU count via nvidia-smi; defaulting to 1" >&2
    total_gpus=1
  fi
  if [[ "${total_gpus}" -le 1 ]]; then
    gpus_list="1"
  else
    gpus_list="1 ${total_gpus}"
  fi
fi
echo "==> gpu counts: ${gpus_list}  (set GPUS_LIST=... to override)"

mkdir -p "${OUT_DIR}"

# --- run sweep + render bar chart -------------------------------------

# shellcheck disable=SC2086 - intentional word splitting on gpus_list / backend_args
python3 "${repo_root}/benchmarks/lbm_lid_driven_cavity/scripts/sweep_and_plot.py" \
  --backends "${backend_args[@]}" \
  --mode sweep+plot \
  --output-dir "${OUT_DIR}" \
  --sizes "${N}" \
  --gpus ${gpus_list} \
  --configs q19:bgk q27:bgk q27:kbc \
  --iters "${ITERS}" \
  --warmup "${WARMUP}" \
  --plot-x config \
  --plot-y mlups \
  --plot-out "${OUT_DIR}/mlups_full_matrix.png" \
  --plot-title "Carbon LBM lid-driven cavity — full local matrix (N=${N})"

echo
echo "==> done"
echo "    reports: ${OUT_DIR}/run-*.json"
echo "    plot:    ${OUT_DIR}/mlups_full_matrix.png"
