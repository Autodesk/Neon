#!/usr/bin/env bash
# Runs the halo-scalability benchmark over a (N, C) sweep on this machine.
# CSV is appended row-by-row, so re-running with new sizes adds without
# overwriting -- delete the file first to start fresh.
#
#   bash benchmarks/halo_scalability/scripts/run_local.sh <build_dir> <csv_path>
#
# Defaults: build, reports/halo_scalability/halo.csv

set -euo pipefail

BUILD_DIR="${1:-build}"
CSV="${2:-reports/halo_scalability/halo.csv}"
BIN="${BUILD_DIR}/benchmarks/halo_scalability/carbon_benchmark_halo_scalability"

if [[ ! -x "${BIN}" ]]; then
  echo "missing binary: ${BIN}" >&2
  echo "build with: cmake --build ${BUILD_DIR} -j --target carbon_benchmark_halo_scalability" >&2
  exit 1
fi

mkdir -p "$(dirname "${CSV}")"

# 2-GPU sweep: this is where the halo work actually happens. Sizes chosen to
# stay well within the per-GPU memory footprint at C=27 and N=384
# (~13 GB / slab) on an 80GB A100.
SIZES=(64 96 128 160 192 224 256 320 384)
COMPONENTS="1,3,5,9,13,19,27"

# Sweep 1: standard semantic across the full (N, C) matrix.
for N in "${SIZES[@]}"; do
  echo ">>> standard, N=${N}, gpus=2, C={${COMPONENTS}}"
  "${BIN}" --N "${N}" --gpus 2 --components "${COMPONENTS}" \
           --semantic standard \
           --iters 200 --warmup 50 --csv "${CSV}" 2>/dev/null | \
    grep -E '^(grid|----|dense|disg)' || true
done

# Sweep 2: lattice semantic over each lattice flavour. Components is fixed by
# the lattice (Q=19 / Q=27); we only sweep N.
for LATTICE in d3q19 d3q19_grouped d3q27 d3q27_grouped; do
  for N in "${SIZES[@]}"; do
    echo ">>> lattice ${LATTICE}, N=${N}, gpus=2"
    "${BIN}" --N "${N}" --gpus 2 \
             --semantic lattice --lattice "${LATTICE}" \
             --iters 200 --warmup 50 --csv "${CSV}" 2>/dev/null | \
      grep -E '^(grid|----|dense|disg)' || true
  done
done

echo "wrote ${CSV}"
