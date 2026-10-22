#!/usr/bin/env bash
# One-command local micro-benchmark sweep (quick mode by default).
#
#   bash benchmarks/scripts/run_local.sh [build_dir] [output_dir] [quick|full]

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BUILD_DIR="${1:-${ROOT}/build}"
OUT_DIR="${2:-}"
MODE="${3:-quick}"

if [[ -n "${OUT_DIR}" ]]; then
  REPORT_DIR="${OUT_DIR}"
else
  REPORT_DIR="${ROOT}/reports/bench-$(date -u +%Y-%m-%d)"
fi

ARGS=(--build-dir "${BUILD_DIR}" --mode "${MODE}" --output-dir "${REPORT_DIR}")

cmake --build "${BUILD_DIR}" -j --target \
  carbon_benchmark_reduce_throughput \
  carbon_benchmark_skeleton_overlap \
  carbon_benchmark_block_sparse

python3 "${ROOT}/benchmarks/scripts/run_benchmarks.py" "${ARGS[@]}"

python3 "${ROOT}/benchmarks/scripts/plot_summary.py" \
  --input-dir "${REPORT_DIR}" \
  --output "${REPORT_DIR}/summary.png" || true
