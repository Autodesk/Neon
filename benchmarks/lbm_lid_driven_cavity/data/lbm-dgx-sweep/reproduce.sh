#!/usr/bin/env bash
#
# benchmarks/lbm_lid_driven_cavity/data/lbm-dgx-sweep/reproduce.sh
#
# Reproduces this result set: runs the sweep described by lbm-dgx-sweep.yml
# (the frozen copy next to this script) and renders every plot in this folder.
#
#   D3Q19 + BGK, fp32, --overlap standard; N = 256..512 step 64; all four
#   grids: Dense and DenseDisg on 1..8 GPUs with Peeraccess, NCCL and MPI
#   (P2P); DenseDistributed and DenseDisgDistributed (one rank per GPU) with
#   NCCL and MPI (P2P), slab on 2..8 GPUs, pencil on 4 and 8, brick on 8;
#   every run with hp-halo off and on; 400 warmup steps, then 5 timed
#   repetitions of 1000 solver steps (840 runs).
#
# Original run: Nov 2026, 8x A100-SXM4-40GB (NVLink), driver 535.161.08,
# images carbon-cuda:local (docker/Dockerfile.cuda) for the builds and
# Peeraccess runs and carbon-cuda-nccl-pinned:local (NCCL 2.27.7) for the
# NCCL / MPI runs.
#
# Usage:
#   ./benchmarks/lbm_lid_driven_cavity/data/lbm-dgx-sweep/reproduce.sh [--plot-only]
#
#   (no flag)    build both trees in Docker, redo every run (force: true
#                overwrites json/), then plot; takes about 6 hours on 8 GPUs
#   --plot-only  only re-render the PNGs from the reports already in json/
#
# Needs docker with the NVIDIA container toolkit, both images above, and
# python3 with PyYAML and matplotlib. Run it from tmux (or similar) because
# the sweep outlives an SSH session.

set -euo pipefail

plot_only=0
case "${1:-}" in
  "") ;;
  --plot-only) plot_only=1 ;;
  *)
    echo "usage: $0 [--plot-only]" >&2
    exit 2
    ;;
esac

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${here}/../../../.." && pwd)"
scripts="${repo_root}/benchmarks/lbm_lid_driven_cavity/scripts"
config="${here}/lbm-dgx-sweep.yml"

if [[ "${plot_only}" -eq 0 ]]; then
  "${scripts}/lbm-dgx-sweep-run.sh" "${config}"
fi

python3 "${scripts}/plotting/plot_scaling.py" --config "${config}" --output-dir "${here}"
python3 "${scripts}/plotting/plot_disg_vs_dense.py" --config "${config}" --output-dir "${here}"
echo "==> plots written to ${here}"
