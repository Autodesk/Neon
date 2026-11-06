#!/usr/bin/env bash
#
# benchmarks/lbm_lid_driven_cavity_distributed/data/lbm-dist-sweep/reproduce.sh
#
# Reproduces this result set: runs the sweep described by lbm-dist-sweep.yml
# (the frozen copy next to this script) and renders every plot in this folder.
#
#   D3Q19 + BGK, fp32, --grid dense_distributed, --overlap standard;
#   N = 256..512 step 64; 1..8 GPUs (one MPI rank each); slab, pencil and
#   brick decompositions; NCCL halos with --hp-halo off and on;
#   400 warmup steps, then 5 timed repetitions of 1000 solver steps.
#
# Usage:
#   ./benchmarks/lbm_lid_driven_cavity_distributed/data/lbm-dist-sweep/reproduce.sh [--plot-only]
#
#   (no flag)    build build-dist/ in Docker, redo every run (force: true
#                overwrites json/), then plot; takes about two hours on 8 GPUs
#   --plot-only  only re-render the PNGs from the reports already in json/
#
# Needs docker with the NVIDIA container toolkit, the image named in the yml,
# and python3 with PyYAML and matplotlib. Run it from tmux (or similar) because
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
scripts="${repo_root}/benchmarks/lbm_lid_driven_cavity_distributed/scripts"
config="${here}/lbm-dist-sweep.yml"

if [[ "${plot_only}" -eq 0 ]]; then
  "${scripts}/lbm-dist-sweep-run.sh" "${config}"
fi

python3 "${scripts}/plotting/plot_dist_scaling.py" --config "${config}" --output-dir "${here}"
echo "==> plots written to ${here}"
