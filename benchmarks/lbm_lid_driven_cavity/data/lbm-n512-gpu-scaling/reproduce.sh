#!/usr/bin/env bash
#
# benchmarks/lbm_lid_driven_cavity/data/lbm-n512-gpu-scaling/reproduce.sh
#
# Reproduces this result set: runs the sweep described by
# lbm-n512-gpu-scaling.yml (the frozen copy next to this script) and renders
# the plot in this folder.
#
#   D3Q19 + BGK, fp32, --grid dense and disg, --overlap standard; N = 512;
#   Peeraccess / NCCL / MPI (P2P) on 1, 2, 4 and 8 GPUs (the MPI transports at
#   1 GPU as one rank on GPU 0); 400 warmup steps, then one timed window of
#   1600 solver steps.
#
# Runs on 8x A100-SXM4-40GB (NVLink), driver 535, images carbon-cuda:local
# (docker/Dockerfile.cuda) for the builds and carbon-cuda-nccl-pinned:local
# (NCCL 2.27.7, CUDA 12.2) for the runs.
#
# Usage:
#   ./benchmarks/lbm_lid_driven_cavity/data/lbm-n512-gpu-scaling/reproduce.sh [--plot-only]
#
#   (no flag)    build build/ and build-mpi/ in Docker, redo every run (force:
#                true overwrites json/), then plot; takes about 15 minutes
#   --plot-only  only re-render the PNG from the reports already in json/
#
# Needs docker with the NVIDIA container toolkit, both images above, and
# python3 with PyYAML and matplotlib.

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
config="${here}/lbm-n512-gpu-scaling.yml"

if [[ "${plot_only}" -eq 0 ]]; then
  "${scripts}/lbm-dgx-sweep-run.sh" "${config}"
fi

python3 "${scripts}/plotting/plot_scaling.py" --config "${config}" --output-dir "${here}"
echo "==> plot written to ${here}"
