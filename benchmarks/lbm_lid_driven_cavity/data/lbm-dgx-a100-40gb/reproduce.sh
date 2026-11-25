#!/usr/bin/env bash
#
# benchmarks/lbm_lid_driven_cavity/data/lbm-dgx-a100-40gb/reproduce.sh
#
# Reproduces this result set: runs the two sweeps described by
# lbm-dgx-a100-40gb.yml and lbm-dgx-a100-40gb-colormap.yml (the frozen copies
# next to this script, sharing json/) and renders every plot in this folder.
#
#   D3Q19 + BGK, fp32, --grid dense and disg, --overlap standard;
#   N = 128..608 step 32; Peeraccess / NCCL / MPI (P2P) at 8 GPUs plus the
#   1-GPU Peeraccess reference (MLUPS-per-GPU vs N plot), and Peeraccess / NCCL
#   on 1, 2, 4 and 8 GPUs (best-family colormap); 400 warmup steps, then one
#   timed window of 1600 solver steps.
#
# Runs on 8x A100-SXM4-40GB (NVLink), driver 535, images carbon-cuda:local
# (docker/Dockerfile.cuda) for the builds and carbon-cuda-nccl-pinned:local
# (NCCL 2.27.7, CUDA 12.2) for the runs.
#
# Usage:
#   ./benchmarks/lbm_lid_driven_cavity/data/lbm-dgx-a100-40gb/reproduce.sh [--plot-only]
#
#   (no flag)    delete json/*.json, build build/ and build-mpi/ in Docker, redo
#                every run (the two sweeps share runs; force: false lets the
#                second reuse the first's), then plot; takes a few hours on 8 GPUs
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
config="${here}/lbm-dgx-a100-40gb.yml"
colormap_config="${here}/lbm-dgx-a100-40gb-colormap.yml"

if [[ "${plot_only}" -eq 0 ]]; then
  rm -f "${here}"/json/*.json
  "${scripts}/lbm-dgx-sweep-run.sh" "${config}"
  SKIP_BUILD=1 "${scripts}/lbm-dgx-sweep-run.sh" "${colormap_config}"
fi

python3 "${scripts}/plotting/plot_scaling.py" --config "${config}" --output-dir "${here}"
python3 "${scripts}/plotting/plot_best_family.py" --config "${colormap_config}" \
  --output-dir "${here}"
echo "==> plots written to ${here}"
