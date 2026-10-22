#!/usr/bin/env bash
# Build + test the Carbon Python (Warp) layer inside a CUDA 12 container.
#
# The pinned Warp fork (nvlukasz/warp @ external-source-support-update2) does not
# compile against CUDA 13 (CUB header reorg + dropped arches), and the dev host
# only ships CUDA 13. This script runs the whole Python validation loop inside a
# CUDA 12.6 devel container instead, where the fork builds cleanly. numpy comes
# from apt (PyPI is unreachable behind the dev-host proxy).
#
# Usage:  scripts/py-test-cuda12.sh [pytest-args...]
#   Env:  CARBON_PY_IMAGE   base image (default nvidia/cuda:12.6.3-devel-ubuntu22.04)
#         CARBON_PY_REBUILD_WARP=1  force a warp rebuild
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
IMAGE="${CARBON_PY_IMAGE:-cuda:12.8.0.ubuntu.24.04}"
BUILD_DIR="build-cuda12-py"

docker run --rm --gpus all \
  -e CARBON_PY_REBUILD_WARP="${CARBON_PY_REBUILD_WARP:-0}" \
  -e BUILD_DIR="${BUILD_DIR}" \
  -v "${ROOT}":/carbon -w /carbon \
  "${IMAGE}" bash -lc '
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
echo "=== apt deps ==="
apt-get update -qq
apt-get install -y -qq --no-install-recommends \
    python3 python3-numpy python3-pytest cmake ninja-build build-essential git \
    > /tmp/apt-install.log 2>&1 || { tail -20 /tmp/apt-install.log; exit 1; }
export PATH=/usr/local/cuda/bin:$PATH

if [ "${CARBON_PY_REBUILD_WARP}" = "1" ] || [ ! -f warp/warp/bin/warp.so ]; then
  echo "=== build warp fork (CUDA $(nvcc --version | sed -n "s/.*release //p" | cut -d, -f1)) ==="
  ( cd warp && python3 build_lib.py --cuda_path=/usr/local/cuda --no_libmathdx --no_standalone )
else
  echo "=== warp already built (warp/warp/bin/warp.so) ==="
fi

echo "=== build libcarbonpy ==="
rm -rf "${BUILD_DIR}"
cmake -S . -B "${BUILD_DIR}" -G Ninja \
  -DCARBON_ENABLE_PYTHON=ON -DCARBON_BUILD_EXAMPLES=OFF -DCARBON_BUILD_BENCHMARKS=OFF
cmake --build "${BUILD_DIR}" -j

echo "=== carbon python tests ==="
export PYTHONPATH=/carbon/warp:/carbon/py
python3 -m pytest -q tests/test_warp_*.py '"$*"'
'
