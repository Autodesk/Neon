#!/usr/bin/env bash
# End-to-end XLB (Carbon backend) validation inside a CUDA 12 container.
#
# Builds the Warp fork + libcarbonpy (like scripts/py-test-cuda12.sh) and then
# pip-installs XLB's runtime deps from the Autodesk Artifactory index (PyPI is
# blocked on the dev host; pip >= 24 is required, so we use the ubuntu-24.04
# based CUDA 12.8 image). XLB itself is installed with --no-deps so the [neon]
# wheel hook never runs and the locally built Warp fork stays on the path.
#
# Usage:  scripts/xlb-test-cuda12.sh [python args, default: -c "import xlb"]
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
IMAGE="${CARBON_PY_IMAGE:-cuda:12.8.0.ubuntu.24.04}"
INDEX_URL="${PIP_INDEX_URL:-https://art-toucan.autodesk.com/artifactory/api/pypi/autodesk-pypi-virtual/simple}"
BUILD_DIR="build-cuda12-py"
# Default: run the import/smoke file. Override by passing a path/args, e.g.
#   scripts/xlb-test-cuda12.sh -m pytest -q XLB/tests/...
PYARGS="${*:-scripts/_xlb_smoke.py}"

# Forward CUDA_VISIBLE_DEVICES only when set to a non-empty value: an *empty*
# value hides every GPU (it is a device list, not a boolean), so passing it
# unconditionally would leave carbon's System with zero GPUs.
CVD_ENV=()
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
  CVD_ENV=(-e "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}")
fi

docker run --rm --gpus all --network host \
  -e PIP_INDEX_URL="${INDEX_URL}" \
  -e PIP_ROOT_USER_ACTION=ignore \
  -e BUILD_DIR="${BUILD_DIR}" \
  -e PYARGS="${PYARGS}" \
  "${CVD_ENV[@]}" \
  -v "${ROOT}":/carbon -w /carbon \
  "${IMAGE}" bash -lc '
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
echo "=== apt deps ==="
apt-get update -qq
apt-get install -y -qq --no-install-recommends \
    python3 python3-pip python3-dev python3-venv cmake ninja-build build-essential git \
    libgl1 libglib2.0-0 > /tmp/apt-install.log 2>&1 || { tail -20 /tmp/apt-install.log; exit 1; }
export PATH=/usr/local/cuda/bin:$PATH
PIP="pip3 install --break-system-packages -q"

echo "=== build warp fork ==="
if [ ! -f warp/warp/bin/warp.so ]; then
  ( cd warp && python3 build_lib.py --cuda_path=/usr/local/cuda --no_libmathdx --no_standalone )
fi
# Warp needs numpy at import; install runtime deps (fork stays on PYTHONPATH, no PyPI warp-lang).
echo "=== pip deps (Artifactory) ==="
# jax/jaxlib pinned to 0.11.0: Artifactory only serves already-cached wheels, and the
# newest (0.11.1 / jaxlib 0.8.2) 403 on-demand. 0.11.0 is cached and satisfies XLB (>=0.8.2).
${PIP} "numpy>=2.1.2" nvtx pydantic "jax==0.11.0" "jaxlib==0.11.0" matplotlib pyvista trimesh numpy-stl h5py pytest

echo "=== build libcarbonpy ==="
rm -rf "${BUILD_DIR}"
cmake -S . -B "${BUILD_DIR}" -G Ninja \
  -DCARBON_ENABLE_PYTHON=ON -DCARBON_BUILD_EXAMPLES=OFF -DCARBON_BUILD_BENCHMARKS=OFF
cmake --build "${BUILD_DIR}" -j

echo "=== install XLB (no deps; use built warp fork) ==="
${PIP} --no-deps -e ./XLB

echo "=== run ==="
export PYTHONPATH=/carbon/warp:/carbon/py
python3 ${PYARGS}
'
