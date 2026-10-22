#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
WARP_BRANCH="${WARP_BRANCH:-external-source-support-update2}"

cd "${ROOT}"
git submodule update --init --recursive warp
git -C warp checkout "${WARP_BRANCH}"

cd warp
python3 build_lib.py
python3 -m pip install -e .

echo "Warp submodule built and installed (branch ${WARP_BRANCH})"
