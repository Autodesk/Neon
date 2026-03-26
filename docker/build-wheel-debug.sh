#!/usr/bin/env bash
# Build a single Neon debug wheel for one Python version.
# Run this script from the Neon repo root (e.g. /workspace inside the container).
#
# Usage:
#   ./docker/build-wheel-debug.sh [--clean] [--local] [--python 3.12] [--source-dir /host/path]
#
# Options:
#   --clean              Remove build/ and dist/ before building.
#   --local              Build only for the current GPU arch (faster; auto-detects via CMake).
#   --python VER         Python version to use (default: 3.12).
#   --source-dir PATH    Host path to the neon repo. Remaps DWARF source paths so GDB
#                        on the host can find source files even though the build ran in
#                        Docker. Example: --source-dir /home/max/repos/neon
#
# Environment:
#   NEON_CUDA_ARCH   Override GPU architectures (e.g. "87")
#
# GPU architectures (when not using --local):
#   x86_64:  70 75 80 86 89 90  (Volta through Hopper)
#   aarch64: 72 87              (Jetson Xavier, Jetson Orin)
#
# Requires: Docker image with the chosen Python version installed.
# Example (from host, neon repo root):
#   cd docker && ./build-run-docker-multi.sh
#   # inside container:
#   ./docker/build-wheel-debug.sh --local --python 3.12 --source-dir /home/max/repos/neon

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_PATH="$SCRIPT_DIR/$(basename "${BASH_SOURCE[0]}")"
NEON_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$NEON_ROOT"

# ---------------------------------------------------------------------------
# GPU architecture lists (by host CPU)
# ---------------------------------------------------------------------------
GPU_ARCHS_X86="70;75;80;86;89;90"
GPU_ARCHS_ARM="72;87"

HOST_ARCH="$(uname -m)"
if [[ "$HOST_ARCH" == "aarch64" ]]; then
    DEFAULT_GPU_ARCHS="$GPU_ARCHS_ARM"
else
    DEFAULT_GPU_ARCHS="$GPU_ARCHS_X86"
fi

CLEAN=false
USE_ALL_ARCHS=true
GPU_ARCHS="${NEON_CUDA_ARCH:-$DEFAULT_GPU_ARCHS}"
PYVER="3.12"
HOST_SOURCE_DIR=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --clean)
            CLEAN=true
            shift
            ;;
        --local)
            USE_ALL_ARCHS=false
            shift
            ;;
        --python)
            PYVER="$2"
            shift 2
            ;;
        --source-dir)
            HOST_SOURCE_DIR="$2"
            shift 2
            ;;
        --help|-h)
            head -28 "$SCRIPT_PATH" | tail -24
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

PYTHON="python${PYVER}"

if ! command -v "$PYTHON" &>/dev/null; then
    echo "ERROR: $PYTHON is not installed in this environment."
    exit 1
fi

export _BUILDING_NEON_WHEEL=1

if [[ -z "$CMAKE_BUILD_PARALLEL_LEVEL" ]]; then
    if [[ "$HOST_ARCH" == "aarch64" ]]; then
        export CMAKE_BUILD_PARALLEL_LEVEL=2
    else
        export CMAKE_BUILD_PARALLEL_LEVEL="$(nproc)"
    fi
fi

echo "=========================================="
echo "Neon DEBUG wheel build"
echo "=========================================="
echo "Host architecture: $HOST_ARCH"
if $USE_ALL_ARCHS; then
    echo "GPU architectures: $GPU_ARCHS"
else
    echo "GPU architectures: auto-detect (--local)"
fi
echo "Python version:    $PYVER ($($PYTHON --version 2>&1))"
echo "Build type:        Debug"
if [[ -n "$HOST_SOURCE_DIR" ]]; then
    echo "Source remap:      $NEON_ROOT -> $HOST_SOURCE_DIR"
fi
echo ""

if $CLEAN; then
    echo "==> Cleaning build artifacts..."
    rm -rf "$NEON_ROOT/build" "$NEON_ROOT/dist"
fi

echo "==> Initializing submodules..."
git submodule update --init --recursive

if [[ ! -d "extern/warp" ]] || [[ -z "$(ls -A extern/warp 2>/dev/null)" ]]; then
    echo "ERROR: Warp submodule missing or empty at extern/warp"
    exit 1
fi

rm -rf "$NEON_ROOT/build" "$NEON_ROOT/dist"
mkdir -p "$NEON_ROOT/build"

echo "==> Building Warp native libs ($PYTHON)..."
(cd extern/warp && "$PYTHON" -m pip install numpy --quiet && "$PYTHON" build_lib.py)

echo "==> Installing build deps ($PYTHON)..."
"$PYTHON" -m pip install build scikit-build-core --quiet

PREFIX_MAP_FLAG=()
if [[ -n "$HOST_SOURCE_DIR" ]]; then
    PREFIX_MAP_FLAG=("--config-setting=cmake.define.NEON_DEBUG_PREFIX_MAP=$NEON_ROOT=$HOST_SOURCE_DIR")
fi

echo "==> Building DEBUG wheel ($PYTHON)..."
if $USE_ALL_ARCHS; then
    "$PYTHON" -m build --wheel \
        "--config-setting=cmake.define.CMAKE_CUDA_ARCHITECTURES=$GPU_ARCHS" \
        --config-setting=cmake.build-type=Debug \
        --config-setting=cmake.define.CMAKE_BUILD_TYPE=Debug \
        --config-setting=install.strip=false \
        --config-setting=cmake.define.NEON_BUILD_FOR_ALL_GPUS=OFF \
        --config-setting=cmake.define.NEON_BUILD_ONLY_FOR_INSTALLED_GPU=OFF \
        --config-setting=cmake.define.NEON_INFO_DEFAULT_OFF=ON \
        "${PREFIX_MAP_FLAG[@]}"
else
    "$PYTHON" -m build --wheel \
        --config-setting=cmake.build-type=Debug \
        --config-setting=cmake.define.CMAKE_BUILD_TYPE=Debug \
        --config-setting=install.strip=false \
        --config-setting=cmake.define.NEON_BUILD_FOR_ALL_GPUS=OFF \
        --config-setting=cmake.define.NEON_INFO_DEFAULT_OFF=ON \
        "${PREFIX_MAP_FLAG[@]}"
fi

# ---------------------------------------------------------------------------
# Verify debug symbols survived packaging
# ---------------------------------------------------------------------------
echo ""
echo "==> Verifying debug symbols in wheel..."
WHEEL_OK=true
TMPDIR_VFY="$(mktemp -d)"
trap 'rm -rf "$TMPDIR_VFY"' EXIT

for whl in "$NEON_ROOT"/dist/*.whl; do
    "$PYTHON" -c "import zipfile,sys; zipfile.ZipFile(sys.argv[1]).extractall(sys.argv[2])" "$whl" "$TMPDIR_VFY"
done

FOUND_SO=false
for so in "$TMPDIR_VFY"/neon/*.so "$TMPDIR_VFY"/neon/**/*.so; do
    [[ -f "$so" ]] || continue
    FOUND_SO=true
    if file "$so" | grep -q "not stripped"; then
        echo "  OK  $(basename "$so") — has debug symbols"
    elif readelf -S "$so" 2>/dev/null | grep -q '\.debug_info'; then
        echo "  OK  $(basename "$so") — has .debug_info section"
    else
        echo "  FAIL $(basename "$so") — debug symbols MISSING (stripped?)"
        WHEEL_OK=false
    fi
done

if ! $FOUND_SO; then
    echo "  WARNING: no .so files found in wheel to verify"
fi

echo ""
echo "=========================================="
if $WHEEL_OK && $FOUND_SO; then
    echo "Done. Debug wheel(s) with symbols in: $NEON_ROOT/dist"
else
    echo "Done. Wheel(s) in: $NEON_ROOT/dist"
    echo "WARNING: Some libraries may be missing debug symbols!"
    echo "  Ensure scikit-build-core >=0.10.7 is installed,"
    echo "  or set install.strip=false in pyproject.toml."
fi
echo "=========================================="
ls -la "$NEON_ROOT"/dist/*.whl 2>/dev/null || echo "No wheel produced."
