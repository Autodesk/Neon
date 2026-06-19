#!/usr/bin/env bash
# Build Neon wheels for Python 3.11, 3.12, 3.13, and 3.14.
# Run this script from the Neon repo root (e.g. /workspace inside the container).
#
# Usage:
#   ./docker/build-wheels-multi.sh [--clean] [--local] [--python VERSION]
#
# Options:
#   --clean            Remove build/ and dist/ before building (clean build).
#   --local            Build only for the current GPU arch (faster; auto-detects via CMake).
#   --python VERSION   Build only for one Python version (3.11, 3.12, 3.13, or 3.14).
#   -p VERSION         Short form of --python.
#
# Environment:
#   NEON_CUDA_ARCH   Override GPU architectures (e.g. "80;87;90")
#
# GPU architectures (when not using --local):
#   x86_64:  70 75 80 86 89 90  (Volta through Hopper)
#   aarch64: 72 87              (Jetson Xavier, Jetson Orin)
#
# Requires: multi-Python Docker image (Dockerfile.wheel-builder.multi).
# Example (from host, neon repo root):
#   cd docker && ./build-run-docker-multi.sh
#   # inside container:
#   ./docker/build-wheels-multi.sh
#   ./docker/build-wheels-multi.sh --clean --local --python 3.11

set -e
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_PATH="$SCRIPT_DIR/$(basename "${BASH_SOURCE[0]}")"
NEON_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$NEON_ROOT"

python_wheel_tag() {
    local version="$1"
    echo "cp${version//./}"
}

report_build_failure() {
    local py="$1"
    local log_file="$2"
    echo ""
    echo "=========================================="
    echo "BUILD FAILED for Python ${py}"
    echo "=========================================="
    if [[ -f "$log_file" ]]; then
        echo "Full log: $log_file"
        echo ""
        echo "Last matching error lines:"
        grep -iE 'error:|fatal error|FAILED:|killed|ninja: build stopped|CMake Error' "$log_file" | tail -40 || true
    fi
    exit 1
}

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

# Defaults
CLEAN=false
USE_ALL_ARCHS=true
GPU_ARCHS="${NEON_CUDA_ARCH:-$DEFAULT_GPU_ARCHS}"
SELECTED_PY=""

SUPPORTED_PYVERSIONS=(3.11 3.12 3.13 3.14)

is_supported_python() {
    local version="$1"
    local py
    for py in "${SUPPORTED_PYVERSIONS[@]}"; do
        if [[ "$py" == "$version" ]]; then
            return 0
        fi
    done
    return 1
}

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
        --python|-p)
            if [[ -z "${2:-}" ]]; then
                echo "ERROR: $1 requires a version (e.g. 3.11)"
                exit 1
            fi
            SELECTED_PY="$2"
            shift 2
            ;;
        --help|-h)
            head -30 "$SCRIPT_PATH" | tail -26
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

if [[ -n "$SELECTED_PY" ]]; then
    if ! is_supported_python "$SELECTED_PY"; then
        echo "ERROR: Unsupported Python version: $SELECTED_PY"
        echo "       Supported versions: ${SUPPORTED_PYVERSIONS[*]}"
        exit 1
    fi
    PYVERSIONS=("$SELECTED_PY")
else
    PYVERSIONS=("${SUPPORTED_PYVERSIONS[@]}")
fi

export _BUILDING_NEON_WHEEL=1

# On aarch64 (Jetson), CUDA compilation is very memory-intensive and the
# OOM killer will terminate nvcc if too many compile jobs run in parallel.
# Default to 2 parallel jobs on ARM (8 GB shared memory), full parallelism on x86.
if [[ -z "$CMAKE_BUILD_PARALLEL_LEVEL" ]]; then
    if [[ "$HOST_ARCH" == "aarch64" ]]; then
        export CMAKE_BUILD_PARALLEL_LEVEL=2
    else
        export CMAKE_BUILD_PARALLEL_LEVEL="$(nproc)"
    fi
fi

DIST_MULTI="$NEON_ROOT/dist-multi"
mkdir -p "$DIST_MULTI"
# Start with empty list; we'll only keep the last dist/ per version
rm -rf "$NEON_ROOT/dist"

# Check how many Pythons are available
AVAILABLE=()
MISSING=()
for py in "${PYVERSIONS[@]}"; do
    if command -v "python${py}" &>/dev/null; then
        AVAILABLE+=("$py")
    else
        MISSING+=("$py")
    fi
done

if [[ -n "$SELECTED_PY" ]] && ! command -v "python${SELECTED_PY}" &>/dev/null; then
    echo "ERROR: python${SELECTED_PY} is not installed in this container."
    exit 1
fi

if [[ ${#MISSING[@]} -gt 0 ]]; then
    echo "=========================================="
    echo "NOTE: Not all Python versions are in this container."
    echo "      Available: ${AVAILABLE[*]:-none}"
    echo "      Missing: ${MISSING[*]}"
    echo ""
    echo "To build wheels for all of 3.11–3.14, use the multi-Python image:"
    echo "  1. Exit this container (exit)"
    echo "  2. From host, in docker/:  ./build-run-docker-multi.sh"
    echo "  3. Inside the new container:  ./docker/build-wheels-multi.sh"
    echo "=========================================="
    echo ""
fi

echo "=========================================="
echo "Neon multi-Python wheel build"
echo "=========================================="
echo "Host architecture: $HOST_ARCH"
if $USE_ALL_ARCHS; then
    echo "GPU architectures: $GPU_ARCHS"
else
    echo "GPU architectures: auto-detect (--local)"
fi
echo "Python versions to try: ${PYVERSIONS[*]}"
echo "Output directory: $DIST_MULTI"
echo ""

if $CLEAN; then
    echo "==> Cleaning build artifacts..."
    rm -rf "$NEON_ROOT/build" "$NEON_ROOT/dist"
    if [[ -n "$SELECTED_PY" ]]; then
        wheel_tag="$(python_wheel_tag "$SELECTED_PY")"
        echo "==> Removing existing wheels for ${wheel_tag} in dist-multi/"
        rm -f "$DIST_MULTI"/*-"${wheel_tag}"-*.whl
    else
        rm -f "$DIST_MULTI"/*.whl
    fi
    mkdir -p "$DIST_MULTI"
fi

echo "==> Initializing submodules..."
git submodule update --init --recursive

if [[ ! -d "extern/warp" ]] || [[ -z "$(ls -A extern/warp 2>/dev/null)" ]]; then
    echo "ERROR: Warp submodule missing or empty at extern/warp"
    exit 1
fi

for py in "${PYVERSIONS[@]}"; do
    if ! command -v "python${py}" &>/dev/null; then
        echo "==> Skipping Python ${py} (not installed)"
        continue
    fi
    echo ""
    echo "=========================================="
    echo "Building for Python ${py}"
    echo "=========================================="
    rm -rf "$NEON_ROOT/build" "$NEON_ROOT/dist"
    mkdir -p "$NEON_ROOT/build"

    echo "==> Building Warp native libs (python${py})..."
    (cd extern/warp && "python${py}" -m pip install numpy --quiet && "python${py}" build_lib.py)
    echo "==> Installing build deps (python${py})..."
    "python${py}" -m pip install build scikit-build-core --quiet
    echo "==> Building wheel (python${py})..."
    LOG_FILE="$NEON_ROOT/build-wheel-py${py//./}.log"
    if $USE_ALL_ARCHS; then
        "python${py}" -m build --wheel \
            "--config-setting=cmake.define.CMAKE_CUDA_ARCHITECTURES=$GPU_ARCHS" \
            --config-setting=cmake.define.NEON_BUILD_FOR_ALL_GPUS=OFF \
            --config-setting=cmake.define.NEON_BUILD_ONLY_FOR_INSTALLED_GPU=OFF \
            --config-setting=cmake.define.NEON_INFO_DEFAULT_OFF=ON \
            2>&1 | tee "$LOG_FILE" || report_build_failure "$py" "$LOG_FILE"
    else
        "python${py}" -m build --wheel \
            --config-setting=cmake.define.NEON_BUILD_FOR_ALL_GPUS=OFF \
            --config-setting=cmake.define.NEON_INFO_DEFAULT_OFF=ON \
            2>&1 | tee "$LOG_FILE" || report_build_failure "$py" "$LOG_FILE"
    fi
    if [[ -d "$NEON_ROOT/dist" ]]; then
        cp -v "$NEON_ROOT"/dist/*.whl "$DIST_MULTI/"
    fi
done

echo ""
echo "=========================================="
echo "Done. Wheels in: $DIST_MULTI"
echo "=========================================="
ls -la "$DIST_MULTI"/*.whl 2>/dev/null || echo "No wheels produced."
