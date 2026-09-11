#!/bin/bash
set -e

# =============================================================================
# Neon Wheel Build Script
# =============================================================================
# Builds a Python wheel containing Neon and bundled Warp.
#
# Usage:
#   ./wheel.sh                    # Build for all common GPU architectures (default)
#   ./wheel.sh --local            # Build only for installed GPU (fastest)
#   ./wheel.sh --arch "80;90"     # Build for specific architectures
#   ./wheel.sh --clean            # Clean build artifacts before building
#   ./wheel.sh --help             # Show this help message
#
# Environment variables:
#   NEON_CUDA_ARCH    Override GPU architectures (e.g., "70;75;80;86;89;90")
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Default settings
BUILD_FOR_ALL_GPUS="ON"
CUSTOM_ARCH=""
CLEAN_BUILD=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --local)
            BUILD_FOR_ALL_GPUS="OFF"
            shift
            ;;
        --arch)
            CUSTOM_ARCH="$2"
            shift 2
            ;;
        --clean)
            CLEAN_BUILD=true
            shift
            ;;
        --help|-h)
            head -20 "$0" | tail -15
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Override with environment variable if set
if [[ -n "$NEON_CUDA_ARCH" ]]; then
    CUSTOM_ARCH="$NEON_CUDA_ARCH"
fi

# Clean build artifacts if requested
if $CLEAN_BUILD; then
    echo "==> Cleaning build artifacts..."
    rm -rf build/ dist/ *.egg-info/
fi

# Step 1: Initialize submodules (Warp)
echo "==> Initializing submodules..."
git submodule update --init --recursive

# Check if Warp submodule exists
if [[ ! -d "extern/warp" ]] || [[ -z "$(ls -A extern/warp 2>/dev/null)" ]]; then
    echo ""
    echo "ERROR: Warp submodule not found or empty at extern/warp"
    echo ""
    echo "To set up the Warp submodule, run:"
    echo "  git submodule add -b external-source-support-update4 https://github.com/massimim/warp.git extern/warp"
    echo "  git submodule update --init --recursive"
    echo ""
    exit 1
fi

# Step 2: Build Warp's native libraries
echo "==> Building Warp native libraries..."
cd extern/warp
pip install numpy --quiet
python build_lib.py
cd "$SCRIPT_DIR"

# Step 3: Install build dependencies
echo "==> Installing build dependencies..."
pip install build scikit-build-core --quiet

# Step 4: Build the wheel with appropriate GPU architecture settings
echo "==> Building Neon wheel..."

# Set flag to skip library loading during wheel build
export _BUILDING_NEON_WHEEL=1

# Use all CPU cores for CMake/Ninja (override with CMAKE_BUILD_PARALLEL_LEVEL if set)
export CMAKE_BUILD_PARALLEL_LEVEL="${CMAKE_BUILD_PARALLEL_LEVEL:-$(nproc)}"

if [[ -n "$CUSTOM_ARCH" ]]; then
    echo "    GPU architectures: $CUSTOM_ARCH"
    echo "    INFO logging default: OFF (can be enabled at runtime)"
    python -m build --wheel \
        --config-setting=cmake.define.NEON_BUILD_FOR_ALL_GPUS=OFF \
        --config-setting=cmake.define.CMAKE_CUDA_ARCHITECTURES="$CUSTOM_ARCH" \
        --config-setting=cmake.define.NEON_INFO_DEFAULT_OFF=ON
elif [[ "$BUILD_FOR_ALL_GPUS" == "ON" ]]; then
    echo "    GPU architectures: all common (70, 75, 80, 86, 89, 90)"
    echo "    INFO logging default: OFF (can be enabled at runtime)"
    python -m build --wheel \
        --config-setting=cmake.define.NEON_BUILD_FOR_ALL_GPUS=ON \
        --config-setting=cmake.define.NEON_INFO_DEFAULT_OFF=ON
else
    echo "    GPU architectures: auto-detect installed GPU"
    echo "    INFO logging default: OFF (can be enabled at runtime)"
    python -m build --wheel \
        --config-setting=cmake.define.NEON_BUILD_FOR_ALL_GPUS=OFF \
        --config-setting=cmake.define.NEON_INFO_DEFAULT_OFF=ON
fi

# Report success
echo ""
echo "==> Build complete!"
echo "    Wheel location: $(ls -1 dist/*.whl 2>/dev/null | tail -1)"
echo ""
echo "    Install with: pip install dist/*.whl"
