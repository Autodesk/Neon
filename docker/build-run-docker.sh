#!/bin/bash

# Script to build and run Docker container for Warp and Neon wheel building
# Usage: ./build-run-docker.sh [cuda-version]
#   cuda-version: "12.8" (default) or "13.1"
#
# Note: This script should be run from the docker/ directory
# It will mount the parent directory (neon root) as /workspace in the container

set -e  # Exit on error

# Enable BuildKit only if buildx is available (required for BuildKit)
# Otherwise use legacy builder so the build still works
if docker buildx version >/dev/null 2>&1; then
    export DOCKER_BUILDKIT=1
else
    export DOCKER_BUILDKIT=0
    echo "Note: Docker buildx not found; using legacy builder (slower). Install buildx for faster builds: https://docs.docker.com/go/buildx/"
fi

# Check if Docker is accessible
if ! docker info >/dev/null 2>&1; then
    echo "Error: Cannot connect to Docker daemon."
    echo ""
    echo "Possible solutions:"
    echo "1. Add your user to the docker group:"
    echo "   sudo usermod -aG docker \$USER"
    echo "   Then log out and log back in."
    echo ""
    echo "2. Start Docker service:"
    echo "   sudo systemctl start docker"
    echo ""
    echo "3. Check Docker is running:"
    echo "   sudo systemctl status docker"
    echo ""
    echo "See docker/TROUBLESHOOTING.md for more details."
    exit 1
fi

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Get the parent directory (neon root) to mount as workspace
NEON_ROOT="$(dirname "$SCRIPT_DIR")"

# Default CUDA version
CUDA_VERSION="${1:-12.8}"

# Validate CUDA version
if [[ "$CUDA_VERSION" != "12.8" && "$CUDA_VERSION" != "13.1" ]]; then
    echo "Error: CUDA version must be '12.8' or '13.1'"
    echo "Usage: $0 [12.8|13.1]"
    exit 1
fi

# Set Dockerfile and image tag based on CUDA version
if [[ "$CUDA_VERSION" == "12.8" ]]; then
    DOCKERFILE="Dockerfile.wheel-builder"
    IMAGE_TAG="neon-warp-builder:12.8"
else
    DOCKERFILE="Dockerfile.wheel-builder.cuda13"
    IMAGE_TAG="neon-warp-builder:13.1"
fi

# Check if Dockerfile exists
if [[ ! -f "$SCRIPT_DIR/$DOCKERFILE" ]]; then
    echo "Error: Dockerfile '$DOCKERFILE' not found in $SCRIPT_DIR"
    exit 1
fi

echo "=========================================="
echo "Building Docker image for CUDA $CUDA_VERSION"
echo "=========================================="
echo "Dockerfile: $DOCKERFILE"
echo "Image tag: $IMAGE_TAG"
echo "Neon root: $NEON_ROOT"
echo ""

# Change to script directory for docker build context
cd "$SCRIPT_DIR"

# Build the Docker image
echo "Building Docker image..."
docker build -f "$DOCKERFILE" -t "$IMAGE_TAG" .

if [[ $? -eq 0 ]]; then
    echo ""
    echo "✓ Docker image built successfully!"
    echo ""
    echo "=========================================="
    echo "Running Docker container"
    echo "=========================================="
    echo "Mounting $NEON_ROOT as /workspace"
    echo ""
    
    # Run the Docker container
    # Mount the neon root directory as /workspace
    docker run --gpus all -it --rm \
        -v "$NEON_ROOT:/workspace" \
        "$IMAGE_TAG"
else
    echo ""
    echo "✗ Docker image build failed!"
    exit 1
fi

