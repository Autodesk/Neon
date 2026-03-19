#!/bin/bash
# Build and run the multi-Python wheel builder container (Python 3.11–3.14).
# Run from the docker/ directory. Neon root is mounted at /workspace.
# Inside the container, run: ./docker/build-wheels-multi.sh

set -e

if docker buildx version >/dev/null 2>&1; then
    export DOCKER_BUILDKIT=1
else
    export DOCKER_BUILDKIT=0
    echo "Note: Docker buildx not found; using legacy builder."
fi

if ! docker info >/dev/null 2>&1; then
    echo "Error: Cannot connect to Docker daemon. See docker/TROUBLESHOOTING.md"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NEON_ROOT="$(dirname "$SCRIPT_DIR")"
DOCKERFILE="Dockerfile.wheel-builder.multi"
IMAGE_TAG="neon-warp-builder:multi"

if [[ ! -f "$SCRIPT_DIR/$DOCKERFILE" ]]; then
    echo "Error: $DOCKERFILE not found in $SCRIPT_DIR"
    exit 1
fi

echo "=========================================="
echo "Multi-Python wheel builder (3.11–3.14)"
echo "=========================================="
echo "Image tag: $IMAGE_TAG"
echo "Neon root: $NEON_ROOT"
echo ""

cd "$SCRIPT_DIR"
echo "Building Docker image..."
docker build -f "$DOCKERFILE" -t "$IMAGE_TAG" .

echo ""
echo "✓ Image built. Running container (mounting $NEON_ROOT as /workspace)..."
echo "  Inside container, run: ./docker/build-wheels-multi.sh"
echo ""

docker run --gpus all -it --rm \
    -v "$NEON_ROOT:/workspace" \
    "$IMAGE_TAG"
