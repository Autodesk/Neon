#!/bin/bash
# Build and run the multi-Python wheel builder container (Python 3.11–3.14).
# Supports both x86_64 and aarch64 (Jetson Orin / ARM) hosts.
#
# Run from anywhere; paths are resolved automatically.
# Inside the container, run: ./docker/build-wheels-multi.sh
#
# On Jetson (aarch64):
#   - Uses CUDA 12.6 base image (matching JetPack 6.x / driver 540.x)
#   - Requires nvidia-container-toolkit for GPU access in Docker
#   - Uses --runtime nvidia (most reliable on Jetson)
#
# On x86_64:
#   - Uses CUDA 12.8 base image (default)
#   - Uses --gpus all

set -e

if ! docker info >/dev/null 2>&1; then
    echo "Error: Cannot connect to Docker daemon."
    echo "  - Is Docker running?   systemctl status docker"
    echo "  - Are you in the docker group?   sudo usermod -aG docker \$USER  (then log out/in)"
    echo "  See docker/TROUBLESHOOTING.md"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NEON_ROOT="$(dirname "$SCRIPT_DIR")"
DOCKERFILE="Dockerfile.wheel-builder.multi"
IMAGE_TAG="neon-warp-builder:multi"
HOST_ARCH="$(uname -m)"
HOST_UID="$(id -u)"
HOST_GID="$(id -g)"
HOST_USER="${USER:-host}"

# When Neon is checked out as a git submodule, .git is a file pointing at the
# parent repo's .git/modules/Neon. Mount the parent so git submodule works.
MOUNT_ROOT="$NEON_ROOT"
CONTAINER_WORKDIR="/workspace"
if [[ -f "$NEON_ROOT/.git" ]] && grep -q '^gitdir:' "$NEON_ROOT/.git"; then
    MOUNT_ROOT="$(dirname "$NEON_ROOT")"
    CONTAINER_WORKDIR="/workspace/$(basename "$NEON_ROOT")"
fi

if [[ ! -f "$SCRIPT_DIR/$DOCKERFILE" ]]; then
    echo "Error: $DOCKERFILE not found in $SCRIPT_DIR"
    exit 1
fi

# ---------------------------------------------------------------------------
# Platform-specific configuration
# ---------------------------------------------------------------------------
case "$HOST_ARCH" in
    aarch64)
        # Jetson Orin (JetPack 6.x): driver 540.x supports CUDA <= 12.6
        BASE_IMAGE="nvidia/cuda:12.6.3-devel-ubuntu22.04"
        PLATFORM_LABEL="aarch64 (Jetson / ARM)"
        GPU_FLAG="--runtime nvidia"
        ;;
    x86_64)
        BASE_IMAGE="nvidia/cuda:12.8.1-devel-ubuntu22.04"
        PLATFORM_LABEL="x86_64"
        GPU_FLAG="--gpus all"
        ;;
    *)
        echo "Error: Unsupported host architecture: $HOST_ARCH"
        exit 1
        ;;
esac

# On aarch64, verify the NVIDIA container runtime is available
if [[ "$HOST_ARCH" == "aarch64" ]]; then
    if ! docker info 2>/dev/null | grep -q "nvidia"; then
        echo "=========================================="
        echo "WARNING: NVIDIA runtime not detected in Docker."
        echo ""
        echo "GPU access inside the container requires nvidia-container-toolkit."
        echo "Install it with:"
        echo "  sudo apt-get install -y nvidia-container-toolkit"
        echo "  sudo nvidia-ctk runtime configure --runtime=docker"
        echo "  sudo systemctl restart docker"
        echo "=========================================="
        echo ""
        echo "Continuing anyway (the build will fail if GPU access is needed)..."
        echo ""
    fi
fi

echo "=========================================="
echo "Multi-Python wheel builder (3.11–3.14)"
echo "=========================================="
echo "  Platform : $PLATFORM_LABEL"
echo "  CUDA base: $BASE_IMAGE"
echo "  Image tag: $IMAGE_TAG"
echo "  Neon root: $NEON_ROOT"
echo "  Mount    : $MOUNT_ROOT -> /workspace"
echo "  Workdir  : $CONTAINER_WORKDIR"
echo "  Run as   : ${HOST_USER} (uid=${HOST_UID} gid=${HOST_GID})"
echo ""

cd "$SCRIPT_DIR"
echo "Building Docker image..."
docker build -f "$DOCKERFILE" --build-arg BASE_IMAGE="$BASE_IMAGE" -t "$IMAGE_TAG" .

echo ""
echo "✓ Image built. Running container..."
echo "  Inside container, run: ./docker/build-wheels-multi.sh"
echo ""

# Start as root briefly to register the host user in /etc/passwd, then drop to that
# user so bind-mounted files are owned by the host user with a normal shell prompt.
docker run $GPU_FLAG -it --rm \
    --user root \
    -e HOST_UID="${HOST_UID}" \
    -e HOST_GID="${HOST_GID}" \
    -e HOST_USER="${HOST_USER}" \
    -e CONTAINER_WORKDIR="${CONTAINER_WORKDIR}" \
    -v "$MOUNT_ROOT:/workspace" \
    -v "$SCRIPT_DIR/entrypoint-host-user.sh:/entrypoint-host-user.sh:ro" \
    --entrypoint /entrypoint-host-user.sh \
    "$IMAGE_TAG"
