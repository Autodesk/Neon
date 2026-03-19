# Docker Container for Building Warp and Neon Wheels

This document describes how to run the wheel-builder containers and build wheels for one or many Python versions.

---

## How to run the container

### 1. Build the image and start an interactive shell

From the **neon repo root** (parent of `docker/`):

```bash
cd docker
./build-run-docker.sh
```

Or for CUDA 13.1:

```bash
./build-run-docker.sh 13.1
```

This will:

- Build the Docker image (only the first time or when the Dockerfile changes).
- Start a container with your **neon repo mounted at `/workspace`**.
- Give you an interactive shell as user **builder** (passwordless `sudo` available).

You are then inside the container at `/workspace` (the neon root). Any change you make under `/workspace` is reflected on your host repo.

### 2. Run a single command instead of a shell

To run one command and exit (no interactive shell):

```bash
cd docker
docker run --gpus all --rm -v "$(dirname $(pwd)):/workspace" neon-warp-builder:12.8 \
  bash -c "cd /workspace && ./wheel.sh"
```

Replace `neon-warp-builder:12.8` with `neon-warp-builder:13.1` if you use the CUDA 13.1 image.

### 3. Manual build and run (without the helper script)

```bash
# From the docker/ directory
docker build -f Dockerfile.wheel-builder -t neon-warp-builder:12.8 .

# Run with neon root mounted at /workspace
docker run --gpus all -it --rm \
  -v "/path/to/neon:/workspace" \
  neon-warp-builder:12.8
```

- **`--gpus all`** – needed for GPU access inside the container.
- **`-v host_path:/workspace`** – mount your neon repo so the container can read/write it.
- **`-it`** – interactive terminal; drop `-it` and add a command for one-off runs.
- **`--rm`** – remove the container when it exits.

### 4. Exiting the container

Type `exit` or press Ctrl+D to leave the container. The container is removed; your repo on the host is unchanged except for any edits or built artifacts (e.g. under `build/`, `dist/`).

---

## Single-Python image (CUDA 12.8 or 13.1)

**Primary recommendation:** Use `Dockerfile.wheel-builder` (CUDA 12.8) for one Python version (3.10).

### Why CUDA 12.8?

1. **Better Compatibility**: CUDA 12.8 has fewer namespace and API compatibility issues compared to CUDA 13.x
2. **Proven Stability**: Many CUDA projects work reliably with CUDA 12.8
3. **Avoids Workarounds**: CUDA 13.x requires workarounds for namespace issues (like the `::cuda` namespace error we fixed)

### Quick Start (single Python)

```bash
cd docker
./build-run-docker.sh          # CUDA 12.8
# or
./build-run-docker.sh 13.1     # CUDA 13.1
```

You will be logged in as user **builder** (not root) with passwordless `sudo`.

### Faster builds (BuildKit)

For faster image builds, install the Docker buildx plugin so the script can use BuildKit:

- **Manjaro / Arch:** `sudo pacman -S docker-buildx`
- **Debian / Ubuntu:** `sudo apt install docker-buildx-plugin`
- **Other:** See [Docker buildx install](https://docs.docker.com/go/buildx/)

Then run `./build-run-docker.sh` as usual; it will automatically use BuildKit when available.

### Building a single wheel inside the container

Once inside the container (you’re at `/workspace`):

```bash
./wheel.sh              # build for all common GPU archs
./wheel.sh --local      # build only for current GPU (faster)
./wheel.sh --clean      # clean then build
```

Or build step by step:

```bash
git submodule update --init --recursive
cd extern/warp && pip install numpy && python build_lib.py && cd ../..
pip install build scikit-build-core
python -m build --wheel
```

Wheels end up in `dist/`.

---

## Building wheels for multiple Python versions (3.11–3.14)

Use the **multi-Python** image and script to build one wheel per Python version (cp311–cp314) in one go.

### 1. Build and run the multi-Python container

From the **docker/** directory:

```bash
./build-run-docker-multi.sh
```

This builds the image `neon-warp-builder:multi` (if needed) and starts a container with the neon repo at `/workspace`. The image includes Python 3.11, 3.12, 3.13, and 3.14 (via deadsnakes PPA).

### 2. Run the multi-Python wheel script inside the container

Inside the container (at `/workspace`):

```bash
./docker/build-wheels-multi.sh
```

Options:

- **`./docker/build-wheels-multi.sh --clean`** – clean `build/` and `dist/` before building.
- **`./docker/build-wheels-multi.sh --local`** – build only for the current GPU architecture (faster).

The script will:

1. Initialize submodules (Warp).
2. For each Python 3.11–3.14: build Warp native libs with that Python, then build the Neon wheel.
3. Put all wheels in **`dist-multi/`** (e.g. `dist-multi/neon_gpu-0.5.2a1-cp311-cp311-linux_x86_64.whl`, etc.).

### 3. One-off run from the host (no interactive shell)

To build all wheels without entering the container:

```bash
cd docker
docker run --gpus all --rm -v "$(dirname $(pwd)):/workspace" neon-warp-builder:multi \
  bash -c "./docker/build-wheels-multi.sh"
```

Ensure the image exists first (run `./build-run-docker-multi.sh` once and exit, or build the image manually).

---

## Building Neon (CMake + wheel) manually inside container

If you prefer to run CMake and the wheel build by hand (single Python 3.10):

```bash
# You're already in /workspace (the neon root directory)
cd /workspace
mkdir -p build && cd build
cmake ../
cmake --build . --target libNeonPy -j $(nproc)

# Build Python wheel
cd /workspace
python3.10 -m build --wheel
```

## Alternative: CUDA 13.1 Container

If you specifically need CUDA 13.x features, use `Dockerfile.wheel-builder.cuda13`.

**Note**: This requires the CUDA 13.x workarounds we implemented in:
- `libNeonPy/CMakeLists.txt`
- `libNeonPy/src/Neon/py/dGrid.cu`
- `libNeonPy/src/Neon/py/mGrid.cu`

```bash
# From the docker/ directory
./build-run-docker.sh 13.1
```

## Container Specifications

### Base Image
- **CUDA 12.8**: `nvidia/cuda:12.8.1-devel-ubuntu22.04` (recommended)
- **CUDA 13.1**: `nvidia/cuda:13.1.1-devel-ubuntu22.04` (alternative)

### Included Tools
- **CMake**: 3.22+ (required: 3.19+)
- **GCC/G++**: 11.x (avoid GCC 13.x due to compatibility issues)
- **Python**: 3.10 (matches existing wheel: `cp310`)
- **CUDA Toolkit**: Full development toolkit
- **Build Tools**: pip, setuptools, wheel, build

### System Requirements
- **C++ Standard**: C++17 (required by Neon)
- **CUDA Version**: 11+ (tested with 12.8 and 13.1)
- **OpenMP**: Included for multi-threading support

## Building Warp from Source

If you need to build Warp from source in the same container:

```bash
# Inside the container
cd /workspace
git clone https://github.com/NVIDIA/warp.git
cd warp
python3.10 -m pip install -e .
```

## Compatibility Notes

1. **GCC Version**: GCC 11.x is used instead of the default GCC 13.x to avoid CMake feature detection issues with CUDA
2. **Python Version**: Python 3.10 matches the existing wheel naming convention (`cp310`)
3. **CUDA Version**: CUDA 12.8 is recommended for maximum compatibility, but 13.1 works with the implemented workarounds

## Troubleshooting

### Docker Permission Issues

If you get "permission denied" errors when running Docker commands, see `TROUBLESHOOTING.md` for detailed solutions.

**Quick fix:**
```bash
sudo usermod -aG docker $USER
# Then log out and log back in
```

### If you encounter namespace errors:
- Ensure you're using CUDA 12.8, or
- Verify the CUDA 13.x workarounds are in place

### If CMake fails:
- Check that GCC 11 is set as default: `gcc --version`
- Verify CUDA is detected: `nvcc --version`

### If Python wheel build fails:
- Ensure all dependencies are installed: `pip list`
- Check that `libNeonPy` was built successfully

### If the script can't find Dockerfiles:
- Make sure you're running the script from the `docker/` directory
- Verify the Dockerfiles are in the same directory as the script

For more detailed troubleshooting, see `TROUBLESHOOTING.md`.

