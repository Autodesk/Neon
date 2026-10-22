# syntax=docker/dockerfile:1
#
# Native HIP (ROCm) build image for carbon.
#
# Covers two carbon configurations:
#   - "hip"      : CARBON_BACKEND=Hip, CARBON_ENABLE_MPI=OFF
#   - "hip-mpi"  : CARBON_BACKEND=Hip, CARBON_ENABLE_MPI=ON   (RCCL transport)
#
# This is the AMD analogue of Dockerfile.cuda: the device .cu TUs are compiled
# by ROCm's HIP toolchain (amdclang++/hipcc) via CMake's HIP language, the host
# .cpp sources stay on the distro g++, and the HIP runtime + hipcub + RCCL all
# come from the rocm/dev `-complete` base at /opt/rocm. Unlike Dockerfile.sycl-rccl
# there is no AdaptiveCpp / conda layer — carbon talks to HIP directly.

FROM rocm/dev-ubuntu-22.04:6.4.4-complete

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
 && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    ninja-build \
    ca-certificates \
    curl \
    libomp-dev \
    openmpi-bin libopenmpi-dev \
    python3 \
 && rm -rf /var/lib/apt/lists/*

# Ubuntu 22.04 apt ships cmake 3.22; carbon's CMAKE_HIP_STANDARD 20 + the HIP
# language support carbon relies on want cmake >= 3.25. `BinPackParameters:
# AlwaysOnePerLine` in .clang-format needs clang-format >= 20.
#
# Both come from upstream rather than pip. pip is avoided deliberately: on a network that
# intercepts TLS to files.pythonhosted.org, every pip layer fails and cannot be repaired by
# pointing pip at a different trust store (see docs/CHANGELOG.md 2026-07). github.com and
# apt.llvm.org serve their genuine certificates, so these two sources verify normally.
ARG CMAKE_VERSION=3.31.6
RUN curl -fsSL "https://github.com/Kitware/CMake/releases/download/v${CMAKE_VERSION}/cmake-${CMAKE_VERSION}-linux-x86_64.tar.gz" \
      | tar xz --strip-components=1 -C /usr/local

# The key is ASCII-armored, which apt >= 2.4 accepts directly as a signed-by target, so no
# gnupg in the image just to dearmor it.
RUN curl -fsSL https://apt.llvm.org/llvm-snapshot.gpg.key \
      -o /usr/share/keyrings/llvm-archive-keyring.asc \
 && echo "deb [signed-by=/usr/share/keyrings/llvm-archive-keyring.asc] http://apt.llvm.org/jammy/ llvm-toolchain-jammy-20 main" \
      > /etc/apt/sources.list.d/llvm-20.list \
 && apt-get update \
 && apt-get install -y --no-install-recommends clang-format-20 \
 && rm -rf /var/lib/apt/lists/* \
 && ln -sf /usr/bin/clang-format-20 /usr/local/bin/clang-format

# /opt/rocm/bin carries hipcc / amdclang++ (the HIP device compiler CMake's HIP
# language binds) plus hipconfig; put it ahead of the distro paths so
# enable_language(HIP) + find_package(hip) resolve against the system ROCm.
ENV ROCM_PATH=/opt/rocm
ENV PATH=/opt/rocm/bin:${PATH}
ENV LD_LIBRARY_PATH=/opt/rocm/lib:${LD_LIBRARY_PATH:-}

# The dist_* tests invoke `mpiexec` inside the container. OpenMPI refuses to run
# as root by default; containers run in an isolated namespace so the usual
# rationale doesn't apply. Set both vars so the dist_* ctest invocations proceed.
ENV OMPI_ALLOW_RUN_AS_ROOT=1 \
    OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1

WORKDIR /work
