#!/usr/bin/env bash
#
# benchmarks/lbm_lid_driven_cavity/scripts/lbm-dgx-a100-40gb-run.sh
#
# Reproduce the Neon-Fig.7 domain-size sweep on an 8x A100-SXM4-40GB (NVLink)
# DGX node. Builds the two Release trees inside Docker and runs the full
# 8-GPU parallel-efficiency matrix for both grid families and both transports:
#
#   Peeraccess (intra-process NVLink P2P)  -> --grid {dense,disg}, no MPI
#   NCCL       (MPI+NCCL, one rank/GPU)    -> --grid {dense,disg}, mpiexec -n 8
#   MPI (P2P)  (CUDA-aware MPI, no NCCL)   -> --grid {dense,disg}, mpiexec -n 8
#                                              with CARBON_DIST_TRANSPORT=mpi
#
# Reference single-GPU (g1) runs are taken with dense only (the plot normalises
# every series to the peak single-GPU dense throughput x 8).
#
# All JSON reports land in data/lbm-dgx-a100-40gb/json/ so the companion
# plotter (lbm-dgx-a100-40gb-plot.py) can render straight from there. Call
# the two back to back:
#
#   ./benchmarks/lbm_lid_driven_cavity/scripts/lbm-dgx-a100-40gb-run.sh
#   python3 benchmarks/lbm_lid_driven_cavity/scripts/lbm-dgx-a100-40gb-plot.py
#
# N=640+ is intentionally excluded: the Q=19 ping-pong population fields do not
# fit a single 40 GB A100 (OOM at N=640; N=608 is the largest size that fits).
# A size that cannot run on one GPU has no 1-GPU baseline, so it is dropped for
# every configuration.
#
# Knobs (env vars, all optional):
#   SIZES="128 160 192 ... 512"           domain edge lengths (step 32)
#   ITERS=2000                            total iterations per run
#   WARMUP=400                            warmup iterations excluded from timing
#   GPUS=8                                multi-GPU count (Peeraccess + NCCL)
#   SM_ARCH=80                            CUDA arch (A100=80, H100=90)
#   IMAGE_INTRA=carbon-cuda:local         image for build + intra-process runs
#   IMAGE_MPI=carbon-cuda-nccl-pinned:local  image for MPI+NCCL runs
#                                         (driver-pinned NCCL; plain carbon-cuda
#                                          fails ncclCommInitRank on this host)
#   SKIP_BUILD=0                          set 1 to reuse existing build trees
#   FORCE=0                               set 1 to re-run cells whose JSON exists

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${repo_root}"

SIZES="${SIZES:-$(seq -s' ' 128 32 608)}"
ITERS="${ITERS:-2000}"
WARMUP="${WARMUP:-400}"
GPUS="${GPUS:-8}"
SM_ARCH="${SM_ARCH:-80}"
IMAGE_INTRA="${IMAGE_INTRA:-carbon-cuda:local}"
IMAGE_MPI="${IMAGE_MPI:-carbon-cuda-nccl-pinned:local}"
SKIP_BUILD="${SKIP_BUILD:-0}"
FORCE="${FORCE:-0}"

PREFIX="lbm-dgx-a100-40gb"
rel_root="benchmarks/lbm_lid_driven_cavity/data/${PREFIX}"
rel_data="${rel_root}/json"
mkdir -p "${repo_root}/${rel_data}"

docker_run() {
  local image="$1" script="$2"
  docker run --rm --gpus all \
    -v "${repo_root}:/src:rw" -w /src \
    "${image}" /bin/bash -lc "${script}"
}

if [[ "${SKIP_BUILD}" -eq 0 ]]; then
  echo "==> building Release trees (SM_${SM_ARCH}) in ${IMAGE_INTRA}"
  docker_run "${IMAGE_INTRA}" "
set -euo pipefail
export PATH=\"/root/.local/bin:\$PATH\"
echo '=== Building CUDA (intra-process) ==='
cmake -S /src -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCARBON_BACKEND=Cuda -DCMAKE_CUDA_ARCHITECTURES=${SM_ARCH}
cmake --build build -j\"\$(nproc)\" --target carbon_benchmark_lbm_lid_driven_cavity
echo '=== Building CUDA+MPI (distributed) ==='
cmake -S /src -B build-mpi -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCARBON_BACKEND=Cuda -DCMAKE_CUDA_ARCHITECTURES=${SM_ARCH} -DCARBON_ENABLE_MPI=ON
cmake --build build-mpi -j\"\$(nproc)\" --target carbon_benchmark_lbm_lid_driven_cavity
echo '=== Build complete ==='
"
else
  echo "==> SKIP_BUILD=1: reusing existing build/ and build-mpi/ trees"
fi

echo "==> Peeraccess sweep (intra-process) in ${IMAGE_INTRA}"
docker_run "${IMAGE_INTRA}" "
set -euo pipefail
INTRA_BIN=\"/src/build/benchmarks/lbm_lid_driven_cavity/carbon_benchmark_lbm_lid_driven_cavity\"
OUT_DIR=\"/src/${rel_data}\"
mkdir -p \"\${OUT_DIR}\"

SIZES=\"${SIZES}\"
GPUS=${GPUS}
FORCE=${FORCE}

run_intra() {
  local grid=\"\$1\" n=\"\$2\" gpus=\"\$3\"
  local label=\"\${grid}-intra\"
  local report=\"\${OUT_DIR}/${PREFIX}-\${label}-n\${n}-g\${gpus}.json\"
  if [[ \"\${FORCE}\" -eq 0 && -f \"\${report}\" ]]; then
    echo \"==> [skip exists] \${report##*/}\"
    return 0
  fi
  local cvd
  cvd=\$(seq -s, 0 \$((gpus - 1)))
  echo \"==> [intra] grid=\${grid} N=\${n} gpus=\${gpus} -> \${report##*/}\"
  CUDA_VISIBLE_DEVICES=\"\${cvd}\" \"\${INTRA_BIN}\" \
    --grid \"\${grid}\" --n \"\${n}\" --iters ${ITERS} --warmup ${WARMUP} \
    --lattice q19 --collision bgk --overlap standard \
    --report-every 0 --label \"\${label}\" --report \"\${report}\"
}

for n in \${SIZES}; do run_intra dense \"\${n}\" 1; done
for n in \${SIZES}; do run_intra dense \"\${n}\" \${GPUS}; done
for n in \${SIZES}; do run_intra disg  \"\${n}\" \${GPUS}; done

echo '=== Peeraccess sweep complete ==='
ls -1 \"\${OUT_DIR}\"/*.json | wc -l
"

echo "==> NCCL sweep (MPI+NCCL) in ${IMAGE_MPI}"
docker_run "${IMAGE_MPI}" "
set -euo pipefail
export OMPI_ALLOW_RUN_AS_ROOT=1 OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1
export NCCL_IB_DISABLE=1

MPI_BIN=\"/src/build-mpi/benchmarks/lbm_lid_driven_cavity/carbon_benchmark_lbm_lid_driven_cavity\"
OUT_DIR=\"/src/${rel_data}\"
mkdir -p \"\${OUT_DIR}\"

SIZES=\"${SIZES}\"
GPUS=${GPUS}
FORCE=${FORCE}

run_mpi() {
  local grid=\"\$1\" n=\"\$2\"
  local label=\"\${grid}-mpi\"
  local report=\"\${OUT_DIR}/${PREFIX}-\${label}-n\${n}-g\${GPUS}.json\"
  if [[ \"\${FORCE}\" -eq 0 && -f \"\${report}\" ]]; then
    echo \"==> [skip exists] \${report##*/}\"
    return 0
  fi
  echo \"==> [mpi] grid=\${grid} N=\${n} gpus=\${GPUS} -> \${report##*/}\"
  mpiexec -n \${GPUS} \"\${MPI_BIN}\" \
    --grid \"\${grid}\" --n \"\${n}\" --iters ${ITERS} --warmup ${WARMUP} \
    --lattice q19 --collision bgk --overlap standard \
    --report-every 0 --label \"\${label}\" --report \"\${report}\"
}

for n in \${SIZES}; do run_mpi dense \"\${n}\"; done
for n in \${SIZES}; do run_mpi disg  \"\${n}\"; done

echo '=== NCCL sweep complete ==='
ls -1 \"\${OUT_DIR}\"/*.json | wc -l
"

echo "==> MPI (P2P) sweep (CUDA-aware MPI, no NCCL) in ${IMAGE_MPI}"
docker_run "${IMAGE_MPI}" "
set -euo pipefail
export OMPI_ALLOW_RUN_AS_ROOT=1 OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1
export NCCL_IB_DISABLE=1

MPI_BIN=\"/src/build-mpi/benchmarks/lbm_lid_driven_cavity/carbon_benchmark_lbm_lid_driven_cavity\"
OUT_DIR=\"/src/${rel_data}\"
mkdir -p \"\${OUT_DIR}\"

SIZES=\"${SIZES}\"
GPUS=${GPUS}
FORCE=${FORCE}

run_mpip2p() {
  local grid=\"\$1\" n=\"\$2\"
  local label=\"\${grid}-mpip2p\"
  local report=\"\${OUT_DIR}/${PREFIX}-\${label}-n\${n}-g\${GPUS}.json\"
  if [[ \"\${FORCE}\" -eq 0 && -f \"\${report}\" ]]; then
    echo \"==> [skip exists] \${report##*/}\"
    return 0
  fi
  echo \"==> [mpip2p] grid=\${grid} N=\${n} gpus=\${GPUS} -> \${report##*/}\"
  mpiexec -n \${GPUS} -x CARBON_DIST_TRANSPORT=mpi \"\${MPI_BIN}\" \
    --grid \"\${grid}\" --n \"\${n}\" --iters ${ITERS} --warmup ${WARMUP} \
    --lattice q19 --collision bgk --overlap standard \
    --report-every 0 --label \"\${label}\" --report \"\${report}\"
}

for n in \${SIZES}; do run_mpip2p dense \"\${n}\"; done
for n in \${SIZES}; do run_mpip2p disg  \"\${n}\"; done

echo '=== MPI (P2P) sweep complete ==='
ls -1 \"\${OUT_DIR}\"/*.json | wc -l
"

docker_run "${IMAGE_INTRA}" "chmod -R a+rw /src/${rel_data}"

echo
echo "==> done. JSON reports in ${rel_data}/ (prefix ${PREFIX}-*.json)"
echo "    now render ${rel_root}/${PREFIX}.png with:"
echo "      python3 benchmarks/lbm_lid_driven_cavity/scripts/lbm-dgx-a100-40gb-plot.py"
