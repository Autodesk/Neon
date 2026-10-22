#!/usr/bin/env bash
#
# Strong-scaling sweep at fixed N=512 across GPU counts 1, 2, 4, 8.
# Runs Peeraccess, NCCL, and MPI (P2P) for both Dense and DenseDisg grids.
#
# Usage:
#   ./benchmarks/lbm_lid_driven_cavity/scripts/lbm-n512-gpu-scaling-run.sh
#
# Knobs (env vars):
#   N=512
#   GPUS_LIST="1 2 4 8"
#   ITERS=2000  WARMUP=400
#   SM_ARCH=80
#   SKIP_BUILD=1   reuse build/ and build-mpi/ trees
#   FORCE=1        overwrite existing JSON

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${repo_root}"

N="${N:-512}"
GPUS_LIST="${GPUS_LIST:-1 2 4 8}"
ITERS="${ITERS:-2000}"
WARMUP="${WARMUP:-400}"
SM_ARCH="${SM_ARCH:-80}"
IMAGE_INTRA="${IMAGE_INTRA:-carbon-cuda:local}"
IMAGE_MPI="${IMAGE_MPI:-carbon-cuda-nccl-pinned:local}"
SKIP_BUILD="${SKIP_BUILD:-1}"
FORCE="${FORCE:-1}"

PREFIX="lbm-n512-scaling"
rel_data="benchmarks/lbm_lid_driven_cavity/data/lbm-n512-gpu-scaling/json"
mkdir -p "${repo_root}/${rel_data}"

docker_run() {
  local image="$1" script="$2"
  docker run --rm --gpus all \
    -v "${repo_root}:/src:rw" -w /src \
    "${image}" /bin/bash -lc "${script}"
}

if [[ "${SKIP_BUILD}" -eq 0 ]]; then
  echo "==> building Release trees (SM_${SM_ARCH})"
  docker_run "${IMAGE_INTRA}" "
set -euo pipefail
cmake -S /src -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCARBON_BACKEND=Cuda -DCMAKE_CUDA_ARCHITECTURES=${SM_ARCH}
cmake --build build -j\"\$(nproc)\" --target carbon_benchmark_lbm_lid_driven_cavity
cmake -S /src -B build-mpi -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCARBON_BACKEND=Cuda -DCMAKE_CUDA_ARCHITECTURES=${SM_ARCH} -DCARBON_ENABLE_MPI=ON
cmake --build build-mpi -j\"\$(nproc)\" --target carbon_benchmark_lbm_lid_driven_cavity
"
fi

echo "==> Peeraccess sweep (dense+disg, N=${N}, GPUS=${GPUS_LIST})"
docker_run "${IMAGE_INTRA}" "
set -euo pipefail
BIN=\"/src/build/benchmarks/lbm_lid_driven_cavity/carbon_benchmark_lbm_lid_driven_cavity\"
OUT=\"/src/${rel_data}\"
GPUS_LIST=\"${GPUS_LIST}\"
FORCE=${FORCE}

run_intra() {
  local grid=\"\$1\" label=\"\$2\" g=\"\$3\"
  local report=\"\${OUT}/${PREFIX}-\${label}-n${N}-g\${g}.json\"
  if [[ \"\${FORCE}\" -eq 0 && -f \"\${report}\" ]]; then
    echo \"==> [skip] \${report##*/}\"
    return 0
  fi
  local cvd
  cvd=\$(seq -s, 0 \$((g - 1)))
  echo \"==> [intra] grid=\${grid} g=\${g} -> \${report##*/}\"
  CUDA_VISIBLE_DEVICES=\"\${cvd}\" \"\${BIN}\" \
    --grid \"\${grid}\" --n ${N} --iters ${ITERS} --warmup ${WARMUP} \
    --lattice q19 --collision bgk --overlap standard \
    --report-every 0 --label \"\${label}\" --report \"\${report}\"
}

for g in \${GPUS_LIST}; do run_intra dense dense-intra \"\${g}\"; done
for g in \${GPUS_LIST}; do run_intra disg  disg-intra  \"\${g}\"; done
"

echo "==> NCCL sweep (dense+disg, N=${N}, GPUS=${GPUS_LIST})"
docker_run "${IMAGE_MPI}" "
set -euo pipefail
export OMPI_ALLOW_RUN_AS_ROOT=1 OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1
export NCCL_IB_DISABLE=1
BIN=\"/src/build-mpi/benchmarks/lbm_lid_driven_cavity/carbon_benchmark_lbm_lid_driven_cavity\"
OUT=\"/src/${rel_data}\"
GPUS_LIST=\"${GPUS_LIST}\"
FORCE=${FORCE}

run_mpi() {
  local grid=\"\$1\" label=\"\$2\" g=\"\$3\"
  local report=\"\${OUT}/${PREFIX}-\${label}-n${N}-g\${g}.json\"
  if [[ \"\${FORCE}\" -eq 0 && -f \"\${report}\" ]]; then
    echo \"==> [skip] \${report##*/}\"
    return 0
  fi
  echo \"==> [mpi] grid=\${grid} g=\${g} -> \${report##*/}\"
  if [[ \"\${g}\" -eq 1 ]]; then
    CUDA_VISIBLE_DEVICES=0 mpiexec -n 1 \"\${BIN}\" \
      --grid \"\${grid}\" --n ${N} --iters ${ITERS} --warmup ${WARMUP} \
      --lattice q19 --collision bgk --overlap standard \
      --report-every 0 --label \"\${label}\" --report \"\${report}\"
  else
    mpiexec -n \"\${g}\" \"\${BIN}\" \
      --grid \"\${grid}\" --n ${N} --iters ${ITERS} --warmup ${WARMUP} \
      --lattice q19 --collision bgk --overlap standard \
      --report-every 0 --label \"\${label}\" --report \"\${report}\"
  fi
}

for g in \${GPUS_LIST}; do run_mpi dense dense-mpi \"\${g}\"; done
for g in \${GPUS_LIST}; do run_mpi disg  disg-mpi  \"\${g}\"; done
"

echo "==> MPI (P2P) sweep (dense+disg, N=${N}, GPUS=${GPUS_LIST})"
docker_run "${IMAGE_MPI}" "
set -euo pipefail
export OMPI_ALLOW_RUN_AS_ROOT=1 OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1
export NCCL_IB_DISABLE=1
BIN=\"/src/build-mpi/benchmarks/lbm_lid_driven_cavity/carbon_benchmark_lbm_lid_driven_cavity\"
OUT=\"/src/${rel_data}\"
GPUS_LIST=\"${GPUS_LIST}\"
FORCE=${FORCE}

run_mpip2p() {
  local grid=\"\$1\" label=\"\$2\" g=\"\$3\"
  local report=\"\${OUT}/${PREFIX}-\${label}-n${N}-g\${g}.json\"
  if [[ \"\${FORCE}\" -eq 0 && -f \"\${report}\" ]]; then
    echo \"==> [skip] \${report##*/}\"
    return 0
  fi
  echo \"==> [mpip2p] grid=\${grid} g=\${g} -> \${report##*/}\"
  if [[ \"\${g}\" -eq 1 ]]; then
    CUDA_VISIBLE_DEVICES=0 mpiexec -n 1 -x CARBON_DIST_TRANSPORT=mpi \"\${BIN}\" \
      --grid \"\${grid}\" --n ${N} --iters ${ITERS} --warmup ${WARMUP} \
      --lattice q19 --collision bgk --overlap standard \
      --report-every 0 --label \"\${label}\" --report \"\${report}\"
  else
    mpiexec -n \"\${g}\" -x CARBON_DIST_TRANSPORT=mpi \"\${BIN}\" \
      --grid \"\${grid}\" --n ${N} --iters ${ITERS} --warmup ${WARMUP} \
      --lattice q19 --collision bgk --overlap standard \
      --report-every 0 --label \"\${label}\" --report \"\${report}\"
  fi
}

for g in \${GPUS_LIST}; do run_mpip2p dense dense-mpip2p \"\${g}\"; done
for g in \${GPUS_LIST}; do run_mpip2p disg  disg-mpip2p  \"\${g}\"; done
"

docker_run "${IMAGE_INTRA}" "chmod -R a+rw /src/${rel_data}"
echo "==> done. JSON in ${rel_data}/"
