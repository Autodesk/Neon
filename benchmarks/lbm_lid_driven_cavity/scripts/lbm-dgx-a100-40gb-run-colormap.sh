#!/usr/bin/env bash
#
# benchmarks/lbm_lid_driven_cavity/scripts/lbm-dgx-a100-40gb-run-colormap.sh
#
# Full (domain-size x GPU-count) matrix for the four LBM "families":
#
#   dense-intra  (Dense       grid, Peeraccess / intra-process NVLink P2P)
#   dense-mpi    (Dense       grid, NCCL / MPI+NCCL, one rank per GPU)
#   disg-intra   (DenseDisg   grid, Peeraccess)
#   disg-mpi     (DenseDisg   grid, NCCL)
#
# This is the data producer for the discrete "best-family" colormap rendered
# by lbm-dgx-a100-40gb-run-colormap.py.  The JSON reports use the *shared*
# lbm-dgx-a100-40gb naming/format (identical to lbm-dgx-a100-40gb-run.sh) and
# land in the same data/lbm-dgx-a100-40gb/json/ folder, so the g8 reports the
# Fig.7 sweep already produced are reused as-is and only the missing GPU-count
# cells (g1/g2/g4, plus the mpi families at g1) are actually run. Existing
# JSON is skipped unless FORCE=1.
#
#   ./benchmarks/lbm_lid_driven_cavity/scripts/lbm-dgx-a100-40gb-run-colormap.sh
#   python3 benchmarks/lbm_lid_driven_cavity/scripts/lbm-dgx-a100-40gb-run-colormap.py
#
# N=640+ is excluded: the Q=19 ping-pong fields do not fit a single 40 GB A100
# (OOM at N=640; N=608 is the largest size that fits on one GPU).
#
# Note on the mpi families at g1: a plain `mpiexec -n 1` grabs ALL visible GPUs
# (world_size==1 is treated as "not distributed" -> get_all_devices()). To get
# an honest single-GPU point we pin CUDA_VISIBLE_DEVICES=0 for that case, so it
# collapses to a single-device run (numerically ~= the intra family at g1).
#
# Knobs (env vars, all optional):
#   SIZES="128 160 ... 608"               domain edge lengths (step 32)
#   GPUS_LIST="1 2 4 8"                    GPU-count axis of the colormap
#   ITERS=2000                            total iterations per run
#   WARMUP=400                            warmup iterations excluded from timing
#   SM_ARCH=80                            CUDA arch (A100=80, H100=90)
#   IMAGE_INTRA=carbon-cuda:local         image for build + intra-process runs
#   IMAGE_MPI=carbon-cuda-nccl-pinned:local  image for MPI+NCCL runs
#   SKIP_BUILD=0                          set 1 to reuse existing build trees
#   FORCE=0                               set 1 to re-run cells whose JSON exists

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${repo_root}"

SIZES="${SIZES:-$(seq -s' ' 128 32 608)}"
GPUS_LIST="${GPUS_LIST:-1 2 4 8}"
ITERS="${ITERS:-2000}"
WARMUP="${WARMUP:-400}"
SM_ARCH="${SM_ARCH:-80}"
IMAGE_INTRA="${IMAGE_INTRA:-carbon-cuda:local}"
IMAGE_MPI="${IMAGE_MPI:-carbon-cuda-nccl-pinned:local}"
SKIP_BUILD="${SKIP_BUILD:-0}"
FORCE="${FORCE:-0}"

# JSON keeps the shared base prefix so cells are reused across sweeps; only the
# .py / .png carry the -run-colormap suffix.
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

echo "==> Peeraccess sweep (intra-process, GPUS_LIST=${GPUS_LIST}) in ${IMAGE_INTRA}"
docker_run "${IMAGE_INTRA}" "
set -euo pipefail
INTRA_BIN=\"/src/build/benchmarks/lbm_lid_driven_cavity/carbon_benchmark_lbm_lid_driven_cavity\"
OUT_DIR=\"/src/${rel_data}\"
mkdir -p \"\${OUT_DIR}\"

SIZES=\"${SIZES}\"
GPUS_LIST=\"${GPUS_LIST}\"
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

for g in \${GPUS_LIST}; do
  for grid in dense disg; do
    for n in \${SIZES}; do run_intra \"\${grid}\" \"\${n}\" \"\${g}\"; done
  done
done

echo '=== Peeraccess sweep complete ==='
ls -1 \"\${OUT_DIR}\"/*.json | wc -l
"

echo "==> NCCL sweep (MPI+NCCL, GPUS_LIST=${GPUS_LIST}) in ${IMAGE_MPI}"
docker_run "${IMAGE_MPI}" "
set -euo pipefail
export OMPI_ALLOW_RUN_AS_ROOT=1 OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1
export NCCL_IB_DISABLE=1

MPI_BIN=\"/src/build-mpi/benchmarks/lbm_lid_driven_cavity/carbon_benchmark_lbm_lid_driven_cavity\"
OUT_DIR=\"/src/${rel_data}\"
mkdir -p \"\${OUT_DIR}\"

SIZES=\"${SIZES}\"
GPUS_LIST=\"${GPUS_LIST}\"
FORCE=${FORCE}

run_mpi() {
  local grid=\"\$1\" n=\"\$2\" gpus=\"\$3\"
  local label=\"\${grid}-mpi\"
  local report=\"\${OUT_DIR}/${PREFIX}-\${label}-n\${n}-g\${gpus}.json\"
  if [[ \"\${FORCE}\" -eq 0 && -f \"\${report}\" ]]; then
    echo \"==> [skip exists] \${report##*/}\"
    return 0
  fi
  echo \"==> [mpi] grid=\${grid} N=\${n} gpus=\${gpus} -> \${report##*/}\"
  if [[ \"\${gpus}\" -eq 1 ]]; then
    # honest single-GPU: pin one device so mpiexec -n 1 cannot grab all GPUs
    CUDA_VISIBLE_DEVICES=0 mpiexec -n 1 \"\${MPI_BIN}\" \
      --grid \"\${grid}\" --n \"\${n}\" --iters ${ITERS} --warmup ${WARMUP} \
      --lattice q19 --collision bgk --overlap standard \
      --report-every 0 --label \"\${label}\" --report \"\${report}\"
  else
    mpiexec -n \"\${gpus}\" \"\${MPI_BIN}\" \
      --grid \"\${grid}\" --n \"\${n}\" --iters ${ITERS} --warmup ${WARMUP} \
      --lattice q19 --collision bgk --overlap standard \
      --report-every 0 --label \"\${label}\" --report \"\${report}\"
  fi
}

for g in \${GPUS_LIST}; do
  for grid in dense disg; do
    for n in \${SIZES}; do run_mpi \"\${grid}\" \"\${n}\" \"\${g}\"; done
  done
done

echo '=== NCCL sweep complete ==='
ls -1 \"\${OUT_DIR}\"/*.json | wc -l
"

docker_run "${IMAGE_INTRA}" "chmod -R a+rw /src/${rel_data}"

echo
echo "==> done. JSON reports in ${rel_data}/ (prefix ${PREFIX}-*.json)"
echo "    now render ${rel_root}/${PREFIX}-run-colormap.png with:"
echo "      python3 benchmarks/lbm_lid_driven_cavity/scripts/lbm-dgx-a100-40gb-run-colormap.py"
