#!/usr/bin/env bash
#
# benchmarks/lbm_lid_driven_cavity/scripts/lbm-dgx-sweep-run.sh
#
# Generic (lattice/collision x precision x domain size x GPU count x grid x
# transport) sweep on the DGX node. Builds the two Release trees inside Docker
# (same as the other DGX scripts) and does one benchmark run per combination
# with --overlap standard. Lattice/collision pairs: q19:bgk (D3Q19 + BGK),
# q27:bgk (D3Q27 + BGK), q27:kbc (D3Q27 + KBC); precisions: fp32, fp64.
# Transports:
#
#   intra   Peeraccess (one process, intra-process NVLink P2P)
#   mpi     NCCL       (mpiexec -n G, halos over NCCL)
#   mpip2p  MPI (P2P)  (mpiexec -n G, CARBON_DIST_TRANSPORT=mpi)
#
# The NCCL and MPI (P2P) transports are skipped at G=1: a single rank has no
# halo exchange, and the efficiency baseline is the 1-GPU dense-intra run.
#
# Timing: after WARMUP solver steps, each run times REPS repetitions of
# ITERS_PER_REP solver steps each (binary --reps), syncing the GPUs between
# repetitions. results.mlups is the MLUPS over all timed steps (what the
# plotters read); results.repetitions holds the per-repetition MLUPS and their
# mean / min / max / stddev.
#
# Reports are named
#   <PREFIX>-<lattice>-<collision>-<precision>-<grid>-<transport>-n<N>-g<G>.json
# so each (lattice, collision, precision) is its own series for the Fig. 7 /
# strong-scaling plotters: pass --prefix <PREFIX>-<lattice>-<collision>-<precision>.
# Each series is normalised to its own 1-GPU dense-intra run. A failing or
# hung run (RUN_TIMEOUT) is logged to failed-{intra,mpi}.txt and the sweep
# moves on; sizes whose 1-GPU run does not fit in GPU memory (e.g. fp64 at
# N=512) fail that way.
#
# Usage:
#   ./benchmarks/lbm_lid_driven_cavity/scripts/lbm-dgx-sweep-run.sh [CONFIG.yml]
#
# CONFIG.yml (see lbm-dgx-sweep.yml) sets the knobs below by their lowercase
# name (configs, precisions, sizes, gpus, ..., run_timeout). List-valued keys
# take a list or a space-separated string; sizes / gpus also accept
# {start, stop, step} (inclusive). Precedence: env var > CONFIG.yml > default.
#
# Knobs (env vars, all optional):
#   CONFIGS="q19:bgk"                     lattice:collision pairs (q19:bgk q27:bgk q27:kbc)
#   PRECISIONS="fp32"                     fp32 and/or fp64
#   SIZES="256 320 384 448 512"           domain edge lengths
#   GPUS_LIST="1 2 3 4 5 6 7 8"           GPU counts
#   GRIDS="dense disg"                    grid families
#   TRANSPORTS="intra mpi mpip2p"         transports
#   WARMUP=400                            untimed solver steps before the first repetition
#   ITERS_PER_REP=400                     solver steps averaged in each timed repetition
#   REPS=4                                timed repetitions per run
#   SM_ARCH=auto                          CUDA arch (A100=80, H100=90); auto = detect
#                                         via nvidia-smi; "80;90" builds for both
#   IMAGE_INTRA=carbon-cuda:local         image for build + intra-process runs
#   IMAGE_MPI=carbon-cuda-nccl-pinned:local  image for MPI runs
#   PREFIX=lbm-dgx-sweep                  report prefix and data/<PREFIX>/ folder
#   SKIP_BUILD=0                          set 1 to reuse existing build trees
#   FORCE=0                               set 1 to redo runs whose JSON exists
#   RUN_TIMEOUT=1800                      per-run timeout in seconds
#   DRY_RUN=0                             set 1 to print the resolved config and exit

set -euo pipefail

if [[ $# -gt 1 ]]; then
  echo "usage: $0 [CONFIG.yml]" >&2
  exit 2
fi
config_path=""
if [[ $# -eq 1 ]]; then
  if [[ ! -f "$1" ]]; then
    echo "error: config file not found: $1" >&2
    exit 2
  fi
  config_path="$(realpath "$1")"
fi

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${repo_root}"

# Every knob as a validated VAR=value line, defaults filled in, from the parser
# the plotters share (lbm_sweep_config.py); an empty config yields the defaults.
config_text=""
if [[ -n "${config_path}" ]]; then
  config_text="$(<"${config_path}")"
fi
config_vars="$(LBM_SWEEP_CONFIG="${config_text}" \
  python3 "${repo_root}/benchmarks/lbm_lid_driven_cavity/scripts/lbm_sweep_config.py")"
while IFS='=' read -r key value; do
  [[ -z "${key}" ]] && continue
  if [[ -z "${!key+x}" ]]; then
    export "${key}=${value}"
  else
    export "${key}"
    echo "==> ${key} taken from the environment"
  fi
done <<< "${config_vars}"

for c in ${CONFIGS}; do
  case "${c}" in
    q19:bgk | q27:bgk | q27:kbc) ;;
    *) echo "error: CONFIGS entry '${c}' is not one of q19:bgk q27:bgk q27:kbc" >&2; exit 1 ;;
  esac
done
for p in ${PRECISIONS}; do
  case "${p}" in
    fp32 | fp64) ;;
    *) echo "error: PRECISIONS entry '${p}' is not fp32 or fp64" >&2; exit 1 ;;
  esac
done
if [[ "${SM_ARCH}" == "auto" ]]; then
  SM_ARCH="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null |
             tr -d '. ' | sort -u | paste -sd';' -)" || true
  if [[ -z "${SM_ARCH}" ]]; then
    echo "error: could not detect the GPU compute capability with nvidia-smi;" \
         "set sm_arch / SM_ARCH explicitly (A100=80, H100=90)" >&2
    exit 1
  fi
fi
if [[ ! "${SM_ARCH}" =~ ^[0-9]+(\;[0-9]+)*$ ]]; then
  echo "error: SM_ARCH must be auto or numbers separated by ';' (got '${SM_ARCH}')" >&2
  exit 1
fi

echo "==> sweep config${config_path:+ (${config_path##*/})}:"
for v in CONFIGS PRECISIONS SIZES GPUS_LIST GRIDS TRANSPORTS WARMUP ITERS_PER_REP REPS SM_ARCH \
         IMAGE_INTRA IMAGE_MPI PREFIX SKIP_BUILD FORCE RUN_TIMEOUT; do
  printf '    %-14s %s\n' "${v}" "${!v}"
done
if [[ "${DRY_RUN:-0}" -eq 1 ]]; then
  exit 0
fi

rel_root="benchmarks/lbm_lid_driven_cavity/data/${PREFIX}"
rel_data="${rel_root}/json"
mkdir -p "${repo_root}/${rel_data}"

docker_run() {
  local image="$1" script="$2"
  shift 2
  docker run --rm --gpus all \
    -v "${repo_root}:/src:rw" -w /src \
    -e CONFIGS -e PRECISIONS -e SIZES -e GPUS_LIST -e GRIDS -e TRANSPORTS \
    -e WARMUP -e ITERS_PER_REP -e REPS \
    -e FORCE -e RUN_TIMEOUT -e PREFIX -e OUT_DIR="/src/${rel_data}" \
    "${image}" /bin/bash -lc "${script}" sweep "$@"
}

if [[ "${SKIP_BUILD}" -eq 0 ]]; then
  echo "==> building Release trees (SM_${SM_ARCH}) in ${IMAGE_INTRA}"
  docker_run "${IMAGE_INTRA}" "
set -euo pipefail
export PATH=\"/root/.local/bin:\$PATH\"
echo '=== Building CUDA (intra-process) ==='
cmake -S /src -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCARBON_BACKEND=Cuda '-DCMAKE_CUDA_ARCHITECTURES=${SM_ARCH}'
cmake --build build -j\"\$(nproc)\" --target carbon_benchmark_lbm_lid_driven_cavity
echo '=== Building CUDA+MPI (distributed) ==='
cmake -S /src -B build-mpi -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCARBON_BACKEND=Cuda '-DCMAKE_CUDA_ARCHITECTURES=${SM_ARCH}' -DCARBON_ENABLE_MPI=ON
cmake --build build-mpi -j\"\$(nproc)\" --target carbon_benchmark_lbm_lid_driven_cavity
echo '=== Build complete ==='
"
else
  echo "==> SKIP_BUILD=1: reusing existing build/ and build-mpi/ trees"
fi

# Runs inside the container as `bash -lc "$sweep_body" sweep <intra|mpi>`.
read -r -d '' sweep_body <<'EOF' || true
set -uo pipefail
mode="$1"
bin_rel="benchmarks/lbm_lid_driven_cavity/carbon_benchmark_lbm_lid_driven_cavity"
if [[ "${mode}" == "intra" ]]; then
  bin="/src/build/${bin_rel}"
  transports=""
  [[ " ${TRANSPORTS} " == *" intra "* ]] && transports="intra"
else
  bin="/src/build-mpi/${bin_rel}"
  transports=""
  for t in mpi mpip2p; do
    [[ " ${TRANSPORTS} " == *" ${t} "* ]] && transports="${transports} ${t}"
  done
  export OMPI_ALLOW_RUN_AS_ROOT=1 OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1
  export NCCL_IB_DISABLE=1
fi

mkdir -p "${OUT_DIR}"
failed_log="${OUT_DIR}/failed-${mode}.txt"
: > "${failed_log}"
iters=$((WARMUP + REPS * ITERS_PER_REP))

for cfg in ${CONFIGS}; do
  lattice="${cfg%%:*}"
  collision="${cfg##*:}"
  for prec in ${PRECISIONS}; do
    series="${PREFIX}-${lattice}-${collision}-${prec}"
    for t in ${transports}; do
      for g in ${GPUS_LIST}; do
        [[ "${t}" != "intra" && "${g}" -eq 1 ]] && continue
        for grid in ${GRIDS}; do
          for n in ${SIZES}; do
            label="${grid}-${t}"
            report="${OUT_DIR}/${series}-${label}-n${n}-g${g}.json"
            if [[ "${FORCE}" -eq 0 && -f "${report}" ]]; then
              echo "==> [skip exists] ${report##*/}"
              continue
            fi
            case "${t}" in
              intra)  cmd=(env CUDA_VISIBLE_DEVICES="$(seq -s, 0 $((g - 1)))" "${bin}") ;;
              mpi)    cmd=(mpiexec -n "${g}" "${bin}") ;;
              mpip2p) cmd=(mpiexec -n "${g}" -x CARBON_DIST_TRANSPORT=mpi "${bin}") ;;
            esac
            echo "==> [${t}] ${lattice}/${collision}/${prec} grid=${grid} N=${n} gpus=${g}" \
                 "-> ${report##*/}"
            if ! timeout "${RUN_TIMEOUT}" "${cmd[@]}" \
                --grid "${grid}" --n "${n}" --iters "${iters}" --warmup "${WARMUP}" \
                --reps "${REPS}" \
                --lattice "${lattice}" --collision "${collision}" --precision "${prec}" \
                --overlap standard \
                --report-every 0 --label "${label}" --report "${report}"; then
              echo "!!> [failed] ${report##*/}"
              rm -f "${report}"
              echo "${report##*/}" >> "${failed_log}"
            fi
          done
        done
      done
    done
  done
done
echo "=== ${mode} sweep complete ($(wc -l < "${failed_log}") failed) ==="
EOF

echo "==> Peeraccess sweep in ${IMAGE_INTRA}"
docker_run "${IMAGE_INTRA}" "${sweep_body}" intra

echo "==> NCCL + MPI (P2P) sweep in ${IMAGE_MPI}"
docker_run "${IMAGE_MPI}" "${sweep_body}" mpi

docker_run "${IMAGE_INTRA}" "chmod -R a+rw /src/${rel_data}"

echo
echo "==> done. JSON reports in ${rel_data}/ (prefix ${PREFIX}-*.json)"
cat "${rel_data}"/failed-*.txt 2>/dev/null | sed 's/^/    FAILED: /' || true
