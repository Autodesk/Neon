#!/usr/bin/env bash
#
# benchmarks/lbm_lid_driven_cavity_distributed/scripts/lbm-dist-sweep-run.sh
#
# Strong-scaling sweep of the distributed LBM benchmark on one multi-GPU node:
# lattice/collision x domain size x decomposition x its GPU counts x transport
# x halo priority. Builds the CUDA+MPI Release tree inside Docker and does one
# `mpiexec -n G` run per combination (one rank per GPU, --grid
# dense_distributed, --overlap OVERLAP).
#
#   decompositions  slab (split z), pencil (y, z), brick (x, y, z), each with its
#                   own GPU counts (GPUS_SLAB / GPUS_PENCIL / GPUS_BRICK, all >= 2)
#   transports      mpi = NCCL; mpip2p = MPI point-to-point (CARBON_DIST_TRANSPORT=mpi)
#   hp_halo         off = normal halo stream; on = --hp-halo (high-priority halo stream)
#
# No 1-GPU run is done: the efficiency baseline of size N is the 1-GPU report
# of another sweep, by default the Dense Peeraccess run of the single-process
# DGX sweep,
#   <REFERENCE_DIR>/<REFERENCE_PREFIX>-<lattice>-<collision>-fp32-<REFERENCE_LABEL>-n<N>-g1.json
# The script warns before the runs if one of them is missing.
#
# Timing: after WARMUP solver steps, each run times REPS repetitions of
# ITERS_PER_REP solver steps (binary --reps); every repetition ends with a GPU
# drain and an MPI barrier, so rank 0's clock covers the slowest rank.
# results.mlups is the MLUPS over all timed steps (what the plotter reads);
# results.repetitions holds the per-repetition MLUPS; config.process_grid the
# rank grid ProcessGrid actually chose.
#
# Reports are named
#   <PREFIX>-<lattice>-<collision>-fp32-<decomposition>-<transport>-hp<off|on>-n<N>-g<G>.json
# in benchmarks/lbm_lid_driven_cavity_distributed/data/<PREFIX>/json/. A failing
# or hung run (RUN_TIMEOUT) is logged to failed.txt there and the sweep moves on.
#
# Usage:
#   ./benchmarks/lbm_lid_driven_cavity_distributed/scripts/lbm-dist-sweep-run.sh [CONFIG.yml]
#
# CONFIG.yml (see lbm-dist-sweep.yml) sets the knobs below by their lowercase
# name. List-valued keys take a list or a space-separated string; sizes and
# gpus_* also accept {start, stop, step} (inclusive). Precedence: env var >
# CONFIG.yml > default.
#
# Knobs (env vars, all optional):
#   CONFIGS="q19:bgk"                     lattice:collision pairs (q19:bgk q27:bgk q27:kbc)
#   SIZES="256 320 384 448 512"           domain edge lengths
#   DECOMPOSITIONS="slab pencil brick"    process decompositions
#   GPUS_SLAB="2 3 4 5 6 7 8"             GPU counts (= MPI ranks) of the slab runs
#   GPUS_PENCIL="4 8"                     ... of the pencil runs
#   GPUS_BRICK="8"                        ... of the brick runs
#   REFERENCE_DIR=benchmarks/lbm_lid_driven_cavity/data/lbm-dgx-sweep/json
#                                         1-GPU baseline reports, relative to the repo root
#   REFERENCE_PREFIX=lbm-dgx-sweep        their report prefix
#   REFERENCE_LABEL=dense-intra           their grid-transport label
#   TRANSPORTS="mpi"                      mpi (NCCL) and/or mpip2p
#   HP_HALO="off on"                      halo stream priority variants
#   OVERLAP=standard                      Skeleton OCC mode of every run
#   WARMUP=400                            untimed solver steps before the first repetition
#   ITERS_PER_REP=400                     solver steps in each timed repetition
#   REPS=4                                timed repetitions per run
#   SM_ARCH=auto                          CUDA arch (A100=80, H100=90); auto = detect
#   IMAGE=carbon-cuda-nccl-pinned:local   image for the build and every run
#   BUILD_DIR=build-dist                  CUDA+MPI build tree under the repo root
#   PREFIX=lbm-dist-sweep                 report prefix and data/<PREFIX>/ folder
#   SKIP_BUILD=0                          set 1 to reuse the existing build tree
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
# the plotter shares (lbm_dist_sweep_config.py); an empty config yields the defaults.
config_text=""
if [[ -n "${config_path}" ]]; then
  config_text="$(<"${config_path}")"
fi
config_vars="$(LBM_SWEEP_CONFIG="${config_text}" \
  python3 "${repo_root}/benchmarks/lbm_lid_driven_cavity_distributed/scripts/lbm_dist_sweep_config.py")"
while IFS='=' read -r key value; do
  [[ -z "${key}" ]] && continue
  if [[ -z "${!key+x}" ]]; then
    export "${key}=${value}"
  else
    export "${key}"
    echo "==> ${key} taken from the environment"
  fi
done <<< "${config_vars}"

check_words() {
  local var="$1" allowed="$2" w
  for w in ${!var}; do
    if [[ " ${allowed} " != *" ${w} "* ]]; then
      echo "error: ${var} entry '${w}' is not one of: ${allowed}" >&2
      exit 1
    fi
  done
}
check_words CONFIGS "q19:bgk q27:bgk q27:kbc"
check_words DECOMPOSITIONS "slab pencil brick"
check_words TRANSPORTS "mpi mpip2p"
check_words HP_HALO "off on"
check_words OVERLAP "none standard extended twoway-extended"
for v in SIZES GPUS_SLAB GPUS_PENCIL GPUS_BRICK WARMUP ITERS_PER_REP REPS RUN_TIMEOUT; do
  if [[ ! "${!v}" =~ ^[0-9]+( [0-9]+)*$ ]]; then
    echo "error: ${v} must be non-negative integers (got '${!v}')" >&2
    exit 1
  fi
done
for v in GPUS_SLAB GPUS_PENCIL GPUS_BRICK; do
  for g in ${!v}; do
    if [[ "${g}" -lt 2 ]]; then
      echo "error: ${v} entries must be >= 2 (the 1-GPU point is the reference run)" >&2
      exit 1
    fi
  done
done
for v in BUILD_DIR PREFIX REFERENCE_PREFIX REFERENCE_LABEL; do
  if [[ ! "${!v}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
    echo "error: ${v} must be a single path component (got '${!v}')" >&2
    exit 1
  fi
done
if [[ ! "${REFERENCE_DIR}" =~ ^[A-Za-z0-9._/-]+$ || "${REFERENCE_DIR}" == /* ||
      "/${REFERENCE_DIR}/" == */../* ]]; then
  echo "error: REFERENCE_DIR must be a path inside the repo (got '${REFERENCE_DIR}')" >&2
  exit 1
fi
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
for v in CONFIGS SIZES DECOMPOSITIONS GPUS_SLAB GPUS_PENCIL GPUS_BRICK REFERENCE_DIR \
         REFERENCE_PREFIX REFERENCE_LABEL TRANSPORTS HP_HALO OVERLAP WARMUP ITERS_PER_REP \
         REPS SM_ARCH IMAGE BUILD_DIR PREFIX SKIP_BUILD FORCE RUN_TIMEOUT; do
  printf '    %-16s %s\n' "${v}" "${!v}"
done
for cfg in ${CONFIGS}; do
  for n in ${SIZES}; do
    ref="${REFERENCE_DIR}/${REFERENCE_PREFIX}-${cfg%%:*}-${cfg##*:}-fp32-${REFERENCE_LABEL}-n${n}-g1.json"
    if [[ ! -f "${repo_root}/${ref}" ]]; then
      echo "warning: missing 1-GPU reference ${ref}; N=${n} gets no efficiency plots" >&2
    fi
  done
done
if [[ "${DRY_RUN:-0}" -eq 1 ]]; then
  exit 0
fi

rel_data="benchmarks/lbm_lid_driven_cavity_distributed/data/${PREFIX}/json"
mkdir -p "${repo_root}/${rel_data}"

docker_run() {
  local script="$1"
  shift
  docker run --rm --gpus all --ipc=host --shm-size=8g \
    -v "${repo_root}:/src:rw" -w /src \
    -e CONFIGS -e SIZES -e DECOMPOSITIONS -e GPUS_SLAB -e GPUS_PENCIL -e GPUS_BRICK \
    -e TRANSPORTS -e HP_HALO -e OVERLAP \
    -e WARMUP -e ITERS_PER_REP -e REPS -e BUILD_DIR \
    -e FORCE -e RUN_TIMEOUT -e PREFIX -e OUT_DIR="/src/${rel_data}" \
    "${IMAGE}" /bin/bash -lc "${script}" sweep "$@"
}

if [[ "${SKIP_BUILD}" -eq 0 ]]; then
  echo "==> building ${BUILD_DIR}/ (CUDA+MPI Release, SM_${SM_ARCH}) in ${IMAGE}"
  docker_run "
set -euo pipefail
export PATH=\"/root/.local/bin:\$PATH\"
cmake -S /src -B \"/src/\${BUILD_DIR}\" -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCARBON_BACKEND=Cuda '-DCMAKE_CUDA_ARCHITECTURES=${SM_ARCH}' -DCARBON_ENABLE_MPI=ON
cmake --build \"/src/\${BUILD_DIR}\" -j\"\$(nproc)\" \
  --target carbon_benchmark_lbm_lid_driven_cavity_distributed
echo '=== Build complete ==='
"
else
  echo "==> SKIP_BUILD=1: reusing ${BUILD_DIR}/"
fi

# Runs inside the container as `bash -lc "$sweep_body" sweep`.
read -r -d '' sweep_body <<'EOF' || true
set -uo pipefail
bin="/src/${BUILD_DIR}/benchmarks/lbm_lid_driven_cavity_distributed/carbon_benchmark_lbm_lid_driven_cavity_distributed"
if [[ ! -x "${bin}" ]]; then
  echo "error: ${bin} not built (run without skip_build)" >&2
  exit 1
fi
export OMPI_ALLOW_RUN_AS_ROOT=1 OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1
export NCCL_IB_DISABLE=1

mkdir -p "${OUT_DIR}"
failed_log="${OUT_DIR}/failed.txt"
: > "${failed_log}"
iters=$((WARMUP + REPS * ITERS_PER_REP))

# run <report> <ranks> <decomposition> <transport> <hp on|off> <lattice> <collision> <n> <label>
run() {
  local report="$1" g="$2" decomp="$3" t="$4" hp="$5" lattice="$6" collision="$7" n="$8"
  local label="$9"
  if [[ "${FORCE}" -eq 0 && -f "${report}" ]]; then
    echo "==> [skip exists] ${report##*/}"
    return
  fi
  local launch=(mpiexec --bind-to none -n "${g}")
  [[ "${t}" == "mpip2p" ]] && launch+=(-x CARBON_DIST_TRANSPORT=mpi)
  local extra=()
  [[ "${hp}" == "on" ]] && extra+=(--hp-halo)
  echo "==> ${lattice}/${collision} ${label} N=${n} gpus=${g} -> ${report##*/}"
  if ! timeout "${RUN_TIMEOUT}" "${launch[@]}" "${bin}" \
      --grid dense_distributed --decomposition "${decomp}" --gpus 1 \
      --n "${n}" --iters "${iters}" --warmup "${WARMUP}" --reps "${REPS}" \
      --lattice "${lattice}" --collision "${collision}" \
      --overlap "${OVERLAP}" "${extra[@]}" \
      --report-every 0 --label "${label}" --report "${report}"; then
    echo "!!> [failed] ${report##*/}"
    rm -f "${report}"
    echo "${report##*/}" >> "${failed_log}"
  fi
}

for cfg in ${CONFIGS}; do
  lattice="${cfg%%:*}"
  collision="${cfg##*:}"
  series="${PREFIX}-${lattice}-${collision}-fp32"
  for decomp in ${DECOMPOSITIONS}; do
    gpus_var="GPUS_${decomp^^}"
    for g in ${!gpus_var}; do
      for n in ${SIZES}; do
        for t in ${TRANSPORTS}; do
          for hp in ${HP_HALO}; do
            label="${decomp}-${t}-hp${hp}"
            run "${OUT_DIR}/${series}-${label}-n${n}-g${g}.json" "${g}" "${decomp}" "${t}" \
                "${hp}" "${lattice}" "${collision}" "${n}" "${label}"
          done
        done
      done
    done
  done
done
echo "=== sweep complete ($(wc -l < "${failed_log}") failed) ==="
EOF

echo "==> distributed sweep in ${IMAGE}"
docker_run "${sweep_body}"

docker_run "chmod -R a+rw /src/${rel_data}"

echo
echo "==> done. JSON reports in ${rel_data}/ (prefix ${PREFIX}-*.json)"
sed 's/^/    FAILED: /' "${rel_data}/failed.txt" 2>/dev/null || true
