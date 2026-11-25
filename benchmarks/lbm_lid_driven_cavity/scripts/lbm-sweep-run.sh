#!/usr/bin/env bash
#
# benchmarks/lbm_lid_driven_cavity/scripts/lbm-sweep-run.sh
#
# Runs the experiments of a sweep YAML (see lbm-sweep.yml). Builds the
# benchmark (in Docker, or on the host with runner: local), then does every run
# the selected experiments need, once: two experiments asking for the same run
# share one JSON report. The plotters in plotting/ take the same YAML.
#
# Usage:
#   ./benchmarks/lbm_lid_driven_cavity/scripts/lbm-sweep-run.sh CONFIG.yml [--only exp1,exp2]
#
#   --only a,b   run only the named experiments of CONFIG.yml
#   DRY_RUN=1    print the resolved config, the experiments and every run, then exit
#   Env vars named after the global keys (RUNNER, FORCE, SKIP_BUILD, SM_ARCH,
#   WARMUP, ITERS_PER_REP, REPS, RUN_TIMEOUT, DATA_DIR, ...) override the YAML.
#
# Grids:
#   dense, disg                      single-process grids (z split across the
#                                    process's GPUs, or one GPU per MPI rank)
#   dense_distributed,               one MPI rank per GPU, split by the
#   disg_distributed                 decompositions slab | pencil | brick
#
# Transports:
#   intra   Peeraccess (one process, intra-process NVLink P2P)
#   mpi     NCCL       (mpiexec -n G, halos over NCCL)
#   mpip2p  MPI (P2P)  (mpiexec -n G, CARBON_DIST_TRANSPORT=mpi)
#
# NCCL and MPI (P2P) are skipped at G=1 unless mpi_single_gpu is set, which
# runs them for dense / disg as one rank pinned to GPU 0 (a plain mpiexec -n 1
# would grab every visible GPU). The distributed grids skip intra.
#
# Timing (global, shared by every experiment): warmup solver steps, then reps
# timed repetitions of iters_per_rep solver steps (binary --reps).
#
# Reports: <data_dir>/json/<lattice>-<collision>-<precision>-<label>-n<N>-g<G>.json
# with <label> = <grid>-<transport> for dense / disg (plus -hpon with hp_halo
# on) and <grid>-<decomposition>-<transport>-hp<off|on> for the distributed
# grids. With force: false an existing report is reused only if its iters,
# warmup, reps and overlap match this sweep; otherwise the run is redone.
# force: true redoes every run.
#
# Progress: <data_dir>/trace.log gets a header, one line per run with a
# completion bar and a footer (follow it with tail -f). Failed or hung runs
# (run_timeout) are listed in <data_dir>/failed-{intra,mpi}.txt and the sweep
# moves on.

set -euo pipefail

usage() {
  echo "usage: $0 CONFIG.yml [--only exp1,exp2]" >&2
  exit 2
}
config_path=""
only=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --only)
      [[ $# -ge 2 ]] || usage
      only="$2"
      shift 2
      ;;
    --only=*)
      only="${1#--only=}"
      shift
      ;;
    -*) usage ;;
    *)
      [[ -z "${config_path}" ]] || usage
      config_path="$1"
      shift
      ;;
  esac
done
[[ -n "${config_path}" ]] || usage
if [[ ! -f "${config_path}" ]]; then
  echo "error: config file not found: ${config_path}" >&2
  exit 2
fi
config_path="$(realpath "${config_path}")"

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${repo_root}"

# Every global key as a validated VAR=value line (env vars override the YAML),
# plus EXPERIMENTS (name:type:runs) and the deduplicated RUNS tokens, from the
# parser the plotters share.
config_vars="$(LBM_SWEEP_CONFIG="$(<"${config_path}")" LBM_SWEEP_ONLY="${only}" \
  python3 "${repo_root}/benchmarks/lbm_lid_driven_cavity/scripts/lbm_sweep_config.py")"
while IFS='=' read -r key value; do
  [[ -z "${key}" ]] && continue
  export "${key}=${value}"
done <<< "${config_vars}"

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
export SM_ARCH

read -r -a runs <<< "${RUNS}"
read -r -a experiments <<< "${EXPERIMENTS}"
json_dir="${repo_root}/${DATA_DIR}/json"
n_intra=0
n_mpi=0
n_existing=0
for r in "${runs[@]}"; do
  IFS=: read -r lattice collision prec grid decomp t hp g n label <<< "${r}"
  if [[ "${t}" == "intra" ]]; then
    n_intra=$((n_intra + 1))
  else
    n_mpi=$((n_mpi + 1))
  fi
  if [[ -f "${json_dir}/${lattice}-${collision}-${prec}-${label}-n${n}-g${g}.json" ]]; then
    n_existing=$((n_existing + 1))
  fi
done

echo "==> sweep config (${config_path##*/}):"
for v in RUNNER WARMUP ITERS_PER_REP REPS OVERLAP REFERENCE_LABEL DATA_DIR SM_ARCH \
         IMAGE_INTRA IMAGE_MPI BUILD_DIR BUILD_MPI_DIR SKIP_BUILD FORCE RUN_TIMEOUT; do
  printf '    %-16s %s\n' "${v}" "${!v}"
done
echo "==> experiments:"
names=()
for e in "${experiments[@]}"; do
  IFS=: read -r name etype count <<< "${e}"
  names+=("${name}")
  printf '    %-20s %-15s %6d runs\n' "${name}" "${etype}" "${count}"
done
echo "==> ${#runs[@]} distinct runs (${n_intra} Peeraccess, ${n_mpi} NCCL / MPI (P2P));" \
     "${n_existing} reports already in ${DATA_DIR}/json"
if [[ "${DRY_RUN:-0}" -eq 1 ]]; then
  for r in "${runs[@]}"; do
    IFS=: read -r lattice collision prec grid decomp t hp g n label <<< "${r}"
    printf '    %-7s %s\n' "${t}" "${lattice}-${collision}-${prec}-${label}-n${n}-g${g}"
  done
  exit 0
fi

if [[ "${RUNNER}" == "docker" ]]; then
  src="/src"
else
  src="${repo_root}"
fi
export NEED_MPI=$((n_mpi > 0 ? 1 : 0))
export SRC="${src}"
export B_INTRA="${src}/${BUILD_DIR}"
export B_MPI="${src}/${BUILD_MPI_DIR}"
export OUT_DIR="${src}/${DATA_DIR}/json"
export DATA_OUT="${src}/${DATA_DIR}"

# in_env IMAGE BODY ARGS...: runs BODY with bash, in IMAGE (runner docker) or on
# the host (runner local), with the variables above.
in_env() {
  local image="$1" body="$2"
  shift 2
  if [[ "${RUNNER}" == "docker" ]]; then
    local tz=()
    [[ -e /etc/localtime ]] && tz=(-v /etc/localtime:/etc/localtime:ro)
    docker run --rm --gpus all "${tz[@]}" \
      -v "${repo_root}:/src:rw" -w /src \
      -e SRC -e B_INTRA -e B_MPI -e OUT_DIR -e DATA_OUT -e NEED_MPI -e SM_ARCH \
      -e RUNS -e OVERLAP -e WARMUP -e ITERS_PER_REP -e REPS -e FORCE -e RUN_TIMEOUT \
      -e RUN_TOTAL -e SWEEP_START \
      "${image}" /bin/bash -lc "${body}" sweep "$@"
  else
    bash -c "${body}" sweep "$@"
  fi
}

read -r -d '' build_body <<'EOF' || true
set -euo pipefail
export PATH="${HOME:-/root}/.local/bin:${PATH}"
gen=()
if command -v ninja > /dev/null; then
  gen=(-G Ninja)
fi
target=carbon_benchmark_lbm_lid_driven_cavity
echo '=== Building CUDA (intra-process) ==='
cmake -S "${SRC}" -B "${B_INTRA}" "${gen[@]}" -DCMAKE_BUILD_TYPE=Release \
  -DCARBON_BACKEND=Cuda "-DCMAKE_CUDA_ARCHITECTURES=${SM_ARCH}"
cmake --build "${B_INTRA}" -j"$(nproc)" --target "${target}"
if [[ "${NEED_MPI}" -eq 1 ]]; then
  echo '=== Building CUDA+MPI (distributed) ==='
  cmake -S "${SRC}" -B "${B_MPI}" "${gen[@]}" -DCMAKE_BUILD_TYPE=Release \
    -DCARBON_BACKEND=Cuda "-DCMAKE_CUDA_ARCHITECTURES=${SM_ARCH}" -DCARBON_ENABLE_MPI=ON
  cmake --build "${B_MPI}" -j"$(nproc)" --target "${target}"
fi
echo '=== Build complete ==='
EOF

if [[ "${SKIP_BUILD}" -eq 0 ]]; then
  echo "==> building Release trees (SM_${SM_ARCH}, runner ${RUNNER})"
  in_env "${IMAGE_INTRA}" "${build_body}"
else
  echo "==> skip_build: reusing ${BUILD_DIR}/ and ${BUILD_MPI_DIR}/"
fi

# Runs as `bash -c "$sweep_body" sweep <intra|mpi>`: intra executes the RUNS
# tokens of transport intra, mpi all the others. Progress counters live in
# DATA_OUT/.trace-state so the two phases count as one sweep.
read -r -d '' sweep_body <<'EOF' || true
set -uo pipefail
mode="$1"
bin_rel="benchmarks/lbm_lid_driven_cavity/carbon_benchmark_lbm_lid_driven_cavity"
if [[ "${mode}" == "intra" ]]; then
  bin="${B_INTRA}/${bin_rel}"
else
  bin="${B_MPI}/${bin_rel}"
  export OMPI_ALLOW_RUN_AS_ROOT=1 OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1
  export NCCL_IB_DISABLE=1
fi
trace="${DATA_OUT}/trace.log"
state="${DATA_OUT}/.trace-state"
failed_log="${DATA_OUT}/failed-${mode}.txt"

done_n=0 exec_n=0 exec_ms=0 n_ok=0 n_skip=0 n_redo=0 n_fail=0
if [[ -f "${state}" ]]; then
  while IFS='=' read -r k v; do
    if [[ "${k}" =~ ^(done_n|exec_n|exec_ms|n_ok|n_skip|n_redo|n_fail)$ &&
          "${v}" =~ ^[0-9]+$ ]]; then
      printf -v "${k}" '%s' "${v}"
    fi
  done < "${state}"
fi
save_state() {
  printf 'done_n=%s\nexec_n=%s\nexec_ms=%s\nn_ok=%s\nn_skip=%s\nn_redo=%s\nn_fail=%s\n' \
    "${done_n}" "${exec_n}" "${exec_ms}" "${n_ok}" "${n_skip}" "${n_redo}" "${n_fail}" \
    > "${state}"
}
fmt_dur() {
  local s="$1"
  if (( s >= 3600 )); then
    printf '%dh%02dm' $((s / 3600)) $((s % 3600 / 60))
  else
    printf '%dm%02ds' $((s / 60)) $((s % 60))
  fi
}
trace_line() {
  local width=30 fill pct elapsed eta="ETA --"
  fill=$((done_n * width / RUN_TOTAL))
  pct=$((done_n * 100 / RUN_TOTAL))
  elapsed=$(( $(date +%s) - SWEEP_START ))
  if (( exec_n > 0 )); then
    eta="ETA $(fmt_dur $(( exec_ms / exec_n * (RUN_TOTAL - done_n) / 1000 )))"
  fi
  printf '%s [%s%s] %*d/%d %3d%%  elapsed %s  %-11s | %-26s | %s\n' \
    "$(date '+%F %T')" "$(printf '%*s' "${fill}" '' | tr ' ' '#')" \
    "$(printf '%*s' $((width - fill)) '' | tr ' ' '.')" "${#RUN_TOTAL}" "${done_n}" \
    "${RUN_TOTAL}" "${pct}" "$(fmt_dur "${elapsed}")" "${eta}" "$1" "$2" >> "${trace}"
}

mode_runs=()
for r in ${RUNS}; do
  IFS=: read -r _ _ _ _ _ t _ _ _ _ <<< "${r}"
  if [[ "${mode}" == "intra" && "${t}" == "intra" ]] ||
     [[ "${mode}" != "intra" && "${t}" != "intra" ]]; then
    mode_runs+=("${r}")
  fi
done
if [[ ${#mode_runs[@]} -eq 0 ]]; then
  echo "=== ${mode} sweep: nothing to run ==="
  exit 0
fi
if [[ ! -x "${bin}" ]]; then
  echo "error: ${bin} not built (run without skip_build)" >&2
  echo "$(date '+%F %T') error: ${bin} not built" >> "${trace}"
  exit 1
fi

mkdir -p "${OUT_DIR}"
: > "${failed_log}"
iters=$((WARMUP + REPS * ITERS_PER_REP))
settings_match() {
  grep -Eq "\"iters\": *${iters}," "$1" && grep -Eq "\"warmup\": *${WARMUP}," "$1" &&
    grep -Eq "\"reps\": *${REPS}," "$1" && grep -Eq "\"overlap\": *\"${OVERLAP}\"" "$1"
}
run_log="$(mktemp)"

for r in "${mode_runs[@]}"; do
  IFS=: read -r lattice collision prec grid decomp t hp g n label <<< "${r}"
  name="${lattice}-${collision}-${prec}-${label}-n${n}-g${g}"
  report="${OUT_DIR}/${name}.json"
  redo=0
  if [[ -f "${report}" && "${FORCE}" -eq 0 ]]; then
    if settings_match "${report}"; then
      done_n=$((done_n + 1))
      n_skip=$((n_skip + 1))
      echo "==> [skip exists] ${name}.json"
      trace_line "skip (settings match)" "${name}"
      save_state
      continue
    fi
    redo=1
  fi
  extra=()
  [[ "${hp}" == "on" ]] && extra+=(--hp-halo)
  case "${grid}:${t}" in
    *:intra)  cmd=(env CUDA_VISIBLE_DEVICES="$(seq -s, 0 $((g - 1)))" "${bin}") ;;
    *_distributed:*)
      cmd=(mpiexec --bind-to none -n "${g}")
      [[ "${t}" == "mpip2p" ]] && cmd+=(-x CARBON_DIST_TRANSPORT=mpi)
      cmd+=("${bin}")
      extra+=(--decomposition "${decomp}" --gpus 1) ;;
    *:mpi)    cmd=(mpiexec -n "${g}" "${bin}") ;;
    *:mpip2p) cmd=(mpiexec -n "${g}" -x CARBON_DIST_TRANSPORT=mpi "${bin}") ;;
  esac
  if [[ "${t}" != "intra" && "${g}" -eq 1 ]]; then
    cmd=(env CUDA_VISIBLE_DEVICES=0 "${cmd[@]}")
  fi
  echo "==> [${t}] ${name}.json$([[ "${redo}" -eq 1 ]] && echo ' (settings differ, redo)')"
  t0=$(date +%s%N)
  timeout "${RUN_TIMEOUT}" "${cmd[@]}" \
    --grid "${grid}" --n "${n}" --iters "${iters}" --warmup "${WARMUP}" --reps "${REPS}" \
    --lattice "${lattice}" --collision "${collision}" --precision "${prec}" \
    --overlap "${OVERLAP}" "${extra[@]}" \
    --report-every 0 --label "${label}" --report "${report}" 2>&1 | tee "${run_log}"
  rc=${PIPESTATUS[0]}
  ms=$(( ($(date +%s%N) - t0) / 1000000 ))
  done_n=$((done_n + 1))
  exec_n=$((exec_n + 1))
  exec_ms=$((exec_ms + ms))
  secs="$((ms / 1000)).$((ms % 1000 / 100))s"
  if [[ "${rc}" -eq 0 && -f "${report}" ]]; then
    mlups="$(grep -oE '^MLUPS=[0-9.]+' "${run_log}" | tail -n 1 | cut -d= -f2)"
    if [[ "${redo}" -eq 1 ]]; then
      status=redo
      n_redo=$((n_redo + 1))
    else
      status=ok
      n_ok=$((n_ok + 1))
    fi
    trace_line "$(printf '%-4s %7s MLUPS %7s' "${status}" "${mlups%.*}" "${secs}")" "${name}"
  else
    kind=FAIL
    [[ "${rc}" -eq 124 ]] && kind=TIMEOUT
    n_fail=$((n_fail + 1))
    echo "!!> [failed] ${name}.json"
    rm -f "${report}"
    echo "${name}.json" >> "${failed_log}"
    trace_line "$(printf '%-7s rc=%-3s %7s' "${kind}" "${rc}" "${secs}")" "${name}"
  fi
  save_state
done
rm -f "${run_log}"
echo "=== ${mode} sweep complete ($(wc -l < "${failed_log}") failed) ==="
EOF

trace="${repo_root}/${DATA_DIR}/trace.log"
state="${repo_root}/${DATA_DIR}/.trace-state"
mkdir -p "${json_dir}"
export RUN_TOTAL="${#runs[@]}"
export SWEEP_START="$(date +%s)"
: > "${state}"
{
  echo "======================================================================"
  echo "$(date '+%F %T') sweep start: ${config_path#"${repo_root}/"} (runner ${RUNNER})"
  echo "    experiments: ${names[*]}"
  echo "    ${RUN_TOTAL} runs (${n_intra} Peeraccess, ${n_mpi} NCCL / MPI (P2P)),"
  echo "    ${n_existing} reports already in ${DATA_DIR}/json, force=${FORCE}"
  echo "    timing: warmup ${WARMUP}, ${REPS} x ${ITERS_PER_REP} steps, overlap ${OVERLAP}"
} >> "${trace}"
echo "==> progress: tail -f ${DATA_DIR}/trace.log"

if [[ "${n_intra}" -gt 0 ]]; then
  echo "==> Peeraccess runs (runner ${RUNNER})"
  in_env "${IMAGE_INTRA}" "${sweep_body}" intra
fi
if [[ "${n_mpi}" -gt 0 ]]; then
  echo "==> NCCL + MPI (P2P) runs (runner ${RUNNER})"
  in_env "${IMAGE_MPI}" "${sweep_body}" mpi
fi
if [[ "${RUNNER}" == "docker" ]]; then
  in_env "${IMAGE_INTRA}" "chmod -R a+rw /src/${DATA_DIR}"
fi

declare -A s=([n_ok]=0 [n_skip]=0 [n_redo]=0 [n_fail]=0)
while IFS='=' read -r k v; do
  [[ -n "${k}" ]] && s[${k}]="${v}"
done < "${state}"
rm -f "${state}"
took=$(( $(date +%s) - SWEEP_START ))
{
  printf '%s sweep done: %s ok, %s skipped, %s redone, %s failed; took %dh%02dm%02ds\n' \
    "$(date '+%F %T')" "${s[n_ok]}" "${s[n_skip]}" "${s[n_redo]}" "${s[n_fail]}" \
    $((took / 3600)) $((took % 3600 / 60)) $((took % 60))
  cat "${repo_root}/${DATA_DIR}"/failed-*.txt 2>/dev/null | sed 's/^/    FAILED: /' || true
} | tee -a "${trace}"
echo "==> reports in ${DATA_DIR}/json/; plot with plotting/plot_*.py --config ${config_path#"${repo_root}/"}"
