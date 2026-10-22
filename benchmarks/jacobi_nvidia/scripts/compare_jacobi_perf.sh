#!/usr/bin/env bash
# Run NVIDIA single_gpu Jacobi and carbon jacobi_nvidia at the same size; print side-by-side CSV.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
NV_JACOBI="${ROOT}/multi-gpu-programming-models/single_gpu/jacobi"
CARBON_JACOBI="${ROOT}/carbon/build/benchmarks/jacobi_nvidia/carbon_benchmark_jacobi_nvidia"

NX="${NX:-4096}"
NY="${NY:-4096}"
NITER="${NITER:-100}"
NCCHECK="${NCCHECK:-1}"
GPUS="${GPUS:-1}"

if [[ ! -x "${NV_JACOBI}" ]]; then
  echo "Build NVIDIA reference first: make -C ${ROOT}/multi-gpu-programming-models/single_gpu" >&2
  exit 1
fi
if [[ ! -x "${CARBON_JACOBI}" ]]; then
  echo "Build carbon benchmark first (cmake -DCARBON_BUILD_BENCHMARKS=ON)." >&2
  exit 1
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export SPDLOG_LEVEL="${SPDLOG_LEVEL:-error}"

NV_LINE="$("${NV_JACOBI}" -csv -nx "${NX}" -ny "${NY}" -niter "${NITER}" -nccheck "${NCCHECK}")"
CB_LINE="$("${CARBON_JACOBI}" -csv -nx "${NX}" -ny "${NY}" -niter "${NITER}" -nccheck "${NCCHECK}" -gpus "${GPUS}" --skip-validation 2>&1 | grep '^carbon_jacobi' | tail -1)"

echo "type,nx,ny,niter,nccheck,runtime_seconds"
echo "${NV_LINE}" | awk -F',' '{printf "nvidia_single_gpu,%s,%s,%s,%s,%s\n",$2,$3,$4,$5,$6}'
echo "${CB_LINE}" | awk -F',' '{printf "carbon,%s,%s,%s,%s,%s\n",$2,$3,$4,$5,$6}'

NV_TIME="$(echo "${NV_LINE}" | awk -F',' '{print $6}')"
CB_TIME="$(echo "${CB_LINE}" | awk -F',' '{print $6}')"
python3 - <<PY
nv=float("${NV_TIME}")
cb=float("${CB_TIME}")
print(f"speedup_vs_nvidia_single={nv/cb:.4f}")
PY
