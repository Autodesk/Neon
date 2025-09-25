set -x

DOMAIN_SIZE_LIST="64 128 192 256 320 384 448 512"
GRID_LIST="dGrid bGrid eGrid"
STORAGE_FP_LIST="double float"
COMPUTE_FP_LIST="double float"
OCC="nOCC"

for DOMAIN_SIZE in ${DOMAIN_SIZE_LIST}; do
  for STORAGE_FP in ${STORAGE_FP_LIST}; do
    for COMPUTE_FP in ${COMPUTE_FP_LIST}; do
      for GRID in ${GRID_LIST}; do

        if [ "${STORAGE_FP}_${COMPUTE_FP}" = "double_float" ]; then
          continue
        fi

        echo ./lbm-lid-driven-cavity-flow \
          --deviceType gpu --deviceIds 0 \
          --grid "${GRID}" \
          --domain-size "${DOMAIN_SIZE}" \
          --warmup-iter 10 --max-iter 100 --repetitions 5 \
          --report-filename "lbm-lid-driven-cavity-flow_${DOMAIN_SIZE}_${GRID}_STORAGE_${STORAGE_FP}_COMPUTE_${COMPUTE_FP}" \
          --computeFP "${COMPUTE_FP}" \
          --storageFP "${STORAGE_FP}" \
          --${OCC} --benchmark
      done
    done
  done
done
