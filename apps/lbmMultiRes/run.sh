#!/bin/bash
exe="../../build/bin/app-lbmMultiRes"
numIter=50
deviceId=7

for scale in 1 2 3 4 5 6 7 8 9; do
for dataType in "float" "double"; do
for collideOption in "--storeCoarse" "--storeFine" "--collisionFusedStore" "--fusedFinest --collisionFusedStore"; do
for streamOption in " " "--streamFusedExpl" "--streamFusedCoal" "--streamFuseAll"; do
echo ${exe} --numIter $numIter --deviceType gpu --deviceId $deviceId --problemType lid $problemId --benchmark --dataType $dataType --storeCoarse --re 100 --scale $scale $collideOption $streamOption
${exe} --numIter $numIter --deviceType gpu --deviceId $deviceId --problemType lid $problemId --benchmark --dataType $dataType --storeCoarse --re 100 --scale $scale $collideOption $streamOption
done
done
done 
done