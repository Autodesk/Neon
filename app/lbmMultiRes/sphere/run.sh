#!/bin/bash
exe="../../../build/bin/app-lbmMultiRes"
numIter=50
deviceId=7

for scale in 2 4 6 8 10; do
for dataType in "float" "double"; do
for collideOption in "--storeCoarse" "--storeFine" "--collisionFusedStore" "--fusedFinest --collisionFusedStore"; do
for streamOption in " " "--streamFusedExpl" "--streamFusedCoal" "--streamFuseAll"; do
echo ${exe} --numIter $numIter --deviceType gpu --deviceId $deviceId --problemType sphere --benchmark --dataType $dataType --re 100 --scale $scale $collideOption $streamOption
${exe} --numIter $numIter --deviceType gpu --deviceId $deviceId --problemType sphere --benchmark --dataType $dataType --re 100 --scale $scale $collideOption $streamOption
done
done	
done 
done