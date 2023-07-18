#!/bin/bash
exe="../../build/bin/app-lbmMultiRes"
numIter=50
deviceId=7

for problemId in 1 2 3 4 5 6 7 8 9 10; do
for dataType in "float" "double"; do
echo ${exe} --numIter $numIter --deviceType gpu --deviceId $deviceId --problemId $problemId --benchmark --dataType $dataType
${exe} --numIter $numIter --deviceType gpu --deviceId $deviceId --problemId $problemId --benchmark --dataType $dataType
echo ${exe} --numIter $numIter --deviceType gpu --deviceId $deviceId --problemId $problemId --benchmark --collisionFusedStore --streamFusedExpl --dataType $dataType
${exe} --numIter $numIter --deviceType gpu --deviceId $deviceId --problemId $problemId --benchmark --collisionFusedStore --streamFusedExpl --dataType $dataType
done 
done 