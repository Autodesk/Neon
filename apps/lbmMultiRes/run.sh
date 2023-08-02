#!/bin/bash
exe="../../build/bin/app-lbmMultiRes"
numIter=50
deviceId=7

for problemId in 1 2 3 4 5 6 7 8 9; do
for dataType in "float" "double"; do
echo ${exe} --numIter $numIter --deviceType gpu --deviceId $deviceId --problemId $problemId --benchmark --dataType $dataType --storeCoarse
${exe} --numIter $numIter --deviceType gpu --deviceId $deviceId --problemId $problemId --benchmark --dataType $dataType --storeCoarse

echo ${exe} --numIter $numIter --deviceType gpu --deviceId $deviceId --problemId $problemId --benchmark --dataType $dataType  --storeFine
${exe} --numIter $numIter --deviceType gpu --deviceId $deviceId --problemId $problemId --benchmark --dataType $dataType  --storeFine

echo ${exe} --numIter $numIter --deviceType gpu --deviceId $deviceId --problemId $problemId --benchmark --dataType $dataType  --storeFine  --collisionFusedStore
${exe} --numIter $numIter --deviceType gpu --deviceId $deviceId --problemId $problemId --benchmark --dataType $dataType  --storeFine  --collisionFusedStore

echo ${exe} --numIter $numIter --deviceType gpu --deviceId $deviceId --problemId $problemId --benchmark --dataType $dataType  --collisionFusedStore
${exe} --numIter $numIter --deviceType gpu --deviceId $deviceId --problemId $problemId --benchmark --dataType $dataType  --collisionFusedStore

echo ${exe} --numIter $numIter --deviceType gpu --deviceId $deviceId --problemId $problemId --benchmark --dataType $dataType  --collisionFusedStore  --streamFusedExpl 
${exe} --numIter $numIter --deviceType gpu --deviceId $deviceId --problemId $problemId --benchmark --dataType $dataType  --collisionFusedStore  --streamFusedExpl 

echo ${exe} --numIter $numIter --deviceType gpu --deviceId $deviceId --problemId $problemId --benchmark --dataType $dataType  --collisionFusedStore  --streamFusedCoal
${exe} --numIter $numIter --deviceType gpu --deviceId $deviceId --problemId $problemId --benchmark --dataType $dataType  --collisionFusedStore  --streamFusedCoal

echo ${exe} --numIter $numIter --deviceType gpu --deviceId $deviceId --problemId $problemId --benchmark --dataType $dataType  --collisionFusedStore  --streamFuseAll
${exe} --numIter $numIter --deviceType gpu --deviceId $deviceId --problemId $problemId --benchmark --dataType $dataType  --collisionFusedStore  --streamFuseAll
done 
done