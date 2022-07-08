#!/bin/sh
#VerifyNeonPRUnix.sh ID 2>&1 | tee NeonUnix.log
if [ "$#" -ne 1 ]; then
    echo "******************************"
    echo "Pull Request ID is missing. Usage: "
    echo ">> VerifyNeonPRUnix.sh ID"
    echo "******************************"
    exit
fi
starting_dir="$(cd "$(dirname "$0")" && pwd)"
mkdir temp
cd temp
PR=$1
git clone https://github.com/Autodesk/Neon.git
cd Neon
git fetch origin refs/pull/$PR/head:pull_$PR
git checkout  pull_$PR
mkdir build
cd build
cmake ..
cmake --build . --config Release -j 10
ctest_filename=CTestNeonUnixReport.log
ctest --no-compress-output --output-on-failure -T Test --build-config Release --output-log $ctest_filename
echo "******************************"
echo "Test final report location: $starting_dir/temp/Neon/build/$ctest_filename"
echo "******************************"
cd $starting_dir