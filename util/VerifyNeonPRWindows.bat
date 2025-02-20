@echo off
IF %1.==. GOTO Usage

set starting_dir=%CD%
mkdir temp
cd temp
set PR=%1
git clone https://github.com/Autodesk/Neon.git
cd Neon
git fetch origin refs/pull/%PR%/head:pull_%PR%
git checkout  pull_%PR%
mkdir build
cd build
cmake -G "Visual Studio 16 2019" ..
cmake --build . --config Release -j 10
set ctest_filename=CTestNeonWindowsReport.log
ctest --no-compress-output --output-on-failure -T Test --build-config Release --output-log %ctest_filename%
echo "******************************"
echo "Test final report location: %starting_dir%\temp\Neon\build\%ctest_filename%"
echo "******************************"
cd %starting_dir%
GOTO end

:Usage
  echo "******************************"
  echo "Pull Request ID is missing. Usage: "
  echo ">> VerifyNeonPRWindows.bat ID"
  echo "******************************"
:end