@echo off
rem Runs a cargo command with the CUDA backend, e.g.
rem   build-cuda.cmd build
rem   build-cuda.cmd test
rem   build-cuda.cmd run -- train --epochs 20
rem candle's kernels are compiled by nvcc, which on Windows needs the MSVC host compiler
rem (cl.exe) on PATH, so load the VS 2019 x64 environment first.
setlocal
call "D:\VS\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1 || (echo vcvars64.bat not found & exit /b 1)
if not defined CUDA_PATH set "CUDA_PATH=D:\CUDA\v12.9"
set "PATH=%CUDA_PATH%\bin;%PATH%"

set "SUBCOMMAND=%~1"
if not defined SUBCOMMAND set "SUBCOMMAND=build"
set "REST="
:collect
shift
if "%~1"=="" goto run
set "REST=%REST% %1"
goto collect
:run
cargo %SUBCOMMAND% --release --features cuda%REST%
