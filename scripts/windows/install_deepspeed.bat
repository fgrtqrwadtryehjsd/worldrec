@echo off
setlocal
echo === Step 1: activate conda env worldrec ===
call "E:\Soft\anaconda3\Scripts\activate.bat" worldrec
if errorlevel 1 (
  echo [ERR] conda activate failed
  exit /b 1
)

echo === Step 2: vcvars64 ===
call "C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul
if errorlevel 1 (
  echo [ERR] vcvars64 failed
  exit /b 1
)

echo === Step 3: CUDA_HOME + DS_BUILD env ===
set "CUDA_HOME=E:\Soft\anaconda3\envs\worldrec"
set "PATH=%CUDA_HOME%\bin;%PATH%"
set "DISTUTILS_USE_SDK=1"
set "PYTORCH_NVCC_FLAGS=-DCUDA_HAS_FP16=1 -D__CUDA_NO_HALF_OPERATORS__ -D__CUDA_NO_HALF_CONVERSIONS__ -D__CUDA_NO_HALF2_OPERATORS__"
set "DS_BUILD_AIO=0"
set "DS_BUILD_CFILE=0"
set "DS_BUILD_GDS=0"
set "DS_BUILD_EVOFORMER_ATTN=0"
set "DS_BUILD_FP_QUANTIZER=0"
set "DS_BUILD_INFERENCE_CUTLASS=0"
set "DS_BUILD_TRANSFORMER_INFERENCE=0"

echo === Step 4: verify cl / nvcc ===
where cl
where nvcc

echo === Step 5: pip install deepspeed (try 0.14.5 first) ===
cd /d "E:\zdm\worldrec\worldrec"
pip install deepspeed==0.14.5 --no-build-isolation
echo "PIP_EXIT=%ERRORLEVEL%"
endlocal
