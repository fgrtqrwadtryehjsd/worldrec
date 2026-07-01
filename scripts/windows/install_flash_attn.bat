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

echo === Step 3: CUDA_HOME + env ===
set "CUDA_HOME=E:\Soft\anaconda3\envs\worldrec"
set "PATH=%CUDA_HOME%\bin;%PATH%"
set "DISTUTILS_USE_SDK=1"
set "MAX_JOBS=4"

echo === Step 4: pip install flash-attn (try pre-built wheel) ===
cd /d "E:\zdm\worldrec\worldrec"
pip install flash-attn --no-build-isolation
echo "PIP_EXIT=%ERRORLEVEL%"
endlocal
