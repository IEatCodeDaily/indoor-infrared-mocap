@echo off
setlocal enabledelayedexpansion

:: DEPRECATED WARNING
echo WARNING: This script is deprecated and will be removed in future versions.
echo Please use the new build scripts located in the build directory.

:: ===== Python Check =====

:: First check if we're in a Poetry environment
for /f "tokens=*" %%i in ('poetry env info -p 2^>nul') do set POETRY_VENV=%%i
if defined POETRY_VENV (
    :: Use Poetry's Python
    set PYTHON_EXE=%POETRY_VENV%/Scripts/python.exe
    echo Using Poetry virtual environment Python at: %PYTHON_EXE%
) else (
    :: Fallback to pyenv
    for /f "tokens=*" %%i in ('where pyenv 2^>nul') do set PYENV_PATH=%%i
    if defined PYENV_PATH (
        for /f "tokens=*" %%i in ('pyenv which python') do set PYTHON_EXE=%%i
        if not defined PYTHON_EXE (
            echo Error: Python executable not found via pyenv.
            exit /b 1
        )
    ) else (
        :: Final fallback to system Python
        for /f "tokens=*" %%i in ('where python 2^>nul') do set PYTHON_EXE=%%i
        if not defined PYTHON_EXE (
            echo Error: Python executable not found.
            exit /b 1
        )
    )
)

echo Python executable: %PYTHON_EXE%

:: Get Python paths using the selected Python executable
%PYTHON_EXE% -c "from distutils.sysconfig import get_config_var; print(get_config_var('LIBDIR'))" > tmp_libdir.txt
set /p PYTHON_LIBDIR=<tmp_libdir.txt
echo Python library directory: %PYTHON_LIBDIR%

%PYTHON_EXE% -c "from distutils.sysconfig import get_python_inc; print(get_python_inc())" > tmp_incdir.txt
set /p PYTHON_INCDIR=<tmp_incdir.txt
echo Python include directory: %PYTHON_INCDIR%

%PYTHON_EXE% -c "from distutils.sysconfig import get_python_lib; print(get_python_lib())" > tmp_packages.txt
set /p PYTHON_PACKAGES=<tmp_packages.txt
echo Python packages directory: %PYTHON_PACKAGES%

:: Clean up temporary files
del tmp_libdir.txt tmp_incdir.txt tmp_packages.txt

:: ===== OpenCV Build Environment =====

:: Load environment variables from opencv_paths.env
for /f "tokens=*" %%a in (opencv_build.env) do set %%a

:: Verify paths exist
if not exist "%OPENCV_REPO_PATH%" (
    echo Error: OpenCV repository not found at %OPENCV_REPO_PATH%
    exit /b 1
)
if not exist "%OPENCV_CONTRIB_PATH%" (
    echo Error: OpenCV contrib repository not found at %OPENCV_CONTRIB_PATH%
    exit /b 1
)

:: ===== CUDA Check =====

:: Detect CUDA path from nvcc location
for /f "tokens=*" %%i in ('where nvcc 2^>nul') do (
    set NVCC_PATH=%%i
    :: Extract CUDA path (two directories up from nvcc.exe)
    for %%j in ("!NVCC_PATH!\..\..") do set CUDA_PATH=%%~fj
)

if not defined CUDA_PATH (
    echo Error: CUDA not found! Please ensure CUDA is installed and in PATH
    exit /b 1
)

:: Convert CUDA_PATH backslashes to forward slashes
set CUDA_PATH=%CUDA_PATH:\=/%

:: Get CUDA version
for /f "tokens=* usebackq" %%i in (`nvcc --version ^| findstr "release"`) do (
    set VERSION_LINE=%%i
)
for /f "tokens=5 delims=, " %%a in ("!VERSION_LINE!") do set CUDA_VERSION=%%a
echo Found CUDA %CUDA_VERSION% at %CUDA_PATH%

:: Verify required CUDA libraries exist
if not exist "!CUDA_PATH!\lib\x64\cudart_static.lib" (
    echo Error: cudart_static.lib not found
    exit /b 1
)

:: ===== Vcpkg Check =====

:: Verify vcpkg paths for Eigen, Gflags, Glog, Ceres Solver, and VTK
set EIGEN_DIR=%VCPKG_ROOT%/installed/x64-windows/include/eigen3
set GFLAGS_INCLUDE_DIR=%VCPKG_ROOT%/installed/x64-windows/include
set GFLAGS_LIBRARY_DIR=%VCPKG_ROOT%/installed/x64-windows/lib
set GLOG_INCLUDE_DIR=%VCPKG_ROOT%/installed/x64-windows/include
set GLOG_LIBRARY_DIR=%VCPKG_ROOT%/installed/x64-windows/lib

:: Check for Eigen, Glog, Gflags, Ceres Solver, and VTK availability
if not exist "%EIGEN_DIR%" (
    echo Error: Eigen directory not found at %EIGEN_DIR%. Please install Eigen via vcpkg.
    exit /b 1
)
if not exist "%GFLAGS_INCLUDE_DIR%" (
    echo Error: Gflags directory not found at %GFLAGS_DIR%. Please install Gflags via vcpkg.
    exit /b 1
)
if not exist "%GLOG_INCLUDE_DIR%" (
    echo Error: Glog directory not found at %GLOG_DIR%. Please install Glog via vcpkg.
    exit /b 1
)

:: Run cmake with captured paths
echo Configuring Cmake
cmake -S "%OPENCV_REPO_PATH%" -B build ^
    -D CMAKE_TOOLCHAIN_FILE="%VCPKG_ROOT%/scripts/buildsystems/vcpkg.cmake" ^
    -D CMAKE_BUILD_TYPE=Release ^
    -D CMAKE_INSTALL_PREFIX=install ^
    -D PYTHON3_EXECUTABLE="%PYTHON_EXE%" ^
    -D OPENCV_EXTRA_MODULES_PATH="%OPENCV_CONTRIB_PATH%/modules" ^
    -D HAVE_opencv_python3=ON ^
    -D PYTHON3_LIBRARY="%PYTHON_LIBDIR%/python310.lib" ^
    -D PYTHON3_INCLUDE_DIR="%PYTHON_INCDIR%" ^
    -D PYTHON3_PACKAGES_PATH="%PYTHON_PACKAGES%" ^
    -D PYTHON3_NUMPY_INCLUDE_DIRS="%PYTHON_PACKAGES%/numpy/core/include" ^
    -D CUDA_TOOLKIT_ROOT_DIR="%CUDA_PATH%" ^
    -D CUDA_SDK_ROOT_DIR="%CUDA_PATH%" ^
    -D WITH_CUDA=ON ^
    -D WITH_CUDNN=ON ^
    -D OPENCV_DNN_CUDA=ON ^
    -D CUDA_ARCH_BIN="%CUDA_ARCH%" ^
    -D CUDA_FAST_MATH=ON ^
    -D BUILD_opencv_python3=ON ^
    -D BUILD_opencv_python2=OFF
