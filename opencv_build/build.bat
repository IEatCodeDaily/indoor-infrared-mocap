:: DEPRECATED: This script is deprecated and will be removed in future versions.

@echo off
setlocal enabledelayedexpansion

:: DEPRECATED WARNING
echo WARNING: This script is deprecated and will be removed in future versions.
echo Please use the new build scripts located in the build directory.

:: Parse command line arguments
set BUILD_DEBUG=0
:parse_args
if "%1"=="" goto end_parse
if "%1"=="--debug" set BUILD_DEBUG=1
if "%1"=="-d" set BUILD_DEBUG=1
shift
goto parse_args
:end_parse

:: Get number of processors and use half
for /f "tokens=2 delims==" %%i in ('wmic cpu get NumberOfLogicalProcessors /value') do set TOTAL_PROCESSORS=%%i
set /a NUM_PROCESSORS=%TOTAL_PROCESSORS% / 2
if %NUM_PROCESSORS% LEQ 0 set NUM_PROCESSORS=1
echo Building with %NUM_PROCESSORS% out of %TOTAL_PROCESSORS% processors

:: Build Release configuration
echo Building Release configuration...
cmake --build build --config Release --parallel %NUM_PROCESSORS%
if errorlevel 1 (
    echo Release build failed!
    exit /b 1
)

:: Build Debug if flag is set
if %BUILD_DEBUG%==1 (
    echo Building Debug configuration...
    cmake --build build --config Debug --parallel %NUM_PROCESSORS%
    if errorlevel 1 (
        echo Debug build failed!
        exit /b 1
    )
)

:: Install Release
echo Installing Release configuration...
cmake --install build --config Release
if errorlevel 1 (
    echo Release installation failed!
    exit /b 1
)

:: Install Debug if built
if %BUILD_DEBUG%==1 (
    echo Installing Debug configuration...
    cmake --install build --config Debug
    if errorlevel 1 (
        echo Debug installation failed!
        exit /b 1
    )
)

echo.
echo Build and installation completed successfully!
if %BUILD_DEBUG%==1 (
    echo Built configurations: Release, Debug
) else (
    echo Built configurations: Release
)

exit /b 0