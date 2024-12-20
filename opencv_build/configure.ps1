# Load configs from env files
Write-Host "Loading configurations..."
$config = @{}

# Load opencv_build.env
Get-Content "opencv_build.env" | ForEach-Object {
    if ($_ -match '^([^=]+)=(.*)$') {
        $config[$matches[1]] = $matches[2].Replace('\', '/')
    }
}

Write-Host "`nAnalyzing Python environment..."
$pythonInfo = @{}

# Check if we're using Poetry based on USE_POETRY variable
$usePoetry = $false
if ($config.USE_POETRY -eq "true") {
    if (Get-Command poetry -ErrorAction SilentlyContinue) {
        Write-Host "Poetry found" -ForegroundColor Green
        $usePoetry = $true
    } else {
        Write-Host "Poetry not found, will use system Python" -ForegroundColor Yellow
        $continue = Read-Host "Continue with system Python? (Y/N)"
        if ($continue -ne "Y") {
            throw "Configuration cancelled by user"
        }
    }
}

if ($usePoetry) {
    $venvPath = (poetry env info -p).Replace('\', '/')
    $pythonInfo.Executable = (Join-Path $venvPath "Scripts/python.exe").Replace('\', '/')
} else {
    $pythonInfo.Executable = (Get-Command python).Source.Replace('\', '/')
}
Write-Host "Python executable: $($pythonInfo.Executable)"

# Call the Python script to get Python directories info
Write-Host "Running Python script to get directories info..."
& $pythonInfo.Executable ./scripts/get_python_dirs.py

if ($LASTEXITCODE -ne 0) {
    throw "Python script failed"
}

# Load Python info from JSON file
$pythonDirs = Get-Content -Raw -Path "scripts/get_python_dir.json" | ConvertFrom-Json
$pythonVersion = $pythonDirs.Version
$pythonInfo.LibDir = $pythonDirs.LibDir
$pythonInfo.IncludeDir = $pythonDirs.IncludeDir
$pythonInfo.PackagesPath = $pythonDirs.PackagesPath
$pythonInfo.NumPyInclude = $pythonDirs.NumPyInclude

Write-Host "Using paths:"
Write-Host "Executable: $($pythonInfo.Executable)"
Write-Host "Library dir: $($pythonInfo.LibDir)"
Write-Host "Include dir: $($pythonInfo.IncludeDir)"
Write-Host "Packages dir: $($pythonInfo.PackagesPath)"
Write-Host "NumPy include dir: $($pythonInfo.NumPyInclude)"

# Get CUDA info
$cudaInfo = @{}
$pathEnv = (Get-ChildItem -Path Env:CUDA_PATH*).Value -split ';'
$cudaPaths = $pathEnv | Where-Object { $_ -match 'CUDA\\v\d+\.\d+' }

if ($cudaPaths) {
    $availableCudaVersions = @()
    foreach ($cudaPath in $cudaPaths) {
        $nvccPath = Join-Path $cudaPath 'bin\nvcc.exe'
        if (Test-Path $nvccPath) {
            $nvccOutput = & "$nvccPath" --version
            $cudaVersion = ($nvccOutput | Select-String "release \d+\.\d+").Matches.Value -replace 'release '
            $availableCudaVersions += [PSCustomObject]@{ Path = $cudaPath; Version = $cudaVersion }
        }
    }

    # Output available CUDA versions
    Write-Host "Available CUDA versions:"
    $availableCudaVersions | ForEach-Object { Write-Host "Version: $($_.Version), Path: $($_.Path)" }

    # Check for specified CUDA version
    if ($config.CUDA_VERSION) {
        $specifiedCuda = $availableCudaVersions | Where-Object { $_.Version -eq $config.CUDA_VERSION }
        if ($specifiedCuda) {
            $cudaInfo.Path = ($specifiedCuda.Path).Replace('\', '/')
            $cudaInfo.Version = $specifiedCuda.Version
            Write-Host "Using specified CUDA version $($config.CUDA_VERSION) at $($cudaInfo.Path)"
        } else {
            Write-Host "Specified CUDA version $($config.CUDA_VERSION) not found. Falling back to the first available CUDA version." -ForegroundColor Yellow
            $cudaInfo.Path = ($availableCudaVersions[0].Path).Replace('\', '/')
            $cudaInfo.Version = $availableCudaVersions[0].Version
        }
    } else {
        $cudaInfo.Path = ($availableCudaVersions[0].Path).Replace('\', '/')
        $cudaInfo.Version = $availableCudaVersions[0].Version
    }

    Write-Host "CUDA Toolkit found at $($cudaInfo.Path) with version $($cudaInfo.Version)"
} else {
    throw "CUDA not found. Please ensure CUDA is installed and in PATH."
}

# Get cuDNN info
$cudnnInfo = @{}
if ($config.CUDNN_PATH) {
    $cudnnInfo.Path = $config.CUDNN_PATH.Replace('\', '/')
    $cudnnInfo.Library = (Join-Path $cudnnInfo.Path "lib/x64/cudnn.lib").Replace('\', '/')
    $cudnnInfo.IncludeDir = (Join-Path $cudnnInfo.Path "include").Replace('\', '/')
    Write-Host "cuDNN found at $($cudnnInfo.Path)"
} else {
    throw "CUDNN_PATH not found in environment variables."
}

# Build CMake paths
$cmakePaths = @{
    'OpenCV Repository' = $config.OPENCV_REPO_PATH
    'OpenCV Contrib' = "$($config.OPENCV_CONTRIB_PATH)/modules"
    'VCPKG Toolchain' = "$($config.VCPKG_ROOT)/scripts/buildsystems/vcpkg.cmake"
    'Python Executable' = $pythonInfo.Executable
    'Python Include' = $pythonInfo.IncludeDir
    'Python Packages' = $pythonInfo.PackagesPath
    'Python NumPy' = $pythonInfo.NumPyInclude
    'CUDA Toolkit' = $cudaInfo.Path
    'cuDNN Library' = $cudnnInfo.Library
    'cuDNN Include' = $cudnnInfo.IncludeDir
    'Eigen' = "$($config.VCPKG_ROOT)/installed/x64-windows/include/eigen3"
}

# Verify all paths
Write-Host "`nVerifying all required paths..."
$cmakePaths.GetEnumerator() | ForEach-Object {
    # Convert forward slashes back to backslashes for Test-Path
    $testPath = $_.Value.Replace('/', '\')
    Write-Host "Checking $($_.Key)... " -NoNewline
    if (Test-Path $testPath) {
        Write-Host "Found" -ForegroundColor Green
    } else {
        Write-Host "Not Found" -ForegroundColor Red
        throw "$($_.Key) not found at $($_.Value)"
    }
}

# Build CMake command
$cmakeArgs = @(
    "-S", $config.OPENCV_REPO_PATH
    "-B", "build"
    "-D", "CMAKE_TOOLCHAIN_FILE=$($config.VCPKG_ROOT)/scripts/buildsystems/vcpkg.cmake"
    "-D", "CMAKE_BUILD_TYPE=Release"
    "-D", "CMAKE_INSTALL_PREFIX=install"
    "-D", "PYTHON3_EXECUTABLE=$($pythonInfo.Executable)"
    "-D", "OPENCV_EXTRA_MODULES_PATH=$($config.OPENCV_CONTRIB_PATH)/modules"
    "-D", "HAVE_opencv_python3=ON"
    "-D", "PYTHON3_LIBRARY=$($pythonInfo.LibDir)/python$pythonVersion.lib"
    "-D", "OPENCV_FORCE_PYTHON_LIBS=ON"
    "-D", "PYTHON3_INCLUDE_DIR=$($pythonInfo.IncludeDir)"
    "-D", "PYTHON3_PACKAGES_PATH=$($pythonInfo.PackagesPath)"
    "-D", "PYTHON3_NUMPY_INCLUDE_DIRS=$($pythonInfo.NumPyInclude)"
    "-D", "CUDA_TOOLKIT_ROOT_DIR=$($cudaInfo.Path)"
    "-D", "WITH_CUDA=ON"
    "-D", "WITH_CUDNN=ON"
    "-D", "OPENCV_DNN_CUDA=1"
    "-D", "CUDA_ARCH_BIN=$($cudaInfo.Arch)"
    "-D", "CUDA_FAST_MATH=1"
    "-D", "CUDNN_LIBRARY=$($cudnnInfo.Library)"
    "-D", "CUDNN_INCLUDE_DIR=$($cudnnInfo.IncludeDir)"
)

Write-Host "Configuring OpenCV build with CUDA $($cudaInfo.Version) and cuDNN..."
& cmake $cmakeArgs

if ($LASTEXITCODE -ne 0) {
    throw "CMake configuration failed"
}

Write-Host "`nOpenCV build configured successfully!"
# EOF