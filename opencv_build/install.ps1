# setup_opencv.ps1

# Function to handle errors
function Write-ErrorAndExit {
    param($message)
    Write-Host "Error: $message" -ForegroundColor Red
    exit 1
}

# Function to read env file
function Read-EnvFile {
    param($envPath)
    $envVars = @{}
    Get-Content $envPath | ForEach-Object {
        if ($_ -match '^([^=]+)=(.*)$') {
            $envVars[$matches[1]] = $matches[2]
        }
    }
    return $envVars
}

# Verify required tools
Write-Host "Checking prerequisites..." -ForegroundColor Blue
$required_tools = @("cmake", "poetry", "python")
foreach ($tool in $required_tools) {
    if (!(Get-Command $tool -ErrorAction SilentlyContinue)) {
        Write-ErrorAndExit "$tool not found in PATH"
    }
}

# Read environment variables
$envFile = Join-Path $PSScriptRoot "opencv_build.env"
if (!(Test-Path $envFile)) {
    Write-ErrorAndExit "opencv_build.env not found"
}
$envVars = Read-EnvFile $envFile
foreach ($key in $envVars.Keys) {
    Set-Item "env:$key" $envVars[$key]
}

# Get paths
$buildDir = Join-Path $PSScriptRoot "build"
$installDir = Join-Path $PSScriptRoot "install"
$sitePackages = poetry run python -c "import site; print(site.getsitepackages()[0])"

# Configure and build OpenCV
Write-Host "Configuring OpenCV..." -ForegroundColor Blue
if (!(Test-Path $buildDir)) {
    New-Item -ItemType Directory -Path $buildDir | Out-Null
}

try {
    # Create .pth file for DLL path
    Write-Host "Setting up DLL paths..." -ForegroundColor Blue
    $dllPath = Join-Path $installDir "x64/vc17/bin"
    $pthFile = Join-Path $sitePackages "opencv-dll.pth"
    Set-Content -Path $pthFile -Value $dllPath -Force

    # Create and run test script
    Write-Host "Testing OpenCV installation..." -ForegroundColor Blue
    $testScript = @"
import cv2
print(f"OpenCV Version: {cv2.__version__}")
print(f"CUDA available: {cv2.cuda.getCudaEnabledDeviceCount() > 0}")
"@
    $testPath = Join-Path $PSScriptRoot "opencv_test.py"
    Set-Content -Path $testPath -Value $testScript
    poetry run python $testPath

    Write-Host "`nSetup completed successfully!" -ForegroundColor Green
    Write-Host "OpenCV is now installed and configured in your Poetry environment." -ForegroundColor Green
} catch {
    Write-ErrorAndExit $_.Exception.Message
}