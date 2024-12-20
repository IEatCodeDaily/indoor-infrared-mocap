param (
    [switch]$Debug
)

# Parse command line arguments
$BUILD_DEBUG = if ($Debug) { 1 } else { 0 }

# Get number of processors and use half
$TOTAL_PROCESSORS = (Get-WmiObject -Class Win32_ComputerSystem).NumberOfLogicalProcessors
$NUM_PROCESSORS = [math]::Max([math]::Floor($TOTAL_PROCESSORS * .75), 1)
Write-Output "Building with $NUM_PROCESSORS out of $TOTAL_PROCESSORS processors"

# Build Release configuration
Write-Output "Building Release configuration..."
cmake --build build --config Release --parallel $NUM_PROCESSORS
if ($LASTEXITCODE -ne 0) {
    Write-Output "Release build failed!"
    exit 1
}

# Build Debug if flag is set
if ($BUILD_DEBUG -eq 1) {
    Write-Output "Building Debug configuration..."
    cmake --build build --config Debug --parallel $NUM_PROCESSORS
    if ($LASTEXITCODE -ne 0) {
        Write-Output "Debug build failed!"
        exit 1
    }
}

# Install Release
Write-Output "Installing Release configuration..."
cmake --install build --config Release
if ($LASTEXITCODE -ne 0) {
    Write-Output "Release installation failed!"
    exit 1
}

# Install Debug if built
if ($BUILD_DEBUG -eq 1) {
    Write-Output "Installing Debug configuration..."
    cmake --install build --config Debug
    if ($LASTEXITCODE -ne 0) {
        Write-Output "Debug installation failed!"
        exit 1
    }
}

Write-Output ""
Write-Output "Build and installation completed successfully!"
if ($BUILD_DEBUG -eq 1) {
    Write-Output "Built configurations: Release, Debug"
} else {
    Write-Output "Built configurations: Release"
}

exit 0