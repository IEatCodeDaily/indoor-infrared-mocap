# OpenCV Build Script with Poetry
Note: Only tested for windows

# File Structure
your file structure should be like this
```
C:/path/to/your/project/              # Your project root
├── opencv_build/
│   ├── build/                     # Build output
│   ├── install/                   # Install output
│   ├── configure.bat              # CMake configuration
│   ├── build.bat                  # Build script
│   └── opencv_paths.env           # OpenCV paths configuration
├── src/
├── tests/
├── pyproject.toml
└── .env                          # Project-wide environment vars
```

# Environment Variables
set `USE_POETRY` to `true` if you want to install it with poetry. set to `false` if you want to install it on system's Python. it will refer to whatever the `python` command executable resolve to.

# Steps
1. Configure `opencv_build.env` with all the paths needed
2. Run `configure.ps1`
3. Run `build.ps1`