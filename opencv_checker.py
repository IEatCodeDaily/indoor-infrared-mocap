import cv2
import numpy as np

# Print OpenCV version
print(f"OpenCV Version: {cv2.__version__}")

# Check if CUDA is available
print(f"CUDA enabled: {cv2.cuda.getCudaEnabledDeviceCount() > 0}")

# Check for SFM-related modules
sfm_modules = {
    'sfm': hasattr(cv2, 'sfm'),
    'features2d': hasattr(cv2, 'features2d'),
    'xfeatures2d': hasattr(cv2, 'xfeatures2d')
}
print("\nSFM-related modules:")
for module, available in sfm_modules.items():
    print(f"{module}: {'Available' if available else 'Not available'}")

# Try to import specific SFM functionality
try:
    reconstruct = cv2.sfm.reconstruct
    print("\nSFM reconstruction available")
except AttributeError:
    print("\nSFM reconstruction not available")

# Check build information
print("\nBuild information:")
print(cv2.getBuildInformation())