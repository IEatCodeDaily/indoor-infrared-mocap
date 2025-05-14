import os
import time
from datetime import datetime
from pseyepy import Camera, Stream

fps = 60
duration = 120  # seconds
output_dir = 'footage_video'
output_file = 'calibration_20250514.mp4'

# Ensure the output directory exists
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# initialize all connected cameras
print("Initializing camera")
try:
    c = Camera(fps=fps, resolution=Camera.RES_LARGE, gain=50, exposure=100)
except Exception as e:
    print(f"An error occurred: {e}")
print("Camera initialized")

# wait for the camera to warm up
time.sleep(1)

try:
    # begin saving data to files
    s = Stream(c, file_name=os.path.join(output_dir, output_file), fps=fps)
    print("Stream initialized")

    # Run the stream for a specific duration (e.g., 2 seconds)
    start_time = time.time()

    while time.time() - start_time < duration:
        time.sleep(0.1)  # Adjust the sleep time as needed

    # when finished, close the stream
    print("Stream ending")
    s.end()
    print("Stream ended")

except Exception as e:
    print(f"An error occurred: {e}")

finally:
    c.end()
    print("Camera ended")