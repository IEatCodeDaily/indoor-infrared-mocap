import os
import time
import cv2
from datetime import datetime
from pseyepy import Camera, cam_count
from threading import Thread
from queue import Queue

fps = 60
output_dir = 'footage_video'
frame_width = 640  # Adjust based on your camera resolution
frame_height = 480  # Adjust based on your camera resolution
duration = 10  # seconds

# Ensure the output directory exists
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Initialize the cameras
print("Initializing cameras")
try:
    c = Camera(fps=fps, resolution=Camera.RES_LARGE, gain=63, exposure=100)
    if cam_count() < 4:
        print("At least 4 cameras are required.")
        exit(1)
except Exception as e:
    print(f"An error occurred: {e}")
    exit(1)
print("Cameras initialized")

# Wait for the cameras to warm up
time.sleep(1)

# Shared buffer for frames and timestamps
buffer = [Queue(maxsize=100) for _ in range(4)]  # One queue per camera

def camera_read():
    start_time = time.time()
    while time.time() - start_time < duration:
        try:
            frames, timestamps = c.read()  # Read frames and timestamps
            for i in range(4):  # Assuming 4 cameras
                if not buffer[i].full():
                    buffer[i].put((frames[i], timestamps[i]))
        except Exception as e:
            print(f"An error occurred during camera read: {e}")
        time.sleep(0.005)  # Maintain the desired frame rate

def camera_write(camera_index):
    output_file = f'singlepoint_{camera_index + 1}.mp4'
    timestamp_file = f'singlepoint_{camera_index + 1}.txt'
    output_path = os.path.join(output_dir, output_file)
    timestamp_path = os.path.join(output_dir, timestamp_file)
    start_time = time.time()

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    try:
        print(f"Writing started for camera {camera_index}")

        # Open the timestamp file for writing
        with open(timestamp_path, 'w') as ts_file:
            while True:
                if not buffer[camera_index].empty():
                    frame, timestamp = buffer[camera_index].get()

                    # Write the timestamp to the file
                    ts_file.write(f"{timestamp}\n")

                    # Convert grayscale to BGR if needed and write the frame to the video file
                    # frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                    out.write(frame)
                elif time.time() - start_time >= duration and buffer[camera_index].empty():
                    break

        print(f"Writing finished for camera {camera_index}")

    except Exception as e:
        print(f"An error occurred for camera {camera_index}: {e}")

    finally:
        # Release resources
        out.release()
        print(f"Camera {camera_index} ended")

# Create and start the camera read thread
read_thread = Thread(target=camera_read)
read_thread.start()

# Create and start camera write threads for each camera
write_threads = []
for i in range(4):  # Assuming 4 cameras
    thread = Thread(target=camera_write, args=(i,))
    write_threads.append(thread)
    thread.start()

# Wait for all threads to finish
read_thread.join()
for thread in write_threads:
    thread.join()

# End the camera
c.end()
print("All cameras ended")