# import time
# from pathlib import Path
# from irtracking.system import PointDetector, CameraParamsManager
# import cv2
# import threading

# # Tracker System Prototype

# # Show the live tracking
# SHOW_LIVE_TRACKING = False
# FRAME_COUNT = 500 #Show the first x frames

# # Get video files
# footage_dir = Path("footage_calibration")
# video_files = sorted(footage_dir.glob("*.mp4"))
# print(video_files)

# camera_params_manager = CameraParamsManager()
# trackers = [PointDetector(i, camera_params_manager) for i in range(4)]
# # ... existing code ...

# import multiprocessing as mp
# from typing import List, Tuple
# import numpy as np

# def process_video(video_file, camera_id, params_manager):
#     # Create tracker in the subprocess since we can't pickle cv2.VideoCapture
#     tracker = PointDetector(camera_id, params_manager)
    
#     cap = cv2.VideoCapture(str(video_file))
#     frames = []       
    
#     # Buffer the frames
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break
#         frames.append(frame)

#     cap.release()

#     # Process the buffered frames
#     frame_number = 0
#     average_frame_processing_time = 0
#     for frame in frames:
#         start_time = time.time()

#         # Get the tracked points
#         tracked_points, frame_output = tracker.detect_points(frame)
#         frame_output = cv2.cvtColor(frame_output, cv2.COLOR_GRAY2BGR)

#         # Draw the tracked points
#         for point, confidence in tracked_points:
#             cv2.circle(frame_output, (int(point.x), int(point.y)), 2, (0, 255, 0), -1)

#         # Display the frame if needed
#         if SHOW_LIVE_TRACKING:
#             # Note: In multiprocessing, showing windows might not work as expected
#             # Consider using a queue to send frames back to main process for display
#             cv2.imshow(f"Tracked Points {camera_id}", frame_output)
#             if cv2.waitKey(1) & 0xFF == ord('q'):
#                 break

#         end_time = time.time()
#         frame_processing_time = end_time - start_time
#         average_frame_processing_time += frame_processing_time
#         frame_number += 1
#         if frame_number == FRAME_COUNT:
#             break
    
#     average_frame_processing_time /= frame_number
#     print(f"Video {camera_id} - Average frame processing time: {average_frame_processing_time} seconds")

#     if SHOW_LIVE_TRACKING:
#         cv2.destroyWindow(f"Tracked Points {camera_id}")

# if __name__ == '__main__':
#     # Get video files
#     footage_dir = Path("footage_calibration")
#     video_files = sorted(footage_dir.glob("calibration1_*.mp4"))
#     print(video_files)

#     camera_params_manager = CameraParamsManager()

#     # Create and start processes for each video
#     processes = []
#     for i in range(len(video_files)):
#         process = mp.Process(
#             target=process_video, 
#             args=(video_files[i], i, camera_params_manager)
#         )
#         processes.append(process)
#         process.start()

#     # Wait for all processes to finish
#     for process in processes:
#         process.join()
import rerun as rr
import multiprocessing
import time

def worker_process(recording_id):
    # Connect to the shared recording stream
    rr.init("multi_process_logging", spawn=True)
    rr.connect_tcp()

    # Log some data from the worker process
    for i in range(10):
        rr.log("worker_data", rr.Scalar(i))
        time.sleep(0.1)

if __name__ == "__main__":
    # Initialize the main recording stream
    rr.init("multi_process_logging", spawn=True)
    recording_id = rr.get_recording_id()  # Get the ID of the recording stream

    # Start worker processes
    processes = []
    for _ in range(4):  # Start 4 worker processes
        p = multiprocessing.Process(target=worker_process, args=(recording_id,))
        p.start()
        processes.append(p)

    # Log some data from the main process
    for i in range(10):
        rr.log("main_data", rr.Scalar(i))
        time.sleep(0.1)

    # Wait for all worker processes to finish
    for p in processes:
        p.join()

    print("All processes finished logging.")