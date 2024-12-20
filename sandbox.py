import time
from pathlib import Path
from irtracking.camera.system import PointTracker, CameraParamsManager
import cv2
import threading

# Tracker System Prototype

# Get video files
footage_dir = Path("footage_calibration")
video_files = sorted(footage_dir.glob("*.mp4"))
print(video_files)

camera_params_manager = CameraParamsManager()
trackers = [PointTracker(i, camera_params_manager) for i in range(4)]

def process_video(video_file, tracker):
    cap = cv2.VideoCapture(str(video_file))
    frames = []       
        
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)

    cap.release()

    # Process the buffered frames
    frame_number = 0
    average_frame_processing_time = 0
    for frame in frames:
        start_time = time.time()

        # Feed the frame to the tracker

        # Get the tracked points
        tracked_points, frame_output = tracker.detect_points(frame)
        frame_output = cv2.cvtColor(frame_output, cv2.COLOR_GRAY2BGR)

        # Draw the tracked points
        for point, confidence in tracked_points:
            cv2.circle(frame_output, (int(point.x), int(point.y)), 2, (0, 255, 0), -1)

        # Display the frame
        cv2.imshow(f"Tracked Points {video_files.index(video_file)}", frame_output)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        end_time = time.time()
        frame_processing_time = end_time - start_time
        average_frame_processing_time += frame_processing_time
        #print(f"Video {video_files.index(video_file)} - Frame {frame_number} processing time: {frame_processing_time} seconds")
        frame_number += 1
        if frame_number == 200:
            break
    
    average_frame_processing_time /= frame_number
    print(f"Video {video_files.index(video_file)} - Average frame processing time: {average_frame_processing_time} seconds")

    cv2.destroyWindow(f"Tracked Points {video_files.index(video_file)}")
# Create and start threads for each video
threads = []
for i in range(len(video_files)):
    thread = threading.Thread(target=process_video, args=(video_files[i], trackers[i]))
    threads.append(thread)
    thread.start()

# Wait for all threads to finish
for thread in threads:
    thread.join()
