import cv2
import numpy as np
import queue
from .params import CameraParamsManager, IntrinsicParams, ProcessFlags
from typing import List, Optional, Tuple
import multiprocessing
from multiprocessing import Process, Queue, Event
from multiprocessing.managers import SyncManager
from dataclasses import dataclass
import time

@dataclass
class Point2D:
    x: float
    y: float
    
    @classmethod
    def to_ndarray(cls, points: List['Point2D']) -> np.ndarray:
        return np.array([[point.x, point.y] for point in points], dtype=np.float32)

@dataclass
class Point3D:
    x: float
    y: float
    z: float

class PointDetector:
    def __init__(self, camera_id: int, manager: SyncManager, flags: ProcessFlags):
        self.camera_id = camera_id
        self.morph_kernel = np.array([[0, 1, 0],[1, 1, 1],[0, 1, 0]], np.uint8)

        # Queue structure:
        # input_queue: (timestamp, camera_id, IntrinsicParams, ExtrinsicParams, frame)
        # output_queue: (timestamp, camera_id, IntrinsicParams, ExtrinsicParams, points, processed_frame)
        # viz_queue: same as output_queue but non-blocking
        self.input_queue = Queue(maxsize=10)
        self.output_queue = Queue(maxsize=10)
        self.viz_queue = Queue(maxsize=10)

        self._running = Event()
        self.track_process = None  # Initialize to None, will create when starting
        
        # Initialize timing stats with shared memory
        self.timing_stats = manager.dict({
            'undistort': manager.list(),
            'point_detection': manager.list(),
            'total_processing': manager.list()
        })
        self.max_stats_samples = 100  # Keep last 100 samples

        self.flags = flags  # Store the shared flags instance

    def _update_timing(self, category: str, duration: float):
        """Update timing statistics for a category"""
        stats_list = self.timing_stats[category]
        stats_list.append(duration)
        if len(stats_list) > self.max_stats_samples:
            stats_list.pop(0)
        self.timing_stats[category] = stats_list  # Update the shared dict
    
    def get_timing_stats(self):
        """Get average timing statistics"""
        stats = {}
        for category, times in self.timing_stats.items():
            if times:
                times_list = list(times)  # Convert to regular list for calculations
                if times_list:
                    stats[category] = {
                        'avg': sum(times_list) / len(times_list) * 1000,  # Convert to ms
                        'min': min(times_list) * 1000,
                        'max': max(times_list) * 1000
                    }
        return stats

    def start(self):
        """Start point detection process"""
        if self.track_process is not None:
            # If process exists but is not alive, clean it up
            if not self.track_process.is_alive():
                self.track_process.join()
            else:
                print(f"PointDetector {self.camera_id} is already running")
                return
                
        # Create new process
        self._running.set()
        self.track_process = Process(target=self._detect_loop)
        self.track_process.start()
    
    def stop(self):
        """Stop point detection process"""
        self._running.clear()
        
        if self.track_process is not None:
            self.track_process.join()
            self.track_process = None  # Clear the process reference

    def _detect_loop(self):
        """Continuously detect points in frames"""
        last_ts = 0
        while self._running.is_set():
            
            try:
                # Get input data
                ts, cam_id, intrinsic, extrinsic, frame = self.input_queue.get(timeout=0.1)
                
                # Skip if this frame is older than our last processed frame
                if ts < last_ts:
                    continue
                    
                last_ts = ts

                # Detect points using the provided camera parameters
                points, processed_frame = self.detect_points(frame, intrinsic)
                
                # Create output data bundle
                output_data = (ts, cam_id, intrinsic, extrinsic, points, processed_frame)
                
                try:
                    # Clear old frames from output queue
                    while True:
                        try:
                            self.output_queue.get_nowait()
                        except queue.Empty:
                            break
                            
                    # Send to processing pipeline
                    self.output_queue.put(output_data, timeout=0.1)
                    
                    # Also send to visualization (non-blocking)
                    try:
                        self.viz_queue.put(output_data, block=False)
                    except queue.Full:
                        # If viz queue is full, that's okay - just skip
                        pass
                except queue.Full:
                    print(f"PointDetector {self.camera_id} output queue full")
                    
                    
            except queue.Empty:
                continue

    def detect_points(self, frame: np.ndarray, intrinsic: IntrinsicParams) -> Tuple[List[Tuple[Point2D, float]], np.ndarray]:
        """Detect IR LED points in frame"""
        process_start = time.time()
        
        # Undistort using provided camera parameters
        undistort_start = time.time()
        undistorted = cv2.undistort(
            frame, 
            intrinsic.matrix,
            intrinsic.distortion
        )
        
        # Point detection
        detect_start = time.time()
        
        # Convert to grayscale
        gray = cv2.cvtColor(undistorted, cv2.COLOR_BGR2GRAY)

        # Process image to find points
        kernel = np.array([[-2, -1, -1, -1, -2],
                         [-1,  1,  3,  1, -1],
                         [-1,  3,  4,  3, -1],
                         [-1,  1,  3,  1, -1],
                         [-2, -1, -1, -1, -2]])
        filtered = cv2.filter2D(gray, -1, kernel)
        binary = cv2.threshold(filtered, 124, 255, cv2.THRESH_BINARY)[1]
        morph = cv2.erode(binary, self.morph_kernel, iterations=1)
        
        # Find contours
        contours, _ = cv2.findContours(morph, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        # Process contours to find points
        points = []
        for contour in contours:
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = float(M["m10"] / M["m00"])
                cy = float(M["m01"] / M["m00"])
                area = cv2.contourArea(contour)
                confidence = max(0.0, 1.0 - area / 1000)
                points.append((Point2D(cx, cy), confidence))
        
        process_end = time.time()
        if self.flags.get_flag('timing_enabled'):
            self._update_timing('undistort', process_end - undistort_start)
            self._update_timing('point_detection', process_end - detect_start)
            self._update_timing('total_processing', process_end - process_start)
        
        return points, morph, 
