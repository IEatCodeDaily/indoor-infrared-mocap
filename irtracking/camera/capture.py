import cv2
import numpy as np
import queue
from .params import CameraParamsManager, Point2D
from typing import List, Optional, Tuple
import threading


class Camera:
    def __init__(self, camera_id: int, params_manager: CameraParamsManager):
        self.camera_id = camera_id
        self.params_manager = params_manager
        self.params_manager.add_observer(self.update_params)
        self.frame_queue = queue.Queue(maxsize=2)
        self.points_queue = queue.Queue(maxsize=2)
        self._running = False
        self._capture = None
        
    def start(self, video_source: str):
        """Start camera capture thread"""
        self._capture = cv2.VideoCapture(video_source)
        self._running = True
        self.capture_thread = threading.Thread(target=self._capture_loop)
        self.process_thread = threading.Thread(target=self._process_loop)
        self.capture_thread.start()
        self.process_thread.start()

    def stop(self):
        """Stop camera capture and processing"""
        self._running = False
        if self.capture_thread.is_alive():
            self.capture_thread.join()
        if self.process_thread.is_alive():
            self.process_thread.join()
        if self._capture is not None:
            self._capture.release()

    def _capture_loop(self):
        """Continuously capture frames"""
        while self._running:
            ret, frame = self._capture.read()
            if ret:
                try:
                    self.frame_queue.put(frame, block=False)
                except queue.Full:
                    self.frame_queue.get()  # Remove oldest frame
                    self.frame_queue.put(frame, block=False)
    
    def _process_loop(self):
        """Process frames and detect points"""
        while self._running:
            try:
                frame = self.frame_queue.get(timeout=1.0)
                points = self._detect_points(frame)
                try:
                    self.points_queue.put(points, block=False)
                except queue.Full:
                    self.points_queue.get()  # Remove oldest points
                    self.points_queue.put(points, block=False)
            except queue.Empty:
                continue
    
    def _detect_points(self, frame: np.ndarray) -> List[Point2D]:
        """Detect IR LED points in frame"""
        # Undistort
        undistorted = cv2.undistort(frame, self.camera_matrix, self.dist_coeffs)
        
        # Convert to grayscale if needed
        if len(undistorted.shape) > 2:
            gray = cv2.cvtColor(undistorted, cv2.COLOR_BGR2GRAY)
        else:
            gray = undistorted
        
        # Threshold and find contours
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        points = []
        for contour in contours:
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = float(M["m10"] / M["m00"])
                cy = float(M["m01"] / M["m00"])
                # Calculate confidence based on area or intensity
                area = cv2.contourArea(contour)
                confidence = min(1.0, area / 100)  # Adjust scaling factor as needed
                points.append(Point2D(cx, cy, confidence))
        
        return points