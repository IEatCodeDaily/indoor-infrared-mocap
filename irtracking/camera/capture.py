import cv2
import numpy as np
import queue
from .params import CameraParamsManager, IntrinsicParams
from typing import List, Optional, Tuple
import threading
from dataclasses import dataclass
@dataclass
class Point2D:
    x: float
    y: float

@dataclass
class Point3D:
    x: float
    y: float
    z: float

class PointTracker:
    def __init__(self, camera_id: int, params_manager: CameraParamsManager):
        self.camera_id = camera_id
        self.params_manager = params_manager
        self.intrinsics: IntrinsicParams = params_manager.get_intrinsic_params(self.camera_id)
        self.params_manager.add_observer(self.update_params)
        self.morph_kernel = np.array([[0, 1, 0],[1, 1, 1],[0, 1, 0]], np.uint8)


    def detect_points(self, frame: np.ndarray) -> List[Tuple[Point2D, float]]:
        """Detect IR LED points in frame"""
        # Undistort
        undistorted = cv2.undistort(frame, self.intrinsics.matrix, self.intrinsics.distortion)
        
        # Convert to grayscale
        gray = cv2.cvtColor(undistorted, cv2.COLOR_BGR2GRAY)

        kernel = np.array([[-2, -1, -1, -1, -2],
                            [-1,  1,  3,  1, -1],
                            [-1,  3,  4,  3, -1],
                            [-1,  1,  3,  1, -1],
                            [-2, -1, -1, -1, -2]])
        filtered = cv2.filter2D(gray, -1, kernel)
        # Threshold and find contours
        #binary = cv2.adaptiveThreshold(filtered, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        #binary = cv2.threshold(gray, 125, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        binary = cv2.threshold(filtered, 124, 255, cv2.THRESH_BINARY)[1]
        
        #morph = cv2.morphologyEx(binary, cv2.MORPH_OPEN, self.morph_kernel)
        morph = cv2.erode(binary, self.morph_kernel, iterations=1)
        contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        points = []
        for contour in contours:
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = float(M["m10"] / M["m00"])
                cy = float(M["m01"] / M["m00"])
                # Calculate confidence based on area or intensity
                area = cv2.contourArea(contour)
                confidence = max(0.0, 1.0 - area / 1000)  # Adjust scaling factor as needed
                points.append([Point2D(cx, cy), confidence])
        
        return points, morph

    def update_params(self):
        """Update camera parameters"""
        self.intrinsics = self.params_manager.get_intrinsic_params(self.camera_id)

        self.intrinsics = self.params_manager.get_intrinsic_params(self.camera_id)