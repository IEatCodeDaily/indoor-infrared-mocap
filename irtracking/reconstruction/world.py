from typing import List, Dict, Optional, Tuple
import numpy as np
from ..camera.capture import PointTracker
from ..camera.params import ExtrinsicParams, Point2D, Point3D
import queue
import threading

class WorldReconstructor:
    def __init__(self, cameras: List[PointTracker]):
        self.cameras = cameras
        self.point_buffer = queue.Queue(maxsize=10)
        self._running = False
    
    def start(self):
        """Start 3D reconstruction thread"""
        self._running = True
        self.reconstruct_thread = threading.Thread(target=self._reconstruct_loop)
        self.reconstruct_thread.start()
    
    def _reconstruct_loop(self):
        """Continuously reconstruct 3D points from camera views"""
        while self._running:
            # Collect points from all cameras
            camera_points = []
            for camera in self.cameras:
                try:
                    points = camera.points_queue.get(timeout=0.1)
                    camera_points.append((camera, points))
                except queue.Empty:
                    continue
            
            if len(camera_points) >= 2:  # Need at least 2 cameras for triangulation
                world_points = self._triangulate_points(camera_points)
                try:
                    self.point_buffer.put(world_points, block=False)
                except queue.Full:
                    self.point_buffer.get()
                    self.point_buffer.put(world_points, block=False)
    
    def _triangulate_points(self, camera_points: List[Tuple[PointTracker, List[Point2D]]]) -> List[Point3D]:
        """Triangulate 3D points from multiple camera views"""
        # Implement triangulation using camera poses and detected points
        # Use cv2.triangulatePoints for pairs of cameras
        # Merge results from multiple pairs
        pass
