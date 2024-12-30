from typing import List, Dict, Optional, Tuple
import numpy as np
from ..camera.capture import PointDetector, Point2D, Point3D
from ..camera.params import ExtrinsicParams
import queue
import threading

# Take all the 2D points from the cameras and reconstruct the 3D points using epipolar geometry
class WorldReconstructor:
    def __init__(self, cameras: List[PointDetector]):
        self.cameras: List[PointDetector] = cameras
        self._running = False
        self.world_points: List[Point3D] = []

    def start(self):
        """Start 3D reconstruction thread"""
        self._running = True
        self.reconstruct_thread = threading.Thread(target=self._reconstruct_loop)
        self.reconstruct_thread.start()
    
    def stop(self):
        """Stop 3D reconstruction thread"""
        self._running = False
        self.reconstruct_thread.join()

    def get_points(self) -> List[Point3D]:
        """Get 3D points from the buffer"""
        try:
            return self.point_buffer.get(timeout=0.1)
        except queue.Empty:
            return []
        
    def insert_points(self, points: List[Point3D]):
        """Insert 3D points into the buffer"""
        try:
            self.point_buffer.put(points, block=False)
        except queue.Full:
            self.point_buffer.get()
            self.point_buffer.put(points, block=False)
        

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
    
    def _triangulate_points(self, camera_points: List[Tuple[PointDetector, List[Point2D]]]) -> List[Point3D]:
        """Triangulate 3D points from multiple camera views"""
        # Implement triangulation using camera poses and detected points
        # Use cv2.triangulatePoints for pairs of cameras
        # Merge results from multiple pairs
        
        pass
 
    def triangulate_point(image_points, camera_poses):
        image_points = np.array(image_points)
        cameras = Cameras.instance()
        none_indicies = np.where(np.all(image_points == None, axis=1))[0]
        image_points = np.delete(image_points, none_indicies, axis=0)
        camera_poses = np.delete(camera_poses, none_indicies, axis=0)

        if len(image_points) <= 1:
            return [None, None, None]

        Ps = [] # projection matricies

        for i, camera_pose in enumerate(camera_poses):
            RT = np.c_[camera_pose["R"], camera_pose["t"]]
            P = cameras.camera_params[i]["intrinsic_matrix"] @ RT
            Ps.append(P)

        # https://temugeb.github.io/computer_vision/2021/02/06/direct-linear-transorms.html
        def DLT(Ps, image_points):
            A = []

            for P, image_point in zip(Ps, image_points):
                A.append(image_point[1]*P[2,:] - P[1,:])
                A.append(P[0,:] - image_point[0]*P[2,:])
                
            A = np.array(A).reshape((len(Ps)*2,4))
            B = A.transpose() @ A
            U, s, Vh = linalg.svd(B, full_matrices = False)
            object_point = Vh[3,0:3]/Vh[3,3]

            return object_point

        object_point = DLT(Ps, image_points)

        return object_point


