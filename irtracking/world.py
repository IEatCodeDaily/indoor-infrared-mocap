from typing import List, Dict, Optional, Tuple, NamedTuple
import numpy as np
from .capture import PointDetector, Point2D, Point3D
from .params import ExtrinsicParams, CameraParamsManager, IntrinsicParams, ProcessFlags
from multiprocessing import Queue, Process, Event
from multiprocessing.managers import SyncManager
from queue import Empty, Full       
import cv2
import time


class EpipolarLine(NamedTuple):
    camera_id: int
    line: np.ndarray  # [a, b, c] for ax + by + c = 0
    point_idx: int    # Index of the point this line corresponds to

# Take all the 2D points from the cameras and reconstruct the 3D points using epipolar geometry
class WorldReconstructor:
    def __init__(self, detectors: List[PointDetector], manager: SyncManager, flags: ProcessFlags):
        # Store detector queues and camera count
        self.detector_queues = [(d.output_queue, d.camera_id) for d in detectors]
        self.num_cameras = len(detectors)
        
        # Setup queues
        # output_queue: (ts, Dict[camera_id, IntrinsicParams], Dict[camera_id, ExtrinsicParams], 3Dpoints, epipolarlines)
        # viz_queue: same as output_queue but non-block ing
        self._running = Event()
        self.output_queue = Queue(maxsize=10)
        self.viz_queue = Queue(maxsize=10)
        
        # Object detector reference (set by main system)
        self.object_detector = None
        
        # Create reconstruction process
        self.reconstruct_process = None  # Initialize to None, will create when starting
        
        # Initialize timing stats with shared memory
        self.timing_stats = manager.dict({
            'data_collection': manager.list(),
            'triangulation': manager.list(),
            'total_reconstruction': manager.list(),
            'reconstruction': manager.list()
        })
        self.max_stats_samples = 100  # Keep last 100 samples
        
        self.flags = flags  # Store the shared flags instance

    def start(self):
        """Start 3D reconstruction process"""
        if self.reconstruct_process is not None:
            # If process exists but is not alive, clean it up
            if not self.reconstruct_process.is_alive():
                self.reconstruct_process.join()
            else:
                print("WorldReconstructor is already running")
                return
                
        # Create new process
        self._running.set()
        self.reconstruct_process = Process(target=self._reconstruct_loop)
        self.reconstruct_process.start()
    
    def stop(self):
        """Stop 3D reconstruction process"""
        self._running.clear()
        
        if self.reconstruct_process is not None:
            self.reconstruct_process.join()
            self.reconstruct_process = None  # Clear the process reference

    def _update_timing(self, category: str, duration: float):
        """Update timing statistics for a category"""
        stats_list = self.timing_stats[category]
        # Create new list with updated values to minimize shared memory operations
        new_list = list(stats_list)[-self.max_stats_samples + 1:] + [duration]
        self.timing_stats[category] = new_list

    def _update_all_timings(self, timestamps: dict):
        """Update all timing statistics at once"""
        timings = {
            'data_collection': timestamps['collection_end'] - timestamps['reconstruct_start'],
            'triangulation': timestamps['triangulate_end'] - timestamps['triangulate_start'],
            'total_reconstruction': timestamps['triangulate_end'] - timestamps['reconstruct_start']
        }
        
        # Update all stats in one pass
        for category, duration in timings.items():
            self._update_timing(category, duration)

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

    def _reconstruct_loop(self):
        """Continuously reconstruct 3D points from camera views"""
        last_ts = 0
        while self._running.is_set():
            timestamps = {'reconstruct_start': time.time()}
            
            # Initialize camera data
            camera_points = {}  # Dict[camera_id, points]
            intrinsics = {}    # Dict[camera_id, IntrinsicParams]
            extrinsics = {}    # Dict[camera_id, ExtrinsicParams]
            target_ts = None
            
            # First pass: get initial timestamp from any camera
            for queue, camera_id in self.detector_queues:
                try:
                    ts, cam_id, intrinsic, extrinsic, points, _ = queue.get(timeout=0.1)
                    
                    # Skip if this frame is older than our last processed frame
                    if ts < last_ts:
                        continue
                        
                    target_ts = ts
                    camera_points[cam_id] = points
                    intrinsics[cam_id] = intrinsic
                    extrinsics[cam_id] = extrinsic
                    break
                except Empty:
                    continue
            
            if target_ts is None:
                continue  # No data available from any camera
                
            last_ts = target_ts
            
            # Second pass: synchronize all cameras to target timestamp
            for queue, camera_id in self.detector_queues:
                if camera_id in camera_points:
                    continue  # Skip cameras we already have data for
                    
                # Try to get a frame close to target timestamp
                best_ts_diff = float('inf')
                best_data = None
                
                while True:
                    try:
                        ts, cam_id, intrinsic, extrinsic, points, _ = queue.get(timeout=0.1)
                        ts_diff = abs(ts - target_ts)
                        
                        if ts_diff < best_ts_diff:
                            best_ts_diff = ts_diff
                            best_data = (ts, cam_id, intrinsic, extrinsic, points)
                            
                            # If we found an exact match, stop looking
                            if ts_diff == 0:
                                break
                                
                    except Empty:
                        break
                
                # Use the best matching frame if it's within tolerance
                if best_data and best_ts_diff < 0.1:  # 100ms tolerance
                    ts, cam_id, intrinsic, extrinsic, points = best_data
                    camera_points[cam_id] = points
                    intrinsics[cam_id] = intrinsic
                    extrinsics[cam_id] = extrinsic
                    
            timestamps['collection_end'] = time.time()
            
            if len(camera_points) >= 2:  # Need at least 2 cameras for triangulation
                timestamps['triangulate_start'] = time.time()
                world_points, epipolar_lines = self._triangulate_points(camera_points, intrinsics, extrinsics)
                timestamps['triangulate_end'] = time.time()
                
                if world_points:  # Only send if we have points
                    output_data = (target_ts, intrinsics, extrinsics, world_points, epipolar_lines)
                    try:
                        # Clear old data from output queues
                        while True:
                            try:
                                self.output_queue.get_nowait()
                            except Empty:
                                break
                                
                        # Send to processing pipeline
                        self.output_queue.put(output_data, timeout=0.1)
                        
                        # Also send to visualization (non-blocking)
                        try:
                            self.viz_queue.put(output_data, block=False)
                        except Full:
                            pass
                        
                        # Send to object detector if available
                        if self.object_detector is not None:
                            try:
                                self.object_detector.input_queue.put(output_data, block=False)
                            except Full:
                                pass
                                
                    except Full:
                        pass
                        
                # Update timing stats
                self._update_all_timings(timestamps)
                
                if self.flags.get_flag('timing_enabled'):
                    # Update timing stats
                    self._update_timing('reconstruction', timestamps['triangulate_end'] - timestamps['reconstruct_start'])

    def _triangulate_points(
        self, 
        camera_points: Dict[int, List[Tuple[Point2D, float]]], 
        intrinsics: Dict[int, IntrinsicParams],
        extrinsics: Dict[int, ExtrinsicParams]
    ) -> Tuple[List[Point3D], List[EpipolarLine]]:
        """Triangulate 3D points from multiple camera views using epipolar geometry"""
        if not camera_points:
            return [], []
            
        # Extract points and create projection matrices
        projection_matrices = {}  # Dict[camera_id, matrix]
        point_sets = {}          # Dict[camera_id, points]
        epipolar_lines = []
        
        for cam_id, points in camera_points.items():
            if not points or cam_id not in intrinsics or cam_id not in extrinsics:
                continue
                
            # Create projection matrix
            RT = np.c_[extrinsics[cam_id].R, extrinsics[cam_id].t]
            P = intrinsics[cam_id].matrix @ RT
            projection_matrices[cam_id] = P
            
            # Extract 2D points (use only the points, not confidences)
            point_sets[cam_id] = np.array([[p[0].x, p[0].y] for p in points])
        
        if len(projection_matrices) < 2:
            return [], []
            
        # Find point correspondences using epipolar geometry
        world_points = []
        camera_ids = list(point_sets.keys())
        
        for i in range(len(camera_ids)):
            cam_id1 = camera_ids[i]
            for j in range(i + 1, len(camera_ids)):
                cam_id2 = camera_ids[j]
                
                # Get fundamental matrix between camera pair
                F = cv2.sfm.fundamentalFromProjections(
                    projection_matrices[cam_id1], 
                    projection_matrices[cam_id2]
                )
                
                # For each point in first camera
                for point_idx1, point1 in enumerate(point_sets[cam_id1]):
                    # Compute epipolar line in second camera
                    line = cv2.computeCorrespondEpilines(
                        np.array([point1], dtype=np.float32), 
                        1, F
                    )[0][0]
                    
                    # Store epipolar line
                    epipolar_lines.append(EpipolarLine(
                        camera_id=cam_id2,
                        line=line,
                        point_idx=point_idx1
                    ))
                    
                    # Find closest point in second camera
                    best_distance = float('inf')
                    best_point_idx2 = None
                    
                    for point_idx2, point2 in enumerate(point_sets[cam_id2]):
                        # Calculate distance to epipolar line
                        distance = abs(
                            line[0] * point2[0] + 
                            line[1] * point2[1] + 
                            line[2]
                        ) / np.sqrt(line[0]**2 + line[1]**2)
                        
                        if distance < best_distance and distance < 5:  # 5 pixel threshold
                            best_distance = distance
                            best_point_idx2 = point_idx2
                    
                    if best_point_idx2 is not None:
                        # Triangulate this point pair
                        points_4d = cv2.triangulatePoints(
                            projection_matrices[cam_id1],
                            projection_matrices[cam_id2],
                            point_sets[cam_id1][point_idx1:point_idx1+1].T,
                            point_sets[cam_id2][best_point_idx2:best_point_idx2+1].T
                        )
                        
                        # Convert from homogeneous coordinates
                        point_3d = points_4d[:3] / points_4d[3]
                        point_3d = point_3d.T[0]
                        
                        # Calculate reprojection error
                        error = 0
                        for cam_id, P in projection_matrices.items():
                            if cam_id not in [cam_id1, cam_id2]:
                                continue
                            point_2d = P @ np.append(point_3d, 1)
                            point_2d = point_2d[:2] / point_2d[2]
                            orig_point = point_sets[cam_id][point_idx1 if cam_id == cam_id1 else best_point_idx2]
                            error += np.sum((point_2d - orig_point) ** 2)
                        
                        # Only add point if reprojection error is small enough
                        if error < 100:  # Adjust threshold as needed
                            world_points.append(Point3D(
                                x=float(point_3d[0]),
                                y=float(point_3d[1]),
                                z=float(point_3d[2])
                            ))
        
        return world_points, epipolar_lines


