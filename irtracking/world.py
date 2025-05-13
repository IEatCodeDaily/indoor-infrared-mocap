from typing import List, Dict, Optional, Tuple, NamedTuple
import numpy as np
from .capture import PointDetector, Point2D, Point3D
from .params import ExtrinsicParams, CameraParamsManager, IntrinsicParams, ProcessFlags
from .objects import ObjectDetector
from multiprocessing import Queue, Process, Event, cpu_count
from multiprocessing.managers import SyncManager, BaseManager
from queue import Empty, Full, LifoQueue       
import cv2
import time
from itertools import product
from collections import defaultdict, deque
from sklearn.cluster import DBSCAN
from dataclasses import dataclass

@dataclass
class EpipolarLine:
    camera_id: int
    line: np.ndarray  # [a, b, c] for ax + by + c = 0
    point_idx: int    # Index of the point this line corresponds to
    point_2d: np.ndarray  # The 2D point in the camera's image plane

class MyManager(BaseManager):
    pass
MyManager.register('LifoQueue', LifoQueue)

# Take all the 2D points from the cameras and reconstruct the 3D points using epipolar geometry
class WorldReconstructor:
    def __init__(self, detectors: List[PointDetector], manager: SyncManager = None, flags: ProcessFlags = ProcessFlags(), num_workers=4):
        # Store detector queues and camera count
        self.detector_queues = [(d.output_queue, d.camera_id) for d in detectors]
        self.num_cameras = len(detectors)
        self.lifo_manager = MyManager()
        self.lifo_manager.start()
        # Setup queues
        # output_queue: (ts, Dict[camera_id, IntrinsicParams], Dict[camera_id, ExtrinsicParams], 3Dpoints, epipolarlines)
        # viz_queue: same as output_queue but non-blocking
        self.max_frame_group_buffer_size = 60  # Maximum number of frame numbers to keep in buffer
        self.frame_group_queue = self.lifo_manager.LifoQueue(maxsize=self.max_frame_group_buffer_size+2)

        self.result_queue = Queue(maxsize=60)
        self.output_queue = Queue(maxsize=60)
        self.viz_queue = Queue(maxsize=100)
        
        self.num_workers = num_workers or min(2, cpu_count() - 1)
        self.workers = []
        self._running = Event()

        # Object detector queue reference (set by main system)
        self.object_detector_queue = None
        
        # Create reconstruction process
        self.reconstruct_process = None  # Initialize to None, will create when starting
        
        # Initialize timing stats with shared memory
        self.timing_stats = manager.dict({
            'data_collection': manager.list(),
            'triangulation': manager.list(),
            'total_reconstruction': manager.list(),
            'reconstruction': manager.list()
        }) if manager else None
        self.max_stats_samples = 100  # Keep last 100 samples
        
        self.flags = flags  # Store the shared flags instance
        
        # Frame buffer using dictionary with frame numbers as keys
        self.frame_buffer = {}  # Dict[frame_number, Dict[camera_id, frame_data]]

    def start(self):
        """Start 3D reconstruction process and worker pool"""
        if self.reconstruct_process is not None:
            if not self.reconstruct_process.is_alive():
                self.reconstruct_process.join()
            else:
                print("WorldReconstructor is already running")
                return
        print("Starting WorldReconstructor process and worker pool...")
        self._running.set()
        # Start main process as before
        self.reconstruct_process = Process(target=reconstruct_loop, args=(self._running, self.detector_queues, self.frame_group_queue, self.result_queue, self.frame_buffer, self.max_frame_group_buffer_size, self.num_cameras, self.output_queue, self.viz_queue, self.object_detector_queue))
        self.reconstruct_process.start()
        # Start worker pool
        for _ in range(self.num_workers):
            p = Process(target=_triangulation_worker, args=(self.frame_group_queue, self.result_queue, self._running))
            p.start()
            self.workers.append(p)
        
        print("WorldReconstructor process and workers started")
    
    def stop(self):
        """Stop 3D reconstruction process and worker pool"""
        self._running.clear()
        if self.reconstruct_process is not None:
            self.reconstruct_process.terminate()
            self.reconstruct_process.join()
            self.reconstruct_process = None
        for p in self.workers:
            p.terminate()
            p.join()
        self.workers = []

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

def reconstruct_loop(running, detectors, frame_group_queue, result_queue, frame_buffer, max_buffer_size, num_cameras, output_queue, viz_queue, object_detector_queue):
    """Continuously reconstruct 3D points from camera views using frame numbers"""
    while running.is_set():
        timestamps = {'reconstruct_start': time.time()}
        
        # Step 1: Fill frame buffer from detector queues
        for queue, camera_id in detectors:
            while True:
                try:
                    input_data = queue.get_nowait()
                    frame_number = input_data[-1]
                    if frame_number not in frame_buffer:
                        frame_buffer[frame_number] = {}
                    frame_buffer[frame_number][camera_id] = input_data
                    while len(frame_buffer) > max_buffer_size:
                        oldest_frame = min(frame_buffer.keys())
                        del frame_buffer[oldest_frame]
                except Empty:
                    break
        
        # Step 2: Find frame number with all cameras
        current_frame = None
        for frame_number, frames in frame_buffer.items():
            if len(frames) == num_cameras:
                current_frame = frame_number
                break
        
        # Step 3: Put frame group into queue for workers
        if current_frame is not None:
            frames = frame_buffer[current_frame]
            frame_group_queue.put((current_frame, frames))
            del frame_buffer[current_frame]
        
        # Step 4: Collect results from result_queue and handle as before
        try:
            while True:
                result = result_queue.get_nowait()
                frame_number, ts, intrinsics, extrinsics, world_points, epipolar_lines = result
                output_data = (ts, intrinsics, extrinsics, world_points, epipolar_lines)
                # Output queues as before
                while True:
                    try:
                        output_queue.get_nowait()
                    except Empty:
                        break
                try:
                    output_queue.put(output_data, timeout=0.1)
                except Full:
                    print("Output queue full, dropping frame")
                try:
                    viz_queue.put(output_data, block=False)
                except Full:
                    print("Visualization queue full")
                if object_detector_queue is not None:
                    try:
                        object_detector_queue.put(output_data, block=False)
                    except Full:
                        print("Object detector queue full", flush=True)
                        pass
        except Empty:
            pass
        
        time.sleep(0.001)

def _triangulation_worker(frame_group_queue, result_queue, running):
    while running.is_set():
        try:
            frame_number, frames = frame_group_queue.get(timeout=0.1)
            camera_points = {}
            intrinsics = {}
            extrinsics = {}
            ts = None
            for camera_id, input_data in frames.items():
                ts, cam_id, intrinsic, extrinsic, points, confidences, _, frame_number = input_data
                camera_points[cam_id] = points
                intrinsics[cam_id] = intrinsic
                extrinsics[cam_id] = extrinsic
            world_points, epipolar_lines = triangulate_points(camera_points, intrinsics, extrinsics)
            result_queue.put((frame_number, ts, intrinsics, extrinsics, world_points, epipolar_lines))
        except Empty:
            continue
    time.sleep(0.001)

def hungarian_partial_chains_correspondences(
    camera_points: Dict[int, np.ndarray],
    intrinsics: Dict[int, IntrinsicParams],
    extrinsics: Dict[int, ExtrinsicParams],
    epi_thresh: float = 10.0,
    min_views: int = 2
) -> Tuple[list, List[EpipolarLine]]:
    """Find correspondences by building a graph where each node is a point and edges are epipolar matches.
    Each correspondence is a path of 2-4 points through consecutive cameras."""
    if not camera_points or len(camera_points) < 2:
        return [], []
    
    camera_ids = list(sorted(camera_points.keys()))
    num_cams = len(camera_ids)
    
    # Build graph: node = (cam_id, point_idx), edge = epipolar match
    graph = defaultdict(list)
    epipolar_lines = []
    
    # For each pair of cameras
    for i in range(num_cams):
        for j in range(i + 1, num_cams):
            camA, camB = camera_ids[i], camera_ids[j]
            ptsA, ptsB = camera_points[camA], camera_points[camB]
            
            # Get projection matrices for epipolar geometry
            RT_A = np.c_[extrinsics[camA].R, extrinsics[camA].t]
            RT_B = np.c_[extrinsics[camB].R, extrinsics[camB].t]
            P_A = intrinsics[camA].matrix @ RT_A
            P_B = intrinsics[camB].matrix @ RT_B
            F = cv2.sfm.fundamentalFromProjections(P_A, P_B)
            
            # For each point in camera A
            for idxA, ptA in enumerate(ptsA):
                # Compute epipolar line
                line = cv2.computeCorrespondEpilines(np.array([ptA], dtype=np.float32), 1, F)[0][0]
                
                epipolar_lines.append(EpipolarLine(
                    camera_id=camA,
                    line=line,
                    point_idx=idxA,
                    point_2d=ptA
                ))
                
                # Find matches in camera B
                matches = []
                for idxB, ptB in enumerate(ptsB):
                    dist = abs(line[0]*ptB[0] + line[1]*ptB[1] + line[2]) / np.sqrt(line[0]**2 + line[1]**2)
                    if dist < epi_thresh:
                        matches.append((idxB, dist))
                
                # Sort matches by distance and keep only the best one
                if matches:
                    matches.sort(key=lambda x: x[1])
                    best_match = matches[0]
                    graph[(camA, idxA)].append((camB, best_match[0]))
    
    # Extract paths through the graph
    all_correspondences = []
    
    def find_paths_from_start(start_cam, start_idx, max_length=4):
        """Find all valid paths starting from a given point, up to max_length cameras."""
        paths = []
        current_path = [(start_cam, start_idx)]
        used_cameras = {start_cam}
        
        def extend_path(path, used):
            if len(path) >= min_views:
                # Build correspondence: list of 2D points (or None) for each camera
                corr = [None] * num_cams
                for cam_id, idx in path:
                    cam_idx = camera_ids.index(cam_id)
                    corr[cam_idx] = camera_points[cam_id][idx]
                paths.append(corr)
            
            if len(path) >= max_length:
                return
                
            last_cam, last_idx = path[-1]
            for next_cam, next_idx in graph[(last_cam, last_idx)]:
                if next_cam not in used:
                    extend_path(path + [(next_cam, next_idx)], used | {next_cam})
        
        extend_path(current_path, used_cameras)
        return paths
    
    # Start from each point in each camera
    for cam_id in camera_ids:
        for idx in range(len(camera_points[cam_id])):
            paths = find_paths_from_start(cam_id, idx)
            all_correspondences.extend(paths)
    
    return all_correspondences, epipolar_lines

def compute_multi_view_correspondences(
    camera_points: Dict[int, np.ndarray],
    intrinsics: Dict[int, IntrinsicParams],
    extrinsics: Dict[int, ExtrinsicParams],
    epi_thresh: float = 5.0,  # Reduced from 10.0 for stricter matching
    min_views: int = 2
) -> Tuple[list, List[EpipolarLine]]:
    """Find correspondences across all cameras simultaneously using a multi-view approach.
    Returns both the correspondences and epipolar lines for visualization."""
    if not camera_points or len(camera_points) < 2:
        return [], []
    
    camera_ids = list(sorted(camera_points.keys()))
    num_cams = len(camera_ids)
    epipolar_lines = []
    
    # Build cost matrix for all possible point combinations
    max_points = max(len(pts) for pts in camera_points.values())
    cost_matrix = np.full((num_cams, num_cams, max_points, max_points), float('inf'))
    
    # For each camera pair
    for i in range(num_cams):
        for j in range(i + 1, num_cams):
            camA, camB = camera_ids[i], camera_ids[j]
            ptsA, ptsB = camera_points[camA], camera_points[camB]
            
            # Get fundamental matrix
            RT_A = np.c_[extrinsics[camA].R, extrinsics[camA].t]
            RT_B = np.c_[extrinsics[camB].R, extrinsics[camB].t]
            P_A = intrinsics[camA].matrix @ RT_A
            P_B = intrinsics[camB].matrix @ RT_B
            F = cv2.sfm.fundamentalFromProjections(P_A, P_B)
            
            # Compute epipolar distances
            for idxA, ptA in enumerate(ptsA):
                line = cv2.computeCorrespondEpilines(np.array([ptA], dtype=np.float32), 1, F)[0][0]
                
                # # Store epipolar line for visualization
                # epipolar_lines.append(EpipolarLine(
                #     camera_id=camA,
                #     line=line,
                #     point_idx=idxA,
                #     point_2d=ptA
                # ))
                
                for idxB, ptB in enumerate(ptsB):
                    dist = abs(line[0]*ptB[0] + line[1]*ptB[1] + line[2]) / np.sqrt(line[0]**2 + line[1]**2)
                    cost_matrix[i, j, idxA, idxB] = dist
                    cost_matrix[j, i, idxB, idxA] = dist  # Symmetric
    
    # Find consistent correspondences using graph-based approach
    correspondences = []
    used_points = set()
    
    def find_consistent_group(start_cam, start_idx):
        """Find a group of consistent points across all cameras starting from a given point."""
        group = [(start_cam, start_idx)]
        used = {(start_cam, start_idx)}
        
        # Try to extend the group to other cameras
        for cam in range(num_cams):
            if cam == start_cam:
                continue
                
            best_match = None
            best_cost = float('inf')
            
            # Find best matching point in this camera
            for idx in range(len(camera_points[camera_ids[cam]])):
                if (cam, idx) in used:
                    continue
                    
                # Compute total cost of adding this point
                total_cost = 0
                for cam2, idx2 in group:
                    cost = cost_matrix[cam2, cam, idx2, idx]
                    if cost > epi_thresh:
                        total_cost = float('inf')
                        break
                    total_cost += cost
                
                if total_cost < best_cost:
                    best_cost = total_cost
                    best_match = (cam, idx)
            
            if best_match is not None:
                group.append(best_match)
                used.add(best_match)
        
        return group if len(group) >= min_views else None
    
    # Find correspondences starting from each point
    for cam in range(num_cams):
        for idx in range(len(camera_points[camera_ids[cam]])):
            if (cam, idx) in used_points:
                continue
                
            group = find_consistent_group(cam, idx)
            if group is not None:
                # Convert group to correspondence format
                corr = [None] * num_cams
                for cam_id, idx in group:
                    corr[cam_id] = camera_points[camera_ids[cam_id]][idx]
                correspondences.append(corr)
                used_points.update(group)
    
    return correspondences, epipolar_lines

def triangulate_correspondences(
    correspondences: list,
    intrinsics: Dict[int, IntrinsicParams],
    extrinsics: Dict[int, ExtrinsicParams],
    reproj_thresh: float = 10.0
) -> list:
    """Triangulate each correspondence group (2D points per camera) to 3D points.
    Uses reprojection error to select the best triangulated point."""
    world_points = []
    camera_ids = list(intrinsics.keys())
    
    for corr_idx, corr in enumerate(correspondences):
        # corr: list of 2D points (one per camera, or None)
        valid = [(i, pt) for i, pt in enumerate(corr) if pt is not None]
        if len(valid) < 2:
            print(f"Correspondence {corr_idx}: Skipping - only {len(valid)} valid points")
            continue
            
        # Get valid camera poses and points
        valid_camera_poses = []
        valid_points = []
        for idx, pt in valid:
            cam_id = camera_ids[idx]
            valid_camera_poses.append({
                "R": extrinsics[cam_id].R,
                "t": extrinsics[cam_id].t,
                "intrinsic_matrix": intrinsics[cam_id].matrix
            })
            valid_points.append(pt)
            
        # Try different combinations of views for triangulation
        best_error = float('inf')
        best_point = None
        
        
        # Try all possible pairs of views
        for i in range(len(valid_points)):
            for j in range(i+1, len(valid_points)):
                pts1 = np.array([valid_points[i]]).T
                pts2 = np.array([valid_points[j]]).T
                
                RT1 = np.c_[valid_camera_poses[i]["R"], valid_camera_poses[i]["t"]]
                RT2 = np.c_[valid_camera_poses[j]["R"], valid_camera_poses[j]["t"]]
                P1 = valid_camera_poses[i]["intrinsic_matrix"] @ RT1
                P2 = valid_camera_poses[j]["intrinsic_matrix"] @ RT2
                
                try:
                    X_hom = cv2.triangulatePoints(P1, P2, pts1, pts2)
                    X = X_hom[:, 0] / X_hom[3, 0]
                    point_3d = X[:3]
                    
                    # Calculate reprojection error across all views
                    error = 0
                    for k, (pt, cam_pose) in enumerate(zip(valid_points, valid_camera_poses)):
                        projected, _ = cv2.projectPoints(
                            np.expand_dims(point_3d, axis=0).astype(np.float32),
                            cam_pose["R"],
                            cam_pose["t"],
                            cam_pose["intrinsic_matrix"],
                            np.array([])
                        )
                        projected = projected[0, 0]
                        error += np.sum((pt - projected) ** 2)
                    error = error / len(valid_points)
                    
                    if error < best_error:
                        best_error = error
                        best_point = point_3d
                except Exception as e:
                    print(f"Error triangulating points {i} and {j}: {str(e)}")
                    continue
        
        if best_error < reproj_thresh:
            world_points.append(Point3D(
                x=float(best_point[0]),
                y=float(best_point[1]),
                z=float(best_point[2])
            ))
        # else:
        #     print(f"Correspondence {corr_idx}: Best error {best_error:.2f} exceeds threshold {reproj_thresh}")
            
    return world_points

def temporal_smooth_points(points_history, window=3):
    """Simple moving average smoothing for 3D points over time."""
    if len(points_history) < window:
        return points_history[-1]
    arr = np.array(points_history[-window:])
    return np.mean(arr, axis=0)

def cluster_world_points(world_points, eps=20.0, min_samples=2):
    """Cluster 3D points using DBSCAN and return centroids of clusters with enough members."""
    if not world_points:
        return []
    pts = np.array([[p.x, p.y, p.z] for p in world_points])
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(pts)
    labels = db.labels_
    unique_labels = set(labels)
    clustered_points = []
    for label in unique_labels:
        if label == -1:
            continue  # Noise
        members = pts[labels == label]
        if len(members) >= min_samples:
            centroid = np.mean(members, axis=0)
            clustered_points.append(Point3D(float(centroid[0]), float(centroid[1]), float(centroid[2])))
    return clustered_points

def triangulate_points( 
    camera_points: Dict[int, np.ndarray], 
    intrinsics: Dict[int, IntrinsicParams],
    extrinsics: Dict[int, ExtrinsicParams]
) -> Tuple[List[Point3D], List[EpipolarLine]]:
    """Multi-view triangulation with improved correspondence finding."""
    reproj_thresh = 100.0
    correspondences, epipolar_lines = compute_multi_view_correspondences(
        camera_points, intrinsics, extrinsics
    )
    world_points = triangulate_correspondences(correspondences, intrinsics, extrinsics, reproj_thresh)
    # print(f"Input: {sum(len(pts) for pts in camera_points.values())} points, "
    #       f"Correspondences: {len(correspondences)}, "
    #       f"Triangulated: {len(world_points)}")
    return world_points, epipolar_lines
