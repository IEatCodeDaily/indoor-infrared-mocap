from typing import List
import time
import rerun as rr
import rerun.blueprint as rrb
from queue import Full, Empty
from multiprocessing import Queue, Process, Event, Pool
import threading
import numpy as np
from .capture import PointDetector
from .world import WorldReconstructor
from .objects import ObjectDetector
from .params import ProcessFlags
from multiprocessing.managers import SyncManager
import os
from scipy.spatial.transform import Rotation

# TODO:
# - Rewrite how visualization work: have 1 type of process that can handle all types of data with 1 queue. Each queue entry will contain the type of data and the data itself.

class OutputCollector:
    def __init__(self, detectors: List[PointDetector],
                 world_reconstructor: WorldReconstructor,
                 object_detector: ObjectDetector,
                 manager: SyncManager = None,
                 flags: ProcessFlags = None):
        # Store only the necessary queues
        self.detector_queues = [(d.viz_queue, d.camera_id) for d in detectors]
        self.world_queue = world_reconstructor.viz_queue
        self.object_queue = object_detector.viz_queue
        self.num_cameras = len(detectors)
        
        # Add frame queues from system
        # frame_queues: (timestamp, camera_ids, frames)
        self.frame_queues = Queue(maxsize=32)  # Increased from 8 to 32
        # Max size increased to handle more frames before dropping
        
        # Setup process control
        self._running = Event()
        self._initialized = Event()
        self.collect_process = None
        self.flags = flags or ProcessFlags()
        
        # Add rate limiting
        self.last_frame_time = time.time()
        self.min_frame_interval = 1.0 / 30.0  # Maximum 30 fps for visualization
        
        # Frame processing pool
        self.frame_pool = None
        self.num_frame_processors = len(detectors)
        

    @rr.shutdown_at_exit
    def _frame_collection_loop(self):
        rr.init("IR Tracking System")
        rr.connect_tcp()
        while self._running.is_set():
            try:
                ts, camera_ids, frames = self.frame_queues.get_nowait()
                for camera_id, frame in zip(camera_ids, frames):
                    rr.set_time_seconds("timestamp", ts)
                    rr.log(f"cameras/camera{camera_id}/image", 
                          rr.Image(frame, color_model="BGR").compress(jpeg_quality=75))
                    #print(f"Frame {ts} logged")
            except Empty:
                #print("Frame queue is empty")
                pass

    @rr.shutdown_at_exit
    def _detector_collection_loop(self, detector_queue: Queue, camera_id: int):
        rr.init("IR Tracking System")
        rr.connect_tcp()
        while self._running.is_set():
            try:
                ts, cam_id, intrinsic, extrinsic, points, confidences, processed_frame, frame_number = detector_queue.get_nowait()
                rr.set_time_seconds("timestamp", ts)
                
                quat_xyzw = Rotation.from_matrix(extrinsic.R.T).as_quat()

                # Update camera transform and frustum in 3D view
                rr.log(f"cameras/camera{cam_id}",
                        rr.Transform3D(
                            translation=-np.matmul(extrinsic.R.T, extrinsic.t).flatten(),
                            rotation=rr.Quaternion(xyzw=quat_xyzw),
                            from_parent=False
                        ))
                
                # Log camera intrinsics
                rr.log(f"cameras/camera{cam_id}",
                        rr.Pinhole(
                            resolution=[640, 480],  # Adjust if your resolution is different
                            focal_length=[intrinsic.matrix[0,0], intrinsic.matrix[1,1]],
                            principal_point=[intrinsic.matrix[0,2], intrinsic.matrix[1,2]],
                            image_plane_distance=300
                        ))
                
                if len(points) > 0:
                    # Points are already in numpy array format
                    points_array = points
                    
                    # Normalize confidences
                    if len(confidences) > 0:
                        conf_min = confidences.min()
                        conf_max = confidences.max()
                        if conf_max > conf_min:
                            confidences = (confidences - conf_min) / (conf_max - conf_min)
                        else:
                            confidences = np.ones_like(confidences)
                    
                    # Create color array based on confidence
                    colors = np.column_stack([
                        1.0 - confidences,  # Red channel
                        confidences,        # Green channel
                        np.zeros_like(confidences),  # Blue channel
                        np.ones_like(confidences)    # Alpha channel
                    ])
                    
                    # Log 2D points with confidence-based coloring
                    rr.log(f"cameras/camera{cam_id}/keypoints", 
                            rr.Points2D(
                                positions=points_array,
                                radii=2,
                                colors=colors
                            ))
                    
                    # --- 3D projection rays visualization ---
                    camera_center = -np.matmul(extrinsic.R.T, extrinsic.t).flatten()
                    line_segments = []
                    for pt in points_array:
                        pt_hom = np.array([pt[0], pt[1], 1.0])
                        ray_dir = np.matmul(extrinsic.R.T, np.matmul(np.linalg.inv(intrinsic.matrix), pt_hom))
                        ray_dir = ray_dir / np.linalg.norm(ray_dir)
                        end_point = camera_center + 10000 * ray_dir  # Lengthen as needed
                        line_segments.append([camera_center.tolist(), end_point.tolist()])
                    if line_segments:
                        rr.log(f"world/projection_rays/camera{cam_id}",
                               rr.LineStrips3D(
                                   line_segments,
                                   colors=[1.0, 0.5, 0.0, 0.5],  # Orange, semi-transparent
                                   radii=1.5
                               ))
                else:
                    # Clear rays if no points
                    rr.log(f"cameras/camera{cam_id}/projection_rays", rr.LineStrips3D([], colors=[]))
                
            except Empty:
                continue  # Move to next camera when queue is empty
                        
            # Add a small sleep to prevent CPU overload
            #time.sleep(0.001)


    @rr.shutdown_at_exit
    def _world_collection_loop(self):
        rr.init("IR Tracking System")
        rr.connect_tcp()
        while self._running.is_set():
            try:
                ts, intrinsics, extrinsics, world_points, epipolar_lines = self.world_queue.get_nowait()
                rr.set_time_seconds("timestamp", ts)
                
                if world_points:
                    # Convert points to numpy array
                    points_array = np.array([[p.x, p.y, p.z] for p in world_points])
                    
                    # Calculate relative z heights and normalize to 0-1 range
                    z_heights = points_array[:, 2]  # Get z coordinates
                    z_min = z_heights.min()
                    z_max = z_heights.max()
                    if z_max > z_min:
                        z_normalized = (z_heights - z_min) / (z_max - z_min)
                    else:
                        z_normalized = np.ones_like(z_heights)
                        
                    # Create color array based on relative height
                    # Blue (cold) for low points to red (hot) for high points
                    colors = np.column_stack([
                        z_normalized,        # Red increases with height
                        np.zeros_like(z_normalized),  # Green fixed at 0
                        1.0 - z_normalized,  # Blue decreases with height
                        np.ones_like(z_normalized)    # Alpha fixed at 1
                    ])
                    
                    # Log 3D points
                    rr.log("world/points", 
                          rr.Points3D(
                              positions=points_array,
                              radii=15,
                              colors=colors
                          ))
                    
                else:
                    # empty world points
                    rr.log("world/points", rr.Points3D(positions=[], radii=[], colors=[]))
                    
                # # Log epipolar lines as 3D rays
                # if epipolar_lines:
                #     # Create line segments for each epipolar line
                #     line_segments = []
                #     for epipolar_line in epipolar_lines:
                #         # Get camera center and calculate ray direction
                #         cam_id = epipolar_line.camera_id
                #         camera_center = -np.matmul(extrinsics[cam_id].R.T, extrinsics[cam_id].t).flatten()
                        
                #         # Convert 2D point to 3D ray direction
                #         pt_hom = np.array([epipolar_line.point_2d[0], epipolar_line.point_2d[1], 1.0])
                #         ray_dir = np.matmul(extrinsics[cam_id].R.T, 
                #                           np.matmul(np.linalg.inv(intrinsics[cam_id].matrix), pt_hom))
                #         ray_dir = ray_dir / np.linalg.norm(ray_dir)
                        
                #         # Create line segment from camera center
                #         end_point = camera_center + 10000 * ray_dir
                #         line_segments.append([camera_center.tolist(), end_point.tolist()])
                    
                #     # Log all epipolar lines
                #     rr.log("world/epipolar_lines",
                #           rr.LineStrips3D(
                #               line_segments,
                #               colors=[1.0, 0.5, 0.0, 0.5],  # Orange, semi-transparent
                #               radii=0.1
                #           ))
                # else:
                #     # Clear epipolar lines if none exist
                #     rr.log("world/epipolar_lines", rr.LineStrips3D([], colors=[]))
                    
            except Empty:
                pass
                
        
    @rr.shutdown_at_exit
    def _object_collection_loop(self):
        """Collect and visualize detected objects"""
        rr.init("IR Tracking System")
        rr.connect_tcp()
        while self._running.is_set():
            try:
                ts, detected_objects = self.object_queue.get_nowait()
                rr.set_time_seconds("timestamp", ts)
                
                for i, obj in enumerate(detected_objects):
                    # Draw object center and heading
                    rr.log(f"world/objects/{obj.obj.name}_{i}/center",
                          rr.Points3D(
                              positions=[obj.position.tolist()],
                              radii=0.02,
                              colors=[0.2, 1.0, 0.2, 1.0]  # Green
                          ))
                    
                    # Draw heading arrow
                    heading_length = 0.1  # 10cm arrow
                    heading_end = obj.position + heading_length * np.array([
                        np.cos(obj.rotation[0]),
                        np.sin(obj.rotation[0]),
                        0
                    ])
                    
                    rr.log(f"world/objects/{obj.obj.name}_{i}/heading",
                          rr.LineStrips3D(
                              [[obj.position.tolist(), heading_end.tolist()]],
                              colors=[0.2, 1.0, 0.2, 0.8]  # Semi-transparent green
                          ))
                    
                    # Draw LED points
                    led_positions = [[p.x, p.y, p.z] for p in obj.led_points]
                    rr.log(f"world/objects/{obj.obj.name}_{i}/leds",
                          rr.Points3D(
                              positions=led_positions,
                              radii=0.01,
                              colors=[1.0, 0.2, 0.2, 1.0]  # Red
                          ))
                    
                    # Draw confidence indicator
                    confidence_radius = 0.05 * obj.confidence  # Scale radius by confidence
                    rr.log(f"world/objects/{obj.obj.name}_{i}/confidence",
                          rr.Points3D(
                              positions=[obj.position.tolist()],
                              radii=confidence_radius,
                              colors=[1.0, 1.0, 0.2, 0.3]  # Semi-transparent yellow
                          ))
                    
                    # Draw velocity vector if available
                    if hasattr(obj, 'velocity'):
                        velocity_end = obj.position + obj.velocity
                        rr.log(f"world/objects/{obj.obj.name}_{i}/velocity",
                              rr.LineStrips3D(
                                  [[obj.position.tolist(), velocity_end.tolist()]],
                                  colors=[0.2, 0.2, 1.0, 0.8]  # Semi-transparent blue
                              ))
                
            except Empty:
                time.sleep(0.001)  # Small sleep to prevent busy waiting

    def start(self):
        """Start the output collection process"""
        if self.collect_process is not None:
            # If process exists but is not alive, clean it up
            if not self.collect_process.is_alive():
                self.collect_process.join()
            else:
                print("Output collector is already running")
                return False
                
        # Create new process
        self._running.set()
        self._initialized.clear()
        self.collect_process = Process(target=self._collect_loop)
        self.collect_process.start()
        
        # Wait for rerun to be initialized
        self._initialized.wait()
        return True

    def stop(self):
        """Stop the output collection process"""
        self._running.clear()
        self._initialized.clear()
        
        # Clean up frame pool if it exists
        if self.frame_pool:
            self.frame_pool.close()
            self.frame_pool.join()
            self.frame_pool = None
        
        if self.collect_process is not None:
            # First terminate the main process
            self.collect_process.terminate()
            self.collect_process.join(timeout=5.0)
            
            # Force kill if still running
            if self.collect_process.is_alive():
                self.collect_process.kill()
                self.collect_process.join(timeout=1.0)
                
            self.collect_process = None
            
        # Clear all queues to prevent blocking
        for queue, _ in self.detector_queues:
            while not queue.empty():
                try:
                    queue.get_nowait()
                except Empty:
                    break
                    
        while not self.world_queue.empty():
            try:
                self.world_queue.get_nowait()
            except Empty:
                break
                
        while not self.object_queue.empty():
            try:
                self.object_queue.get_nowait()
            except Empty:
                break
                
        while not self.frame_queues.empty():
            try:
                self.frame_queues.get_nowait()
            except Empty:
                break

    @rr.shutdown_at_exit
    def _collect_loop(self):
        """Initialize rerun and create processes"""
        # Initialize main rerun connection and setup views
        self._setup_rerun()
        
        # Initialize processes
        processes = []
        # Create multiple frame collection processes (one per camera)
        for _ in range(self.num_cameras):
            processes.append(Process(target=self._frame_collection_loop))
        
        for queue, camera_id in self.detector_queues:
            processes.append(Process(target=self._detector_collection_loop, args=(queue, camera_id)))
        
        # Add other collection processes
        processes.extend([
            Process(target=self._world_collection_loop),
            Process(target=self._object_collection_loop)
        ])
        
        # Start the processes
        for process in processes:
            process.start()
            
        # Keep main process alive and wait for child processes
        while self._running.is_set():
            time.sleep(0.1)  # Reduced sleep time for more responsive cleanup
            
        # Clean up processes
        for process in processes:
            process.terminate()  # Force terminate if needed
            process.join()

    def _setup_rerun(self):
        """Setup rerun visualization spaces and views"""
        try:
            # Initialize rerun blueprint with better layout
            blueprint = rrb.Blueprint(
                rrb.Vertical(
                    rrb.Spatial3DView(
                        name="world",
                        origin="/",
                        
                    ),
                    rrb.Horizontal(
                        *[rrb.Spatial2DView(
                                name=f"camera{i}",
                                origin=f"cameras/camera{i}"
                            ) for i in range(self.num_cameras)],
                        column_shares=[1] * self.num_cameras
                    ),
                    row_shares=[2, 1]
                )
            )
            
            # Initialize rerun
            rr.init("IR Tracking System", spawn=True, default_blueprint=blueprint)
            
            rr.set_time_seconds("timestamp", time.time())
            # Setup world view coordinates
            rr.log("/", rr.ViewCoordinates.RBU, static=True)
            
            # Setup camera coordinate systems
            for i in range(self.num_cameras):
                rr.log(f"cameras/camera{i}", rr.ViewCoordinates.RDF, static=True)  # Right-Down-Forward
            
            rr.log("world", rr.ViewCoordinates.RBU, static=True)  # Set an up-axis
            rr.log("world/origin", rr.Transform3D(translation=[0, 0, 0]))
            rr.log("world/origin/xyz", rr.Arrows3D(
                    origins=[[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                    vectors=[[50, 0, 0], [0, 50, 0], [0, 0, 50]],
                    colors=[[255, 0, 0], [0, 255, 0], [0, 0, 255]],
                    labels=["X", "Y", "Z"],
                    show_labels=False
                )
            )

            self._initialized.set()
            
        except Exception as e:
            print(f"Error setting up rerun: {e}")
            self._initialized.set()  # Set anyway to prevent hanging

    def add_frame(self, ts: float, camera_ids: List[int], frames: List[np.ndarray]):
        """Add frames to the visualization queue"""
        current_time = time.time()
        # Rate limit the visualization
        if current_time - self.last_frame_time < self.min_frame_interval:
            return None
            
        try:
            self.frame_queues.put_nowait((ts, camera_ids, frames))
            self.last_frame_time = current_time
        except Full:
            print("Dropping frame due to visualization queue full")
            return Full    

