from typing import List
import time
import rerun as rr
import rerun.blueprint as rrb
from queue import Empty, Full
from multiprocessing import Queue, Process, Event
import threading
import numpy as np
from .capture import PointDetector
from .world import WorldReconstructor
from .objects import ObjectDetector
from .params import ProcessFlags
from multiprocessing.managers import SyncManager

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
        self.frame_queues = Queue(maxsize=3)
        
        # Setup process control
        self._running = Event()
        self._initialized = Event()
        self.collect_process = None  # Initialize to None, will create when starting
        self.flags = flags  # Store the shared flags instance
        
    @rr.shutdown_at_exit
    def _frame_collection_loop(self):
        rr.init("IR Tracking System")
        rr.connect_tcp()
        while self._running.is_set():
            try:
                ts, camera_ids, frames = self.frame_queues.get_nowait()
                for camera_id, frame in zip(camera_ids, frames):
                    rr.set_time_seconds("frame", ts)
                    rr.log(f"cameras/camera{camera_id}/image", 
                          rr.Image(frame, color_model="BGR").compress(jpeg_quality=75))
                    #print(f"Frame {ts} logged")
            except Empty:
                #print("Frame queue is empty")
                pass

    @rr.shutdown_at_exit
    def _detector_collection_loop(self):
        rr.init("IR Tracking System")
        rr.connect_tcp()
        while self._running.is_set():
            for queue, camera_id in self.detector_queues:
                try:
                    ts, cam_id, intrinsic, extrinsic, points, processed_frame = queue.get_nowait()
                    rr.set_time_seconds("frame", ts)
                    
                    # Update camera transform and frustum in 3D view
                    rr.log(f"/cameras/camera{cam_id}",
                          rr.Transform3D(
                              translation=extrinsic.t.tolist(),
                              mat3x3=extrinsic.R.tolist(),
                              from_parent=True
                          ))
                    
                    # Log camera intrinsics
                    rr.log(f"cameras/camera{cam_id}",
                          rr.Pinhole(
                              resolution=[640, 480],  # Adjust if your resolution is different
                              focal_length=[intrinsic.matrix[0,0], intrinsic.matrix[1,1]],
                              principal_point=[intrinsic.matrix[0,2], intrinsic.matrix[1,2]]
                          ))
                    
                    # Log processed frame with detected points
                    # rr.log(f"cameras/camera{cam_id}/processed", 
                    #       rr.Image(processed_frame, color_model="BGR").compress(jpeg_quality=75))
                    
                    if points:
                        # Convert points to numpy array for easier handling
                        points_array = np.array([[p[0].x, p[0].y] for p in points])
                        confidences = np.array([p[1] for p in points])
                        
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
                        
                except Empty:
                    #print("Detector queue is empty")
                    pass

    @rr.shutdown_at_exit
    def _world_collection_loop(self):
        rr.init("IR Tracking System")
        rr.connect_tcp()
        while self._running.is_set():
            try:
                ts, intrinsics, extrinsics, world_points, epipolar_lines = self.world_queue.get_nowait()
                rr.set_time_seconds("frame", ts)
                
                if world_points:
                    # Convert points to numpy array
                    points_array = np.array([[p.x, p.y, p.z] for p in world_points])
                    
                    # Log 3D points
                    rr.log("world/points", 
                          rr.Points3D(
                              positions=points_array,
                              radii=0.01,
                              colors=[0.2, 0.8, 1.0, 1.0]
                          ))
                    
                    # Log point connections if there are multiple points
                    if len(points_array) > 1:
                        rr.log("world/connections",
                              rr.LineStrips3D(
                                  [points_array],
                                  colors=[0.5, 0.5, 1.0, 0.5]
                              ))
                    
                # Log epipolar lines
                for epipolar_line in epipolar_lines:
                    cam_id = epipolar_line.camera_id
                    line = epipolar_line.line
                    
                    # Create points along the epipolar line
                    x = np.linspace(0, 640, 100)  # Adjust range based on image width
                    y = (-line[2] - line[0] * x) / line[1]
                    
                    # Filter points within image bounds
                    mask = (y >= 0) & (y < 480)  # Adjust based on image height
                    x = x[mask]
                    y = y[mask]
                    
                    if len(x) > 0:
                        points = np.column_stack([x, y])
                        rr.log(f"cameras/camera{cam_id}/epipolar_lines",
                              rr.LineStrips2D(
                                  [points],
                                  colors=[1.0, 0.5, 0.0, 0.5]  # Orange, semi-transparent
                              ))
                    
            except Empty:
                #print("World queue is empty")
                pass
        
    @rr.shutdown_at_exit
    def _object_collection_loop(self):
        """Collect and visualize detected objects"""
        rr.init("IR Tracking System")
        rr.connect_tcp()
        while self._running.is_set():
            try:
                ts, detected_objects = self.object_queue.get_nowait()
                rr.set_time_seconds("world", ts)
                
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
                        np.cos(obj.rotation),
                        np.sin(obj.rotation),
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
                    
                    # Draw connections between LEDs
                    rr.log(f"world/objects/{obj.obj.name}_{i}/connections",
                          rr.LineStrips3D(
                              [led_positions],
                              colors=[1.0, 0.2, 0.2, 0.5]  # Semi-transparent red
                          ))
                    
                    # Add object info text
                    rr.log(f"world/objects/{obj.obj.name}_{i}/info",
                          rr.Text(
                              f"{obj.obj.name}\nconf: {obj.confidence:.2f}",
                              position=obj.position.tolist()
                          ))
                    
            except Empty:
                #print("Object queue is empty")
                pass

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
        
        if self.collect_process is not None:
            self.collect_process.join()
            self.collect_process = None  # Clear the process reference

    @rr.shutdown_at_exit
    def _collect_loop(self):
        """Initialize rerun and create processes"""
        # Initialize main rerun connection and setup views
        self._setup_rerun()
        
        # Initialize processes
        processes = [
            Process(target=self._frame_collection_loop),
            Process(target=self._detector_collection_loop),
            Process(target=self._world_collection_loop),
            Process(target=self._object_collection_loop)
        ]
        
        # Start the processes
        for process in processes:
            process.start()
            
        # Keep main process alive and wait for child processes
        while self._running.is_set():
            time.sleep(0.4)
            
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
                        origin="/"
                    ),
                    rrb.Horizontal(
                        *[rrb.Spatial2DView(
                                name=f"camera{i}",
                                origin=f"/cameras/camera{i}"
                            ) for i in range(self.num_cameras)],
                        column_shares=[1] * self.num_cameras
                    ),
                    row_shares=[2, 1]
                )
            )
            
            # Initialize rerun
            rr.init("IR Tracking System", spawn=True, default_blueprint=blueprint)
            
            # Setup world view coordinates
            rr.log("/", rr.ViewCoordinates.RIGHT_HAND_Y_UP, static=True)
            
            # Setup camera coordinate systems
            for i in range(self.num_cameras):
                rr.log(f"cameras/camera{i}", rr.ViewCoordinates.RDF, static=True)  # Right-Down-Forward
            
            # Add world origin marker
            rr.log("world/origin", rr.Transform3D(translation=[0, 0, 0]))
            
            self._initialized.set()
            
        except Exception as e:
            print(f"Error setting up rerun: {e}")
            self._initialized.set()  # Set anyway to prevent hanging

    def add_frame(self, ts: float, camera_ids: List[int], frames: List[np.ndarray]):
        """Add frames to the visualization queue"""
        try:
            self.frame_queues.put_nowait((ts, camera_ids, frames))
        except Full:
            pass    
