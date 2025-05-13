import json
import matplotlib.pyplot as plt
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from multiprocessing import Process, Queue, Event
from multiprocessing.managers import SyncManager
from queue import Empty, Full
from .capture import Point3D
from .params import ProcessFlags
import time
import cv2

# Objects dataclass
@dataclass
class Obj:
    name: str
    type: str
    dimensions: int
    metadata: Dict[str, Any]
    points: np.ndarray  # contains 2D or 3D points
    type: str
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'Obj':
        return cls(
            name=data['name'],
            type=data['type'],
            dimensions=data['dimensions'],
            metadata=data['metadata'],
            points=np.array(data['points'], dtype=np.float64),
            type=data['type']
        )
    
    def to_dict(self) -> Dict:
        return {
            'name': self.name,
            'type': self.type,
            'dimensions': self.dimensions,
            'metadata': self.metadata,
            'points': self.points.tolist(),
            'type': self.type
        }

@dataclass
class DetectedObject:
    obj: Obj
    position: List[float]  # Center position [x, y, z]
    rotation: List[float]  # Rotation [x, y, z]
    confidence: float    # Detection confidence [0-1]
    led_points: List[Point3D]  # Detected LED points

class ObjectManager:
    def __init__(self, config_path: Path = Path("config/objects.json")):
        self.path = config_path
        self.objects = []
        print(f"Loading objects from {self.path}")
        with open(self.path, 'r') as f:
            data = json.load(f)
        for obj_data in data:
            self.objects.append(Obj.from_dict(obj_data))
        print(f"Loaded {len(self.objects)} objects: {[obj.name for obj in self.objects]}")

class ObjectDetector:
    def __init__(self, object_manager: ObjectManager, manager: SyncManager, flags: ProcessFlags):
        self.objects = object_manager.objects  # This is just a list that can be pickled
        self.flags = flags  # Store the shared flags instance
        
        # Queue structure:
        # input_queue: (ts, Dict[camera_id, IntrinsicParams], Dict[camera_id, ExtrinsicParams], world_points, epipolar_lines)
        # output_queue: (ts, List[DetectedObject])
        # viz_queue: same as output_queue but non-blocking
        self._running = Event()
        self.input_queue = Queue(maxsize=5)
        self.output_queue = Queue(maxsize=5)
        self.viz_queue = Queue(maxsize=10)
        
        # Create detection process
        self.detect_process = None  # Initialize to None, will create when starting
        
        # Initialize timing stats with shared memory
        self.timing_stats = manager.dict({
            'pattern_matching': manager.list(),
            'total_detection': manager.list(),
            'tracking': manager.list()
        })
        self.max_stats_samples = 100  # Keep last 100 samples
        
        # Initialize object tracker
        self.tracker = ObjectTracker()

    def start(self):
        """Start object detection process"""
        if self.detect_process is not None:
            # If process exists but is not alive, clean it up
            if not self.detect_process.is_alive():
                self.detect_process.join()
            else:
                print("ObjectDetector is already running")
                return
                
        # Create new process
        self._running.set()
        self.detect_process = Process(target=self._detect_loop)
        self.detect_process.start()
        print("ObjectDetector process started")

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
        
    def stop(self):
        """Stop object detection process"""
        self._running.clear()
        
        if self.detect_process is not None:
            self.detect_process.join()
            self.detect_process = None  # Clear the process reference
        
    def _detect_loop(self):
        """Continuously detect objects from 3D points"""
        import sys
        print("ObjectDetector loop started", flush=True)
        while self._running.is_set():
            try:
                ts, intrinsics, extrinsics, world_points, _ = self.input_queue.get(timeout=0.1)
                print(f"Received {len(world_points)} points from world reconstructor", flush=True)
                
                if not world_points:
                    continue
                    
                # Collect timestamps
                timestamps = {
                    'detect_start': time.time()
                }
                
                # Detect objects from 3D points
                detected_objects = self._detect_objects(world_points)
                timestamps['detect_end'] = time.time()
                
                if detected_objects:
                    print(f"Detected {len(detected_objects)} objects: {[obj.obj.name for obj in detected_objects]}", flush=True)
                
                # Track objects
                tracking_start = time.time()
                tracked_objects = self.tracker.update(detected_objects, ts)
                timestamps['tracking_end'] = time.time()
                
                if tracked_objects:
                    print(f"Tracking {len(tracked_objects)} objects: {[obj.obj.name for obj in tracked_objects]}", flush=True)
                    output_data = (ts, tracked_objects)
                    try:
                        # Send to processing pipeline
                        self.output_queue.put(output_data, block=False)
                        # Also send to visualization (non-blocking)
                        self.viz_queue.put(output_data, block=False)
                    except Full:
                        print("ObjectDetector output queue full", flush=True)
                
            except Empty:
                continue
                
    def _detect_objects(self, points: List[Point3D]) -> List[DetectedObject]:
        """Detect objects by matching LED patterns in 3D points"""
        
        detected_objects = []
        if not points:
            return detected_objects
            
        print(f"Detecting objects from {len(points)} points")
        points_array = np.array([[p.x, p.y, p.z] for p in points])
        
        # Calculate pairwise distances between all points (in mm)
        distances = np.sqrt(np.sum((points_array[:, None] - points_array[None, :]) ** 2, axis=2))
        
        # For each object type we're looking for
        for obj in self.objects:
            if obj.type == "drone":
                #print(f"Looking for object pattern: {obj.name}")
                # Get the pattern points
                pattern = obj.points
                
                # Calculate pairwise distances in the pattern (in mm)
                pattern_distances = np.sqrt(np.sum((pattern[:, None] - pattern[None, :]) ** 2, axis=2))
                print(f"Pattern distances:\n{pattern_distances}")
                
                # Try to find matching distance patterns
                used_points = set()
                
                # For each point as a potential first LED
                for i in range(len(points)):
                    if i in used_points:
                        continue
                        
                    print(f"\nTrying point {i} as first LED")
                    # Find points that could be other LEDs based on distances
                    potential_matches = []
                    for pattern_idx in range(len(pattern)):
                        matches = []
                        for j in range(len(points)):
                            if j == i or j in used_points:
                                continue
                            # Check if distance matches any pattern distance (with tolerance)
                            dist = distances[i, j]
                            pattern_dists = pattern_distances[pattern_idx]
                            # Use 20mm tolerance
                            if any(abs(dist - pd) < 20 for pd in pattern_dists):
                                matches.append(j)
                                print(f"Point {j} matches pattern point {pattern_idx} with distance {dist:.1f}mm")
                        potential_matches.append(matches)
                    
                    # If we found potential matches for all pattern points
                    if all(potential_matches):
                        print(f"Found potential matches for all pattern points: {potential_matches}")
                        # Try all combinations of potential matches
                        from itertools import product
                        for match_combination in product(*potential_matches):
                            if len(set(match_combination)) != len(match_combination):
                                continue  # Skip if same point used multiple times
                                
                            # Get matched points
                            matched_points = [points[i]] + [points[idx] for idx in match_combination]
                            matched_array = np.array([[p.x, p.y, p.z] for p in matched_points])
                            
                            # Calculate center position
                            center = np.mean(matched_array, axis=0)
                            
                            # Calculate rotation (heading) from first two points
                            direction = matched_array[1] - matched_array[0]
                            heading = np.arctan2(direction[1], direction[0])
                            
                            # Calculate match confidence based on distance errors
                            errors = []
                            for p1_idx in range(len(matched_array)):
                                for p2_idx in range(p1_idx + 1, len(matched_array)):
                                    actual_dist = np.linalg.norm(matched_array[p1_idx] - matched_array[p2_idx])
                                    pattern_dist = np.linalg.norm(pattern[p1_idx] - pattern[p2_idx])
                                    errors.append(abs(actual_dist - pattern_dist))
                            confidence = np.exp(-np.mean(errors) * 0.01)  # Adjusted for mm scale
                            
                            print(f"Match confidence: {confidence:.3f}")
                            if confidence > 0.5:  # Only accept high confidence detections
                                print(f"Found object {obj.name} with confidence {confidence:.3f}")
                                detected_objects.append(DetectedObject(
                                    obj=obj,
                                    position=center,
                                    rotation=[heading, 0, 0],
                                    confidence=confidence,
                                    led_points=matched_points
                                ))
                                used_points.update([i] + list(match_combination))
                                break
        
        print(f"Detected {len(detected_objects)} objects")
        return detected_objects

class ObjectTracker:
    def __init__(self):
        # State transition matrix (position, velocity, rotation, angular velocity)
        self.F = np.array([
            [1, 0, 0, 1, 0, 0, 0, 0],  # x = x + vx
            [0, 1, 0, 0, 1, 0, 0, 0],  # y = y + vy
            [0, 0, 1, 0, 0, 1, 0, 0],  # z = z + vz
            [0, 0, 0, 1, 0, 0, 0, 0],  # vx = vx
            [0, 0, 0, 0, 1, 0, 0, 0],  # vy = vy
            [0, 0, 0, 0, 0, 1, 0, 0],  # vz = vz
            [0, 0, 0, 0, 0, 0, 1, 1],  # theta = theta + omega
            [0, 0, 0, 0, 0, 0, 0, 1]   # omega = omega
        ])
        
        # Measurement matrix (we only measure position and rotation)
        self.H = np.array([
            [1, 0, 0, 0, 0, 0, 0, 0],  # x
            [0, 1, 0, 0, 0, 0, 0, 0],  # y
            [0, 0, 1, 0, 0, 0, 0, 0],  # z
            [0, 0, 0, 0, 0, 0, 1, 0]   # theta
        ])
        
        # Process noise covariance
        self.Q = np.eye(8) * 0.1
        
        # Measurement noise covariance
        self.R = np.eye(4) * 0.1
        
        # Tracked objects dictionary: {object_id: (kalman_filter, last_update_time)}
        self.tracked_objects = {}
        self.next_id = 0
        
        # Maximum time between updates before considering track lost
        self.max_time_between_updates = 0.5  # seconds
        
    def update(self, detected_objects: List[DetectedObject], timestamp: float) -> List[DetectedObject]:
        """Update tracked objects with new detections"""
        # Convert detections to measurement format
        measurements = []
        for obj in detected_objects:
            measurement = np.array([
                obj.position[0],  # x
                obj.position[1],  # y
                obj.position[2],  # z
                obj.rotation[0]   # theta
            ])
            measurements.append((measurement, obj))
            
        # Update existing tracks
        updated_objects = []
        for track_id, (kf, last_update) in self.tracked_objects.items():
            # Check if track is too old
            if timestamp - last_update > self.max_time_between_updates:
                continue
                
            # Predict
            kf.predict()
            
            # Find best matching detection
            best_match = None
            best_match_dist = float('inf')
            for measurement, obj in measurements:
                # Calculate distance between predicted and measured position
                predicted_pos = kf.x[:3]  # First 3 elements are position
                measured_pos = measurement[:3]
                dist = np.linalg.norm(predicted_pos - measured_pos)
                
                if dist < best_match_dist:
                    best_match_dist = dist
                    best_match = (measurement, obj)
            
            # Update if good match found
            if best_match is not None and best_match_dist < 0.5:  # 50cm threshold
                measurement, obj = best_match
                kf.update(measurement)
                measurements.remove((measurement, obj))
                
                # Create updated object
                state = kf.x
                updated_obj = DetectedObject(
                    obj=obj.obj,
                    position=state[:3],
                    rotation=[state[6], 0, 0],  # Only track yaw for now
                    confidence=obj.confidence,
                    led_points=obj.led_points
                )
                updated_objects.append(updated_obj)
                
                # Update last update time
                self.tracked_objects[track_id] = (kf, timestamp)
        
        # Create new tracks for unmatched detections
        for measurement, obj in measurements:
            # Initialize Kalman filter
            kf = cv2.KalmanFilter(8, 4)  # 8 state variables, 4 measurements
            kf.transitionMatrix = self.F
            kf.measurementMatrix = self.H
            kf.processNoiseCov = self.Q
            kf.measurementNoiseCov = self.R
            
            # Initialize state
            initial_state = np.zeros(8)
            initial_state[:3] = measurement[:3]  # Position
            initial_state[6] = measurement[3]    # Rotation
            kf.statePre = initial_state
            kf.statePost = initial_state
            
            # Add to tracked objects
            self.tracked_objects[self.next_id] = (kf, timestamp)
            self.next_id += 1
            
            # Add to updated objects
            updated_objects.append(obj)
        
        return updated_objects

class ObjectsManager:
    def __init__(self, path: Path = 'config/objects.json'):
        self.objects: List[Obj] = []
        self.path = path
        self.load_objects()

    def add_object(self, obj: Obj):
        self.objects.append(obj)

    def remove_object(self, name: str):
        for obj in self.objects:
            if obj.name == name:
                self.objects.remove(obj)

    def get_object(self, name: str) -> Obj:
        for obj in self.objects:
            if obj.name == name:
                return obj
        return None

    def get_object_points(self, name: str) -> List[np.ndarray]:
        obj = self.get_object(name)
        if obj is not None:
            return obj.points
        return None

    def plot_object(self, name: str):
        obj = self.get_object(name)
        if obj is not None:
            fig, ax = plt.subplots()
            # Set plot properties
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.2)
                    
            # Draw coordinate axes
            ax.axhline(y=0, color='gray', alpha=0.3, linestyle='-')
            ax.axvline(x=0, color='gray', alpha=0.3, linestyle='-')

            # Set background colors
            ax.set_facecolor('black')
            fig.patch.set_facecolor('black')

            for point in obj.points:
                if obj.dimensions == 2:
                    ax.scatter(point[0], point[1])
                elif obj.dimensions == 3:
                    ax.scatter(point[0], point[1], point[2])
                # Add point coordinates as labels
                ax.annotate(f'({point[0]},{point[1]})', (point[0], point[1]), 
                   xytext=(5, 5), textcoords='offset points',
                   color='blue', fontsize=8)
            plt.show()

    def plot_objects(self):
        fig, ax = plt.subplots()
        for obj in self.objects:
            for point in obj.points:
                if obj.dimensions == 2:
                    ax.scatter(point[0], point[1])
                elif obj.dimensions == 3:
                    ax.scatter(point[0], point[1], point[2])
        plt.show()

    def save_objects(self):
        with open(self.path, 'w') as f:
            json.dump([obj.__dict__ for obj in self.objects], f)

    def load_objects(self):
        self.objects = []
        with open(self.path, 'r') as f:
            data = json.load(f)
        for obj_data in data:
            self.objects.append(Obj.from_dict(obj_data))


