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
# Objects dataclass
@dataclass
class Obj:
    name: str
    type: str
    dimensions: int
    metadata: Dict[str, Any]
    points: np.ndarray  # contains 2D or 3D points
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'Obj':
        return cls(
            name=data['name'],
            type=data['type'],
            dimensions=data['dimensions'],
            metadata=data['metadata'],
            points=np.array(data['points'], dtype=np.float64)
        )
    
    def to_dict(self) -> Dict:
        return {
            'name': self.name,
            'type': self.type,
            'dimensions': self.dimensions,
            'metadata': self.metadata,
            'points': self.points.tolist()
        }

@dataclass
class DetectedObject:
    obj: Obj
    position: np.ndarray  # Center position [x, y, z]
    rotation: float      # Rotation around vertical axis (heading)
    confidence: float    # Detection confidence [0-1]
    led_points: List[Point3D]  # Detected LED points

class ObjectManager:
    def __init__(self, config_path: Path = Path("config/objects.json")):
        self.path = config_path
        self.objects = []
        with open(self.path, 'r') as f:
            data = json.load(f)
        for obj_data in data:
            self.objects.append(Obj.from_dict(obj_data))

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
            'total_detection': manager.list()
        })
        self.max_stats_samples = 100  # Keep last 100 samples

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
        while self._running.is_set():
            try:
                
                # Get world reconstruction data
                ts, intrinsics, extrinsics, world_points, _ = self.input_queue.get(timeout=0.1)
                
                if not world_points:
                    continue
                    
                # Collect timestamps
                timestamps = {
                    'detect_start': time.time()
                }
                

                # Detect objects from 3D points
                detected_objects = self._detect_objects(world_points, timestamps)
                timestamps['detect_end'] = time.time()
                
                if detected_objects:
                    output_data = (ts, detected_objects)
                    try:
                        # Send to processing pipeline
                        self.output_queue.put(output_data, block=False)
                        # Also send to visualization (non-blocking)
                        self.viz_queue.put(output_data, block=False)
                    except Full:
                        print("ObjectDetector output queue full")
                
                
                
            except Empty:
                continue
                
    def _detect_objects(self, points: List[Point3D]) -> List[DetectedObject]:
        """Detect objects by matching LED patterns in 3D points"""
        
        detected_objects = []
        points_array = np.array([[p.x, p.y, p.z] for p in points])
        
        # Calculate pairwise distances between all points
        distances = np.sqrt(np.sum((points_array[:, None] - points_array[None, :]) ** 2, axis=2))
        
        # For each object type we're looking for
        for obj in self.objects:
            # Get the pattern points
            pattern = obj.points
            
            # Calculate pairwise distances in the pattern
            pattern_distances = np.sqrt(np.sum((pattern[:, None] - pattern[None, :]) ** 2, axis=2))
            
            # Try to find matching distance patterns
            used_points = set()
            
            # For each point as a potential first LED
            for i in range(len(points)):
                if i in used_points:
                    continue
                    
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
                        if any(abs(dist - pd/1000) < 0.02 for pd in pattern_dists):  # Convert mm to m, 2cm tolerance
                            matches.append(j)
                    potential_matches.append(matches)
                
                # If we found potential matches for all pattern points
                if all(potential_matches):
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
                                pattern_dist = np.linalg.norm(pattern[p1_idx] - pattern[p2_idx]) / 1000  # mm to m
                                errors.append(abs(actual_dist - pattern_dist))
                        confidence = np.exp(-np.mean(errors) * 10)  # Convert errors to confidence score
                        
                        if confidence > 0.5:  # Only accept high confidence detections
                            detected_objects.append(DetectedObject(
                                obj=obj,
                                position=center,
                                rotation=heading,
                                confidence=confidence,
                                led_points=matched_points
                            ))
                            used_points.update([i] + list(match_combination))
                            break
        
        return detected_objects

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


