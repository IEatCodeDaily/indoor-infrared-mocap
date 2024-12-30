from typing import List
<<<<<<< HEAD
from .capture import *
from .params import *
from pathlib import Path
import json
from pseyepy import Camera

class TrackerSystem:
    def __init__(self, cameras: Camera):
        self.params_manager = CameraParamsManager()
        self.trackers: List[PointDetector] = []
        self.cameras = cameras
    
    def _system_loop(self):
        """Continuously capture frames from cameras and detect points"""
        while self._running:
            for camera in self.cameras:
                frame = camera.get_frame()
                self.frame_in_queue.put(frame)

    def start_all(self, video_sources: List[str]):
        """Start all cameras with their respective video sources"""
        if len(video_sources) != len(self.trackers):
=======
from .capture import Camera
from .params import CameraParamsManager
from pathlib import Path
import json

class CameraSystem:
    def __init__(self, params_file: str):
        self.cameras: List[Camera] = []
        self._load_camera_params(params_file)
    
    def _load_camera_params(self, params_file: str):
        """Load camera parameters from JSON file"""
        params_path = Path(params_file)
        if not params_path.exists():
            raise FileNotFoundError(f"Camera parameters file not found: {params_file}")
        
        with params_path.open('r') as f:
            params_list = json.load(f)
            
        for cam_id, params in enumerate(params_list):
            camera_params = CameraParamsManager.from_dict(params)
            camera = Camera(
                camera_id=cam_id,
                camera_matrix=camera_params.intrinsic_matrix,
                dist_coeffs=camera_params.distortion_coef,
                rotation=camera_params.rotation
            )
            self.cameras.append(camera)
    
    def start_all(self, video_sources: List[str]):
        """Start all cameras with their respective video sources"""
        if len(video_sources) != len(self.cameras):
>>>>>>> 24f5e0fd324c64481b858610791092b677e3c71f
            raise ValueError(f"Number of video sources ({len(video_sources)}) "
                           f"does not match number of cameras ({len(self.cameras)})")
        
        for camera, source in zip(self.cameras, video_sources):
            camera.start(source)
    
    def stop_all(self):
        """Stop all cameras"""
        for camera in self.cameras:
            camera.stop()
    