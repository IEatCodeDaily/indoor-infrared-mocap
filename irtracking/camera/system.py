from typing import List
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
            raise ValueError(f"Number of video sources ({len(video_sources)}) "
                           f"does not match number of cameras ({len(self.cameras)})")
        
        for camera, source in zip(self.cameras, video_sources):
            camera.start(source)
    
    def stop_all(self):
        """Stop all cameras"""
        for camera in self.cameras:
            camera.stop()
    