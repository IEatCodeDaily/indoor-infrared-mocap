from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, List, Callable, Any
import numpy as np
import json
from multiprocessing import Event


@dataclass
class IntrinsicParams:
    matrix: np.ndarray
    distortion: np.ndarray
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'IntrinsicParams':
        return cls(
            matrix=np.array(data['intrinsic_matrix'], dtype=np.float64),
            distortion=np.array(data['distortion_coef'], dtype=np.float64)
        )
    
    def to_dict(self) -> Dict:
        return {
            'intrinsic_matrix': self.matrix.tolist(),
            'distortion_coef': self.distortion.tolist()
        }

@dataclass
class ExtrinsicParams:
    t: np.ndarray  # 3D position vector
    R: np.ndarray  # 3x3 rotation matrix
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'ExtrinsicParams':
        return cls(
            t=np.array(data['t'], dtype=np.float64),
            R=np.array(data['R'], dtype=np.float64)
        )
    
    def to_dict(self) -> Dict:
        return {
            't': self.t.tolist(),
            'R': self.R.tolist()
        }

class CameraParamsManager:
    def __init__(self, config_dir: Optional[Path] = Path("config")):
        self.config_dir = config_dir
        self.intrinsic: Dict[int, IntrinsicParams] = {}
        self.extrinsic: Dict[int, ExtrinsicParams] = {}
        self._load_params()
    
    def _load_params(self) -> None:
        """Load both intrinsic and extrinsic parameters on initialization"""
        self.load_intrinsic_params()
        self.load_extrinsic_params()
    
    def load_intrinsic_params(self, filename: Path = "camera-intrinsic.json") -> None:
        """Load and parse intrinsic parameters"""
        path = self.config_dir / Path(filename)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        
        with path.open('r') as f:
            data = json.load(f)
            
        self.intrinsic.clear()
        for cam_data in data:
            camera_id = cam_data['camera_id']
            self.intrinsic[camera_id] = IntrinsicParams.from_dict(cam_data)
    
    def load_extrinsic_params(self, filename: Path = "camera-extrinsic.json") -> None:
        """Load and parse extrinsic parameters"""
        path = self.config_dir / Path(filename)
        if not path.exists():
            return  # Extrinsic params are optional
        
        with path.open('r') as f:
            data = json.load(f)
            
        self.extrinsic.clear()
        for cam_data in data:
            camera_id = cam_data['camera_id']
            self.extrinsic[camera_id] = ExtrinsicParams.from_dict(cam_data)
            
    def get_intrinsic_params(self, camera_id: int = None) -> Optional[IntrinsicParams]:
        """Get intrinsic parameters for a specific camera"""
        if camera_id == None:
            return self.intrinsic
        return self.intrinsic.get(camera_id)

    def get_extrinsic_params(self, camera_id: int = None) -> Optional[ExtrinsicParams]:
        """Get extrinsic parameters for a specific camera"""
        if camera_id == None:
            return self.extrinsic
        return self.extrinsic.get(camera_id)
    

    def get_all_camera_ids(self) -> list[int]:
        """Get list of all available camera IDs"""
        return list(self.intrinsic.keys())
    
    def save_intrinsic_params(self, filename: str = "camera-intrinsic.json") -> None:
        """Save intrinsic parameters to JSON file"""
        path = self.config_dir / filename
        data = []
        
        # Create data list from current parameters
        for camera_id, params in self.intrinsic.items():
            cam_data = params.to_dict()
            cam_data['camera_id'] = camera_id
            data.append(cam_data)
        
        # Save to file (overwriting any existing content)
        with path.open('w') as f:
            json.dump(data, f, indent=4)
    
    def save_extrinsic_params(self, filename: str = "camera-extrinsic.json") -> None:
        """Save extrinsic parameters to JSON file"""
        path = self.config_dir / filename
        data = []
        
        # Create data list from current parameters
        for camera_id, params in self.extrinsic.items():
            cam_data = params.to_dict()
            cam_data['camera_id'] = camera_id
            data.append(cam_data)
        
        # Save to file (overwriting any existing content)
        with path.open('w') as f:
            json.dump(data, f, indent=4)
    
    def update_intrinsic(self, intrinsic_params: Dict[int, IntrinsicParams]) -> None:
        """Update intrinsic parameters"""
        self.intrinsic = intrinsic_params
    
    def update_extrinsic(self, extrinsic_params: Dict[int, ExtrinsicParams]) -> None:
        """Update extrinsic parameters"""
        self.extrinsic = extrinsic_params

class ProcessFlags:
    def __init__(self):
        self.timing_stats = Event()
        self.debug_mode = Event()
        self.visualization_enabled = Event()
        
    def set_flag(self, flag_name: str, state: bool):
        """Set a flag's state"""
        if hasattr(self, flag_name):
            flag = getattr(self, flag_name)
            if state:
                flag.set()
            else:
                flag.clear()
                
    def get_flag(self, flag_name: str) -> bool:
        """Get a flag's state"""
        if hasattr(self, flag_name):
            flag = getattr(self, flag_name)
            return flag.is_set()
        return False
        
    def set_flags(self, **kwargs):
        """Set multiple flags at once"""
        for flag_name, state in kwargs.items():
            self.set_flag(flag_name, state)

