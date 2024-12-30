from typing import List
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
            raise ValueError(f"Number of video sources ({len(video_sources)}) "
                           f"does not match number of cameras ({len(self.cameras)})")
        
        for camera, source in zip(self.cameras, video_sources):
            camera.start(source)
    
    def stop_all(self):
        """Stop all cameras"""
        for camera in self.cameras:
            camera.stop()
    