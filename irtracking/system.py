from typing import List, Dict, Tuple
from .capture import PointDetector
from .params import CameraParamsManager, ExtrinsicParams, ProcessFlags
from .collector import OutputCollector
from pathlib import Path
import cv2
import time
from threading import Thread, Event
from queue import Full, Empty
from multiprocessing import Queue
from multiprocessing.managers import SyncManager
import numpy as np
from collections import deque
#from pseyepy import Camera

# TODO:
# - Add pseyepy
# - tidy up everything and aggregate all localization systems into one

class LocalizationSystem:
    def __init__(self,
                 cameras: List[cv2.VideoCapture],
                 detectors: List[PointDetector],
                 output_collector: OutputCollector,
                 params_manager: CameraParamsManager,
                 manager: SyncManager,
                 flags: ProcessFlags = None,
                 buffer_size: int = 30):
        self.cameras = cameras
        self.detectors = detectors
        self.output_collector = output_collector
        self.params_manager = params_manager
        self._running = Event()
        self.frame_delay = 0.01  # 10ms default delay
        self.feed_process = None
        self.flags = flags or ProcessFlags()  # Use provided flags or create new ones
        self.frame_counter = 0  # Single frame counter for all cameras
        # Initialize timing stats with shared memory
        self.timing_stats = manager.dict({
            'frame_capture': manager.list(),
            'param_fetch': manager.list(),
            'total_feed': manager.list(),
            'buffer_wait': manager.list()
        })
        self.max_stats_samples = 100
        # New attributes for preloaded frames
        self.preloaded_frames = []
        self.current_frame_index = 0
        self.total_frames = 0
    
    def _preload_all_frames(self):
        """Preload all frames from all cameras into memory"""
        print("Preloading all frames...")
        self.preloaded_frames = []
        self.current_frame_index = 0
        
        # Get total frames from first camera (assuming all videos have same length)
        self.total_frames = int(self.cameras[0].get(cv2.CAP_PROP_FRAME_COUNT))
        
        for frame_idx in range(self.total_frames):
            frames = []
            for camera in self.cameras:
                ret, frame = camera.read()
                if not ret:
                    print(f"Failed to read frame {frame_idx} from a camera")
                    return False
                frames.append(frame)
            self.preloaded_frames.append((frame_idx, time.time(), frames))
        
        print(f"Successfully preloaded {len(self.preloaded_frames)} frames")
        return True

    def _update_timing(self, category: str, duration: float):
        """Update timing statistics for a category"""
        if not self.flags.get_flag('timing_stats'):
            return
        
        stats_list = self.timing_stats[category]
        stats_list.append(duration)
        if len(stats_list) > self.max_stats_samples:
            stats_list.pop(0)
        self.timing_stats[category] = stats_list  # Update the shared dict
    
    def get_timing_stats(self):
        """Get average timing statistics"""
        if not self.flags.get_flag('timing_stats'):
            return {}
        
        stats = {}
        for category, times in self.timing_stats.items():
            if times:
                avg_time = sum(times) / len(times)
                stats[category] = {
                    'avg': avg_time * 1000,  # Convert to ms
                    'min': min(times) * 1000,
                    'max': max(times) * 1000
                }
        return stats
    
    def _feed_loop(self):
        """Feed preloaded frames to detectors."""
        params = {}
        for i in range(len(self.cameras)):
            intrinsic = self.params_manager.get_intrinsic_params(i)
            extrinsic = self.params_manager.get_extrinsic_params(i)
            if intrinsic is None:
                print(f"No intrinsic parameters for camera {i}")
                continue
            if extrinsic is None:
                extrinsic = ExtrinsicParams(
                    R=np.eye(3),
                    t=np.zeros(3)
                )
            params[i] = (intrinsic, extrinsic)
        
        while self._running.is_set() and self.current_frame_index < len(self.preloaded_frames):
            feed_start = time.time()
            
            frame_number, ts, frames = self.preloaded_frames[self.current_frame_index]
            self.current_frame_index += 1
            
            if len(frames) != len(self.cameras):
                print(f"Frame set incomplete: got {len(frames)}/{len(self.cameras)}")
                continue
            
            for detector, frame, cam_id in zip(self.detectors, frames, range(len(frames))):
                if detector.camera_id not in params:
                    continue
                intrinsic, extrinsic = params[detector.camera_id]
                try:
                    while detector.input_queue.qsize() > 10:
                        try:
                            detector.input_queue.get_nowait()
                        except Empty:
                            break
                    detector.input_queue.put(
                        (ts, detector.camera_id, intrinsic, extrinsic, frame, frame_number),
                        timeout=0.1
                    )
                except Full:
                    print(f"PointDetector {detector.camera_id} input queue full")
            
            try:
                self.output_collector.add_frame(ts, [detector.camera_id for detector in self.detectors], frames)
            except Full:
                print("Visualization queue full")
            
            elapsed = time.time() - feed_start
            if elapsed < self.frame_delay:
                time.sleep(self.frame_delay - elapsed)
    
    def start(self):
        """Start the system"""
        if self.feed_process is not None and self.feed_process.is_alive():
            print("LocalizationSystem is already running")
            return
        
        # Preload all frames first
        if not self._preload_all_frames():
            print("Failed to preload frames")
            return
        
        self._running.set()
        self.feed_process = Thread(target=self._feed_loop)
        self.feed_process.start()
        print("System started with preloaded frames")
    
    def stop(self):
        """Stop the system"""
        self._running.clear()
        if self.feed_process is not None:
            self.feed_process.join()
            self.feed_process = None
        # Clear preloaded frames to free memory
        self.preloaded_frames = []
        self.current_frame_index = 0
    
    def start_all(self, video_sources: List[str]):
        """Start all cameras with their respective video sources"""
        if len(video_sources) != len(self.detectors):
            raise ValueError(f"Number of video sources ({len(video_sources)}) "
                           f"does not match number of cameras ({len(self.cameras)})")
        
        for camera, source in zip(self.cameras, video_sources):
            camera.open(source) 
    
    def stop_all(self):
        """Stop all cameras"""
        for camera in self.cameras:
            camera.release()
    

