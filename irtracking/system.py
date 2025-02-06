from typing import List
from .capture import PointDetector
from .params import CameraParamsManager, ExtrinsicParams
from .collector import OutputCollector
from pathlib import Path
import cv2
import time
from threading import Thread, Event
import queue
from multiprocessing import Queue
from multiprocessing.managers import SyncManager
import numpy as np
#from pseyepy import Camera

class LocalizationSystem:
    def __init__(self,
                 cameras: List[cv2.VideoCapture],
                 detectors: List[PointDetector],
                 output_collector: OutputCollector,
                 params_manager: CameraParamsManager,
                 manager: SyncManager):
        self.cameras = cameras
        self.detectors = detectors
        self.output_collector = output_collector
        self.params_manager = params_manager
        self._running = Event()
        self.frame_delay = 0.03  # 30ms default delay
        self.feed_process = None  # Initialize to None, will create when starting
        
        # Initialize timing stats with shared memory
        self.timing_stats = manager.dict({
            'frame_capture': manager.list(),
            'param_fetch': manager.list(),
            'total_feed': manager.list()
        })
        self.max_stats_samples = 100  # Keep last 100 samples
    
    def _update_timing(self, category: str, duration: float):
        """Update timing statistics for a category"""
        self.timing_stats[category].append(duration)
        if len(self.timing_stats[category]) > self.max_stats_samples:
            self.timing_stats[category].pop(0)
    
    def get_timing_stats(self):
        """Get average timing statistics"""
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
    
    def start(self):
        """Start the system"""
        if self.feed_process is not None:
            # If thread exists but is not alive, clean it up
            if not self.feed_process.is_alive():
                self.feed_process.join()
            else:
                print("LocalizationSystem is already running")
                return
                
        # Create new thread
        self._running.set()
        self.feed_process = Thread(target=self._feed_loop)
        self.feed_process.start()
    
    def stop(self):
        """Stop the system"""
        self._running.clear()
        
        if self.feed_process is not None:
            self.feed_process.join()
            self.feed_process = None  # Clear the thread reference
    
    def set_frame_delay(self, delay: float):
        """Set the delay between frames in seconds"""
        self.frame_delay = delay
    
    def _feed_loop(self):
        """Continuously feed frames from video sources to detectors"""
        last_ts = 0
        while self._running.is_set():
            feed_start = time.time()
            
            # Read frames from all cameras
            capture_start = time.time()
            frames = []
            ts = time.time()
            camera_ids = []
            
            # Only proceed if we can read from all cameras
            all_frames_read = True
            for i, camera in enumerate(self.cameras):
                ret, frame = camera.read()
                if not ret:
                    # If we reach the end of any video, reset all videos
                    print(f"Resetting videos to start (camera {i} reached end)")
                    for cam in self.cameras:
                        cam.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    all_frames_read = False
                    break
                frames.append(frame)
                camera_ids.append(i)
            
            if not all_frames_read:
                continue
                
            self._update_timing('frame_capture', time.time() - capture_start)

            # If we have frames from all cameras, feed them to detectors
            if len(frames) == len(self.cameras):
                # Get camera parameters once for all cameras
                param_start = time.time()
                params = {}
                for i in range(len(self.cameras)):
                    intrinsic = self.params_manager.get_intrinsic_params(i)
                    extrinsic = self.params_manager.get_extrinsic_params(i)
                    
                    if intrinsic is None:
                        print(f"No intrinsic parameters for camera {i}")
                        continue
                        
                    # Extrinsic parameters might be None for some cameras
                    if extrinsic is None:
                        extrinsic = ExtrinsicParams(
                            R=np.eye(3),
                            t=np.zeros(3)
                        )
                    params[i] = (intrinsic, extrinsic)
                    
                self._update_timing('param_fetch', time.time() - param_start)

                # Send frames to all detectors simultaneously
                for detector, frame in zip(self.detectors, frames):
                    if detector.camera_id not in params:
                        continue
                        
                    intrinsic, extrinsic = params[detector.camera_id]
                    
                    try:
                        # Try to clear old frames from input queue
                        while True:
                            try:
                                detector.input_queue.get_nowait()
                            except queue.Empty:
                                break
                                
                        # Send timestamp, camera ID, parameters, and frame
                        detector.input_queue.put(
                            (ts, detector.camera_id, intrinsic, extrinsic, frame),
                            timeout=0.1
                        )
                    except queue.Full:
                        print(f"PointDetector {detector.camera_id} input queue full")
                
                self._update_timing('total_feed', time.time() - feed_start)

                # Send frames to visualization queue
                try:
                    # Clear old frames from visualization queue
                    while True:
                        try:
                            self.output_collector.frame_queues.get_nowait()
                        except queue.Empty:
                            break
                            
                    self.output_collector.add_frame(ts, camera_ids, frames)
                except queue.Full:
                    print("Visualization queue full")
                    

            # Maintain frame rate
            elapsed = time.time() - feed_start
            if elapsed < self.frame_delay:
                time.sleep(self.frame_delay - elapsed)

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
    

