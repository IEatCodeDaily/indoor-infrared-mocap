import tkinter as tk
from tkinter import ttk
import cv2
from pathlib import Path
import os
import time 
from multiprocessing import Manager
from multiprocessing.managers import SyncManager
from irtracking.system import *
from irtracking.capture import *
from irtracking.world import *
from irtracking.collector import *
from irtracking.params import *
from irtracking.objects import *
import multiprocessing

# Add debug prints for pseyepy import
print("Attempting to import pseyepy...")
try:
    import pseyepy
    print("Successfully imported pseyepy")
    print(f"pseyepy version: {pseyepy.__version__ if hasattr(pseyepy, '__version__') else 'unknown'}")
    print(f"pseyepy path: {pseyepy.__file__}")
except ImportError as e:
    print(f"Failed to import pseyepy: {e}")
    raise


# TODO:
# - Fix ProcessFlags and TimingStats

class TrackingGUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("IR Tracking System")
        
        # Initialize multiprocessing manager
        self.manager = Manager()
        if self.manager is None:
            raise RuntimeError("Failed to create multiprocessing manager")    

        # Create tkinter variable for GUI checkbox
        self.timing_enabled_var = tk.BooleanVar(value=True)
        
        # Initialize system components
        self.params_manager = CameraParamsManager()
        base_path = Path(os.path.dirname(os.path.dirname(__file__)))
        self.object_manager = ObjectManager(config_path=base_path / "config" / "objects.json")
        
        # Initialize video sources
        self.video_sources = []
        self.cameras = []
        self.pseyepy_cameras = []
        self.is_live_mode = False
        
        # Create shared flags using the manager
        self.process_flags = ProcessFlags()
        self.process_flags.timing_stats.clear()
        
        # Initialize components and share the SAME flags instance
        self.detectors = []
        self.world_reconstructor = None
        self.object_detector = None
        self.output_collector = None
        self.system = None
        
        # Initialize timing update
        self.timing_update_running = False
        
        self.setup_gui()
        
    def setup_gui(self):
        # Create main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Video Tracking Group
        video_tracking_frame = ttk.LabelFrame(main_frame, text="Video Tracking", padding="5")
        video_tracking_frame.grid(row=0, column=0, columnspan=2, pady=5, sticky=(tk.W, tk.E))
        
        # Control buttons for video tracking
        ttk.Button(video_tracking_frame, text="Start", command=self.start_video_tracking).grid(row=0, column=0, padx=5)
        ttk.Button(video_tracking_frame, text="Stop", command=self.stop_system).grid(row=0, column=1, padx=5)
        
        # Live Tracking Group
        live_tracking_frame = ttk.LabelFrame(main_frame, text="Live Tracking", padding="5")
        live_tracking_frame.grid(row=1, column=0, columnspan=2, pady=5, sticky=(tk.W, tk.E))
        
        # Control buttons for live tracking
        ttk.Button(live_tracking_frame, text="Start", command=self.start_live_tracking).grid(row=0, column=0, padx=5)
        ttk.Button(live_tracking_frame, text="Stop", command=self.stop_system).grid(row=0, column=1, padx=5)
        
        # Timing toggle button
        ttk.Checkbutton(main_frame, text="Show Timing Stats", 
                       variable=self.timing_enabled_var,
                       command=self.toggle_timing_stats).grid(row=2, column=0, columnspan=2, pady=5)
        
        # Video settings frame
        settings_frame = ttk.LabelFrame(main_frame, text="Video Settings", padding="5")
        settings_frame.grid(row=3, column=0, columnspan=2, pady=10, sticky=(tk.W, tk.E))
        
        # Add video controls
        ttk.Label(settings_frame, text="Frame delay (ms):").grid(row=0, column=0, padx=5)
        self.delay_var = tk.StringVar(value="10")
        delay_entry = ttk.Entry(settings_frame, textvariable=self.delay_var, width=10)
        delay_entry.grid(row=0, column=1, padx=5)
        
        # Apply settings button
        ttk.Button(settings_frame, text="Apply", command=self.apply_settings).grid(row=1, column=0, columnspan=2, pady=5)
        
        # Camera parameters frame
        params_frame = ttk.LabelFrame(main_frame, text="Camera Parameters", padding="5")
        params_frame.grid(row=4, column=0, columnspan=2, pady=10, sticky=(tk.W, tk.E))
        
        # Add reload parameters button
        ttk.Button(params_frame, text="Reload Parameters", command=self.reload_parameters).grid(row=0, column=0, pady=5)
        
        # Timing statistics frame
        self.timing_frame = ttk.LabelFrame(main_frame, text="Pipeline Timing (ms)", padding="5")
        self.timing_frame.grid(row=5, column=0, columnspan=2, pady=10, sticky=(tk.W, tk.E))
        
        # Create text widget for timing stats
        self.timing_text = tk.Text(self.timing_frame, height=30, width=50)
        timing_scrollbar = ttk.Scrollbar(self.timing_frame, orient='vertical', command=self.timing_text.yview)
        self.timing_text.configure(yscrollcommand=timing_scrollbar.set)
        
        # Grid layout for text widget and scrollbar
        self.timing_text.grid(row=0, column=0, padx=5, pady=5, sticky=(tk.N, tk.S, tk.E, tk.W))
        timing_scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))

    def initialize_system(self, live_mode=False):
        # Initialize detectors
        self.detectors = [
            PointDetector(i, self.manager, flags=self.process_flags) 
            for i in range(len(self.cameras) if not live_mode else self.pseyepy_cameras.num_cameras)
        ]
        
        # Initialize world reconstructor
        self.world_reconstructor = WorldReconstructor(self.detectors, self.manager, flags=self.process_flags)
        
        # Initialize object detector
        self.object_detector = ObjectDetector(self.object_manager, self.manager, flags=self.process_flags)
        
        # Initialize output collector
        self.output_collector = OutputCollector(
            detectors=self.detectors,
            world_reconstructor=self.world_reconstructor,
            object_detector=self.object_detector,
            manager=self.manager,
            flags=self.process_flags
        )
        
        # Initialize localization system
        self.system = LocalizationSystem(
            cameras=self.cameras if not live_mode else self.pseyepy_cameras,
            detectors=self.detectors,
            output_collector=self.output_collector,
            params_manager=self.params_manager,
            manager=self.manager,
            flags=self.process_flags,
            live_mode=live_mode
        )

    def start_video_tracking(self):
        """Start tracking with video files"""
        if self.is_live_mode:
            self.stop_system()
            
        # Get video sources from calibration footage
        base_path = Path(os.path.dirname(os.path.dirname(__file__))) / "footage_calibration"
        self.video_sources = [
            str(base_path / f"calibration1_{i}.mp4")
            for i in range(4)  # 4 cameras (0-3)
        ]
        
        # Open video files
        self.cameras = []
        for src in self.video_sources:
            cap = cv2.VideoCapture(src)
            if not cap.isOpened():
                raise ValueError(f"Could not open video file: {src}")
            self.cameras.append(cap)
            
        print(f"Loaded {len(self.cameras)} video sources")
        
        # Initialize system components
        self.initialize_system(live_mode=False)
        
        # Start the system
        self.start_system()
        self.is_live_mode = False

    def start_live_tracking(self):
        """Start tracking with pseyepy cameras"""
        if not self.is_live_mode:
            self.stop_system()
            
        # Initialize pseyepy cameras       
        try:
            print("Initializing pseyepy cameras...")
            print(f"Available camera count: {pseyepy.cam_count()}")
            self.pseyepy_cameras = pseyepy.Camera(fps=60, resolution=pseyepy.Camera.RES_LARGE, gain=10, exposure=100)
            print("Successfully created pseyepy Camera object")
        except Exception as e:
            print(f"Failed to initialize pseyepy cameras: {e}")
            print(f"Error type: {type(e)}")
            import traceback
            traceback.print_exc()
            raise
            
        if not self.pseyepy_cameras:
            raise RuntimeError("No pseyepy cameras could be initialized")
            
        # For live mode, self.cameras is not used, but we clear it for safety
        self.cameras = []
        
        print(f"Initialized {self.pseyepy_cameras.num_cameras} pseyepy cameras")
        
        # Initialize system components
        self.initialize_system(live_mode=True)
        
        # Start the system
        self.start_system()
        self.is_live_mode = True

    def start_system(self):
        """Start all system components"""
        # Start all components with flags
        if not self.output_collector.start():
            print("Failed to start output collector")
            return
        
        # Wait for output collector to initialize
        while not self.output_collector._initialized.is_set():
            time.sleep(0.1)

        print("Output collector initialized")

        # Start detectors
        for detector in self.detectors:
            detector.start()
        
        print("Detectors initialized")

        # Start object detector first
        self.object_detector.start()
        print("Object detector initialized")

        # Connect world reconstructor to object detector BEFORE starting it
        self.world_reconstructor.object_detector_queue = self.object_detector.input_queue
        
        # Now start world reconstructor
        self.world_reconstructor.start()
        print("World reconstructor initialized")
        
        # Start the localization system (which will feed frames with parameters)
        self.system.start()
        
        print("Localization system initialized")
        # Start timing stats update
        self.timing_update_running = True
        self.update_timing_stats()

    def stop_system(self):
        """Stop all system components"""
        # Stop timing stats update
        self.timing_update_running = False
        
        # Stop the localization system
        if self.system:
            self.system.stop()
            print("Localization system stopped")
        
        # Stop detectors
        for detector in self.detectors:
            detector.stop()
        print("Detectors stopped")
        
        # Stop world reconstructor
        if self.world_reconstructor:
            self.world_reconstructor.stop()
            print("World reconstructor stopped")
        
        # Stop object detector
        if self.object_detector:
            self.object_detector.stop()
            print("Object detector stopped")
        
        # Stop output collector
        if self.output_collector:
            self.output_collector.stop()
            print("Output collector stopped")
            
        # Stop pseyepy cameras if in live mode
        if self.is_live_mode:
            for camera in self.pseyepy_cameras:
                camera.stop()
            self.pseyepy_cameras = []
            
        # Release OpenCV captures
        for camera in self.cameras:
            camera.release()
        self.cameras = []
        
        # Clear system components
        self.detectors = []
        self.world_reconstructor = None
        self.object_detector = None
        self.output_collector = None
        self.system = None

    def apply_settings(self):
        """Apply video settings"""
        try:
            delay = int(self.delay_var.get())
            # Update frame delay in localization system
            if self.system:
                self.system.set_frame_delay(delay / 1000.0)  # convert to seconds
        except ValueError:
            print("Invalid delay value")
    
    def reload_parameters(self):
        """Reload camera parameters from files and update all components"""
        # Reload parameters
        self.params_manager.load_intrinsic_params()
        self.params_manager.load_extrinsic_params()
        
        # Update world reconstructor's shared parameters
        if self.world_reconstructor:
            self.world_reconstructor.update_params(self.params_manager)
        
        print("Camera parameters reloaded")
    
    def __del__(self):
        """Cleanup when the GUI is closed"""
        self.stop_system()

    def toggle_timing_stats(self):
        """Toggle timing statistics display"""
        enabled = self.timing_enabled_var.get()
        self.timing_enabled_var.set(not enabled)  # Toggle the GUI variable
        
        # This single call will affect ALL processes
        self.process_flags.set_flag('timing_stats', not enabled)
        
        if not enabled:  # If we're enabling stats
            self.timing_frame.grid(row=5, column=0, columnspan=2, pady=10, sticky=(tk.W, tk.E))
            if self.timing_update_running:
                self.update_timing_stats()
        else:  # If we're disabling stats
            self.timing_frame.grid_remove()
            
    def update_timing_stats(self):
        """Update timing statistics display"""
        if not self.timing_update_running or not self.process_flags.get_flag("timing_stats"):
            return
            

        try:
            # Clear text widget
            self.timing_text.delete(1.0, tk.END)
            
            # System timing
            system_stats = self.system.get_timing_stats()
            self.timing_text.insert(tk.END, "System:\n")
            for category, stats in system_stats.items():
                self.timing_text.insert(tk.END, f"  {category}:\n")
                self.timing_text.insert(tk.END, f"    avg: {stats['avg']:.2f} ms\n")
                self.timing_text.insert(tk.END, f"    min: {stats['min']:.2f} ms\n")
                self.timing_text.insert(tk.END, f"    max: {stats['max']:.2f} ms\n")
            
            # Point detector timing
            self.timing_text.insert(tk.END, "\nPoint Detectors:\n")
            for i, detector in enumerate(self.detectors):
                stats = detector.get_timing_stats()
                self.timing_text.insert(tk.END, f"  Camera {i}:\n")
                for category, timing in stats.items():
                    self.timing_text.insert(tk.END, f"    {category}:\n")
                    self.timing_text.insert(tk.END, f"      avg: {timing['avg']:.2f} ms\n")
                    self.timing_text.insert(tk.END, f"      min: {timing['min']:.2f} ms\n")
                    self.timing_text.insert(tk.END, f"      max: {timing['max']:.2f} ms\n")
            
            # World reconstructor timing
            world_stats = self.world_reconstructor.get_timing_stats()
            self.timing_text.insert(tk.END, "\nWorld Reconstructor:\n")
            for category, stats in world_stats.items():
                self.timing_text.insert(tk.END, f"  {category}:\n")
                self.timing_text.insert(tk.END, f"    avg: {stats['avg']:.2f} ms\n")
                self.timing_text.insert(tk.END, f"    min: {stats['min']:.2f} ms\n")
                self.timing_text.insert(tk.END, f"    max: {stats['max']:.2f} ms\n")
            
            # Object detector timing
            object_stats = self.object_detector.get_timing_stats()
            self.timing_text.insert(tk.END, "\nObject Detector:\n")
            for category, stats in object_stats.items():
                self.timing_text.insert(tk.END, f"  {category}:\n")
                self.timing_text.insert(tk.END, f"    avg: {stats['avg']:.2f} ms\n")
                self.timing_text.insert(tk.END, f"    min: {stats['min']:.2f} ms\n")
                self.timing_text.insert(tk.END, f"    max: {stats['max']:.2f} ms\n")
                
        except Exception as e:
            print(f"Error updating timing stats: {e}")
            
        # Schedule next update
        if self.timing_update_running:
            self.root.after(1000, self.update_timing_stats)  # Update every second
        
def main():
    # Create and run GUI
    root = tk.Tk()
    app = TrackingGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
