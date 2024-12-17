import numpy as np
import pyvista as pv
import ipywidgets as widgets
from IPython.display import display

# Set up PyVista for Jupyter
pv.set_jupyter_backend('trame')
pv.set_plot_theme('document')

# Create a simple test visualization first
def test_visualization():
    pl = pv.Plotter(notebook=True, window_size=[1024, 768])
    # Add a sphere at origin
    sphere = pv.Sphere(radius=0.5, center=(0, 0, 0))
    pl.add_mesh(sphere, color='red')
    pl.add_axes()
    pl.show_grid()
    return pl

print("Testing basic visualization...")
test_pl = test_visualization()

# If that works, let's modify the CameraVisualizer to be simpler first
class SimpleCameraVisualizer:
    def __init__(self):
        # Just use first two cameras for testing
        self.positions = np.array([
            [ 0.99382967, 2.59333162, 2.95762625],
            [-1.84272602, 0.87304613, 3.97750837],
            [ 1.38108843, -2.52164672, 3.08849918],
            [ 3.17016849, -0.55979842, 3.5615908 ]
        ])
        
        self.pl = pv.Plotter(notebook=True, window_size=[1024, 768])
        
    def show(self):
        # Add coordinate axes and grid
        self.pl.add_axes()
        self.pl.show_grid()
        
        # Add spheres at camera positions
        for pos in self.positions:
            # Create sphere at camera position
            sphere = pv.Sphere(radius=0.2, center=pos)
            self.pl.add_mesh(sphere, color='blue')
            
            # Add text label
            self.pl.add_point_labels([pos], [f"Camera at {pos}"])
        
        # Add connecting lines between cameras
        for i in range(len(self.positions)-1):
            line = pv.Line(self.positions[i], self.positions[i+1])
            self.pl.add_mesh(line, color='red', line_width=2)
        
        # Set camera position for better view
        self.pl.camera_position = 'xy'
        self.pl.reset_camera()
        
        return self.pl

print("\nTesting camera visualization...")
viz = SimpleCameraVisualizer()
viz.show()