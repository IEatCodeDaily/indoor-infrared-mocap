from core.camera.system import CameraSystem
from core.reconstruction.world import WorldReconstructor
from interfaces.flask_server.server import app as flask_app

def main():
    # Initialize the system
    camera_system = CameraSystem("config/camera_config.json")
    camera_system.initialize_cameras()
    
    reconstructor = WorldReconstructor(camera_system.cameras)
    
    # Start Flask server
    flask_app.run(host='0.0.0.0', port=5000)

if __name__ == "__main__":
    main()