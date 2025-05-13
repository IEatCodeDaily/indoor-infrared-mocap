from irtracking.capture import PointDetector, Point2D
from irtracking.world import WorldReconstructor
from irtracking.params import IntrinsicParams, ExtrinsicParams
import numpy as np
import cv2
import json
from pathlib import Path
import rerun as rr
import rerun.blueprint as rrb
from scipy import optimize
from scipy.spatial import transform
import time
import copy

# Load camera parameters
def load_camera_params():
    # Load intrinsic parameters
    with open('config/camera-intrinsic.json', 'r') as f:
        intrinsic_data = json.load(f)
    
    intrinsics = {}
    for cam_data in intrinsic_data:
        camera_id = cam_data['camera_id']
        intrinsics[camera_id] = IntrinsicParams(
            matrix=np.array(cam_data['intrinsic_matrix'], dtype=np.float64),
            distortion=np.array(cam_data['distortion_coef'], dtype=np.float64)
        )
    
    # Initialize extrinsic parameters
    with open('config/camera-extrinsic.json', 'r') as f:
        extrinsic_data = json.load(f)
    
    extrinsics = {}
    for cam_data in extrinsic_data:
        camera_id = cam_data['camera_id']
        extrinsics[camera_id] = ExtrinsicParams(
            R=np.array(cam_data['R'], dtype=np.float64),
            t=np.array(cam_data['t'], dtype=np.float64)
        )
    
    return intrinsics, extrinsics

# Load calibration videos
def load_calibration_videos():
    videos = []
    for i in range(4):  # Assuming 4 cameras
        cap = cv2.VideoCapture(f'.//footage_calibration//calibration_pose_1_{i}.mp4')
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video file calibration_pose_1_{i}.mp4")
        videos.append(cap)
    return videos

# Test point detection
def detect_points(frame, intrinsic):
    detector = PointDetector(0)  # camera_id doesn't matter for testing
    points, confidence, processed = detector.detect_points(frame, intrinsic)
    if len(points) == 0:
        points = np.array([])
        confidence = np.array([])
    return points, processed, confidence

# Test triangulation
def find_point_correspondance_and_object_points(points_2d, camera_poses, intrinsics):
    """
    Find corresponding points using epipolar geometry and triangulate them
    Args:
        points_2d: Dict of 2D points for each camera
        camera_poses: Dict of camera extrinsic parameters
        intrinsics: Dict of camera intrinsic parameters
        frames: Optional frames for visualization
    Returns:
        errors: Reprojection errors
        object_points: 3D points
        frames: Updated frames with visualizations
    """
    # Convert dict to list format
    image_points = []
    for i in range(len(camera_poses)):
        if i in points_2d and points_2d[i] is not None and len(points_2d[i]) > 0:
            image_points.append(points_2d[i].tolist())
        else:
            image_points.append([])

    # Remove None points if any
    for image_points_i in image_points:
        try:
            image_points_i.remove([None, None])
        except:
            pass

    # Initialize correspondences with points from first camera
    correspondances = [[[i]] for i in image_points[0]]

    # Compute projection matrices
    Ps = [] # projection matrices
    for i in range(len(camera_poses)):
        RT = np.hstack([camera_poses[i].R, camera_poses[i].t.reshape(3,1)])
        P = intrinsics[i].matrix @ RT
        Ps.append(P)

    # Initialize root points from first camera
    root_image_points = [{"camera": 0, "point": point} for point in image_points[0]]
    epipolar_lines_list = []

    # Find correspondences using epipolar geometry
    for i in range(1, len(camera_poses)):
        epipolar_lines = []
        for root_image_point in root_image_points:
            F = cv2.sfm.fundamentalFromProjections(Ps[root_image_point["camera"]], Ps[i])
            line = cv2.computeCorrespondEpilines(np.array([root_image_point["point"]], dtype=np.float32), 1, F)
            epipolar_lines.append(line[0,0].tolist())
        epipolar_lines_list.append(epipolar_lines)
        not_closest_match_image_points = np.array(image_points[i])
        points = np.array(image_points[i])

        for j, [a, b, c] in enumerate(epipolar_lines):
            distances_to_line = np.array([])
            if len(points) != 0:
                distances_to_line = np.abs(a*points[:,0] + b*points[:,1] + c) / np.sqrt(a**2 + b**2)

            possible_matches = points[distances_to_line < 5].copy() if len(points) > 0 else []

            if len(possible_matches) == 0:
                for possible_group in correspondances[j]:
                    possible_group.append([None, None])
            else:
                distances_to_line = distances_to_line[distances_to_line < 5]
                possible_matches_sorter = distances_to_line.argsort()
                possible_matches = possible_matches[possible_matches_sorter]
                
                not_closest_match_image_points = np.array([p for p in not_closest_match_image_points 
                                                         if not np.any(np.all(p == possible_matches[0]))])
                
                new_correspondances_j = []
                for possible_match in possible_matches:
                    temp = copy.deepcopy(correspondances[j])
                    for possible_group in temp:
                        possible_group.append(possible_match.tolist())
                    new_correspondances_j += temp
                correspondances[j] = new_correspondances_j

        for point in not_closest_match_image_points:
            root_image_points.append({"camera": i, "point": point})
            temp = [[[None, None]] * i]
            temp[0].append(point.tolist())
            correspondances.append(temp)

    # Triangulate points using correspondences
    object_points = []
    errors = []
    for image_points_i in correspondances:
        object_points_i = triangulate_points(image_points_i, camera_poses, intrinsics)

        if np.all(object_points_i == None):
            continue

        errors_i = calculate_reprojection_errors(image_points_i, object_points_i, camera_poses, intrinsics)

        object_points.append(object_points_i[np.argmin(errors_i)])
        errors.append(np.min(errors_i))

    return np.array(errors), np.array(object_points), epipolar_lines_list

def calculate_reprojection_errors(image_points, object_points, camera_poses, intrinsics):
    """Calculate reprojection errors for multiple points"""
    errors = np.array([])
    for image_points_i, object_point in zip(image_points, object_points):
        error = calculate_reprojection_error(image_points_i, object_point, camera_poses, intrinsics)
        if error is None:
            continue
        errors = np.concatenate([errors, [error]])

    return errors

def calculate_reprojection_error(image_points, object_point, camera_poses, intrinsics):
    """Calculate reprojection error for a single point"""
    image_points = np.array(image_points)
    none_indices = np.where(np.all(image_points == None, axis=1))[0]
    image_points = np.delete(image_points, none_indices, axis=0)
    camera_ids = list(camera_poses.keys())
    valid_camera_ids = np.delete(camera_ids, none_indices)

    if len(image_points) <= 1:
        return None

    image_points_t = image_points.transpose((0,1))

    errors = np.array([])
    for i, cam_id in enumerate(valid_camera_ids):
        if np.all(image_points[i] == None, axis=0):
            continue
        projected_img_points, _ = cv2.projectPoints(
            np.expand_dims(object_point, axis=0).astype(np.float32), 
            cv2.Rodrigues(camera_poses[cam_id].R)[0], 
            camera_poses[cam_id].t, 
            intrinsics[cam_id].matrix, 
            None
        )
        projected_img_point = projected_img_points[:,0,:][0]
        errors = np.concatenate([errors, (image_points_t[i]-projected_img_point).flatten() ** 2])
    
    return errors.mean()

def drawlines(img, line):
    """Draw epipolar lines on image"""
    height, width = img.shape[:2]
    r = line
    x0, y0 = map(int, [0, -r[2]/r[1]])
    x1, y1 = map(int, [width, -(r[2]+r[0]*width)/r[1]])
    color = tuple(np.random.randint(0,255,3).tolist())
    img = cv2.line(img, (x0,y0), (x1,y1), color, 1)
    return img

def triangulate_point(image_points, camera_poses, intrinsics):
    """
    Triangulate a single point from multiple views using DLT
    Args:
        image_points: Dict of 2D points from different views
        camera_poses: Dict of camera extrinsic parameters
        intrinsics: Dict of camera intrinsic parameters
    Returns:
        3D point if successful, empty list otherwise
    """
    image_points = np.array(image_points)
    none_indices = np.where(np.all(image_points == None, axis=1))[0]
    image_points = np.delete(image_points, none_indices, axis=0)
    camera_ids = list(camera_poses.keys())
    valid_camera_ids = np.delete(camera_ids, none_indices)

    if len(image_points) <= 1:
            return []
    
    Ps = [] # projection matrices
    for cam_id in valid_camera_ids:
        RT = np.hstack([camera_poses[cam_id].R, camera_poses[cam_id].t.reshape(3,1)])
        P = intrinsics[cam_id].matrix @ RT
        Ps.append(P)

    def DLT(Ps, image_points):
        A = []
        for P, image_point in zip(Ps, image_points):
            A.append(image_point[1]*P[2,:] - P[1,:])
            A.append(P[0,:] - image_point[0]*P[2,:])
            
        A = np.array(A).reshape((len(Ps)*2,4))
        B = A.transpose() @ A
        U, s, Vh = np.linalg.svd(B, full_matrices = False)
        object_point = Vh[3,0:3]/Vh[3,3]

        return object_point

    object_point = DLT(Ps, image_points)
    return object_point

def triangulate_points(image_points, camera_poses, intrinsics):
    """
    Triangulate multiple points
    Args:
        image_points: List of 2D points for each view
        camera_poses: Dict of camera extrinsic parameters
        intrinsics: Dict of camera intrinsic parameters
    Returns:
        Array of 3D points
    """
    object_points = []
    for image_points_i in image_points:
        object_point = triangulate_point(image_points_i, camera_poses, intrinsics)
        object_points.append(object_point)
    
    return np.array(object_points)

def bundle_adjustment(points_2d, initial_extrinsics, intrinsics):
    """
    Optimize camera poses using bundle adjustment
    Args:
        points_2d: Dict of 2D points for each camera
        initial_extrinsics: Dict of initial camera extrinsic parameters
        intrinsics: Dict of camera intrinsic parameters
    Returns:
        Optimized extrinsics dictionary
    """
    def params_to_extrinsics(params):
        """Convert optimization parameters to extrinsics dictionary"""
        num_cameras = int(len(params) / 6)  # Each camera has 6 parameters (3 for rotation, 3 for translation)
        extrinsics = {}
        
        for i in range(num_cameras):
            # Get 6 parameters for each camera (no offset needed)
            rot_vec = params[i * 6:i * 6 + 3]
            t = params[i * 6 + 3:i * 6 + 6]
            extrinsics[i] = ExtrinsicParams(
                R=transform.Rotation.from_rotvec(rot_vec).as_matrix(),
                t=t
            )
        return extrinsics

    def residual_function(params):
        """Calculate reprojection errors for optimization"""
        # Convert parameters to extrinsics
        extrinsics = params_to_extrinsics(params)
        
        # Triangulate points with current extrinsics
        world_points, _ = triangulate_points(points_2d, intrinsics, extrinsics)
        
        # Calculate reprojection errors
        errors = []
        for world_point in world_points:
            for cam_id in sorted(points_2d.keys()):
                if points_2d[cam_id] is None or len(points_2d[cam_id]) == 0:
                    continue
                
                # Project 3D point to this camera
                R = extrinsics[cam_id].R
                t = extrinsics[cam_id].t
                K = intrinsics[cam_id].matrix
                
                # Project point
                point_cam = R @ world_point + t
                point_2d = K @ point_cam
                point_2d = point_2d[:2] / point_2d[2]
                
                # Calculate error - points_2d[cam_id] is now a Nx2 array
                observed = points_2d[cam_id][0]  # Take first point
                error = np.linalg.norm(point_2d - observed)
                errors.append(error)
        
        return np.array(errors)

    # Convert initial extrinsics to parameter vector
    init_params = []
    for cam_id in sorted(initial_extrinsics.keys()):
        # Convert rotation matrix to rotation vector
        rot_vec = transform.Rotation.from_matrix(initial_extrinsics[cam_id].R).as_rotvec()
        # Ensure translation is flattened and converted to list
        t = initial_extrinsics[cam_id].t.flatten().tolist()
        # Extend parameters list with rotation vector and translation
        init_params.extend(rot_vec.tolist())
        init_params.extend(t)
    
    init_params = np.array(init_params, dtype=np.float64)

    # Optimize using Levenberg-Marquardt algorithm
    result = optimize.least_squares(
        residual_function,
        init_params,
        loss='huber',  # Robust loss function to handle outliers
        verbose=2
    )

    # Convert optimized parameters back to extrinsics
    optimized_extrinsics = params_to_extrinsics(result.x)
    return optimized_extrinsics

def calibrate_cameras(points_2d, intrinsics, initial_poses=None):
    """
    Calibrate camera poses using only bundle adjustment
    Args:
        points_2d: Dict of detected points for each camera (as numpy arrays)
        intrinsics: Dict of camera intrinsic parameters
        initial_poses: Dict of initial extrinsic parameters (optional)
    Returns:
        Updated extrinsic parameters
    """
    # Format points for bundle adjustment
    formatted_points = {}
    for cam_id, points in points_2d.items():
        if points is not None and len(points) > 0:  # Only include cameras with detected points
            formatted_points[cam_id] = points  # Points are already numpy arrays

    # Initialize camera poses
    initial_poses_list = {}
    for cam_id in sorted(points_2d.keys()):  # Use points_2d keys to ensure we only include cameras with points
        if cam_id in initial_poses:
            # Use provided initial pose
            initial_poses_list[cam_id] = initial_poses[cam_id]  # Pass ExtrinsicParams directly
        else:
            # Fallback to default initialization
            initial_poses_list[cam_id] = ExtrinsicParams(
                R=np.eye(3),
                t=np.array([0.1, 0.1, 0.1]) if cam_id > 0 else np.zeros(3)
            )

    # Run bundle adjustment
    optimized_poses = bundle_adjustment(formatted_points, initial_poses_list, intrinsics)

    return optimized_poses

def visualize_results(frame_num, frames, processed_frames, points_2d, confidence, world_points, extrinsics, intrinsics, epipolar_lines_list):
    """Visualize results in rerun"""
    # Log timestamp
    rr.set_time_sequence("frame", frame_num)
    
    # Log camera frames and detected 2D points
    for i, frame in enumerate(frames):
        # Log camera frame
        rr.log(f"cameras/camera{i}/image", rr.Image(frame, color_model="BGR").compress(jpeg_quality=75))
        rr.log(f"cameras/camera{i}/image_processed", rr.Image(processed_frames[i], color_model="BGR").compress(jpeg_quality=75))
        
        # Log detected 2D points if available
        if i in points_2d and points_2d[i] is not None and len(points_2d[i]) > 0:
            points_array = points_2d[i]  # Already numpy array
            confidences = confidence[i]  # Assuming equal confidence for now
            
            # Create color array based on confidence
            colors = np.column_stack([
                1.0 - confidences,  # Red channel
                confidences,        # Green channel
                np.zeros_like(confidences),  # Blue channel
                np.ones_like(confidences)    # Alpha channel
            ])
            
            rr.log(f"cameras/camera{i}/points", 
                  rr.Points2D(
                      positions=points_array,
                      colors=colors,
                      radii=3
                  ))
        else:
            rr.log(f"cameras/camera{i}/points", 
                  rr.Points2D(
                      positions=np.array([[None, None]]),
                      colors=np.array([[0, 0, 0, 0]]),
                      radii=3
                  ))
        
        # Log intrinsic parameters
        rr.log(f"cameras/camera{i}/params/int/matrix", 
               rr.TextLog(f"Intrinsic Matrix:\n{np.array2string(intrinsics[i].matrix, precision=4, suppress_small=True)}"))
        rr.log(f"cameras/camera{i}/params/int/distortion", 
               rr.TextLog(f"Distortion Coeffs:\n{np.array2string(intrinsics[i].distortion, precision=4, suppress_small=True)}"))
        
        # Log extrinsic parameters
        if i in extrinsics:
            rr.log(f"cameras/camera{i}/params/ext/rotation", 
                   rr.TextLog(f"Rotation Matrix:\n{np.array2string(extrinsics[i].R, precision=4, suppress_small=True)}"))
            rr.log(f"cameras/camera{i}/params/ext/translation", 
                   rr.TextLog(f"Translation Vector:\n{np.array2string(extrinsics[i].t, precision=4, suppress_small=True)}"))
            
            # Also log euler angles for easier interpretation
            euler = transform.Rotation.from_matrix(extrinsics[i].R).as_euler('xyz', degrees=True)
            rr.log(f"cameras/camera{i}/params/ext/euler_angles", 
                   rr.TextLog(f"Euler Angles (xyz, degrees):\n{np.array2string(euler, precision=2, suppress_small=True)}"))
    
    # Log camera poses in 3D space
    for i, extrinsic in extrinsics.items():
        # Update camera transform and frustum in 3D view
        rr.log(f"cameras/camera{i}",
                rr.Transform3D(
                    translation=extrinsic.t.tolist(),
                    mat3x3=extrinsic.R.tolist(),
                    from_parent=False
                ))
        
        # Log camera intrinsics
        rr.log(f"cameras/camera{i}",
                rr.Pinhole(
                    resolution=[640, 480],  # Adjust if your resolution is different
                    focal_length=[intrinsics[i].matrix[0,0], intrinsics[i].matrix[1,1]],
                    principal_point=[intrinsics[i].matrix[0,2], intrinsics[i].matrix[1,2]],
                    camera_xyz=rr.ViewCoordinates.RUF,
                    #image_plane_distance=10,
                    
                ))
    
    # Log 3D points
    if world_points is not None and len(world_points) > 0:
        points_array = np.array([[p[0], p[1], p[2]] for p in world_points])
        rr.log("world/points",
               rr.Points3D(
                   positions=points_array,
                   colors=[0.0, 1.0, 0.0, 1.0],  # Green points
                   radii=0.01  # 1cm radius
               ))
    else:
        rr.log("world/points",
               rr.Points3D(
                   positions=np.array([[None, None, None]]),
                   colors=np.array([[0, 0, 0, 0]]),
                   radii=0.01  # 1cm radius
               ))
    

def setup_rerun(num_cameras = 4):
    # Initialize rerun blueprint with better layout
            blueprint = rrb.Blueprint(
                rrb.Vertical(
                    rrb.Spatial3DView(
                        name="world",
                        origin="/",
                    ),
                    rrb.Horizontal(
                        *[rrb.Spatial2DView(
                                name=f"camera{i}",
                                origin=f"cameras/camera{i}"
                            ) for i in range(num_cameras)],
                        column_shares=[1] * num_cameras
                    ),
                    row_shares=[2, 1]
                )
            )
            
            # Initialize rerun
            rr.init("IR Tracking Calibration", spawn=True, default_blueprint=blueprint)
            
            rr.set_time_seconds("timestamp", time.time())
            # Setup world view coordinates
            rr.log("/", rr.ViewCoordinates.RUF, static=True)
            
            # Setup camera coordinate systems
            for i in range(num_cameras):
                rr.log(f"cameras/camera{i}", rr.ViewCoordinates.RUF, static=True)  # Right-Down-Forward
            
            rr.log("world", rr.ViewCoordinates.RUF, static=True)  # Set an up-axis
            rr.log("world/origin", rr.Transform3D(translation=[0, 0, 0]))
            rr.log("world/origin/xyz", rr.Arrows3D(
                    origins=[[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                    vectors=[[50, 0, 0], [0, 50, 0], [0, 0, 50]],
                    colors=[[255, 0, 0], [0, 255, 0], [0, 0, 255]],
                    labels=["X", "Y", "Z"],
                    show_labels=False
                )
            )
            

def main():
    setup_rerun()
    
    # Load parameters
    intrinsics, extrinsics = load_camera_params()
        
    # Load videos
    videos = load_calibration_videos()
    print("videos loaded")
    
    frame_num = 0
    try:
        while True:
            # Read frames from all cameras
            frames = []
            points_2d = {}
            confidences = {}
            processed_frames = []

            for i, cap in enumerate(videos):
                ret, frame = cap.read()
                if not ret:
                    print(f"End of video {i}")
                    return
                
                frames.append(frame)
                
                # Detect points
                points, processed, confidence = detect_points(frame, intrinsics[i])
                points_2d[i] = points
                confidences[i] = confidence
                processed_frames.append(cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR))

            world_points = []
            if len(points_2d) >= 2:
                
                # Find correspondences and triangulate points
                errors, world_points, epipolar_lines_list = find_point_correspondance_and_object_points(points_2d, extrinsics, intrinsics)
                print(f"Frame {frame_num}: Detected {len(world_points)} 3D points")
                
                # Calibrate cameras if we have enough points
                if len(points_2d) >= 2:
                    # Use previous extrinsics as initialization
                    extrinsics = calibrate_cameras(points_2d, intrinsics, initial_poses=extrinsics)
            
            visualize_results(frame_num, frames, processed_frames, points_2d, confidences, world_points, extrinsics, intrinsics, epipolar_lines_list)
            
            frame_num += 1
            
            # Slow down playback for better visualization
            if cv2.waitKey(100) & 0xFF == ord('q'):  # 100ms delay
                break
                
    finally:
        # Clean up
        for cap in videos:
            cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 
