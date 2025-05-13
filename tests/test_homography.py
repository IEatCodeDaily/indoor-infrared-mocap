import cv2
import numpy as np

def find_homography(image, pattern_image):
    """
    Find homography between an image and a pattern image using feature matching.
    
    Args:
        image: Input image (from video frame)
        pattern_image: Reference calibration pattern image
        
    Returns:
        H: Homography matrix
        status: Mask indicating which points were used for homography
    """
    # Convert images to grayscale
    img1_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(pattern_image, cv2.COLOR_BGR2GRAY)
    
    # Initialize SIFT detector
    sift = cv2.SIFT_create()
    
    # Find keypoints and descriptors
    kp1, des1 = sift.detectAndCompute(img1_gray, None)
    kp2, des2 = sift.detectAndCompute(img2_gray, None)
    
    # FLANN parameters
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    
    # FLANN matcher
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)
    
    # Apply Lowe's ratio test
    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)
    
    # Get corresponding points
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    
    # Find homography
    H, status = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    
    return H, status

def draw_matches(image, pattern_image, H):
    """
    Draw the homography transformation result.
    
    Args:
        image: Input image
        pattern_image: Reference pattern image
        H: Homography matrix
    
    Returns:
        result: Image with visualization of the homography matching
    """
    h, w = pattern_image.shape[:2]
    pts = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)
    dst = cv2.perspectiveTransform(pts, H)
    
    # Draw the transformed pattern corners on the image
    result = image.copy()
    result = cv2.polylines(result, [np.int32(dst)], True, (0, 255, 0), 3)
    
    return result

if __name__ == "__main__":
    # Example usage (you can modify this for your specific needs)
    cap = cv2.VideoCapture("calibration1_1.mp4")
    pattern = cv2.imread("pattern.png")  # Load your calibration pattern image
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        H, status = find_homography(frame, pattern)
        result = draw_matches(frame, pattern, H)
        
        cv2.imshow("Homography Matching", result)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows() 