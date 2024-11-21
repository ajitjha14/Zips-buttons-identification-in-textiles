import numpy as np
from scipy.spatial.transform import Rotation as R

def compute_rotation_matrix(camera_points, world_points):
    """
    Computes the rotation matrix that aligns points from the camera frame
    to the world (or robot) frame.
    
    Parameters:
    - camera_points: np.array of shape (N, 3), where each row is a 3D point in the camera frame.
    - world_points: np.array of shape (N, 3), where each row is the corresponding 3D point in the world frame.
    
    Returns:
    - R_matrix: Rotation matrix (3x3).
    """
    # Ensure the input points have the same shape
    assert camera_points.shape == world_points.shape, "Point arrays must have the same shape"
    
    # Compute the centroids of the points
    camera_centroid = np.mean(camera_points, axis=0)
    world_centroid = np.mean(world_points, axis=0)
    
    # Center the points around the centroids
    camera_centered = camera_points - camera_centroid
    world_centered = world_points - world_centroid
    
    # Compute the covariance matrix
    H = np.dot(camera_centered.T, world_centered)
    
    # Singular Value Decomposition (SVD) to compute rotation matrix
    U, S, Vt = np.linalg.svd(H)
    R_matrix = np.dot(Vt.T, U.T)
    
    # Ensure a right-handed coordinate system
    if np.linalg.det(R_matrix) < 0:
        Vt[-1, :] *= -1
        R_matrix = np.dot(Vt.T, U.T)
    
    return R_matrix

# Placeholder for three points in the camera frame (replace these with actual values)
camera_points = np.array([
    [-0.07, -0.046, 0.455],  # Example point 1 in camera frame
    [0.06, -0.094, 0.468],   # Example point 2 in camera frame
    [-0.09, -0.083, 0.463]   # Example point 3 in camera frame
])

# Placeholder for corresponding points in the world frame (replace these with actual values)
world_points = np.array([
    [0.417, 0.07, -0.005],   # Corresponding point 1 in world frames
    [0.47, -0.06, -0.005],  # Corresponding point 2 in world frame
    [0.453, 0.09, -0.005]    # Corresponding point 3 in world frame
])

# Compute the rotation matrix using the provided points
R_matrix = compute_rotation_matrix(camera_points, world_points)

# Manually set the translation vector from camera to world frame
T_vector = np.array([0.275, 0.002, 0.43])

# Print the results
print("Rotation Matrix (R):\n", R_matrix)
print("Translation Vector (T):\n", T_vector)

# Example point in the camera frame
camera_point = np.array([-0.127, -0.121, 0.471])

# Transform the point from the camera frame to the world frame
def transform_point(camera_point, R_matrix, T_vector):
    """
    Transforms a 3D point from the camera frame to the world frame.

    Parameters:
    - camera_point: np.array of shape (3,), point in the camera frame.
    - R_matrix: np.array of shape (3, 3), rotation matrix.
    - T_vector: np.array of shape (3,), translation vector.

    Returns:
    - world_point: np.array of shape (3,), point in the world frame.
    """
    world_point = np.dot(R_matrix, camera_point) + T_vector
    return world_point

# Transform the example camera point to the world frame
world_point = transform_point(camera_point, R_matrix, T_vector)

# Print the transformed point
print("Camera Frame Point:", camera_point)
print("Transformed Point in World Frame:", world_point)
