import numpy as np
from scipy.spatial.transform import Rotation as R

def find_rotation_matrix(camera_pos_robot, object_camera, object_robot):
    """
    Find the rotation matrix between the camera frame and the robot base frame.

    Parameters:
    camera_pos_robot (np.array): Camera position relative to robot base [x, y, z].
    object_camera (np.array): Object position in the camera frame [x, y, z].
    object_robot (np.array): Object position relative to the robot base [x, y, z].

    Returns:
    np.array: Rotation matrix (3x3) from camera frame to robot base frame.
    """
    # Calculate vectors from the camera to the object in both frames
    vector_camera = object_camera - np.array([0, 0, 0])  # Object position in camera frame (relative to camera)
    vector_robot = object_robot - camera_pos_robot       # Object position in robot frame (relative to camera position in robot frame)

    # Normalize the vectors
    vector_camera_normalized = vector_camera / np.linalg.norm(vector_camera)
    vector_robot_normalized = vector_robot / np.linalg.norm(vector_robot)

    # Compute the rotation matrix from camera frame to robot base frame
    cross_product = np.cross(vector_camera_normalized, vector_robot_normalized)
    dot_product = np.dot(vector_camera_normalized, vector_robot_normalized)

    # Check for zero vector length, which would cause division by zero
    if np.linalg.norm(cross_product) == 0:
        print("Vectors are collinear; rotation may not be defined correctly.")
        return np.eye(3)  # Return identity matrix if vectors are collinear

    # Calculate skew-symmetric cross-product matrix
    skew_sym_matrix = np.array([[0, -cross_product[2], cross_product[1]],
                                [cross_product[2], 0, -cross_product[0]],
                                [-cross_product[1], cross_product[0], 0]])

    # Calculate the rotation matrix using Rodrigues' rotation formula
    identity_matrix = np.eye(3)
    rotation_matrix = identity_matrix + skew_sym_matrix + skew_sym_matrix @ skew_sym_matrix * ((1 - dot_product) / (np.linalg.norm(cross_product) ** 2))

    return rotation_matrix

def apply_rotation_translation(rotation_matrix, camera_pos_robot, object_camera):
    """
    Apply rotation matrix and translation to convert object coordinates 
    from the camera frame to the robot base (world) frame.

    Parameters:
    rotation_matrix (np.array): Rotation matrix from camera to robot base frame (3x3).
    camera_pos_robot (np.array): Camera position relative to robot base [x, y, z].
    object_camera (np.array): Object position in the camera frame [x, y, z].

    Returns:
    np.array: Transformed object position in the robot base (world) frame.
    """
    # Rotate the object in the camera frame
    rotated_object = rotation_matrix @ object_camera

    # Translate to the world frame using the camera position
    object_world = rotated_object + camera_pos_robot
    return object_world

# Define initial positions and parameters
camera_pos_robot = np.array([0.425, 0, 0.32])
object_camera = np.array([0.08, -0.05, 0.45])    # Initial object position in camera frame
object_robot = np.array([0.46, 0.015, -0.01])    # Initial object position in robot base frame

# Calculate rotation matrix using initial positions
rotation_matrix = find_rotation_matrix(camera_pos_robot, object_camera, object_robot)
print("Rotation Matrix from Camera to Robot Base Frame:\n", rotation_matrix)

# Test the transformation on the initial object to verify if it matches the given coordinates
initial_object_transformed = apply_rotation_translation(rotation_matrix, camera_pos_robot, object_camera)
print("Transformed initial object position in world coordinates:", initial_object_transformed)

# New object position in camera frame to test the transformation
new_object_camera = np.array([-0.04, -0.09, 0.47])  # Example coordinates in camera frame

# Calculate new object position in the robot base (world) frame
new_object_world = apply_rotation_translation(rotation_matrix, camera_pos_robot, new_object_camera)
print("New object position in world coordinates:", new_object_world)
