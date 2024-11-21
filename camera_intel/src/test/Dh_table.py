import numpy as np

def dh_transform(a, alpha, d, theta):
    return np.array([[np.cos(theta), -np.sin(theta) * np.cos(alpha), np.sin(theta) * np.sin(alpha), a * np.cos(theta)],
                     [np.sin(theta), np.cos(theta) * np.cos(alpha), -np.cos(theta) * np.sin(alpha), a * np.sin(theta)],
                     [0, np.sin(alpha), np.cos(alpha), d],
                     [0, 0, 0, 1]])

# Initialize transformation matrix
T = np.eye(4)

# Define your DH parameters for each joint (a, alpha, d, theta)
dh_parameters = [
    # (a, alpha, d, theta)
    (0.0, 0, 0.072, 0),          # Waist
    (0.0, np.pi/2, 0.03865, 0),  # Shoulder
    (0.04975, 0, 0.25, 0),       # Elbow
    (0.175, 0, 0.0, 0),          # Forearm Roll
    (0.075, 0, 0.0, 0),          # Wrist Angle
    (0.065, 0, 0.0, 0),          # Wrist Rotate
    (0.043, 0, 0.0, 0),          # EE Arm
    (0.0055, 0, 0.0, 0),         # Gripper
    (0.02, 0, 0.055, 0)          # Camera Joint
]

# Calculate the transformation from base to camera
for (a, alpha, d, theta) in dh_parameters:
    T_next = dh_transform(a, alpha, d, theta)
    T = np.dot(T, T_next)

# Now T contains the transformation from base to camera
# To get the transformation from camera to base, take the inverse
T_camera_to_base = np.linalg.inv(T)

print("Transformation Matrix from Camera to Base:")
print(T_camera_to_base)

