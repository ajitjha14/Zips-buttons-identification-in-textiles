import os
import glob
import numpy as np
import pyrealsense2 as rs
import cv2  # Import OpenCV for image processing

# Start RealSense pipeline
pipeline = rs.pipeline()
cfg = rs.config()
cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
pipeline.start(cfg)

# Intrinsic camera parameters (from the provided intrinsic matrix)
fx = 603.63747782
fy = 604.91600466
cx = 334.45612024
cy = 243.55865732

# Transformation matrix (from camera to robot base)
T_camera_to_world = np.array([
    [1, 0, 0, 0.08],  # Translation along X-axis
    [0, 1, 0, 0.00],  # No translation along Y-axis
    [0, 0, 1, 0.065], # Translation along Z-axis
    [0, 0, 0, 1]      # Homogeneous coordinate
])

def get_world_coordinates(x_center, y_center):
    try:
        # Get frames
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()

        # Ensure we have a depth frame
        if not depth_frame:
            print("No depth frame received.")
            return None

        # Get depth at the object's center
        depth = depth_frame.get_distance(int(x_center), int(y_center))

        # Convert 2D image coordinates to 3D camera coordinates
        X_camera = (x_center - cx) * depth / fx
        Y_camera = (y_center - cy) * depth / fy
        Z_camera = depth

        # Convert to homogeneous coordinates
        camera_coords = np.array([X_camera, Y_camera, Z_camera, 1])

        # Transform to world coordinates
        world_coords = np.dot(T_camera_to_world, camera_coords)
        return world_coords[:3]  # Return X_world, Y_world, Z_world

    except RuntimeError as e:
        print(f"Runtime error: {e}")
        return None
    except Exception as e:
        print(f"Error: {e}")
        return None

# Set the input folder containing the images and output folder
input_folder = "/home/amir/Desktop/test/Taken pictures"  # Folder containing the images
output_folder = "/home/amir/Desktop/test/Detection_results"  # Path to save the world coordinates

# Create the output directory if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Open a file to save world coordinates
output_file = os.path.join(output_folder, 'world_coordinates.txt')
with open(output_file, 'w') as f:
    # Get all image files in the input folder
    image_paths = glob.glob(os.path.join(input_folder, "*.jpg"))  # Adjust extension if needed, e.g., .png, .jpeg

    for image_path in image_paths:
        print(f"Processing {image_path}...")

        # Read the original image
        image = cv2.imread(image_path)

        # For this example, assume we manually define the center point (x_center, y_center)
        # You can modify this part to get the center from user input or some other method
        x_center = image.shape[1] / 2  # Image width / 2 (center X)
        y_center = image.shape[0] / 2  # Image height / 2 (center Y)

        # Get the world coordinates for the center
        world_coords = get_world_coordinates(x_center, y_center)
        if world_coords is not None:
            X_world, Y_world, Z_world = world_coords
            f.write(f"Image: {os.path.basename(image_path)}, "
                    f"Center: ({x_center:.2f}, {y_center:.2f}), "
                    f"World Coordinates: (X={X_world:.2f}, Y={Y_world:.2f}, Z={Z_world:.2f})\n")
            print(f"World Coordinates for ({x_center:.2f}, {y_center:.2f}): "
                  f"X={X_world:.2f}, Y={Y_world:.2f}, Z={Z_world:.2f}")
        else:
            print(f"Could not get world coordinates for center: ({x_center:.2f}, {y_center:.2f})")

print(f"All world coordinates saved to {output_file}")

# Stop the RealSense pipeline
pipeline.stop()

