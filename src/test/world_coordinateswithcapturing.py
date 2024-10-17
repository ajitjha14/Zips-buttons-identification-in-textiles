import os
import glob
import numpy as np
import pyrealsense2 as rs
import cv2  # Import OpenCV for image processing
from ultralytics import YOLO

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

# Load the YOLO model
model = YOLO("/home/amir/Desktop/test/best.pt")  # Path to your trained YOLOv8 model

# Set the input folder containing the images and output folder
input_folder = "/home/amir/Desktop/test"  # Folder containing the images
output_folder = "/home/amir/Desktop/test/Detection_results"  # Path to save the detection results

# Create the output directory if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Open a file to save world coordinates
output_file = os.path.join(output_folder, 'world_coordinates.txt')
with open(output_file, 'w') as f:
    # Get all image files in the input folder
    image_paths = glob.glob(os.path.join(input_folder, "*.jpg"))  # Adjust extension if needed, e.g., .png, .jpeg

    # Counter for saved images
    img_counter = 1

    while True:
        # Wait for frames from the camera
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            print("No color frame received.")
            continue

        # Convert the frame to a NumPy array
        color_image = np.asanyarray(color_frame.get_data())

        # Display the image
        cv2.imshow("Camera Feed", color_image)

        # Run inference on the current frame
        results = model.predict(color_image)

        # Save each result manually
        for idx, result in enumerate(results):
            # Extract bounding boxes, labels, and confidence scores
            bboxes = result.boxes.xyxy  # Get bounding box coordinates
            confidences = result.boxes.conf  # Get confidence scores
            labels = result.boxes.cls  # Get class labels
            
            for i in range(len(bboxes)):
                bbox = bboxes[i]  # Get the bounding box
                confidence = confidences[i].item()  # Get the confidence score
                label = labels[i].item()  # Get the class label

                # Get the coordinates of the bounding box
                x_min, y_min, x_max, y_max = map(int, bbox)  # Convert to int

                # Calculate the center of the bounding box
                x_center = (x_min + x_max) / 2
                y_center = (y_min + y_max) / 2

                # Get the world coordinates for the center
                world_coords = get_world_coordinates(x_center, y_center)
                if world_coords is not None:
                    X_world, Y_world, Z_world = world_coords
                    f.write(f"Image: {img_counter}, Label: {label}, Confidence: {confidence:.2f}, "
                            f"Center: ({x_center:.2f}, {y_center:.2f}), World Coordinates: (X={X_world:.2f}, Y={Y_world:.2f}, Z={Z_world:.2f})\n")
                    print(f"World Coordinates for ({x_center:.2f}, {y_center:.2f}): X={X_world:.2f}, Y={Y_world:.2f}, Z={Z_world:.2f}")
                else:
                    print(f"Could not get world coordinates for center: ({x_center:.2f}, {y_center:.2f})")

                # Draw the bounding box and label on the image
                cv2.rectangle(color_image, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)  # Draw rectangle in blue
                label_text = f"{model.names[label]}: {confidence:.2f}"  # Get class name
                cv2.putText(color_image, label_text, (x_min, y_min - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)  # Draw label

        # Check for key presses
        key = cv2.waitKey(1)
        if key == ord('s'):  # If 's' is pressed, save the image
            img_name = f"color_img{img_counter}.jpg"
            cv2.imwrite(os.path.join(output_folder, img_name), color_image)
            print(f"{img_name} saved!")
            img_counter += 1
        elif key == ord('q'):  # If 'q' is pressed, quit the program
            print("Exiting...")
            break

        # Save the modified image to the output folder
        output_image_path = os.path.join(output_folder, f"detection_{img_counter}.jpg")
        cv2.imwrite(output_image_path, color_image)
        print(f"Saved detection result to {output_image_path}")

print(f"All world coordinates saved to {output_file}")

# Stop the RealSense pipeline
pipeline.stop()
cv2.destroyAllWindows()

