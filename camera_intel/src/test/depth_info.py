import numpy as np
import cv2
import pyrealsense2 as rs
import os
from ultralytics import YOLO

# Intrinsic parameters (replace with your actual camera parameters)
fx = 603.63747782  # Focal length in x direction (replace with actual value)
fy = 604.91600466  # Focal length in y direction (replace with actual value)
cx = 334.45612024  # Principal point x-coordinate (replace with actual value)
cy = 243.55865732  # Principal point y-coordinate (replace with actual value)

# Constants
BASE_TO_TOOL_X = 0.1  # Distance from base to tool tip in x direction (in meters)
TOOL_TO_CAMERA_X = 0.05  # Approx distance between tool tip and camera in x direction (in meters)
TOOL_TO_CAMERA_Z_Y = 0.49  # Distance between tool and camera in the y direction for Z projection

def calculate_xyz(x_pixel, y_pixel, depth):
    """
    Calculate 3D camera coordinates (X, Y, Z) from pixel coordinates and depth.
    """
    Z = depth  # Depth is the Z-coordinate
    X = (x_pixel - cx) * Z / fx  # X-coordinate in 3D
    Y = (y_pixel - cy) * Z / fy  # Y-coordinate in 3D
    return X, Y, Z

def calculate_global_xyz(camera_x, camera_y, camera_z):
    """
    Calculate global XYZ coordinates based on camera XYZ and known constants.
    """
    # Check if camera_z is greater than the camera to tool distance
    if camera_z > TOOL_TO_CAMERA_Z_Y:
        z_x = np.sqrt(camera_z**2 - TOOL_TO_CAMERA_Z_Y**2)
    else:
        z_x = 0  # Handle invalid depth

    global_x = BASE_TO_TOOL_X + TOOL_TO_CAMERA_X + z_x
    global_y = -camera_x  # Global Y is negated to align with the global frame
    global_z = TOOL_TO_CAMERA_X  # Global Z is constant

    return global_x, global_y, global_z

def main():
    # Configure Intel RealSense pipeline for both depth and color streams
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    # Start the pipeline
    pipeline.start(config)

    # Set the target directory
    save_dir = "/home/amir/Desktop/test/Taken pictures"
    output_dir = "/home/amir/Desktop/test/Detection_results"
    
    # Ensure directories exist
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    # Load the YOLO model
    model = YOLO("/home/amir/Desktop/test/best.pt")  # Path to your YOLO model

    img_counter = 1

    try:
        while True:
            # Wait for frames from the camera
            frames = pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()

            if not depth_frame or not color_frame:
                print("No frame received.")
                continue

            # Convert the frames to NumPy arrays
            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())

            # Display the image
            cv2.imshow("Camera Feed", color_image)

            # Check for key presses
            key = cv2.waitKey(1)
            if key == ord('s'):  # If 's' is pressed, save the image
                img_name = f"color_img{img_counter}.jpg"
                img_path = os.path.join(save_dir, img_name)
                cv2.imwrite(img_path, color_image)
                print(f"{img_name} saved at {save_dir}!")

                # Run YOLO detection on the saved image
                results = model.predict(img_path)

                # Initialize a list to store coordinates and class names
                detection_results = []
                object_counter = 1  # Initialize object counter

                # Save detection results with bounding boxes
                for result in results:
                    bboxes = result.boxes.xyxy  # Bounding box coordinates
                    confidences = result.boxes.conf  # Confidence scores
                    labels = result.boxes.cls  # Class labels

                    for i in range(len(bboxes)):
                        x_min, y_min, x_max, y_max = map(int, bboxes[i])
                        confidence = confidences[i].item()
                        label = labels[i].item()
                        class_name = model.names[label]

                        # Calculate the center of the bounding box
                        x_center = (x_min + x_max) / 2
                        y_center = (y_min + y_max) / 2

                        # Get the depth at the center of the bounding box
                        depth_at_center = depth_frame.get_distance(int(x_center), int(y_center))

                        # Debugging: Print the depth values for each detected object
                        print(f"Depth at center for {class_name}: {depth_at_center:.3f} meters")

                        # Calculate the 3D camera coordinates (X, Y, Z)
                        camera_X, camera_Y, camera_Z = calculate_xyz(x_center, y_center, depth_at_center)

                        # Calculate the global coordinates (X, Y, Z)
                        global_X, global_Y, global_Z = calculate_global_xyz(camera_X, camera_Y, camera_Z)

                        # Debugging: Print global coordinates for each object
                        print(f"Global Coordinates for {class_name}: X={global_X:.2f}, Y={global_Y:.2f}, Z={global_Z:.2f}")

                        detection_results.append((class_name, x_center, y_center, confidence, global_X, global_Y, global_Z))

                        # Draw bounding boxes and labels on the image with the object number
                        cv2.rectangle(color_image, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
                        label_text = f"#{object_counter} {class_name}: {confidence:.2f}, Global X: {global_X:.2f}, Y: {global_Y:.2f}, Z: {global_Z:.2f}"
                        cv2.putText(color_image, label_text, (x_min, y_min - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

                        object_counter += 1  # Increment the object counter

                # Save the result image with detections to the output folder
                output_image_path = os.path.join(output_dir, img_name)
                cv2.imwrite(output_image_path, color_image)
                print(f"Detection result saved at {output_image_path}!")

                # Save detection results to a text file with the object number
                coords_file_path = os.path.join(output_dir, f"{img_name.replace('.jpg', '.txt')}")
                with open(coords_file_path, 'w') as f:
                    object_counter = 1  # Reset the object counter for the text file
                    for class_name, x_center, y_center, confidence, global_X, global_Y, global_Z in detection_results:
                        f.write(f"#{object_counter} Class: {class_name}, Confidence: {confidence:.2f}, "
                                f"Center: ({x_center:.2f}, {y_center:.2f}), "
                                f"Global Coordinates: X={global_X:.2f}, Y={global_Y:.2f}, Z={global_Z:.2f}\n")
                        object_counter += 1

                print(f"Detection results saved to {coords_file_path}")

                img_counter += 1

            elif key == ord('q'):  # If 'q' is pressed, quit the program
                print("Exiting...")
                break

    finally:
        # Stop streaming
        pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

