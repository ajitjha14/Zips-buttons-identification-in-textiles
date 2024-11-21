import numpy as np
import cv2
import pyrealsense2 as rs
import os
from ultralytics import YOLO

# Intrinsic parameters (replace with your actual camera parameters)
fx = 1360.51470  # Focal length in x direction
fy = 1364.45918  # Focal length in y direction
cx = 996.430781  # Principal point x-coordinate
cy = 574.664317  # Principal point y-coordinate

def calculate_xyz(x_pixel, y_pixel, depth):
    """
    Calculate 3D world coordinates (X, Y, Z) from pixel coordinates and depth.
    """
    Z = depth  # Depth is the Z-coordinate
    X = (x_pixel - cx) * (Z / fx)  # X-coordinate in 3D
    Y = (y_pixel - cy) * (Z / fy)   # Y-coordinate in 3D
    return X, Y, Z

def main():
    # Configure Intel RealSense pipeline for both depth and color streams
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 1920, 1080, rs.format.bgr8, 30)

    # Start the pipeline
    pipeline.start(config)

    # Access the depth sensor and set it to short range
    depth_sensor = pipeline.get_active_profile().get_device().first_depth_sensor()
    
    # Set the depth sensor to short range mode if supported
    if depth_sensor.supports(rs.option.visual_preset):
        depth_sensor.set_option(rs.option.visual_preset, rs.l500_visual_preset.short_range)  # Short Range Preset for L515

    # Set the target directories
    save_dir = "/home/amir/Desktop/test/Taken pictures"
    output_dir = "/home/amir/Desktop/test/Detection_results"
    
    # Ensure directories exist
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    # Load the YOLO model
    model = YOLO("/home/amir/Desktop/test/best.pt")  # Path to your YOLO model

    # Scaling factors for depth and color frame dimensions
    depth_width, depth_height = 640, 480  # Depth frame resolution
    color_width, color_height = 1920, 1080  # Color frame resolution
    x_scale = depth_width / color_width
    y_scale = depth_height / color_height

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

                        # Scale the center coordinates to match the depth frame resolution
                        x_center_depth = int(x_center * x_scale)
                        y_center_depth = int(y_center * y_scale)

                        # Get the depth at the scaled center of the bounding box
                        depth_at_center = depth_frame.get_distance(x_center_depth, y_center_depth)

                        # Calculate the 3D coordinates (X, Y, Z)
                        X, Y, Z = calculate_xyz(x_center, y_center, depth_at_center)

                        detection_results.append((class_name, x_center, y_center, confidence, X, Y, Z))

                        # Draw bounding boxes and labels on the image
                        cv2.rectangle(color_image, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
                        label_text = f"{class_name}: {confidence:.2f}, X: {X:.2f}, Y: {Y:.2f}, Z: {Z:.2f}"
                        cv2.putText(color_image, label_text, (x_min, y_min - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

                # Save the result image with detections to the output folder
                output_image_path = os.path.join(output_dir, img_name)
                cv2.imwrite(output_image_path, color_image)
                print(f"Detection result saved at {output_image_path}!")

                # Save detection results to a text file
                coords_file_path = os.path.join(output_dir, f"{img_name.replace('.jpg', '.txt')}")
                with open(coords_file_path, 'w') as f:
                    for class_name, x_center, y_center, confidence, X, Y, Z in detection_results:
                        f.write(f"Class: {class_name}, Confidence: {confidence:.2f}, Center: ({x_center:.2f}, {y_center:.2f}), "
                                f"3D Coordinates: X={X:.2f}, Y={Y:.2f}, Z={Z:.2f}\n")
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

