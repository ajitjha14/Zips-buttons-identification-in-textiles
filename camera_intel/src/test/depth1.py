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
    if depth_sensor.supports(rs.option.visual_preset):
        depth_sensor.set_option(rs.option.visual_preset, rs.l500_visual_preset.short_range)  # Short Range Preset for L515

    # Set the target directories
    save_dir = "/home/amir/Desktop/test/Taken pictures"
    output_dir = "/home/amir/Desktop/test/Detection_results"
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    # Load the YOLO model
    model = YOLO("/home/amir/Desktop/test/best.pt")

    # Scaling factors for depth and color frame dimensions
    depth_width, depth_height = 640, 480  # Depth frame resolution
    color_width, color_height = 1920, 1080  # Color frame resolution
    x_scale = depth_width / color_width
    y_scale = depth_height / color_height

    # List of classes to ignore
    ignore_classes = {"jacket", "shirt"}

    img_counter = 1

    try:
        while True:
            frames = pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()

            if not depth_frame or not color_frame:
                print("No frame received.")
                continue

            # Convert frames to NumPy arrays
            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())
            cv2.imshow("Camera Feed", color_image)

            key = cv2.waitKey(1)
            if key == ord('s'):
                img_name = f"color_img{img_counter}.jpg"
                img_path = os.path.join(save_dir, img_name)
                cv2.imwrite(img_path, color_image)
                print(f"{img_name} saved at {save_dir}!")

                results = model.predict(img_path)
                detection_results = []

                for result in results:
                    bboxes = result.boxes.xyxy
                    confidences = result.boxes.conf
                    labels = result.boxes.cls

                    for i in range(len(bboxes)):
                        x_min, y_min, x_max, y_max = map(int, bboxes[i])
                        confidence = confidences[i].item()
                        label = labels[i].item()
                        class_name = model.names[label]

                        if class_name.lower() in ignore_classes:
                            continue

                        x_center = (x_min + x_max) / 2
                        y_center = (y_min + y_max) / 2
                        x_center_depth = int(x_center * x_scale)
                        y_center_depth = int(y_center * y_scale)
                        depth_at_center = depth_frame.get_distance(x_center_depth, y_center_depth)
                        X, Y, Z = calculate_xyz(x_center, y_center, depth_at_center)
                        detection_results.append((class_name, x_center, y_center, confidence, X, Y, Z))

                        cv2.rectangle(color_image, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
                        label_text = f"{class_name}: {confidence:.2f}, X: {X:.3f}, Y: {Y:.3f}, Z: {Z:.3f}"
                        cv2.putText(color_image, label_text, (x_min, y_min - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

                output_image_path = os.path.join(output_dir, img_name)
                cv2.imwrite(output_image_path, color_image)
                print(f"Detection result saved at {output_image_path}!")

                # Display pop-up window with the detection result image
                cv2.imshow("Detection Result", color_image)
                cv2.waitKey(0)  # Press any key to close the detection result window
                cv2.destroyWindow("Detection Result")

                img_counter += 1

            elif key == ord('q'):
                print("Exiting...")
                break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

