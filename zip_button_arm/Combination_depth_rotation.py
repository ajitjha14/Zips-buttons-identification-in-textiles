import numpy as np
import cv2
import pyrealsense2 as rs
import os
from ultralytics import YOLO
from scipy.spatial.transform import Rotation as R

# Intrinsic parameters for the camera
fx = 1360.51470
fy = 1364.45918
cx = 996.430781
cy = 574.664317

# Extrinsic parameters: Rotation matrix and translation vector from camera to world frame
def compute_rotation_matrix(camera_points, world_points):
    camera_centroid = np.mean(camera_points, axis=0)
    world_centroid = np.mean(world_points, axis=0)
    camera_centered = camera_points - camera_centroid
    world_centered = world_points - world_centroid
    H = np.dot(camera_centered.T, world_centered)
    U, _, Vt = np.linalg.svd(H)
    R_matrix = np.dot(Vt.T, U.T)
    if np.linalg.det(R_matrix) < 0:
        Vt[-1, :] *= -1
        R_matrix = np.dot(Vt.T, U.T)
    return R_matrix

camera_points = np.array([[-0.07, -0.046, 0.455], [0.06, -0.094, 0.468], [-0.09, -0.083, 0.463]])
world_points = np.array([[0.417, 0.07, -0.005], [0.47, -0.06, -0.005], [0.453, 0.09, -0.005]])
R_matrix = compute_rotation_matrix(camera_points, world_points)
T_vector = np.array([0.275, 0.0, 0.43])

def transform_point(camera_point, R_matrix, T_vector):
    world_point = np.dot(R_matrix, camera_point) + T_vector
    return world_point

def calculate_xyz(x_pixel, y_pixel, depth):
    Z = depth
    X = (x_pixel - cx) * (Z / fx)
    Y = (y_pixel - cy) * (Z / fy)
    return X, Y, Z

def main():
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 1920, 1080, rs.format.bgr8, 30)
    pipeline.start(config)

    depth_sensor = pipeline.get_active_profile().get_device().first_depth_sensor()
    if depth_sensor.supports(rs.option.visual_preset):
        depth_sensor.set_option(rs.option.visual_preset, rs.l500_visual_preset.short_range)

    save_dir = "/home/amir/Desktop/test/Taken pictures"
    output_dir = "/home/amir/Desktop/test/Detection_results"
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    model = YOLO("/home/amir/Desktop/test/best.pt")
    depth_width, depth_height = 640, 480
    color_width, color_height = 1920, 1080
    x_scale = depth_width / color_width
    y_scale = depth_height / color_height
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

                        camera_point = np.array([X, Y, Z])
                        world_point = transform_point(camera_point, R_matrix, T_vector)

                        detection_results.append((class_name, x_center, y_center, confidence, *world_point))

                        cv2.rectangle(color_image, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
                        label_text = f"{class_name}: {confidence:.2f}, X: {world_point[0]:.3f}, Y: {world_point[1]:.3f}, Z: {world_point[2]:.3f}"
                        cv2.putText(color_image, label_text, (x_min, y_min - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

                output_image_path = os.path.join(output_dir, img_name)
                cv2.imwrite(output_image_path, color_image)
                print(f"Detection result saved at {output_image_path}!")

                cv2.imshow("Detection Result", color_image)
                cv2.waitKey(0)
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

