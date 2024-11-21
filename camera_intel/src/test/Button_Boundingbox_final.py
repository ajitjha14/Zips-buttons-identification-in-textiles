#!/usr/bin/env python3

import sys
import numpy as np
import cv2
import pyrealsense2 as rs
import os
from ultralytics import YOLO
from scipy.spatial.transform import Rotation as R
import tkinter as tk
from tkinter import messagebox
import threading
from collections import defaultdict
from interbotix_common_modules.common_robot.robot import robot_shutdown, robot_startup
from interbotix_xs_modules.xs_robot.arm import InterbotixManipulatorXS
import time

# Intrinsic parameters for the camera
fx = 1360.51470
fy = 1364.45918
cx = 996.430781
cy = 574.664317

# Define the function to set end-effector pose and joint position
def detecting_home_position(bot):
    bot.arm.set_ee_pose_components(x=0.3, z=0.55)
    bot.arm.set_single_joint_position(joint_name='wrist_angle', position=np.pi / 1.5)

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

camera_points = np.array([[-0.07, -0.046, 0.455],
                          [0.06, -0.094, 0.468],
                          [-0.09, -0.083, 0.463]])
world_points = np.array([[0.417, 0.07, -0.005],
                         [0.47, -0.06, -0.005],
                         [0.453, 0.09, -0.005]])
R_matrix = compute_rotation_matrix(camera_points, world_points)
T_vector = np.array([0.32, 0.0, 0.43])

def transform_point(camera_point, R_matrix, T_vector):
    return np.dot(R_matrix, camera_point) + T_vector

def calculate_xyz(x_pixel, y_pixel, depth):
    """Calculates the 3D position in the camera frame given pixel coordinates and depth."""
    if depth <= 0:
        print(f"Invalid depth: {depth}. Skipping this point.")
        return None
    Z = depth
    X = (x_pixel - cx) * (Z / fx)
    Y = (y_pixel - cy) * (Z / fy)
    return X, Y, Z

# Move robot to the center of the bounding box
def move_robot_to_center(bot, center_x, center_y, center_z):
    target_z = center_z + 0.2  # Maintain 0.2m offset from object in Z
    print(f"[DEBUG] Moving to center - Calculated coordinates: X={center_x:.3f}, Y={center_y:.3f}, Z={target_z:.3f}")
    bot.arm.set_ee_pose_components(x=center_x, y=center_y, z=target_z)
    print("Returning to detecting position...")
    time.sleep(5)
    detecting_home_position(bot)

# Move robot around the bounding box in counterclockwise sequence
def move_robot_around_corners(bot, corners, loops=1):
    ordered_corners = corners + [corners[0]]  # Adds the first corner again to return to start point
    for loop in range(loops):
        print(f"[DEBUG] Starting loop {loop + 1}")
        for i, corner in enumerate(ordered_corners):
            x_world, y_world, z_world = corner
            target_z = z_world + 0.2  # Maintain 0.2m offset from object in Z
            if abs(x_world) > 1 or abs(y_world) > 1:
                print(f"Corner {i} is out of reach: X: {x_world:.3f}, Y: {y_world:.3f}")
                continue
            print(f"[DEBUG] Loop {loop + 1}: Moving to corner {i} - X: {x_world:.3f}, Y: {y_world:.3f}, Z={target_z:.3f}")
            bot.arm.set_ee_pose_components(x=x_world, y=y_world, z=target_z)
            time.sleep(0.05)  # Adjust this for your robot’s movement speed

    print("[DEBUG] Returning to detecting position.")
    detecting_home_position(bot)

# Display UI in a separate thread for selecting detected coordinates and movement type
selection_active = False  # Control flag for selection menu

def display_ui(detection_results, bot):
    global selection_active
    if selection_active:
        return

    selection_active = True
    root = tk.Tk()
    root.title("Select Coordinate and Movement Type for Robot")

    tk.Label(root, text="Detected Coordinates").pack()
    listbox = tk.Listbox(root, width=100)
    listbox.pack()

    def on_select():
        selection = listbox.curselection()
        if selection:
            idx = selection[0]
            label, class_name, x_center, y_center, confidence, x_world, y_world, z_world, corners = detection_results[idx]

            # Create a new dialog window
            move_window = tk.Toplevel(root)
            move_window.title("Select Movement Type")

            tk.Label(move_window, text="Where would you like to move?").pack(pady=10)

            # Function to handle the choice
            def move_to_center():
                move_robot_to_center(bot, x_world, y_world, z_world)
                move_window.destroy()
                close_menu(root)

            def move_to_bounding_box():
                move_robot_around_corners(bot, corners)
                move_window.destroy()
                close_menu(root)

            # Buttons for movement options
            tk.Button(move_window, text="Center", command=move_to_center, width=15).pack(side=tk.LEFT, padx=20, pady=10)
            tk.Button(move_window, text="Bounding Box", command=move_to_bounding_box, width=15).pack(side=tk.RIGHT, padx=20, pady=10)
        else:
            close_menu(root)

    for detection in detection_results:
        label, class_name, x_center, y_center, confidence, x_world, y_world, z_world, corners = detection
        listbox.insert(tk.END, f"{label} - {class_name} - Confidence: {confidence:.2f} - X: {x_world:.3f}, Y: {y_world:.3f}, Z: {z_world:.3f}")

    tk.Button(root, text="Select", command=on_select).pack(pady=10)
    root.protocol("WM_DELETE_WINDOW", lambda: close_menu(root))
    root.mainloop()

def close_menu(root):
    global selection_active
    root.destroy()
    selection_active = False

def main():
    bot = InterbotixManipulatorXS(
        robot_model='wx250s',
        group_name='arm',
        gripper_name='gripper',
    )
    robot_startup()
    detecting_home_position(bot)

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
    last_update_time = time.time()

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

            current_time = time.time()
            if current_time - last_update_time >= 0.5 and not selection_active:
                # Reset class counts for hashtags with each detection update
                class_counts = defaultdict(int)

                results = model(color_image)
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

                        # Increment the counter for the class
                        class_counts[class_name] += 1
                        hashtag = f"#{class_name}_{class_counts[class_name]}"

                        # Calculate bounding box center for this detection only once
                        x_center = (x_min + x_max) / 2
                        y_center = (y_min + y_max) / 2
                        x_center_depth = int(x_center * x_scale)
                        y_center_depth = int(y_center * y_scale)

                        # Ensure x and y are within depth frame boundaries
                        if not (0 <= x_center_depth < depth_width and 0 <= y_center_depth < depth_height):
                            print("Center coordinates out of bounds. Skipping this detection.")
                            continue

                        # Get depth at the center point
                        depth_at_center = depth_frame.get_distance(x_center_depth, y_center_depth)
                        if depth_at_center > 0:
                            camera_coords = calculate_xyz(x_center, y_center, depth_at_center)
                            if camera_coords:
                                world_coords = transform_point(np.array(camera_coords), R_matrix, T_vector)
                                x_world, y_world, z_world = world_coords
                                
                                # Append each detection's unique coordinates to detection_results
                                corners = [
                                    transform_point(
                                        np.array(calculate_xyz(x, y, depth_frame.get_distance(int(x * x_scale), int(y * y_scale)))),
                                        R_matrix,
                                        T_vector
                                    )
                                    for (x, y) in [(x_min, y_min), (x_min, y_max), (x_max, y_max), (x_max, y_min), (x_min, y_min)]
                                    if 0 <= int(x * x_scale) < depth_width and 0 <= int(y * y_scale) < depth_height and
                                    calculate_xyz(x, y, depth_frame.get_distance(int(x * x_scale), int(y * y_scale))) is not None
                                ]
                                detection_results.append((hashtag, class_name, x_center, y_center, confidence, x_world, y_world, z_world, corners))
                                
                                # Display each bounding box and label
                                label_text = f"{hashtag} - {class_name}: {confidence:.2f}, X: {x_world:.3f}, Y: {y_world:.3f}, Z: {z_world:.3f}"
                                cv2.putText(color_image, label_text, (x_min, y_min - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                                cv2.rectangle(color_image, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)

                last_update_time = current_time
                cv2.imshow("Live Detection", color_image)

            key = cv2.waitKey(1)
            if key == ord('s'):
                img_name = f"color_img{img_counter}.jpg"
                img_path = os.path.join(save_dir, img_name)
                cv2.imwrite(img_path, color_image)
                print(f"{img_name} saved at {save_dir}!")
                threading.Thread(target=display_ui, args=(detection_results, bot)).start()
                img_counter += 1

            elif key == ord('q'):
                print("Exiting...")
                break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
        robot_shutdown()

if __name__ == "__main__":
    main()

