import numpy as np
import cv2
import pyrealsense2 as rs
import os
from ultralytics import YOLO

def main():
    # Configure Intel RealSense pipeline
    pipe = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    
    # Set the target directory
    save_dir = "/home/amir/Desktop/test/Taken pictures"
    output_dir = "/home/amir/Desktop/test/Detection_results"
    
    # Ensure directories exist
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    # Load the YOLO model
    model = YOLO("/home/amir/Desktop/test/best.pt")  # Path to your YOLO model

    try:
        pipe.start(cfg)
    except Exception as e:
        print(f"Failed to start the Intel RealSense pipeline: {e}")
        return

    img_counter = 1

    while True:
        # Wait for frames from the camera
        frames = pipe.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            print("No color frame received.")
            continue

        # Convert the frame to a NumPy array
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
                    detection_results.append((class_name, x_center, y_center, confidence))

                    # Draw bounding boxes and labels on the image
                    cv2.rectangle(color_image, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
                    label_text = f"{class_name}: {confidence:.2f}"
                    cv2.putText(color_image, label_text, (x_min, y_min - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            # Save the result image with detections to the output folder
            output_image_path = os.path.join(output_dir, img_name)
            cv2.imwrite(output_image_path, color_image)
            print(f"Detection result saved at {output_image_path}!")

            # Save detection results to a text file
            coords_file_path = os.path.join(output_dir, f"{img_name.replace('.jpg', '.txt')}")
            with open(coords_file_path, 'w') as f:
                for class_name, x_center, y_center, confidence in detection_results:
                    f.write(f"Class: {class_name},  Confidence: {confidence:.2f},Center: ({x_center:.2f}, {y_center:.2f})\n")
            print(f"Detection results saved to {coords_file_path}")

            # Display the results in a pop-up window
            result_msg = "Detection Results:\n" + "\n".join([f"Class: {class_name} , Confidence: {confidence:.2f}, Center: ({x:.2f}, {y:.2f})" for class_name, x, y, confidence in detection_results])
            cv2.imshow("Detection Results", color_image)
            cv2.displayOverlay("Detection Results", result_msg,10000)  # Show for 3 seconds
            
            img_counter += 1

        elif key == ord('q'):  # If 'q' is pressed, quit the program
            print("Exiting...")
            break

    # Stop streaming and close windows
    pipe.stop()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

