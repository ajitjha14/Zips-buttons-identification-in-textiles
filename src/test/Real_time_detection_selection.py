import os
import time
import cv2
import re
import numpy as np
import pyrealsense2 as rs
from ultralytics import YOLO

# Sort helper function to extract numbers from filenames
def natural_sort_key(s):
    return [int(text) if text.isdigit() else text for text in re.split(r'(\d+)', s)]

def realtime_detection():
    # Set directories for images and detection results
    save_dir = "/home/amir/Desktop/test/Taken pictures auto"
    output_dir = "/home/amir/Desktop/test/Real time detection results"
    
    # Ensure the directories exist
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    # Configure Intel RealSense pipeline
    pipe = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    # Load the YOLO model
    model = YOLO("/home/amir/Desktop/test/best.pt")  # Path to your YOLO model

    try:
        pipe.start(cfg)
    except Exception as e:
        print(f"Failed to start the Intel RealSense pipeline: {e}")
        return

    img_counter = 1
    start_time = time.time()

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
        key = cv2.waitKey(1)

        # Check if 4 seconds have passed to save an image
        if time.time() - start_time > 4:
            img_name = f"color_img{img_counter}.jpg"
            img_path = os.path.join(save_dir, img_name)
            cv2.imwrite(img_path, color_image)
            print(f"Image '{img_name}' saved!")

            # Run YOLO detection on the saved image
            results = model.predict(img_path)

            # Draw detections on the image and save it
            detection_results = []
            for result in results:
                bboxes = result.boxes.xyxy  # Bounding box coordinates
                confidences = result.boxes.conf  # Confidence scores
                labels = result.boxes.cls  # Class labels

                with open(os.path.join(output_dir, f"{os.path.splitext(img_name)[0]}.txt"), "w") as txt_file:
                    for i in range(len(bboxes)):
                        x_min, y_min, x_max, y_max = map(int, bboxes[i])
                        confidence = confidences[i].item()
                        label = labels[i].item()
                        class_name = model.names[label]

                        # Save detection result in text format
                        txt_file.write(f"{class_name} {confidence:.2f} {x_min} {y_min} {x_max} {y_max}\n")

                        # Draw bounding boxes and labels on the image
                        cv2.rectangle(color_image, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
                        label_text = f"{class_name}: {confidence:.2f}"
                        cv2.putText(color_image, label_text, (x_min, y_min - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

                        # Store detection result for later use
                        detection_results.append((class_name, x_min, y_min, x_max, y_max, confidence))

            # Save the result image in the detection folder
            output_image_path = os.path.join(output_dir, img_name)
            cv2.imwrite(output_image_path, color_image)
            print(f"Detection image '{output_image_path}' and text file saved!")
            img_counter += 1

            start_time = time.time()  # Reset the timer

        if key == ord('q'):  # Exit the real-time detection loop
            print("Exiting real-time detection and entering selection mode...")
            break

    pipe.stop()
    cv2.destroyAllWindows()
    
    # Enter the image review mode to select which image to keep
    review_images(save_dir, output_dir)

def review_images(save_dir, output_dir):
    # Get a list of saved images
    images = [f for f in os.listdir(save_dir) if f.endswith('.jpg')]
    images.sort(key=natural_sort_key)

    if not images:
        print("No images found!")
        return

    current_index = 0
    while True:
        # Display detection images for review
        detection_img_path = os.path.join(output_dir, images[current_index])

        if os.path.exists(detection_img_path):
            img = cv2.imread(detection_img_path)
            cv2.imshow(f"Image {current_index + 1}/{len(images)}: {images[current_index]}", img)
        else:
            print(f"Detection result for '{images[current_index]}' not found. Skipping.")

        # User navigation and selection input
        key = cv2.waitKey(0)

        if key == ord('d'):  # Move to the next image
            cv2.destroyAllWindows()
            current_index = (current_index + 1) % len(images)
        elif key == ord('a'):  # Move to the previous image
            cv2.destroyAllWindows()
            current_index = (current_index - 1) % len(images)
        elif key == ord('s'):  # Select the current image
            cv2.destroyAllWindows()
            break
        elif key == ord('q'):  # Quit the review process
            print("Selection process aborted.")
            return

    selected_image = images[current_index]
    
    # Delete all images except the selected one
    for image in images:
        if image != selected_image:
            os.remove(os.path.join(save_dir, image))
            os.remove(os.path.join(output_dir, image))
            txt_file = os.path.join(output_dir, f"{os.path.splitext(image)[0]}.txt")
            if os.path.exists(txt_file):
                os.remove(txt_file)

    print(f"Image '{selected_image}' and its result have been kept.")
    print("All other images and results have been deleted.")

if __name__ == "__main__":
    realtime_detection()

