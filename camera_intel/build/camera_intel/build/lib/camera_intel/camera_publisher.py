import cv2
import numpy as np
import pyrealsense2 as rs
from ultralytics import YOLO

# Camera parameters
intrinsic_matrix = np.array([[603.63747782, 0, 334.45612024],
                             [0, 604.91600466, 243.55865732],
                             [0, 0, 1]])
focal_length = intrinsic_matrix[0, 0]  # Assuming fx = fy for simplicity

# Configure Intel RealSense pipeline
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipeline.start(config)

# Load YOLO model
model = YOLO("/home/amir/Desktop/test/training test - 2/runs/detect/train4/weights/best.pt")

# Define objects to ignore
ignore_objects = ["jacket", "shirt"]  # Add more objects to ignore as needed

# Initialize an object counter
object_counter = 0

# Use an initial arbitrary depth (in meters)
initial_depth = 0.6  # Adjust this value as needed

# Define expected widths (in meters) for certain objects
# Modify these values based on your expected sizes for Zips
expected_widths = {
    'zip': 0.2,  # Example expected width for a zip in meters
}

while True:
    # Capture frame from the camera
    frames = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()
    if not color_frame:
        continue

    # Convert the frame to a NumPy array
    color_image = np.asanyarray(color_frame.get_data())

    # Run YOLO detection
    results = model.predict(color_image)

    # Process detections
    for result in results:
        bboxes = result.boxes.xyxy  # Bounding box coordinates
        confidences = result.boxes.conf  # Confidence scores
        labels = result.boxes.cls  # Class labels

        for i in range(len(bboxes)):
            x_min, y_min, x_max, y_max = map(int, bboxes[i])
            confidence = confidences[i].item()
            label = labels[i].item()
            class_name = model.names[label]

            # Skip specific objects
            if class_name.lower() in ignore_objects:
                continue  # Ignore this object

            # Increment the object counter for each detected object
            object_counter += 1

            # Calculate the width of the bounding box in pixels
            detected_bbox_width_px = x_max - x_min

            # Calculate real width W of the object based on initial depth
            W = initial_depth * (detected_bbox_width_px / focal_length)

            # Estimate new depth based on expected width
            # If the object is a zip, use the expected width for calculation
            expected_width = expected_widths.get(class_name.lower(), 0.2)  # Default to 0.2m if not found
            new_depth = (W * focal_length) / detected_bbox_width_px if detected_bbox_width_px > 0 else initial_depth

            # Adjust new_depth based on comparison with expected width
            if detected_bbox_width_px > expected_width * focal_length / initial_depth: 
                new_depth *= 0.9  # Reduce depth estimate if bounding box is larger than expected
            else:
                new_depth *= 1.1  # Increase depth estimate if bounding box is smaller than expected

            # Prepare the label with the object counter
            label_text = f"ID: {object_counter}, {class_name}: {confidence:.2f}, Depth: {new_depth:.2f}m, Width: {W:.2f}m"
            cv2.rectangle(color_image, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
            cv2.putText(color_image, label_text, (x_min, y_min - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            # Print the detection information to the terminal
            print(label_text)

    # Display the image with detections
    cv2.imshow("Real-Time Detection", color_image)
    if cv2.waitKey(1) == ord('q'):
        break

# Cleanup
pipeline.stop()
cv2.destroyAllWindows()

