import os
import time
import cv2
import re
import numpy as np
import pyrealsense2 as rs
from ultralytics import YOLO
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

# Sort helper function to extract numbers from filenames
def natural_sort_key(s):
    return [int(text) if text.isdigit() else text for text in re.split(r'(\d+)', s)]

class RealTimeDetectionNode(Node):
    def __init__(self):
        super().__init__('real_time_detection_node')
        
        # Create a publisher
        self.publisher_ = self.create_publisher(String, 'detected_objects', 10)
        
        # Timer to control the publish rate
        self.timer = self.create_timer(0.1, self.publish_detections)  # Publish every 2 seconds

        # Configure Intel RealSense pipeline
        self.pipe = rs.pipeline()
        cfg = rs.config()
        cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

        # Load the YOLO model
        self.model = YOLO("/home/amir/Desktop/test/best.pt")  # Path to your YOLO model
        
        try:
            self.pipe.start(cfg)
        except Exception as e:
            self.get_logger().error(f"Failed to start the Intel RealSense pipeline: {e}")
            rclpy.shutdown()

        # Initialize timing for publishing
        self.last_publish_time = time.time()

        # List to hold the detected objects
        self.detected_objects = []

    def publish_detections(self):
        # Wait for frames from the camera
        frames = self.pipe.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            self.get_logger().warn("No color frame received.")
            return

        # Convert the frame to a NumPy array
        color_image = np.asanyarray(color_frame.get_data())

        # Run YOLO detection on the current frame
        results = self.model.predict(color_image)

        # Clear the detected objects list for new detections
        self.detected_objects.clear()

        # Process detections
        for result in results:
            bboxes = result.boxes.xyxy  # Bounding box coordinates
            confidences = result.boxes.conf  # Confidence scores
            labels = result.boxes.cls  # Class labels

            for i in range(len(bboxes)):
                x_min, y_min, x_max, y_max = map(int, bboxes[i])
                confidence = confidences[i].item()
                label = labels[i].item()
                class_name = self.model.names[label]

                # Calculate the center of the bounding box
                center_x = (x_min + x_max) / 2
                center_y = (y_min + y_max) / 2

                # Add to the detected objects list
                self.detected_objects.append((class_name, center_x, center_y))

                # Optionally, draw bounding boxes and labels on the image
                cv2.rectangle(color_image, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
                label_text = f"{class_name}: {confidence:.2f}"
                cv2.putText(color_image, label_text, (x_min, y_min - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # Publish detected objects
        if time.time() - self.last_publish_time >= 2:
            for class_name, center_x, center_y in self.detected_objects:
                message = f"{class_name}, {center_x:.2f}, {center_y:.2f}"
                self.publisher_.publish(String(data=message))
                self.get_logger().info(f'Publishing: "{message}"')
            self.last_publish_time = time.time()  # Update the last publish time

        # Display the image with detections
        cv2.imshow("Real-Time Detection", color_image)
        if cv2.waitKey(1) == ord('q'):  # Exit the loop if 'q' is pressed
            self.get_logger().info("Exiting real-time detection.")
            self.pipe.stop()
            cv2.destroyAllWindows()
            rclpy.shutdown()

def main(args=None):
    rclpy.init(args=args)
    real_time_detection_node = RealTimeDetectionNode()
    
    try:
        rclpy.spin(real_time_detection_node)
    except KeyboardInterrupt:
        pass
    finally:
        # Stop the pipeline and close windows
        real_time_detection_node.pipe.stop()
        cv2.destroyAllWindows()
        rclpy.shutdown()

if __name__ == "__main__":
    main()

