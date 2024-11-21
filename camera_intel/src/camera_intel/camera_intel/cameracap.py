import numpy as np
import cv2
import pyrealsense2 as rs

def main():
    # Configure Intel RealSense pipeline
    pipe = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    
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
            cv2.imwrite(img_name, color_image)
            print(f"{img_name} saved!")
            img_counter += 1
        elif key == ord('q'):  # If 'q' is pressed, quit the program
            print("Exiting...")
            break

    # Stop streaming
    pipe.stop()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

