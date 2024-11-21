import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

class IntelSubscriber(Node):
    def __init__(self):
        super().__init__("intel_subscriber")
        self.subscription_rgb = self.create_subscription(Image, "rgb_frame", self.rgb_frame_callback, 10)
        self.br_rgb = CvBridge()

    def rgb_frame_callback(self, data):
        self.get_logger().info("Receiving RGB frame")
        current_frame = self.br_rgb.imgmsg_to_cv2(data)

        # Display the image in the existing window opened by the publisher
        cv2.imshow("Camera Feed", current_frame)


def main(args=None):
    rclpy.init(args=args)
    intel_subscriber = IntelSubscriber()
    
    try:
        rclpy.spin(intel_subscriber)
    except KeyboardInterrupt:
        pass
    finally:
        intel_subscriber.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

