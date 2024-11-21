import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PointStamped
from tf2_ros import Buffer, TransformListener, LookupException, ConnectivityException, ExtrapolationException
import tf2_geometry_msgs  # Import to ensure PointStamped is recognized by tf2_ros
import time  # Required for adding sleep to allow TF initialization

class CameraPositionTransformer(Node):
    def __init__(self):
        super().__init__('camera_position_transformer')

        # Initialize a TF buffer and listener to manage and listen for transforms
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Add a delay to allow TF frames to become available
        self.get_logger().info("Waiting for transforms to be available...")
        time.sleep(5)  # Wait for frames to be published and available in the TF tree

    def get_camera_to_base_transform(self):
        try:
            # Attempt to look up the transform from `wx250s/base_link` to `camera_link`
            transform = self.tf_buffer.lookup_transform(
                'wx250s/base_link', 'camera_link', rclpy.time.Time(), timeout=rclpy.duration.Duration(seconds=2.0)
            )
            self.get_logger().info("Transform from base_link to camera_link acquired successfully.")
            return transform
        except (LookupException, ConnectivityException, ExtrapolationException) as e:
            # Log the error for troubleshooting
            self.get_logger().error(f'Error in getting transform: {e}')
            return None

    def transform_object_position(self, object_position_camera_frame):
        try:
            # Define the object position as a PointStamped message in `camera_link` frame
            object_point = PointStamped()
            object_point.header.frame_id = 'camera_link'
            object_point.header.stamp = self.get_clock().now().to_msg()
            object_point.point.x = object_position_camera_frame[0]
            object_point.point.y = object_position_camera_frame[1]
            object_point.point.z = object_position_camera_frame[2]

            # Transform object position from `camera_link` to `wx250s/base_link` frame
            object_in_base_frame = self.tf_buffer.transform(object_point, 'wx250s/base_link')
            self.get_logger().info(
                f'Object position in wx250s/base_link frame: x={object_in_base_frame.point.x}, '
                f'y={object_in_base_frame.point.y}, z={object_in_base_frame.point.z}'
            )
            return object_in_base_frame
        except (LookupException, ConnectivityException, ExtrapolationException) as e:
            # Log if there's an issue with transforming the coordinates
            self.get_logger().error(f'Error transforming object position: {e}')
            return None

def main(args=None):
    # Initialize the ROS2 Python client library
    rclpy.init(args=args)
    
    # Instantiate the CameraPositionTransformer node
    node = CameraPositionTransformer()
    
    # Get and display the transform from the camera to the robot base
    camera_to_base_transform = node.get_camera_to_base_transform()
    if camera_to_base_transform:
        node.get_logger().info(f'Camera position relative to wx250s/base_link: {camera_to_base_transform}')

    # Define an example object position in the camera frame to transform
    object_position = [0.1, -0.05, 0.4]  # Replace with detected object coordinates from the camera
    object_in_base_frame = node.transform_object_position(object_position)
    if object_in_base_frame:
        node.get_logger().info(
            f'Object position in wx250s/base_link frame: x={object_in_base_frame.point.x}, '
            f'y={object_in_base_frame.point.y}, z={object_in_base_frame.point.z}'
        )

    # Keep the node running until it is stopped by the user
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

