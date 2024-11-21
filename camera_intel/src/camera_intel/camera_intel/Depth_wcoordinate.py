import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PointStamped
from tf2_ros import Buffer, TransformListener, LookupException, ConnectivityException, ExtrapolationException

class ObjectPositionTransformer(Node):
    def __init__(self):
        super().__init__('object_position_transformer')

        # Create a buffer and transform listener
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Object coordinates in the camera frame
        self.object_camera_frame = [0.08, -0.05, 0.45]  # Replace with actual coordinates

        # Transform the object position
        self.get_object_world_position()

    def get_object_world_position(self):
        try:
            # Wait for the transform to become available
            transform = self.tf_buffer.lookup_transform('world_frame', 'camera_frame', rclpy.time.Time())

            # Create a PointStamped message for the object's camera frame position
            object_point = PointStamped()
            object_point.header.frame_id = 'camera_frame'
            object_point.point.x = self.object_camera_frame[0]
            object_point.point.y = self.object_camera_frame[1]
            object_point.point.z = self.object_camera_frame[2]

            # Transform the object position to the world frame
            object_world = self.tf_buffer.transform(object_point, 'world_frame')

            # Print the object position in the world frame
            self.get_logger().info(f'Object position in world frame: x={object_world.point.x}, '
                                   f'y={object_world.point.y}, z={object_world.point.z}')

        except (LookupException, ConnectivityException, ExtrapolationException) as e:
            self.get_logger().error(f'Error in transforming coordinates: {e}')

def main(args=None):
    rclpy.init(args=args)
    node = ObjectPositionTransformer()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

