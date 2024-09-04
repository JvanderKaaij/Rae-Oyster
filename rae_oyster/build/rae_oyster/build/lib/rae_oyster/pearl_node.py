import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge


class ImageProcessor(Node):

    def __init__(self):
        super().__init__('image_processor')

        # Create a subscription to the camera topic
        self.subscription = self.create_subscription(
            Image,
            '/rae/left_back/image_raw/compressed',  # The topic you are subscribing to
            self.image_callback,
            10)

        self.subscription  # prevent unused variable warning

        # Initialize CV Bridge
        self.br = CvBridge()

    def image_callback(self, msg):
        self.get_logger().info('Received image')

        # Convert ROS Image message to OpenCV2 format
        current_frame = self.br.imgmsg_to_cv2(msg, "bgr8")

        # Process the frame using OpenCV2 (or other image processing libraries)
        processed_frame = self.process_image(current_frame)

        # Implement further processing with image_pipeline components if needed
        # Example: Save or publish the processed frame

    def process_image(self, frame):
        # Implement your image processing logic here
        return frame


def main(args=None):
    rclpy.init(args=args)

    image_processor = ImageProcessor()

    rclpy.spin(image_processor)

    # Shutdown and cleanup
    image_processor.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()