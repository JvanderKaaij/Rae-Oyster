import cv2
import numpy as np
import rclpy
import math
from pupil_apriltags import Detector

from rclpy.node import Node
from sensor_msgs.msg import CompressedImage, Image, CameraInfo
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
from PIL import Image as PILImage, ImageDraw
from pathlib import Path


class ImageProcessor(Node):

    def __init__(self):
        super().__init__('image_processor')
        self.br = CvBridge()

        self.should_exit = False  # Control flag for exiting

        # Create a subscription to the camera topic
        self.publisher_image = self.create_publisher(Image, '/lcd', 10)

        self.subscription = self.create_subscription(
            CompressedImage,
            '/rae/right/image_raw/compressed',  # The topic you are subscribing to
            self.image_callback,
            10)

        self.subscription  # prevent unused variable warning


    def image_callback(self, msg):
        np_arr = np.frombuffer(msg.data, np.uint8)
        # Convert ROS Compressed Image message to OpenCV2 format
        current_frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        at_detector = Detector(
            families="tag36h11",
            nthreads=1,
            quad_decimate=1.0,
            quad_sigma=0.0,
            refine_edges=1,
            decode_sharpening=0.25,
            debug=0
        )

        at_detector.detect(current_frame)

        cv2.imshow('Binary Threshold Control', processed_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            # If 'q' is pressed, exit the loop and close the window
            cv2.destroyAllWindows()
            self.should_exit = True  # Set the exit flag


def main(args=None):
    rclpy.init(args=args)
    print('lets go!')
    image_processor = ImageProcessor()

    try:
        while rclpy.ok() and not image_processor.should_exit:
            rclpy.spin_once(image_processor)  # Process callbacks
    finally:
        print('doing shutdown')
        # Shutdown and cleanup
        image_processor.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
