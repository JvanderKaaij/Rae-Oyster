import cv2
import numpy as np
import rclpy
import math

from rclpy.node import Node
from sensor_msgs.msg import CompressedImage, Image, CameraInfo
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
from PIL import Image as PILImage, ImageDraw
from pathlib import Path


class ImageProcessor(Node):

    def __init__(self):
        super().__init__('image_processor')
        # Create a subscription to the camera topic
        self.br = CvBridge()

        self.publisher_image = self.create_publisher(Image, '/lcd', 10)

        self.subscription = self.create_subscription(
            CompressedImage,
            '/rae/right/image_raw/compressed',  # The topic you are subscribing to
            self.image_callback,
            10)

        self.subscription  # prevent unused variable warning
        # Initialize CV Bridge
        self.should_exit = False  # Control flag for exiting
        self.twist_publish = self.create_publisher(Twist, '/cmd_vel', rclpy.qos.qos_profile_system_default)
        # self.subscription_info = self.create_subscription(
        #     CameraInfo,
        #     '/rae/right/camera_info',
        #     self.listener_callback,
        #     10)

        cv2.namedWindow('Binary Threshold Control')
        self.threshold = 253
        self.forward_speed = 0

        cv2.createTrackbar('Threshold', 'Binary Threshold Control', self.threshold, 255, self.update_threshold)
        cv2.createTrackbar('DriveSpeed', 'Binary Threshold Control', self.forward_speed, 10, self.update_forward_speed)

        self.right_camera_info = (640, 400)
        self.cam_width = self.right_camera_info[0]
        self.cam_height = self.right_camera_info[1]

        self.cam_center = (self.cam_width/2, self.cam_height/2)

        self.twist_message = Twist()
        self.key_timer = self.create_timer(0.1, self.update_input)

        self.turn_speed = 2
        self.angular_z = 0.0

    def update_input(self):
        key = cv2.waitKey(1)
        print(self.angular_z)
        self.twist_message.angular.z = self.angular_z
        self.twist_message.linear.x = self.forward_speed * 0.05
        self.twist_publish.publish(self.twist_message)

    def image_callback(self, msg):
        np_arr = np.frombuffer(msg.data, np.uint8)
        # Convert ROS Compressed Image message to OpenCV2 format
        current_frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        # Process the frame using OpenCV2 (or other image processing libraries)
        processed_frame = self.process_image(current_frame)
        # Step 4: Create a trackbar in the window
        cv2.imshow('Binary Threshold Control', processed_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            # If 'q' is pressed, exit the loop and close the window
            cv2.destroyAllWindows()
            self.should_exit = True  # Set the exit flag

    def update_threshold(self, x):
        self.threshold = x

    def update_forward_speed(self, x):
        self.forward_speed = x

    def process_image(self, frame):
        # Implement your image processing logic here
        # Convert to gray scale image.
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # binary_image = cv2.adaptiveThreshold(
        #     frame_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2
        # )
        #
        self.threshold = cv2.getTrackbarPos('Threshold', 'Binary Threshold Control')
        _, binary_image = cv2.threshold(frame_gray, self.threshold, 255, cv2.THRESH_BINARY)

        blurred = cv2.GaussianBlur(binary_image, (5, 5), 0)

        # Perform Canny edge detection
        edges = cv2.Canny(blurred, 50, 150)

        # Use the Probabilistic Hough Line Transform
        lines = cv2.HoughLinesP(
            edges,  # Input edge image
            rho=1,  # Distance resolution in pixels
            theta=np.pi / 180,  # Angle resolution in radians
            threshold=100,  # Minimum number of votes
            minLineLength=25,  # Minimum length of a line in pixels
            maxLineGap=10  # Maximum allowed gap between points on the same line
        )

        color_image = cv2.cvtColor(binary_image, cv2.COLOR_GRAY2BGR)

        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]  # Extract line segment endpoints
                cv2.line(color_image, (x1, y1), (x2, y2), (255, 0, 0), 2)

        self.angular_z = 0.0
        # Draw the lines on the original image
        if lines is not None:
            important = self.most_important_line(lines)
            cv2.line(color_image, (important[0], important[1]), (important[2], important[3]), (0, 255, 0), 2)
            line_theta = math.atan2(important[3] - important[1], important[2] - important[0])
            line_angle = line_theta * (180 / math.pi)

            angle_sign = math.copysign(1, line_angle)
            angle_diff = 90 - abs(line_angle)

            self.angular_z = (angle_sign * angle_diff) * self.turn_speed

        return color_image


    def most_important_line(self, lines):
        min_distance = float('inf')
        best_line = None
        top_center = (self.right_camera_info[0] // 2, 0)
        for line in lines:
            for x1, y1, x2, y2 in line:
                # Calculate the midpoint of the current line segment
                midpoint = ((x1 + x2) // 2, (y1 + y2) // 2)
                distance = math.sqrt((midpoint[0] - top_center[0]) ** 2 + (midpoint[1] - top_center[1]) ** 2)
                if distance < min_distance:
                    min_distance = distance
                    best_line = (x1, y1, x2, y2)

        return best_line

    def listener_callback(self, msg):
        self.get_logger().info(f'Received camera info: {msg}')

    def clamp(self, value, min_value, max_value):
        return max(min(value, max_value), min_value)


def main(args=None):
    rclpy.init(args=args)

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
