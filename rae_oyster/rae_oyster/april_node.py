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


class Line:
    def __init__(self, x1, y1, x2, y2):
        if y1 < y2:
            self.x1 = x1
            self.y1 = y1
            self.x2 = x2
            self.y2 = y2
        else:
            self.x1 = x2
            self.y1 = y2
            self.x2 = x1
            self.y2 = y1


class Vector2:
    def __init__(self, x, y):
        self.x = x
        self.y = y


class Rectangle:
    def __init__(self, x, y, w, h):
        self.x = x
        self.y = y
        self.w = w
        self.h = h


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
        self.roi_rect = Rectangle(60, 0, 520, 180)

        self.twist_message = Twist()
        self.key_timer = self.create_timer(0.1, self.update_input)

        self.turn_speed = 2
        self.angular_z = 0.0

        # Define the codec and create VideoWriter object.
        # You can use different codecs. Here we use 'XVID' with .avi file format.
        fourcc = cv2.VideoWriter_fourcc(*'MP4V')

        # Create VideoWriter object. Specify filename, codec, fps, and frame size.
        self.out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (640, 400))

    def update_input(self):
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
        self.out.write(processed_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            # If 'q' is pressed, exit the loop and close the window
            self.out.release()
            cv2.destroyAllWindows()
            self.should_exit = True  # Set the exit flag


    def update_threshold(self, x):
        self.threshold = x

    def update_forward_speed(self, x):
        self.forward_speed = x

    def process_image(self, frame):
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        self.threshold = cv2.getTrackbarPos('Threshold', 'Binary Threshold Control')
        _, binary_image = cv2.threshold(frame_gray, self.threshold, 255, cv2.THRESH_BINARY)

        # Extract the region of interest (ROI)
        roi = binary_image[self.roi_rect.y:self.roi_rect.y + self.roi_rect.h, self.roi_rect.x:self.roi_rect.x + self.cam_width]

        blurred = cv2.GaussianBlur(roi, (5, 5), 0)

        # Perform Canny edge detection
        edges = cv2.Canny(blurred, 50, 150)

        # Use the Probabilistic Hough Line Transform
        lines = cv2.HoughLinesP(
            edges,  # Input edge image
            rho=1,  # Distance resolution in pixels
            theta=np.pi / 180,  # Angle resolution in radians
            threshold=50,  # Minimum number of votes
            minLineLength=25,  # Minimum length of a line in pixels
            maxLineGap=15  # Maximum allowed gap between points on the same line
        )

        color_image = cv2.cvtColor(binary_image, cv2.COLOR_GRAY2BGR)

        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]  # Extract line segment endpoints
                directed_line = Line(x1, y1, x2, y2)
                cv2.line(color_image, (self.roi_rect.x + directed_line.x1, self.roi_rect.y + directed_line.y1), (self.roi_rect.x + directed_line.x2, self.roi_rect.y + directed_line.y2), (255, 0, 0), 2)

        self.angular_z = 0.0
        line_angle = 0.0
        # Draw the lines on the original image
        if lines is not None:
            important = self.most_important_line(lines)

            cv2.line(color_image, (important.x1, important.y1), (important.x2, important.y2), (0, 255, 0), 2)

            line_theta = math.atan2(important.y2 - important.y1, important.x2 - important.x1)

            line_angle = line_theta * (180 / math.pi)

            angle_sign = math.copysign(1, line_angle - 90)
            print(angle_sign)
            angle_diff = 90 - line_angle

            theta_radians = math.radians(angle_diff)  # Convert to radians

            # Calculate spread
            spread = math.sin(theta_radians) ** 2

            angle_multiplier = 5.0

            self.angular_z = angle_sign * min(spread * angle_multiplier, 1.0)

        self.draw_line_angle(color_image, line_angle, (0, 255, 0))  # detected line angle
        self.draw_line_angle(color_image, self.angular_z, (0, 0, 255))  # driving angle
        self.draw_line_angle(color_image, 0, (255, 0, 0))  # zero Point

        cv2.putText(color_image, f'rae angle: {self.angular_z}', (self.roi_rect.x, self.right_camera_info[1]-30), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 0, 255), 1, cv2.LINE_AA)
        cv2.putText(color_image, f'line angle: {line_angle}', (self.roi_rect.x, self.right_camera_info[1] - 50),cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.rectangle(color_image, (self.roi_rect.x, self.roi_rect.y), (self.roi_rect.x + self.roi_rect.w, self.roi_rect.y + self.roi_rect.h), (130, 0, 255), 2)  # Draw rectangle

        return color_image

    def draw_line_angle(self, image, angle, color):
        return self.draw_line_theta(image, angle * (math.pi / 180), color)

    def draw_line_theta(self, image, theta, color):
        line_length = 120

        offset_x2 = int(line_length * math.cos(theta))
        offset_y2 = int(line_length * math.sin(theta))

        end_point_x2 = self.cam_center[0] + offset_x2
        end_point_y2 = self.cam_center[1] + offset_y2

        cv2.line(image, (320, 200), (int(end_point_x2), int(end_point_y2)), color, 2)
        return image

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
                    best_line = Line(self.roi_rect.x + x1, self.roi_rect.y + y1, self.roi_rect.x + x2, self.roi_rect.y + y2)

        return best_line

    def listener_callback(self, msg):
        self.get_logger().info(f'Received camera info: {msg}')

    def clamp(self, value, min_value, max_value):
        return max(min(value, max_value), min_value)


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
