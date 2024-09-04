import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage, Image, CameraInfo
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

        self.timer = self.create_timer(0.1, self.draw_eyes)

        haar_path = 'rae_oyster/resource/haarcascade_frontalface_default.xml'
        complete_path = Path.cwd()/haar_path

        # self.subscription_info = self.create_subscription(
        #     CameraInfo,
        #     '/rae/right/camera_info',
        #     self.listener_callback,
        #     10)

        self.right_camera_info = (640, 400)
        self.cam_width = self.right_camera_info[0]
        self.cam_height = self.right_camera_info[1]

        self.cam_center = (self.cam_width/2, self.cam_height/2)
        self.face_off_center = (0, 0)

        self.face_cascade = cv2.CascadeClassifier(str(complete_path))

    def image_callback(self, msg):
        np_arr = np.frombuffer(msg.data, np.uint8)
        # Convert ROS Compressed Image message to OpenCV2 format
        current_frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        # Process the frame using OpenCV2 (or other image processing libraries)
        processed_frame = self.process_image(current_frame)

        cv2.imshow('Processed Image', processed_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            # If 'q' is pressed, exit the loop and close the window
            cv2.destroyAllWindows()
            self.should_exit = True  # Set the exit flag

    def process_image(self, frame):
        # Implement your image processing logic here
        # Convert to gray scale image.
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        ## Detect faces & returns positions of faces as Rect(x,y,w,h).
        face_rects = self.face_cascade.detectMultiScale(frame_gray, 1.3, 5)

        # Draw rectangles representing the detected faces.
        for (x, y, w, h) in face_rects:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        if len(face_rects) > 0:
            (x, y, w, h) = face_rects[0]
            face_center = (x + w/2, y + h/2)
            self.face_off_center = (self.cam_center[0] - face_center[0], self.cam_center[1] - face_center[1])
            print(self.face_off_center)

        return frame

    def create_eyes_image(self, draw):

        center_x = 40
        center_y = 40
        eye_radius = 20
        margin = center_x - eye_radius
        x_offset = self.clamp(self.face_off_center[0], -margin, margin)
        y_offset = self.clamp(self.face_off_center[1], -margin, margin)

        left_eye_center = (center_x + x_offset, center_y - y_offset)
        right_eye_center = (120 + x_offset, center_y - y_offset)

        pupil_radius = 8

        draw.ellipse(
            (left_eye_center[0] - eye_radius, left_eye_center[1] - eye_radius,
             left_eye_center[0] + eye_radius, left_eye_center[1] + eye_radius),
            fill="white"
        )

        draw.ellipse(
            (left_eye_center[0] - pupil_radius, left_eye_center[1] - pupil_radius,
             left_eye_center[0] + pupil_radius, left_eye_center[1] + pupil_radius),
            fill="black"
        )

        draw.ellipse(
            (right_eye_center[0] - eye_radius, right_eye_center[1] - eye_radius,
             right_eye_center[0] + eye_radius, right_eye_center[1] + eye_radius),
            fill="white"
        )

        draw.ellipse(
            (right_eye_center[0] - pupil_radius, right_eye_center[1] - pupil_radius,
             right_eye_center[0] + pupil_radius, right_eye_center[1] + pupil_radius),
            fill="black"
        )

    def draw_eyes(self):
        img = PILImage.new('RGB', (160, 80), "black")
        draw = ImageDraw.Draw(img)

        self.create_eyes_image(draw)

        img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

        # Convert OpenCV image to ROS image and publish
        img_msg = self.br.cv2_to_imgmsg(img_cv, encoding="bgr8")
        self.publisher_image.publish(img_msg)

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
