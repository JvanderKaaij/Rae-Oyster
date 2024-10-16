import cv2
import numpy as np
import rclpy
import math
from pyapriltags import Detector

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
        self.tag_size = 1.0
        self.camera_params = (6000, 6000, 4208/2, 3120/2)

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
        color_frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        gray_frame = cv2.cvtColor(color_frame, cv2.COLOR_BGR2GRAY)

        at_detector = Detector(
            families="tag36h11",
            nthreads=1,
            quad_decimate=1.0,
            quad_sigma=0.0,
            refine_edges=1,
            decode_sharpening=0.25,
            debug=0
        )

        tags = at_detector.detect(gray_frame, True, camera_params=self.camera_params, tag_size=self.tag_size)

        for tag in tags:
            print(tag.tag_id)
            self._draw_pose(color_frame,
                            self.camera_params,
                            tag.pose_R,
                            tag.pose_t)
            self.triangulate(tag.pose_t)

        cv2.imshow('April Tags', color_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            # If 'q' is pressed, exit the loop and close the window
            cv2.destroyAllWindows()
            self.should_exit = True  # Set the exit flag


    def triangulate(self, transform):
        t_x, t_y, t_z = transform
        print(t_z)


    def _draw_pose(self, overlay, cam_params, pose_r, pose_t):
        opoints = np.array([
            -1, -1, 0,
            1, -1, 0,
            1, 1, 0,
            -1, 1, 0,
            -1, -1, -2,
            1, -1, -2,
            1, 1, -2,
            -1, 1, -2
        ]).reshape(8, 3) * 0.5 * self.tag_size

        edges = np.array([
            0, 1,
            1, 2,
            2, 3,
            3, 0,
            0, 4,
            1, 5,
            2, 6,
            3, 7,
            4, 5,
            5, 6,
            6, 7,
            7, 4
        ]).reshape(-1, 2)

        fx, fy, cx, cy = cam_params

        K = np.array([fx, 0, cx, 0, fy, cy, 0, 0, 1], dtype=np.float32).reshape(3, 3)

        dcoeffs = np.zeros(5)

        ipoints, _ = cv2.projectPoints(opoints, pose_r, pose_t, K, dcoeffs)

        ipoints = np.round(ipoints).astype(int)

        ipoints = [tuple(pt) for pt in ipoints.reshape(-1, 2)]

        for i, j in edges:
            cv2.line(overlay, ipoints[i], ipoints[j], (0, 255, 0), 2, 16)

def main(args=None):
    rclpy.init(args=args)
    print('lets go! openCV!')
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
