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
from sympy import symbols, Eq, solve

class AprilTag:
    def __init__(self, id, x, y, z):
        self.id = id
        self.x = x
        self.y = y
        self.z = z
        self.d = 0


class ImageProcessor(Node):

    def __init__(self):
        super().__init__('image_processor')

        # self.tags = (
        #     AprilTag(0, -2.0, 0.0, 0.0),
        #     AprilTag(1, 0.0, 0.17, 1.5),
        #     AprilTag(2, -2.0, 0.0, 2.0),
        #     AprilTag(3, 1.5, 0.11, 2.0),
        #     AprilTag(4, 1.5, 0.0, 0.0)
        # )

        self.tags = (
            AprilTag(0, 0, 0, 1.5),
            AprilTag(4, 1.5, 0, 2)
        )

        field_rect = ((-2, 2), (2, 2), (2, -2), (-2, -2))

        self.field_min_x = min(point[0] for point in field_rect)
        self.field_max_x = max(point[0] for point in field_rect)
        self.field_min_y = min(point[1] for point in field_rect)
        self.field_max_y = max(point[1] for point in field_rect)

        self.br = CvBridge()

        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        self.out = cv2.VideoWriter('april_output.avi', fourcc, 20.0, (640, 400))

        self.should_exit = False  # Control flag for exiting
        # self.tag_size = 0.08
        self.tag_size = 0.16
        self.camera_params = (284.659, 284.659, 320.594, 200.622)

        self.camera_matrix = np.array([
            [284.659, 0, 320.594],
            [0, 284.659, 200.622],
            [0, 0, 1]
        ])

        self.dist_coeffs = np.array([-5.7565, 17.2065, -0.0008, -0.0003, 4.1285, -5.4157, 15.1905, 10.3328])

        # Create a subscription to the camera topic
        self.publisher_image = self.create_publisher(Image, '/lcd', 10)

        self.subscription = self.create_subscription(
            CompressedImage,
            '/rae/right/image_raw/compressed',  # The topic you are subscribing to
            self.image_callback,
            10)

        # self.subscription_camera_info = self.create_subscription(
        #     CameraInfo,
        #     "/rae/right/camera_info",  # The topic you are subscribing to
        #     self.camera_info_callback,
        #     10)

    def shutdown(self):
        print('shutting down video out')
        self.out.release()

    def camera_info_callback(self, data):
        # This function will be called whenever a new message is received on /rae/right/camera_info
        print("Received camera info:")
        print(f"Width: {data.width}, Height: {data.height}")
        print(f"Distortion Model: {data.distortion_model}")
        print(f"Camera Matrix (K): {data.k}")
        # print(f"Distortion Coefficients: {data.D}")


    def image_callback(self, msg):
        np_arr = np.frombuffer(msg.data, np.uint8)
        # Convert ROS Compressed Image message to OpenCV2 format
        color_frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        # Undistort the image
        undistorted_img = cv2.undistort(color_frame, self.camera_matrix, self.dist_coeffs)

        gray_frame = cv2.cvtColor(undistorted_img, cv2.COLOR_BGR2GRAY)

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

        robot_position = None

        matches = self._handle_tags(tags, undistorted_img)

        if len(matches) >= 2:
            robot_position = self._triangulate(matches[0], matches[1])


        if robot_position:
            cv2.putText(undistorted_img, f'rae pos X: {robot_position[0]:.1f}, Y: {robot_position[1]:.1f}', (100, 100) ,cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 0, 255), 1, cv2.LINE_AA)

        self.out.write(undistorted_img)

        cv2.imshow('April Tags', undistorted_img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            # If 'q' is pressed, exit the loop and close the window
            cv2.destroyAllWindows()
            self.out.release()
            self.should_exit = True  # Set the exit flag

    def _handle_tags(self, tags, undistorted_img):
        matches = []
        print(f"encountered {len(tags)} tags:")
        for tag in tags:
            match = self._handle_tag(tag)
            print(f"encountered tag {tag.tag_id}")
            if match:
                match.d = np.linalg.norm(tag.pose_t)
                print(f'tag: {tag.tag_id} distance: {match.d}')
                matches.append(match)

            self._draw_pose(undistorted_img,
                            self.camera_params,
                            tag.pose_R,
                            tag.pose_t)
        return matches
    def _handle_tag(self, tag):
        match_tag = next((x for x in self.tags if x.id == tag.tag_id), None)
        if match_tag:
            return match_tag
        return None

    def _invert_pose(self, R, t):
        # Invert the rotation
        R_inv = R.T
        # Invert the translation
        t_inv = -np.dot(R_inv, t)
        return R_inv, t_inv

    def _triangulate(self, tag_one, tag_two):
        x1 = tag_one.x
        x2 = tag_two.x
        y1 = tag_one.z
        y2 = tag_two.z
        d1 = tag_one.d
        d2 = tag_two.d

        x, y = symbols('x y')
        eq1 = Eq((x - x1) ** 2 + (y - y1) ** 2, d1 ** 2)
        eq2 = Eq((x - x2) ** 2 + (y - y2) ** 2, d2 ** 2)

        solution = solve((eq1, eq2), (x, y))
        sol1 = self._aabb(solution[0])
        sol2 = self._aabb(solution[1])

        if sol1:
            return solution[0]
        elif sol2:
            return solution[1]
        print("Something went wrong, no solution found")

    def _triangulate_old(self, transform, rotation):
        R_cam_to_tag, t_cam_to_tag = self._invert_pose(rotation, transform)
        return R_cam_to_tag, t_cam_to_tag


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
            cv2.line(overlay, ipoints[i], ipoints[j], (0, 255, 0), 1, 16)

    def _aabb(self, coord):
        px = coord[0]
        py = coord[1]
        if self.field_min_x <= px <= self.field_max_x and self.field_min_y <= py <= self.field_max_y:
            return True
        return False

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
        image_processor.shutdown()
        image_processor.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
