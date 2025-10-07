#!/usr/bin/env python3

from ultralytics import YOLO
import os
import copy
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import TransformStamped, PoseStamped
from cv_bridge import CvBridge
from tf2_ros import TransformBroadcaster
from yolov8_msgs.msg import InferenceResult
from yolov8_msgs.msg import Yolov8Inference

import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R

bridge = CvBridge()

# ArUco dictionary definitions
ARUCO_DICT = {
    "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
    "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
    "DICT_4X4_250": cv2.aruco.DICT_4X4_250,
    "DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
    "DICT_5X5_50": cv2.aruco.DICT_5X5_50,
    "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
    "DICT_5X5_250": cv2.aruco.DICT_5X5_250,
    "DICT_5X5_1000": cv2.aruco.DICT_5X5_1000,
    "DICT_6X6_50": cv2.aruco.DICT_6X6_50,
    "DICT_6X6_100": cv2.aruco.DICT_6X6_100,
    "DICT_6X6_250": cv2.aruco.DICT_6X6_250,
    "DICT_6X6_1000": cv2.aruco.DICT_6X6_1000,
    "DICT_7X7_50": cv2.aruco.DICT_7X7_50,
    "DICT_7X7_100": cv2.aruco.DICT_7X7_100,
    "DICT_7X7_250": cv2.aruco.DICT_7X7_250,
    "DICT_7X7_1000": cv2.aruco.DICT_7X7_1000,
    "DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL
}


def my_estimatePoseSingleMarkers(corners, marker_size, mtx, distortion):
    """Custom implementation replacing deprecated estimatePoseSingleMarkers"""
    marker_points = np.array([[-marker_size / 2, marker_size / 2, 0],
                              [marker_size / 2, marker_size / 2, 0],
                              [marker_size / 2, -marker_size / 2, 0],
                              [-marker_size / 2, -marker_size / 2, 0]], dtype=np.float32)
    rvecs = []
    tvecs = []

    for c in corners:
        success, rvec, tvec = cv2.solvePnP(marker_points, c, mtx, distortion, False, cv2.SOLVEPNP_IPPE_SQUARE)
        if success:
            rvecs.append(rvec)
            tvecs.append(tvec)

    return rvecs, tvecs


class CombinedDetectionVisualizer(Node):

    def __init__(self):
        super().__init__('combined_detection_visualizer')

        # YOLO model initialization
        self.model = YOLO(os.environ['HOME'] + '/moveit2_obb/src/yolov8_obb/scripts/best.pt')
        self.yolov8_inference = Yolov8Inference()

        # ArUco parameters
        self.declare_parameter("aruco_dictionary_name", "DICT_6X6_250")
        self.declare_parameter("aruco_marker_side_length", 0.04)  # 4cm markers
        self.declare_parameter("camera_frame", "camera_link")
        self.declare_parameter("show_preview_window", True)  # Enable/disable OpenCV preview window

        aruco_dictionary_name = self.get_parameter("aruco_dictionary_name").get_parameter_value().string_value
        self.aruco_marker_side_length = self.get_parameter(
            "aruco_marker_side_length").get_parameter_value().double_value
        self.camera_frame = self.get_parameter("camera_frame").get_parameter_value().string_value
        self.show_preview = self.get_parameter("show_preview_window").get_parameter_value().bool_value

        # Initialize ArUco detection system
        if ARUCO_DICT.get(aruco_dictionary_name, None) is None:
            self.get_logger().error(f"ArUco dictionary '{aruco_dictionary_name}' is not supported")
            return

        self.aruco_dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT[aruco_dictionary_name])
        self.aruco_parameters = cv2.aruco.DetectorParameters()
        self.aruco_detector = cv2.aruco.ArucoDetector(self.aruco_dictionary, self.aruco_parameters)

        # Camera calibration parameters
        self.mtx = np.array([[1029.5579833984375, 0.0, 644.6905517578125],
                             [0.0, 1029.5579833984375, 366.17266845703125],
                             [0.0, 0.0, 1.0]])
        self.dist = np.array([6.6431074142456055, -68.78340148925781, -1.1983673175564036e-05, -0.0007913305889815092, 276.66741943359375])

        # ROS2 setup
        self.subscription = self.create_subscription(
            Image,
            '/tb4_jazzy/oakd/rgb/image_raw',
            #'/oak/rgb/image_raw',
            self.camera_callback,
            10)

        # Publishers
        self.yolov8_pub = self.create_publisher(Yolov8Inference, "/Yolov8_Inference", 1)
        self.combined_img_pub = self.create_publisher(Image, "/combined_detection_result", 1)
        self.aruco_pose_pub = self.create_publisher(PoseStamped, "/aruco_pose", 1)

        # TF broadcaster
        self.tf_broadcaster = TransformBroadcaster(self)

        # Visualization colors
        self.yolo_color = (0, 255, 0)  # Green for YOLO boxes
        self.aruco_color = (255, 0, 0)  # Blue for ArUco markers
        self.text_color = (255, 255, 255)  # White for text
        self.axes_length = 0.03  # Length of coordinate axes

        self.get_logger().info(f"Combined Detection Visualizer initialized")
        self.get_logger().info(f"ArUco dictionary: {aruco_dictionary_name}")
        self.get_logger().info(f"Preview window: {'Enabled' if self.show_preview else 'Disabled'}")

    def draw_yolo_detections(self, img, results):
        """Draw YOLO detection results on the image"""
        yolo_detections = []

        for r in results:
            if r.obb is not None:
                boxes = r.obb
                for box in boxes:
                    # Get bounding box coordinates
                    b = box.xyxyxyxy[0].to('cpu').detach().numpy().copy()
                    c = box.cls
                    conf = box.conf.to('cpu').detach().numpy()[0] if box.conf is not None else 0.0
                    class_name = self.model.names[int(c)]

                    # Convert to integer coordinates for drawing
                    points = b.reshape(-1, 2).astype(np.int32)

                    # Draw rotated bounding box
                    cv2.polylines(img, [points], isClosed=True, color=self.yolo_color, thickness=2)

                    # Add class name and confidence
                    label = f"YOLO: {class_name} ({conf:.2f})"
                    label_pos = (int(points[0][0]), int(points[0][1]) - 10)

                    # Draw text background
                    (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    cv2.rectangle(img,
                                  (label_pos[0] - 2, label_pos[1] - text_height - 4),
                                  (label_pos[0] + text_width + 2, label_pos[1] + 4),
                                  self.yolo_color, -1)

                    # Draw text
                    cv2.putText(img, label, label_pos, cv2.FONT_HERSHEY_SIMPLEX,
                                0.6, self.text_color, 2)

                    # Store detection info
                    detection_info = {
                        'class_name': class_name,
                        'confidence': conf,
                        'coordinates': b.tolist()
                    }
                    yolo_detections.append(detection_info)

                    # Create and store inference result for publishing
                    inference_result = InferenceResult()
                    inference_result.class_name = class_name
                    inference_result.coordinates = copy.copy(b.reshape(1, 8)[0].tolist())
                    self.yolov8_inference.yolov8_inference.append(inference_result)

        return yolo_detections

    def draw_aruco_detections(self, img, image_stamp):
        """Detect and draw ArUco markers on the image"""
        # Convert to grayscale for ArUco detection
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Detect markers
        corners, ids, rejected = self.aruco_detector.detectMarkers(gray)
        aruco_detections = []

        if ids is not None and len(ids) > 0:
            # Draw detected markers with custom color
            cv2.aruco.drawDetectedMarkers(img, corners, ids, self.aruco_color)

            try:
                # Estimate poses
                rvecs, tvecs = my_estimatePoseSingleMarkers(
                    corners, self.aruco_marker_side_length, self.mtx, self.dist)

                # Process each detected marker
                for i, marker_id in enumerate(ids.flatten()):
                    if i < len(rvecs):
                        # Draw coordinate axes
                        cv2.drawFrameAxes(img, self.mtx, self.dist,
                                          rvecs[i], tvecs[i], self.axes_length)

                        # Calculate distance and angles
                        distance = np.linalg.norm(tvecs[i])

                        # Convert rotation vector to Euler angles for display
                        rotation_matrix, _ = cv2.Rodrigues(rvecs[i])
                        r = R.from_matrix(rotation_matrix)
                        euler_angles = r.as_euler('xyz', degrees=True)

                        # Publish TF and pose with image timestamp
                        self.publish_marker_tf(marker_id, rvecs[i], tvecs[i], image_stamp)
                        self.publish_marker_pose(marker_id, rvecs[i], tvecs[i], image_stamp)

                        # ... rest of your visualization code remains the same ...

            except Exception as e:
                self.get_logger().error(f"Error in ArUco pose estimation: {str(e)}")

        return aruco_detections

    def add_status_overlay(self, img, yolo_count, aruco_count):
        """Add status information overlay to the image"""
        # Status text
        status_lines = [
            f"YOLO Detections: {yolo_count}",
            f"ArUco Markers: {aruco_count}",
            f"Frame: {img.shape[1]}x{img.shape[0]}",
            "Green=YOLO, Blue=ArUco"
        ]

        # Position at top-left corner
        start_x, start_y = 10, 30

        for i, line in enumerate(status_lines):
            y_pos = start_y + i * 25

            # Background rectangle
            (text_width, text_height), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(img,
                          (start_x - 5, y_pos - text_height - 5),
                          (start_x + text_width + 5, y_pos + 5),
                          (0, 0, 0), -1)  # Black background

            # Text
            cv2.putText(img, line, (start_x, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    def publish_marker_tf(self, marker_id, rvec, tvec, image_stamp):
        """Publish TF transform for detected marker"""
        try:
            t = TransformStamped()
            # Use the image timestamp instead of current time
            t.header.stamp = image_stamp  # Changed from self.get_clock().now().to_msg()
            t.header.frame_id = self.camera_frame
            t.child_frame_id = f"aruco_marker_{marker_id}"

            t.transform.translation.x = float(tvec[0][0])
            t.transform.translation.y = float(tvec[1][0])
            t.transform.translation.z = float(tvec[2][0])

            rotation_matrix, _ = cv2.Rodrigues(rvec)
            r = R.from_matrix(rotation_matrix)
            quat = r.as_quat()

            t.transform.rotation.x = quat[0]
            t.transform.rotation.y = quat[1]
            t.transform.rotation.z = quat[2]
            t.transform.rotation.w = quat[3]

            self.tf_broadcaster.sendTransform(t)
        except Exception as e:
            self.get_logger().error(f"TF publish error: {str(e)}")

    def publish_marker_pose(self, marker_id, rvec, tvec, image_stamp):
        """Publish pose message for detected marker"""
        try:
            pose_msg = PoseStamped()
            pose_msg.header.stamp = image_stamp  # Use image timestamp
            pose_msg.header.frame_id = self.camera_frame

            pose_msg.pose.position.x = float(tvecs[i][0][0])
            pose_msg.pose.position.y = float(tvecs[i][1][0])
            pose_msg.pose.position.z = float(tvecs[i][2][0])

            rotation_matrix, _ = cv2.Rodrigues(rvec)
            r = R.from_matrix(rotation_matrix)
            quat = r.as_quat()

            pose_msg.pose.orientation.x = quat[0]
            pose_msg.pose.orientation.y = quat[1]
            pose_msg.pose.orientation.z = quat[2]
            pose_msg.pose.orientation.w = quat[3]

            self.aruco_pose_pub.publish(pose_msg)
        except Exception as e:
            self.get_logger().error(f"Pose publish error: {str(e)}")

    def camera_callback(self, data):
        """Main callback function that processes camera images"""
        try:
            # Convert ROS image to OpenCV
            img = bridge.imgmsg_to_cv2(data, "bgr8")

            # Store the image timestamp
            image_stamp = data.header.stamp

            # Create a copy for combined visualization
            combined_img = img.copy()

            # Set up YOLO inference message
            self.yolov8_inference.header.frame_id = "inference"
            self.yolov8_inference.header.stamp = self.get_clock().now().to_msg()

            # Perform YOLO detection
            results = self.model(img, conf=0.90)
            yolo_detections = self.draw_yolo_detections(combined_img, results)

            # Perform ArUco detection with image timestamp
            aruco_detections = self.draw_aruco_detections(combined_img, image_stamp)

            # Add status overlay
            self.add_status_overlay(combined_img, len(yolo_detections), len(aruco_detections))

            # Publish YOLO results
            self.yolov8_pub.publish(self.yolov8_inference)
            self.yolov8_inference.yolov8_inference.clear()

            # Publish combined visualization
            combined_msg = bridge.cv2_to_imgmsg(combined_img, "bgr8")
            combined_msg.header.stamp = image_stamp  # Use original timestamp
            self.combined_img_pub.publish(combined_msg)

            # Show preview window if enabled
            #if self.show_preview:
             #   cv2.imshow('Combined Detection Preview', combined_img)
             #   cv2.waitKey(1)  # Non-blocking

        except Exception as e:
            self.get_logger().error(f"Error in camera callback: {str(e)}")



if __name__ == '__main__':
    rclpy.init(args=None)
    try:
        node = CombinedDetectionVisualizer()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        rclpy.shutdown()