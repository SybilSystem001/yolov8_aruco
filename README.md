# yolov8_aruco
Ros2 jazzy package for yolov8 detection of objects with arUco tags, it detects and bound the items with a box and publishes its position in a 3D space if they have an arUCo tag.
``
colcon build --symlink-install --executor sequential --cmake-args
``
``
ros2 launch yolov8_obb yolov8_obb.launch.py
``
if after building an error of some executable file not appearing happens the current fix is go to the file direction and make it executable manually
# Requirements for the ROS2 ArUco + YOLO detection project
It is worth nothing that currently ubuntu noble works with a controlled environment so some of the dependancies such as the Ulralytics module should be installed and run in python a virtual environment.
# Core dependencies
numpy==1.26.4               # Compatible with your current setup and ROS2 requirements
opencv-contrib-python==4.8.0.76  # Includes ArUco, compatible with NumPy 1.x
scipy==1.11.3               # For spatial transformations and numerical operations
transformations3d==0.3.1    # Optional: For advanced 3D transformations, if used
