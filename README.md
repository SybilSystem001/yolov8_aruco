# yolov8_aruco
Ros2 jazzy package for yolov8 detection of objects with arUco tags, it detects and bound the items with a box and publishes its position in a 3D space if they have an arUCo tag.
``
colcon build --symlink-install --executor sequential --cmake-args
``
``
ros2 launch yolov8_obb yolov8_obb.launch.py
``
if after building an error of some executable file not appearing happens the current fix is go to the file direction and make it executable manually
