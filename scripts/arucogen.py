import cv2
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
for i in range(3):
    marker = cv2.aruco.generateImageMarker(aruco_dict, i, 200)
    cv2.imwrite(f'test_marker_{i}.png', marker)
