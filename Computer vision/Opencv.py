#OpenCV --> Real time computer vision library(fast and optimized)
# Best for:
# Face detection, object detection
# Edge detection, filtering
# Working with cameras/video streams
# Feature extraction (ORB, SIFT, etc.)

import cv2

img = cv2.imread("Sample.jpg") 

cv2.imshow("Image", img)
cv2.waitKey(0) #Waits until key '0' is pressed. Because OpenCV is mainly used for real time applications like CCTV
cv2.destroyAllWindows()

