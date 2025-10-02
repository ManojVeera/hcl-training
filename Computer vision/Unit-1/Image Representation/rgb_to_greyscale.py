import cv2
import numpy as np

# Read image in RGB
img = cv2.imread("flower color.jpeg")

# Manual grayscale conversion
gray = 0.299 * img[:,:,2] + 0.587 * img[:,:,1] + 0.114 * img[:,:,0]
gray = gray.astype(np.uint8)

cv2.imshow("Grayscale", gray)
cv2.waitKey(0)
cv2.destroyAllWindows()
