import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("flower color.jpeg", 0)
_, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

kernel = np.ones((5,5), np.uint8)

# Erosion
eroded = cv2.erode(binary, kernel, iterations=1)

plt.subplot(1,2,1), plt.imshow(binary, cmap='gray'), plt.title("Original")
plt.subplot(1,2,2), plt.imshow(eroded, cmap='gray'), plt.title("Eroded")
plt.show()
