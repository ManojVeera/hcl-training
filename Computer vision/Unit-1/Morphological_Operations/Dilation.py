import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read image in grayscale
img = cv2.imread("flower color.jpeg", 0)  

# Convert to binary (thresholding)
_, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

# Structuring element (kernel)
kernel = np.ones((5,5), np.uint8)

# Dilation
dilated = cv2.dilate(binary, kernel, iterations=1)

# Show result
plt.subplot(1,2,1), plt.imshow(binary, cmap='gray'), plt.title("Original")
plt.subplot(1,2,2), plt.imshow(dilated, cmap='gray'), plt.title("Dilated")
plt.show()
