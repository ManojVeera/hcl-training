import cv2
import numpy as np
from matplotlib import pyplot as plt

# Read image in grayscale
img = cv2.imread('Flower color.jpeg', 0)

# Sobel X and Y
sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)  # Gx
sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)  # Gy

# Combine X and Y
sobel_combined = cv2.magnitude(sobelx, sobely)

# Display
plt.subplot(1,3,1), plt.imshow(sobelx, cmap='gray'), plt.title('Sobel X')
plt.subplot(1,3,2), plt.imshow(sobely, cmap='gray'), plt.title('Sobel Y')
plt.subplot(1,3,3), plt.imshow(sobel_combined, cmap='gray'), plt.title('Sobel Combined')
plt.show()
