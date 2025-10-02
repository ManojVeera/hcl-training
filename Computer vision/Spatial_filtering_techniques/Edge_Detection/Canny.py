import cv2
import numpy as np
from matplotlib import pyplot as plt

# Read image in grayscale
img = cv2.imread('Flower color.jpeg', 0)

# Step 1: Gaussian smoothing (remove noise)
gaussian_filtered = cv2.GaussianBlur(img, (3,3), sigmaX=1)

# Step 2: Sobel Gradients
sobelx = cv2.Sobel(gaussian_filtered, cv2.CV_64F, 1, 0, ksize=3)  # Gx
sobely = cv2.Sobel(gaussian_filtered, cv2.CV_64F, 0, 1, ksize=3)  # Gy

# Step 3: Gradient Magnitude
sobel_combined = cv2.magnitude(sobelx, sobely)
sobel_combined = np.uint8(sobel_combined)  # convert for Canny input
_, sobel_thresh = cv2.threshold(sobel_combined, 100, 255, cv2.THRESH_BINARY)

edges = cv2.Canny(np.uint8(sobel_thresh), 50, 150)
# Display results
plt.figure(figsize=(12,4))
plt.subplot(1,4,1), plt.imshow(sobelx, cmap='gray'), plt.title('Sobel X')
plt.subplot(1,4,2), plt.imshow(sobely, cmap='gray'), plt.title('Sobel Y')
plt.subplot(1,4,3), plt.imshow(sobel_combined, cmap='gray'), plt.title('Sobel Combined')
plt.subplot(1,4,4), plt.imshow(edges, cmap='gray'), plt.title('Canny on Sobel')
plt.show()
