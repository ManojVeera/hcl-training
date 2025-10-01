import cv2
import numpy as np
from matplotlib import pyplot as plt

# Read image in grayscale
img = cv2.imread('Flower color.jpeg', 0)
# Define Prewitt kernels
kernelx = np.array([[ -1, 0, 1],
                    [ -1, 0, 1],
                    [ -1, 0, 1]])
kernely = np.array([[ 1, 1, 1],
                    [ 0, 0, 0],
                    [ -1, -1, -1]])

prewittx = cv2.filter2D(img, -1, kernelx)
prewitty = cv2.filter2D(img, -1, kernely)
prewitt_combined = cv2.magnitude(prewittx.astype(float), prewitty.astype(float))

plt.subplot(1,3,1), plt.imshow(prewittx, cmap='gray'), plt.title('Prewitt X')
plt.subplot(1,3,2), plt.imshow(prewitty, cmap='gray'), plt.title('Prewitt Y')
plt.subplot(1,3,3), plt.imshow(prewitt_combined, cmap='gray'), plt.title('Prewitt Combined')
plt.show()
