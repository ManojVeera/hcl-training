import cv2
import numpy as np
from matplotlib import pyplot as plt

# Read image in grayscale
img = cv2.imread('Flower color.jpeg', 0)

# Define Roberts kernels
kernel_roberts_x = np.array([[1, 0],
                             [0, -1]])
kernel_roberts_y = np.array([[0, 1],
                             [-1, 0]])

robertsx = cv2.filter2D(img, -1, kernel_roberts_x)
robertsy = cv2.filter2D(img, -1, kernel_roberts_y)
roberts_combined = cv2.magnitude(robertsx.astype(float), robertsy.astype(float))

plt.subplot(1,3,1), plt.imshow(robertsx, cmap='gray'), plt.title('Roberts X')
plt.subplot(1,3,2), plt.imshow(robertsy, cmap='gray'), plt.title('Roberts Y')
plt.subplot(1,3,3), plt.imshow(roberts_combined, cmap='gray'), plt.title('Roberts Combined')
plt.show()
