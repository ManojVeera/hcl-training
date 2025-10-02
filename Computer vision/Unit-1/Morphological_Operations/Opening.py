import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("flower color.jpeg", 0)
_, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

kernel = np.ones((5,5), np.uint8)

# Opening
opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

plt.subplot(1,2,1), plt.imshow(binary, cmap='gray'), plt.title("Original")
plt.subplot(1,2,2), plt.imshow(opening, cmap='gray'), plt.title("Opening")
plt.show()
