import cv2
import numpy as np
import matplotlib.pyplot as plt

img_path = '/Users/veerasenthilkumarr/Documents/Manoj/HCL/MVTec AD/leather/test/cut/003.png'
img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

edges = cv2.Canny(img, 50, 150)

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
closing = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel)

contours, _ = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
filled_mask = np.zeros_like(opening)
if contours:
    largest_contour = max(contours, key=cv2.contourArea)
    cv2.drawContours(filled_mask, [largest_contour], -1, (255), thickness=cv2.FILLED)

plt.figure(figsize=(12, 8))
plt.subplot(2, 2, 1)
plt.imshow(img, cmap='gray')
plt.title("Original Image")
plt.axis('off')

plt.subplot(2, 2, 2)
plt.imshow(edges, cmap='gray')
plt.title("Canny Edge Detection")
plt.axis('off')

plt.subplot(2, 2, 3)
plt.imshow(closing, cmap='gray')
plt.title("After Morphological Closing")
plt.axis('off')

plt.subplot(2, 2, 4)
plt.imshow(filled_mask, cmap='gray')
plt.title("Final Cleaned & Filled Mask")
plt.axis('off')

plt.tight_layout()
plt.show()
