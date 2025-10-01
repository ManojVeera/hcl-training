import cv2
import numpy as np

# Load grayscale image
img = cv2.imread("flower color.jpeg", cv2.IMREAD_GRAYSCALE)

# Apply Gaussian blur (low-pass)
low_pass = cv2.GaussianBlur(img, (5,5), sigmaX=1.0)

# High-pass = Original - Low-pass
high_pass = cv2.subtract(img, low_pass) 

cv2.imwrite("high_pass_subtract.jpg", high_pass)

sharpened = cv2.add(img, high_pass)
cv2.imwrite("sharpened.jpg", sharpened)