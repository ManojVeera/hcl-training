import cv2
import matplotlib.pyplot as plt
import numpy as np

# download the image
# https://www.mathworks.com/help/images/peppers.png

# Upload the image in your Files then read

img = cv2.imread('/content/peppers.png')

# Convert BGR to RGB for Matplotlib display

img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Split the image into its R, G, B channels
b, g, r = cv2.split(img)

# Create a figure with subplots
plt.figure(figsize=(12, 6))

# Original RGB image
plt.subplot(2, 2, 1)
plt.imshow(img_rgb)
plt.title('Original RGB Image')
plt.axis('off')

# Red Channel
plt.subplot(2, 2, 2)
plt.imshow(r, cmap='gray')
plt.title('Red Channel')
plt.axis('off')
