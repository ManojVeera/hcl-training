import cv2
import numpy as np

# Read image in grayscale
img = cv2.imread("/Users/veerasenthilkumarr/Documents/Manoj/hcl-training/Computer vision/Image Representation/flower greyscale.jpeg", cv2.IMREAD_GRAYSCALE)

# Set threshold value
T = 127  

# Create an empty binary image with the same size
binary_img = np.zeros_like(img)

# Manual thresholding (without cv2.threshold)
for i in range(img.shape[0]):       # loop over rows
    for j in range(img.shape[1]):   # loop over columns
        if img[i, j] > T:
            binary_img[i, j] = 255  # White
        else:
            binary_img[i, j] = 0    # Black

# Show images
cv2.imshow("Grayscale Image", img)
cv2.imshow("Binary Image", binary_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
