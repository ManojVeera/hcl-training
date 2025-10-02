import cv2

# Load grayscale image
img = cv2.imread("flower color.jpeg", 0)  # 0 = grayscale

# Apply Gaussian filter (3x3 kernel, sigma=1)
gaussian_filtered = cv2.GaussianBlur(img, (3,3), sigmaX=1)

# Save the result
cv2.imwrite("gaussian_filtered.jpg", gaussian_filtered)
