import cv2

# Load grayscale image
img = cv2.imread("flower color.jpeg", cv2.IMREAD_GRAYSCALE)

# Apply Laplacian filter
laplacian = cv2.Laplacian(img, cv2.CV_64F)  # Use float64 to handle negative values

# Convert back to 8-bit image
laplacian_8u = cv2.convertScaleAbs(laplacian)

# Save result
cv2.imwrite("laplacian.jpg", laplacian_8u)
