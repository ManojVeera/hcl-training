import cv2

# Load grayscale image
img = cv2.imread("flower color.jpeg", 0)  # 0 = grayscale

# Apply mean filter (3x3 kernel)
mean_filtered = cv2.blur(img, (3,3))  # or (5,5) for larger kernel

# Save the result
cv2.imwrite("mean_filtered.jpg", mean_filtered)

