import cv2

# Read image in grayscale
img = cv2.imread("flower color.jpeg", cv2.IMREAD_GRAYSCALE)

# Apply threshold
# Parameters: source, threshold_value, max_value, threshold_type
_, binary_img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

# Show results
cv2.imshow("Grayscale Image", img)
cv2.imshow("Binary Image", binary_img)

cv2.waitKey(0)
cv2.destroyAllWindows()
