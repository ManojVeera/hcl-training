import cv2
import matplotlib.pyplot as plt

# Read image in grayscale
img = cv2.imread("/Users/veerasenthilkumarr/Documents/Manoj/hcl-training/Computer vision/Unit-1/Tresholding/With shadow.jpeg", 0)

# Adaptive thresholding
adaptive_mean = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                      cv2.THRESH_BINARY, 11, 2)
adaptive_gaussian = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                          cv2.THRESH_BINARY, 11, 2)

# Save outputs
cv2.imwrite("adaptive_mean_output.png", adaptive_mean)
cv2.imwrite("adaptive_gaussian_output.png", adaptive_gaussian)

# Show side by side
plt.figure(figsize=(12,5))
plt.subplot(1,3,1), plt.imshow(img, cmap='gray'), plt.title("Original"), plt.axis("off")
plt.subplot(1,3,2), plt.imshow(adaptive_mean, cmap='gray'), plt.title("Adaptive Mean"), plt.axis("off")
plt.subplot(1,3,3), plt.imshow(adaptive_gaussian, cmap='gray'), plt.title("Adaptive Gaussian"), plt.axis("off")
plt.tight_layout()
plt.show()
