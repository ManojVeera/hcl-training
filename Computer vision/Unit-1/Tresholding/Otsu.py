import cv2
import matplotlib.pyplot as plt

# Read image in grayscale
img = cv2.imread("/Users/veerasenthilkumarr/Documents/Manoj/hcl-training/Computer vision/Unit-1/Tresholding/image.png", 0)

# Apply Otsu's thresholding
thresh_val, otsu_thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Save output image
cv2.imwrite("otsu_output.png", otsu_thresh)

# Display
plt.figure(figsize=(10,4))
plt.subplot(1,2,1), plt.imshow(img, cmap='gray')
plt.title("Original"), plt.axis("off")

plt.subplot(1,2,2), plt.imshow(otsu_thresh, cmap='gray')
plt.title(f"Otsu Thresholding (t={thresh_val:.2f})"), plt.axis("off")

plt.tight_layout()
plt.show()
