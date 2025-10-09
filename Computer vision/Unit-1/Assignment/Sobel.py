import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

image_path = '/Users/veerasenthilkumarr/Documents/Manoj/HCL/MVTec AD/transistor/test/bent_lead/001.png'
output_dir = "out-hands-on"
os.makedirs(output_dir, exist_ok=True)

img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

blurred = cv2.GaussianBlur(img, (5, 5), 1.4)

kernel = np.array([[0, -1, 0],
                   [-1, 5, -1],
                   [0, -1, 0]])
sharpened = cv2.filter2D(blurred, -1, kernel)

alpha = 2.0
beta = 0
adjusted = cv2.convertScaleAbs(sharpened, alpha=alpha, beta=beta)

equalized = cv2.equalizeHist(adjusted)
sobelx = cv2.Sobel(equalized, cv2.CV_64F, 1, 0, ksize=3)
sobely = cv2.Sobel(equalized, cv2.CV_64F, 0, 1, ksize=3)
sobel_combined = cv2.magnitude(sobelx, sobely)
sobel_combined = np.uint8(np.clip(sobel_combined, 0, 255))

_, sobel_thresh = cv2.threshold(sobel_combined, 100, 255, cv2.THRESH_BINARY)
canny_on_sobel = cv2.Canny(sobel_thresh, 50, 150)

combined_output = np.hstack((sobel_combined, canny_on_sobel))
cv2.imwrite(os.path.join(output_dir, "sobel_and_canny_on_sobel.png"), combined_output)

plt.figure(figsize=(15,6))

plt.subplot(1,3,1)
plt.imshow(img, cmap='gray')
plt.title("Original Image")
plt.axis('off')

plt.subplot(1,3,2)
plt.imshow(sobel_combined, cmap='gray')
plt.title("Sobel Edge Detection")
plt.axis('off')

plt.subplot(1,3,3)
plt.imshow(canny_on_sobel, cmap='gray')
plt.title("Canny on Sobel Output")
plt.axis('off')

plt.tight_layout()
plt.show()

print("Saved combined output at:")
print(os.path.join(output_dir, "sobel_and_canny_on_sobel.png"))
