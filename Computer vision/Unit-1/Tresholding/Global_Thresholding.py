import cv2
import matplotlib.pyplot as plt
import numpy as np

img = cv2.imread("image.png", 0)   # grayscale

#_, global_thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

def global_threshold(image, thresh=127, max_val=255):
    # Create a copy for output
    result = np.zeros_like(image)
    rows, cols = image.shape
    
    for i in range(rows):
        for j in range(cols):
            if image[i, j] > thresh:
                result[i, j] = max_val
            else:
                result[i, j] = 0
    return result

# Apply manual threshold
manual_thresh = global_threshold(img, 127, 255)
plt.subplot(1,2,1), plt.imshow(img, cmap='gray'), plt.title("Original")
#plt.subplot(1,2,2), plt.imshow(global_thresh, cmap='gray'), plt.title("Global Thresholding")
plt.subplot(1,2,2), plt.imshow(manual_thresh, cmap='gray'), plt.title("Global Thresholding")
plt.show()

img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
thresh_bgr = cv2.cvtColor(manual_thresh, cv2.COLOR_GRAY2BGR)

side_by_side = cv2.hconcat([img_bgr, thresh_bgr])

cv2.imwrite("global_threshold_result.png", side_by_side)

print("✅ Output saved as 'global_threshold_result.png'")
