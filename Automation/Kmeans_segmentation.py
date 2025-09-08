import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

image = cv2.imread("/Users/veerasenthilkumarr/Documents/Manoj/hcl-training/Automation/Monarch_In_May-2.jpg")
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

pixels = image_rgb.reshape((-1,3)).astype(np.float32)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)

k = 10  # change for detail
_, labels, centers = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

segmented = centers.astype(np.uint8)[labels.flatten()].reshape(image_rgb.shape)

plt.subplot(1,2,1); plt.imshow(image_rgb); plt.title("Original"); plt.axis("off")
plt.subplot(1,2,2); plt.imshow(segmented); plt.title(f"Segmented (k={k})"); plt.axis("off")
plt.show()
