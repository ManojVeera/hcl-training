import cv2
import numpy as np
import matplotlib.pyplot as plt

# Step 1: Read the image
img = cv2.imread('Coins.jpg')  # replace with your image path
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (11,11), 0)
# Step 2: Apply thresholding (Otsu's method works well for coins)
_, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# Step 3: Perform Connected Components Analysis
# connectivity=8 for 8-connectivity
num_labels, labels = cv2.connectedComponents(binary, connectivity=8)

print("Number of connected components (including background):", num_labels)
print("Number of coins detected:", num_labels - 1)  # exclude background

# Step 4: Visualize each component with random colors
label_hue = np.uint8(179 * labels / np.max(labels))
blank_ch = 255 * np.ones_like(label_hue)
colored_label = cv2.merge([label_hue, blank_ch, blank_ch])
colored_label = cv2.cvtColor(colored_label, cv2.COLOR_HSV2BGR)
colored_label[label_hue == 0] = 0  # background set to black

plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.title("Original")
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.subplot(1,2,2)
plt.title("Connected Components")
plt.imshow(colored_label)
plt.axis('off')
plt.show()
