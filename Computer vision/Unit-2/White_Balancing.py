import cv2
import numpy as np
import matplotlib.pyplot as plt

def white_balance_gray_world(img):
    result = img.copy().astype(np.float32)
    avg_b = np.mean(result[:, :, 0])
    avg_g = np.mean(result[:, :, 1])
    avg_r = np.mean(result[:, :, 2])
    avg_gray = (avg_b + avg_g + avg_r) / 3
    result[:, :, 0] *= (avg_gray / avg_b)
    result[:, :, 1] *= (avg_gray / avg_g)
    result[:, :, 2] *= (avg_gray / avg_r)
    return np.clip(result, 0, 255).astype(np.uint8)

image = cv2.imread('/Users/veerasenthilkumarr/Downloads/build1.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
balanced_image = white_balance_gray_world(image)

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1); plt.imshow(image); plt.title('Original Image'); plt.axis('off')
plt.subplot(1, 2, 2); plt.imshow(balanced_image); plt.title('White Balanced Image'); plt.axis('off')
plt.show()
