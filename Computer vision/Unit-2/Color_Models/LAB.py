import cv2
import numpy as np
from skimage import color
import matplotlib.pyplot as plt

image = cv2.imread('/Users/veerasenthilkumarr/Downloads/peppers.png')
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
lab_image = color.rgb2lab(image_rgb)

l_channel = lab_image[:,:,0]
a_channel = lab_image[:,:,1]
b_channel = lab_image[:,:,2]

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(l_channel, cmap='gray'); axes[0].set_title('L* Channel (Lightness)'); axes[0].axis('off')
axes[1].imshow(a_channel, cmap='gray'); axes[1].set_title('a* Channel (Green-Red)'); axes[1].axis('off')
axes[2].imshow(b_channel, cmap='gray'); axes[2].set_title('b* Channel (Blue-Yellow)'); axes[2].axis('off')
plt.tight_layout()
plt.show()
