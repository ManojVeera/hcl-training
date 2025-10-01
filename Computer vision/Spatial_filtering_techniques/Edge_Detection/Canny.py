import cv2
import numpy as np
from matplotlib import pyplot as plt

# Read image in grayscale
img = cv2.imread('Flower color.jpeg', 0)

edges = cv2.Canny(img, threshold1=100, threshold2=200)
plt.imshow(edges, cmap='gray'), plt.title('Canny Edge')
plt.show()
