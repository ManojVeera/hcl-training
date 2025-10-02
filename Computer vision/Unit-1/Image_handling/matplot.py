import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Read image (PNG, JPG, etc.)
img = mpimg.imread("Sample.jpg")

print(img.shape)   # (height, width, channels)

# Show image
plt.imshow(img)
plt.axis("off")   # Hide axes
plt.show()

#Notes:
#Returns a NumPy array.
#Colors are in RGB.
#Very handy for plotting multiple images in subplots.