#More of a general image editing library (like Photoshop lite).
#Supports many formats (JPEG, PNG, BMP, GIF, TIFF, WebP, etc.).
#Reads image as a PIL Image object (RGB format).
#Best for:
#Opening/saving in different formats
#Adding text, drawing shapes
#Cropping, rotating
#Converting formats
from PIL import Image

img = Image.open("Sample.jpg")

# Resize
resized = img.resize((300, 200))

# Convert to grayscale
gray = img.convert("L")

# Save in different format
gray.save("output.png")
