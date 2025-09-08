from skimage import io, transform, img_as_ubyte

# Read
img = io.imread("Sample.jpg")
print(img.shape)

# Resize
resized = transform.resize(img, (225, 225))

# Convert float → uint8
resized_uint8 = img_as_ubyte(resized)

# Save
io.imsave("output.jpg", resized_uint8)
print("Saved successfully!")

#Notes:
#Returns float arrays in range [0, 1].
#Has powerful utilities for filters, segmentation, morphology.