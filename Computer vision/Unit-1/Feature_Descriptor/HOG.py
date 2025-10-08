import cv2
import numpy as np

# --- HOG for Pedestrian Detection ---

# Load the image
# Make sure you have an image named 'Pedestrian.png' in the same folder
image = cv2.imread('Pedestrian.png') 
if image is None:
    print("Error: Could not read Pedestrian.png.")
    exit()

# It's good practice to work on a copy if you want to keep the original
output_image = image.copy()

# Resize for faster processing and better detection (optional)
output_image = cv2.resize(output_image, (640, 480))

# 1. Initialize the HOG descriptor/person detector
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# 2. Detect people in the image
(rects, weights) = hog.detectMultiScale(output_image, winStride=(4, 4), padding=(8, 8), scale=1.05)

# 3. Draw the bounding boxes for the detected people
for (x, y, w, h) in rects:
    # We need the coordinates for the resized image, let's draw on it
    # The rects are in the format (x, y, width, height)
    end_x = x + w
    end_y = y + h
    cv2.rectangle(output_image, (x, y), (end_x, end_y), (0, 255, 0), 2)

print(f"Found {len(rects)} people.")

# Show the result
cv2.imshow("HOG Pedestrian Detection", output_image)

# --- SAVE THE OUTPUT IMAGE ---
# This line will save the image with the green boxes to a file
cv2.imwrite("hog_detection_output.jpg", output_image)
print("Output image saved as hog_detection_output.jpg")

cv2.waitKey(0)
cv2.destroyAllWindows()