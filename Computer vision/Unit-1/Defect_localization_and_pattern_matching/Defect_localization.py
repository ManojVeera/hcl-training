import cv2
import numpy as np

# --- Defect Localization Pipeline ---

# --- 1. Preprocessing and Alignment ---
def align_images(image, template, max_features=500, good_match_percent=0.15):
    """Aligns `image` to match the `template` image."""
    # Convert images to grayscale
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

    # Detect ORB features and compute descriptors.
    orb = cv2.ORB_create(max_features)
    kp1, des1 = orb.detectAndCompute(image_gray, None)
    kp2, des2 = orb.detectAndCompute(template_gray, None)

    # Safeguard: If descriptors are None (not enough keypoints found)
    if des1 is None or des2 is None:
        print("Error: Could not detect enough features in one of the images.")
        return image

    # Match features.
    matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
    matches = matcher.match(des1, des2)   # ✅ Removed None
    matches = list(matches)               # ✅ Ensure it's a list

    # Sort matches by score
    matches.sort(key=lambda x: x.distance)

    # Remove not so good matches
    num_good_matches = int(len(matches) * good_match_percent)
    matches = matches[:num_good_matches]

    # Extract location of good matches
    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)

    for i, match in enumerate(matches):
        points1[i, :] = kp1[match.queryIdx].pt
        points2[i, :] = kp2[match.trainIdx].pt

    # Find homography
    h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)

    if h is None:
        print("Warning: Homography could not be computed. Returning original image.")
        return image

    # Use homography to warp image
    height, width, channels = template.shape
    image_aligned = cv2.warpPerspective(image, h, (width, height))

    return image_aligned

# --- Main script ---
# Load images
golden_template = cv2.imread("golden_template.png")
test_image = cv2.imread("test_image_with_defect.png")

if golden_template is None or test_image is None:
    print("Error: Could not load one of the images. Make sure the file paths are correct.")
    exit()

print("Aligning images...")
# Align the test image to the golden template
test_image_aligned = align_images(test_image, golden_template)

# --- 2. Difference Imaging and Defect Detection ---
# Convert to grayscale for difference calculation
golden_template_gray = cv2.cvtColor(golden_template, cv2.COLOR_BGR2GRAY)
test_image_aligned_gray = cv2.cvtColor(test_image_aligned, cv2.COLOR_BGR2GRAY)

# Calculate the absolute difference
diff_image = cv2.absdiff(golden_template_gray, test_image_aligned_gray)

# --- 3. Post-processing ---
# Threshold the difference image
_, thresh = cv2.threshold(diff_image, 30, 255, cv2.THRESH_BINARY)

# Dilate the thresholded image to fill in holes
kernel = np.ones((5,5), np.uint8)
dilated = cv2.dilate(thresh, kernel, iterations=2)

# --- 4. Contour Analysis ---
# Find contours of the defects
contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# --- 5. Localization and Output ---
# Draw bounding boxes around defects on the original test image
output_image = test_image.copy()
print(f"Found {len(contours)} potential defects.")
for contour in contours:
    # Filter out small contours that might be noise
    if cv2.contourArea(contour) > 100:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(output_image, (x, y), (x + w, y + h), (0, 0, 255), 2)

# Display the results
cv2.imshow("Golden Template", golden_template)
cv2.imshow("Test Image", test_image)
cv2.imshow("Difference", diff_image)
cv2.imshow("Defects Found", output_image)

cv2.waitKey(0)
cv2.destroyAllWindows()
