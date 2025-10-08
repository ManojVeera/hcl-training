import cv2
import numpy as np

# --- SIFT Object Recognition ---
# Goal: Find the 'object.jpg' inside the 'scene.jpg'

# Load the images
object_img = cv2.imread('object.jpeg', cv2.IMREAD_GRAYSCALE)
scene_img = cv2.imread('scene.jpg', cv2.IMREAD_GRAYSCALE)

if object_img is None or scene_img is None:
    print("Error: Could not read one of the images.")
    exit()

# 1. Initialize the SIFT detector
sift = cv2.SIFT_create()

# 2. Find keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(object_img, None)
kp2, des2 = sift.detectAndCompute(scene_img, None)

# 3. Match descriptors using a Brute-Force Matcher
bf = cv2.BFMatcher()
matches = bf.knnMatch(des1, des2, k=2)

# 4. Apply Lowe's ratio test
good_matches = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good_matches.append(m)

# 5. Find the object if enough good matches are found
MIN_MATCH_COUNT = 10
if len(good_matches) > MIN_MATCH_COUNT:
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    matchesMask = mask.ravel().tolist()

    h, w = object_img.shape
    pts = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)
    dst = cv2.perspectiveTransform(pts, M)

    scene_img_color = cv2.cvtColor(scene_img, cv2.COLOR_GRAY2BGR)
    scene_img_with_box = cv2.polylines(scene_img_color, [np.int32(dst)], True, (0, 255, 0), 3, cv2.LINE_AA)

    print("Object Found!")
else:
    print(f"Not enough matches are found - {len(good_matches)}/{MIN_MATCH_COUNT}")
    matchesMask = None
    scene_img_with_box = cv2.cvtColor(scene_img, cv2.COLOR_GRAY2BGR)

# Draw the matches
draw_params = dict(matchColor=(0, 255, 0),
                   singlePointColor=None,
                   matchesMask=matchesMask,
                   flags=2)

result_img = cv2.drawMatches(cv2.cvtColor(object_img, cv2.COLOR_GRAY2BGR),
                             kp1,
                             scene_img_with_box,
                             kp2,
                             good_matches,
                             None,
                             **draw_params)

# --- Show and Save Output ---
cv2.imshow("SIFT Object Recognition", result_img)
cv2.imwrite("sift_output.jpg", result_img)  # Save the result
cv2.waitKey(0)
cv2.destroyAllWindows()

print("Result image saved as 'sift_output.jpg'")
