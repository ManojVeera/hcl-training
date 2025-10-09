import cv2
import numpy as np

# --- ORB Real-Time Feature Matching ---
# Goal: Match features from 'template.jpg' to the live webcam feed

# Load the template image
template = cv2.imread('hcl1.jpeg.jpg', cv2.IMREAD_GRAYSCALE)
if template is None:
    print("Error: Could not read template.jpg.")
    exit()

# 1. Initialize the ORB detector
# We can specify the number of features to find, e.g., 1000
orb = cv2.ORB_create(nfeatures=1000)

# 2. Find keypoints and descriptors for the template image (only once)
kp1, des1 = orb.detectAndCompute(template, None)

# 3. Initialize Brute-Force Matcher with Hamming distance
# NORM_HAMMING is used for binary descriptors like ORB
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# Start webcam capture
cap = cv2.VideoCapture(0)

print("Starting real-time matching... Press 'q' to quit.")

while True:
    # Read a frame from the webcam
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Find keypoints and descriptors for the current frame
    kp2, des2 = orb.detectAndCompute(frame_gray, None)

    # Match descriptors if the frame has any
    if des2 is not None:
        matches = bf.match(des1, des2)
        
        # Sort them in the order of their distance (best matches first)
        matches = sorted(matches, key = lambda x:x.distance)
        
        # Draw the top 15 matches
        # This connects the matching points between the template and the live frame
        result_img = cv2.drawMatches(template, kp1, frame, kp2, matches[:15], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        
        cv2.imshow('ORB Real-Time Matching', result_img)

    else:
        # If no keypoints are found in the frame, just show the frame
        cv2.imshow('ORB Real-Time Matching', frame)

    # Quit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()