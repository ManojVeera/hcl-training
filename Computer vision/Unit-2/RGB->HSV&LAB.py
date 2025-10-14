# rgb_hsv_lab_segmentation.py
import cv2
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score

# ---- Step 1: Convert RGB to HSV and LAB ----
def convert_rgb_to_hsv_lab(image_path):
    img_bgr = cv2.imread(image_path)
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    return img_bgr, hsv, lab

# ---- Step 2: Segment image (K-means) ----
def segment_image(image, k=3):
    Z = image.reshape((-1, 3))
    Z = np.float32(Z)
    kmeans = KMeans(n_clusters=k, n_init=10)
    labels = kmeans.fit_predict(Z)
    segmented = kmeans.cluster_centers_[labels].reshape(image.shape)
    segmented = np.uint8(segmented)
    return segmented, labels, kmeans.inertia_

# ---- Step 3: Assess segmentation ----
def assess_segmentation(image, labels):
    try:
        sil_score = silhouette_score(image.reshape(-1, 3), labels)
        db_score = davies_bouldin_score(image.reshape(-1, 3), labels)
        return sil_score, db_score
    except:
        return None, None

# ---- Example usage ----
if __name__ == "__main__":
    img_bgr, hsv, lab = convert_rgb_to_hsv_lab("sample_rgb.jpg")

    seg_lab, labels, inertia = segment_image(lab, k=3)
    sil, db = assess_segmentation(lab, labels)

    print(f"Inertia: {inertia:.2f}, Silhouette: {sil}, Davies-Bouldin: {db}")
    cv2.imshow("Original", img_bgr)
    cv2.imshow("LAB Segmented", seg_lab)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
