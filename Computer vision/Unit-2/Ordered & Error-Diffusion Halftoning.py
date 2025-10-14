# halftoning_methods.py
import cv2
import numpy as np

# ---- Ordered Dithering (Bayer matrix) ----
def ordered_halftone(image_gray, n=4):
    bayer = np.array([[0, 8, 2, 10],
                      [12, 4, 14, 6],
                      [3, 11, 1, 9],
                      [15, 7, 13, 5]]) / 16.0
    h, w = image_gray.shape
    tile = np.tile(bayer, (h // n + 1, w // n + 1))
    tile = tile[:h, :w]
    norm = image_gray / 255.0
    halftoned = (norm > tile).astype(np.uint8) * 255
    return halftoned

# ---- Error Diffusion (Floyd–Steinberg) ----
def floyd_steinberg_halftone(image_gray):
    img = image_gray.astype(np.float32) / 255.0
    h, w = img.shape
    for y in range(h - 1):
        for x in range(1, w - 1):
            old = img[y, x]
            new = np.round(old)
            img[y, x] = new
            error = old - new
            img[y, x+1] += error * 7/16
            img[y+1, x-1] += error * 3/16
            img[y+1, x] += error * 5/16
            img[y+1, x+1] += error * 1/16
    return (img * 255).clip(0, 255).astype(np.uint8)

# ---- Example usage ----
if __name__ == "__main__":
    gray = cv2.imread("industrial_sample.jpg", cv2.IMREAD_GRAYSCALE)
    ordered = ordered_halftone(gray)
    fs = floyd_steinberg_halftone(gray)

    cv2.imshow("Original Gray", gray)
    cv2.imshow("Ordered Halftone", ordered)
    cv2.imshow("Floyd-Steinberg Halftone", fs)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
