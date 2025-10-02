# Example of binary image
#Application of OCR using pytesseract
import cv2
import pytesseract

img = cv2.imread("text.jpg", cv2.IMREAD_GRAYSCALE)

_, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

text = pytesseract.image_to_string(binary)

print("Recognized Text:", text)
