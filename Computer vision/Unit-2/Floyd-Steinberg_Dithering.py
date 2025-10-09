import numpy as np
from PIL import Image, ImageFile
import matplotlib.pyplot as plt
import cv2

Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True

GREYSCALE = False
img_name = '/content/Girl_Floyd.jpg'
img = Image.open(img_name)
if GREYSCALE: img = img.convert('L')

width, height = img.size
new_width = 400
new_height = int(height * new_width / width)
img = img.resize((new_width, new_height))

def get_new_val(old_val, nc):
    return np.round(old_val * (nc - 1)) / (nc - 1)

def fs_dither(img, nc):
    arr = np.array(img, dtype=float)/255
    for ir in range(new_height):
        for ic in range(new_width):
            old_val = arr[ir, ic].copy()
            new_val = get_new_val(old_val, nc)
            arr[ir, ic] = new_val
            err = old_val - new_val
            if ic < new_width - 1: arr[ir, ic+1] += err*7/16
            if ir < new_height - 1:
                if ic > 0: arr[ir+1, ic-1] += err*3/16
                arr[ir+1, ic] += err*5/16
                if ic < new_width - 1: arr[ir+1, ic+1] += err/16
    carr = np.array(arr/np.max(arr, axis=(0,1))*255, dtype=np.uint8)
    return Image.fromarray(carr)

def palette_reduce(img, nc):
    arr = np.array(img, dtype=float)/255
    arr = get_new_val(arr, nc)
    carr = np.array(arr/np.max(arr)*255, dtype=np.uint8)
    return Image.fromarray(carr)

for nc in (2,3,4):
    dim = fs_dither(img, nc)
    dim.save(f'/content/dimg-{nc}.jpg')
    rim = palette_reduce(img, nc)
    rim.save(f'/content/rimg-{nc}.jpg')

plt.figure(figsize=(12,6))
d = cv2.imread('/content/dimg-2.jpg'); d = cv2.cvtColor(d, cv2.COLOR_BGR2RGB)
plt.subplot(2,1,1); plt.imshow(d); plt.title('nc = 2 Floyd-Steinberg dithered image'); plt.axis('off')
r = cv2.imread('/content/rimg-2.jpg'); r = cv2.cvtColor(r, cv2.COLOR_BGR2RGB)
plt.subplot(2,1,2); plt.imshow(r); plt.title('Palette reduction without dithering image'); plt.axis('off')
plt.tight_layout()
plt.show()
