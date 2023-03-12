import cv2
import numpy as np
from matplotlib import pyplot as plt

# membaca gambar
img = cv2.imread('image/original.jpg', 0)

# melakukan FFT pada citra
f = np.fft.fft2(img)
fshift = np.fft.fftshift(f)

# Filter Ideal Lowpass
rows, cols = img.shape
crow, ccol = rows // 2, cols // 2

# membuat mask dengan nilai nol di tengah dan nilai satu di bagian lain
mask = np.zeros((rows, cols), np.uint8)
r = 80  # radius lingkaran
cv2.circle(mask, (ccol, crow), r, 1, -1)

# menerapkan mask dan melakukan IFFT
fshift_ILP = fshift * mask
f_ILP = np.fft.ifftshift(fshift_ILP)
img_ILP = np.real(np.fft.ifft2(f_ILP))

# Filter Butterworth Lowpass
D0 = 80  # frekuensi cutoff
n = 2  # orde filter

# membuat filter Butterworth dengan frekuensi potong D0 dan orde n
butterworth_lp = np.zeros((rows, cols))
for i in range(rows):
    for j in range(cols):
        distance = np.sqrt((i - crow) ** 2 + (j - ccol) ** 2)
        butterworth_lp[i, j] = 1 / (1 + (distance / D0) ** (2 * n))

# menerapkan filter dan melakukan IFFT
fshift_BLP = fshift * butterworth_lp
f_BLP = np.fft.ifftshift(fshift_BLP)
img_BLP = np.real(np.fft.ifft2(f_BLP))

# tampilkan gambar
plt.subplot(2, 2, 1), plt.imshow(img, cmap='gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(2, 2, 2), plt.imshow(np.log(1 + np.abs(fshift)), cmap='gray')
plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
plt.subplot(2, 2, 3), plt.imshow(img_ILP, cmap='gray')
plt.title('Ideal Lowpass Filter'), plt.xticks([]), plt.yticks([])
plt.subplot(2, 2, 4), plt.imshow(img_BLP, cmap='gray')
plt.title('Butterworth Lowpass Filter'), plt.xticks([]), plt.yticks([])
plt.show()