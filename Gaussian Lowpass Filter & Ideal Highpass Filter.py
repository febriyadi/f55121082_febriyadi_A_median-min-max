#Import semua library yang dibutuhkan
import cv2
import numpy as np
from matplotlib import pyplot as plt

# Membaca citra
img = cv2.imread('image/max.jpg', 0)

# Menerapkan transformasi Fourier pada citra
f = np.fft.fft2(img)
fshift = np.fft.fftshift(f)

# Menentukan ukuran filter
rows, cols = img.shape
crow, ccol = int(rows/2), int(cols/2)

# Membuat filter Gaussian lowpass
mask_gaussian = np.zeros((rows, cols), np.uint8)
r = 50
d = 2*r
for i in range(rows):
    for j in range(cols):
        distance = np.sqrt((i - crow)**2 + (j - ccol)**2)
        mask_gaussian[i, j] = np.exp(-distance**2 / (2*d**2))

# Menerapkan filter Gaussian lowpass pada spektrum frekuensi
fshift_filtered_gaussian = fshift * mask_gaussian

# Menerapkan inverse transformasi Fourier pada spektrum frekuensi yang difilter
img_filtered_gaussian = np.fft.ifft2(np.fft.ifftshift(fshift_filtered_gaussian))

# Membuat filter Ideal highpass
mask_ideal = np.ones((rows, cols), np.uint8)
r = 50
mask_ideal[crow-r:crow+r, ccol-r:ccol+r] = 0

# Menerapkan filter Ideal highpass pada spektrum frekuensi
fshift_filtered_ideal = fshift * mask_ideal

# Menerapkan inverse transformasi Fourier pada spektrum frekuensi yang difilter
img_filtered_ideal = np.fft.ifft2(np.fft.ifftshift(fshift_filtered_ideal))

# Menampilkan citra asli dan citra yang telah difilter
plt.subplot(231),plt.imshow(img, cmap = 'gray')
plt.title('Original'), plt.xticks([]), plt.yticks([])
plt.subplot(232),plt.imshow(np.abs(img_filtered_gaussian), cmap = 'gray')
plt.title('Gaussian Lowpass Filter'), plt.xticks([]), plt.yticks([])
plt.subplot(233),plt.imshow(np.abs(img_filtered_ideal), cmap = 'gray')
plt.title('Ideal Highpass Filter'), plt.xticks([]), plt.yticks([])
plt.show()