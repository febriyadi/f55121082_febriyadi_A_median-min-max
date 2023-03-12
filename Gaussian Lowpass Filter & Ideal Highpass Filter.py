import cv2
import numpy as np
from matplotlib import pyplot as plt

# Baca gambar masukan
img = cv2.imread('image/max.jpg', cv2.IMREAD_GRAYSCALE)

# Hitung DFT gambar input
dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)

# Geser DFT ke tengah
dft_shift = np.fft.fftshift(dft)

#Buat topeng untuk Gaussian lowpass filter
rows, cols = img.shape
crow, ccol = rows//2, cols//2
D0_lp = 50  # Atur frekuensi cutoff untuk filter lowpass di sini
mask_lp = np.zeros((rows, cols, 2), np.uint8)
for i in range(rows):
    for j in range(cols):
        dist = np.sqrt((i-crow)**2 + (j-ccol)**2)
        mask_lp[i, j] = np.exp(-(dist**2) / (2*(D0_lp**2)))

# Terapkan topeng ke DFT untuk filter lowpass
dft_filtered_lp = dft_shift * mask_lp

# Buat topeng untuk filter highpass Ideal
D0_hp = 50  # Atur frekuensi cutoff untuk filter highpass di sini
mask_hp = np.ones((rows, cols, 2), np.uint8)
for i in range(rows):
    for j in range(cols):
        dist = np.sqrt((i-crow)**2 + (j-ccol)**2)
        if dist < D0_hp:
            mask_hp[i, j] = 0

# Terapkan topeng ke DFT untuk filter highpass
dft_filtered_hp = dft_shift * mask_hp

# Geser DFT kembali ke asal untuk filter lowpass
dft_filtered_shift_lp = np.fft.ifftshift(dft_filtered_lp)

# Geser DFT kembali ke asal untuk filter highpass
dft_filtered_shift_hp = np.fft.ifftshift(dft_filtered_hp)

# Balikkan DFT untuk mendapatkan gambar yang difilter untuk filter lowpass
img_filtered_lp = cv2.idft(dft_filtered_shift_lp)
img_filtered_lp = cv2.magnitude(img_filtered_lp[:, :, 0], img_filtered_lp[:, :, 1])

# Balikkan DFT untuk mendapatkan gambar yang difilter untuk filter highpass
img_filtered_hp = cv2.idft(dft_filtered_shift_hp)
img_filtered_hp = cv2.magnitude(img_filtered_hp[:, :, 0], img_filtered_hp[:, :, 1])

# Tampilkan gambar yang difilter
plt.subplot(131), plt.imshow(img, cmap='gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(132), plt.imshow(img_filtered_lp, cmap='gray')
plt.title('Gaussian Lowpass Filtered Image'), plt.xticks([]), plt.yticks([])
plt.subplot(133), plt.imshow(img_filtered_hp, cmap='gray')
plt.title('Ideal Highpass Filtered Image'), plt.xticks([]), plt.yticks([])
plt.show()
