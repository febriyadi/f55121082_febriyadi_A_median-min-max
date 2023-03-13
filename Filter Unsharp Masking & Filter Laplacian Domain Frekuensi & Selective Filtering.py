import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('image/unsharp.png', 0)

gaussian = cv2.GaussianBlur(img, (5, 5), 0)
unsharp_img = cv2.addWeighted(img, 1.5, gaussian, -0.5, 0)

rows, cols = img.shape
crow, ccol = rows // 2, cols // 2
f_img = np.fft.fft2(img)
f_shift = np.fft.fftshift(f_img)
laplacian_kernel = np.zeros((rows, cols), np.float32)
laplacian_kernel[crow - 1: crow + 2, ccol - 1: ccol + 2] = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], np.float32)
f_laplacian = f_shift * laplacian_kernel
f_laplacian_img = np.fft.ifftshift(f_laplacian)
laplacian_img = np.fft.ifft2(f_laplacian_img)
laplacian_img = np.abs(laplacian_img)

f_shift = np.fft.fftshift(f_img)
H = np.zeros((rows, cols), np.float32)
D = np.zeros((rows, cols), np.float32)
for i in range(rows):
    for j in range(cols):
        D[i, j] = np.sqrt((i - crow) ** 2 + (j - ccol) ** 2)
        if D[i, j] <= 30:
            H[i, j] = 1
        elif D[i, j] > 30 and D[i, j] <= 80:
            H[i, j] = 0.5 * (1 + np.cos(np.pi * (D[i, j] - 30) / 50))
        else:
            H[i, j] = 0
f_filtered = f_shift * H
f_filtered_img = np.fft.ifftshift(f_filtered)
filtered_img = np.fft.ifft2(f_filtered_img)
filtered_img = np.abs(filtered_img)

# Tampilkan hasil
plt.subplot(2, 2, 1), plt.imshow(img, cmap='gray'), plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(2, 2, 2), plt.imshow(unsharp_img, cmap='gray'), plt.title('Unsharp Masking')
plt.xticks([]), plt.yticks([])
plt.subplot(2, 2, 3), plt.imshow(laplacian_img, cmap='gray'), plt.title('Filter Laplacian di Domain Frekuensi')
plt.xticks([]), plt.yticks([])
plt.subplot(2, 2, 4), plt.imshow(filtered_img, cmap='gray'), plt.title('Selective Filtering')
plt.xticks([]), plt.yticks([])
plt.show()