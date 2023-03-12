import cv2
import numpy as np
from matplotlib import pyplot as plt

# membaca gambar
img = cv2.imread('image/unsharp.png',0)
rows, cols = img.shape

# Filter Unsharp Masking
# terapkan blur Gaussian dengan ukuran kernel 5x5
blur = cv2.GaussianBlur(img, (5,5), 0)
# membuat unsharp mask dengan mengurangi gambar yang sudah diberi blur dari gambar asli
unsharp_mask = cv2.addWeighted(img, 1.5, blur, -0.5, 0)

# Filter Laplacian di Domain Frekuensi
# terapkan Fourier Transform pada gambar
f = np.fft.fft2(img)
# geser komponen frekuensi nol ke pusat spektrum
fshift = np.fft.fftshift(f)
# membuat filter Laplacian di domain frekuensi
laplacian_filter = np.zeros((rows, cols), np.float32)
laplacian_filter[int(rows/2)-1:int(rows/2)+2, int(cols/2)-1:int(cols/2)+2] = np.array([[0,-1,0],[-1,4,-1],[0,-1,0]])
# terapkan filter
laplacian_freq = fshift * laplacian_filter
# terapkan Inverse Fourier Transform
laplacian_img = np.real(np.fft.ifft2(np.fft.ifftshift(laplacian_freq)))

# Selective Filtering
# membuat filter Gaussian dengan deviasi standar 50
gaussian_filter = np.zeros((rows, cols), np.float32)
for i in range(rows):
    for j in range(cols):
        gaussian_filter[i, j] = np.exp(-((i - rows/2)**2 + (j - cols/2)**2) / (2 * 50**2))
# terapkan filter pada gambar asli dan unsharp mask
selective_filtered_img = gaussian_filter * img + (1 - gaussian_filter) * unsharp_mask
selective_filtered_mask = gaussian_filter * unsharp_mask + (1 - gaussian_filter) * img

# tampilkan gambar
plt.subplot(2,3,1),plt.imshow(img, cmap = 'gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(2,3,2),plt.imshow(unsharp_mask, cmap = 'gray')
plt.title('Unsharp Masking Filter'), plt.xticks([]), plt.yticks([])
plt.subplot(2,3,3),plt.imshow(laplacian_img, cmap = 'gray')
plt.title('Laplacian Domain Frequency Filter'), plt.xticks([]), plt.yticks([])
plt.subplot(2,3,5),plt.imshow(selective_filtered_img, cmap = 'gray')
plt.title('Selective Filtering (Original Image)'), plt.xticks([]), plt.yticks([])
plt.subplot(2,3,6),plt.imshow(selective_filtered_mask, cmap = 'gray')
plt.title('Selective Filtering (Unsharp Mask)'), plt.xticks([]), plt.yticks([])
plt.show()