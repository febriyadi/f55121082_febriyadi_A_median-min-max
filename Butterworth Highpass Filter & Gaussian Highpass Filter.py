import cv2
import numpy as np
from matplotlib import pyplot as plt

# membaca gambar
img = cv2.imread('image/max.jpg',0)
rows, cols = img.shape

# membuat filter Butterworth Highpass
# membuat filter highpass Butterworth 2D dengan jari-jari 30 dan orde 2
butterworth_hp = np.zeros((rows, cols))
D = 30
n = 2
for i in range(rows):
    for j in range(cols):
        butterworth_hp[i, j] = 1 / (1 + (D / np.sqrt((i - rows/2)**2 + (j - cols/2)**2))**(2*n))

# terapkan filter
butterworth_hp_img = np.fft.fftshift(np.fft.fft2(img)) * butterworth_hp
butterworth_hp_img = np.real(np.fft.ifft2(np.fft.ifftshift(butterworth_hp_img)))

# membuat filter Gaussian Highpass
# membuat 2D filter Gaussian highpass  with standard dengan standar deviasi 5
gaussian_hp = np.zeros((rows, cols))
sigma = 5
for i in range(rows):
    for j in range(cols):
        gaussian_hp[i, j] = 1 - np.exp(-((i - rows/2)**2 + (j - cols/2)**2) / (2 * sigma**2))

# terapkan filter
gaussian_hp_img = np.fft.fftshift(np.fft.fft2(img)) * gaussian_hp
gaussian_hp_img = np.real(np.fft.ifft2(np.fft.ifftshift(gaussian_hp_img)))

# tampilkan gambar
plt.subplot(2,2,1),plt.imshow(img, cmap = 'gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,2),plt.imshow(butterworth_hp_img, cmap = 'gray')
plt.title('Butterworth Highpass Filter'), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,3),plt.imshow(gaussian_hp_img, cmap = 'gray')
plt.title('Gaussian Highpass Filter'), plt.xticks([]), plt.yticks([])
plt.show()