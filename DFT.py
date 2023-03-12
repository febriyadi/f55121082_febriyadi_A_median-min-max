import cv2
import numpy as np
from matplotlib import pyplot as plt

# Baca gambar input
img = cv2.imread('image/DFT.jpg', cv2.IMREAD_GRAYSCALE)

# Hitung DFT gambar input
dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)

# Geser DFT ke tengah
dft_shift = np.fft.fftshift(dft)

# Hitung spektrum besarnya
magnitude_spectrum = 20 * np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))

# Tampilkan gambar input dan spektrum besarnya
plt.subplot(121), plt.imshow(img, cmap='gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(magnitude_spectrum, cmap='gray')
plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
plt.show()
