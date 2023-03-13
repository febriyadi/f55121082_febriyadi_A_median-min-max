# FEBRIYADI
# F55121082

import cv2
import numpy as np
from matplotlib import pyplot as plt

# Load image in grayscale mode
img = cv2.imread('image/max.jpg', cv2.IMREAD_GRAYSCALE)

# Butterworth  highpass filter
def butterworth_highpass_filter(img, cutoff, order):
    M, N = img.shape
    center = (M//2, N//2)
    u, v = np.meshgrid(np.arange(N)-center[1], np.arange(M)-center[0])
    D = np.sqrt(u**2 + v**2)
    H = 1 / (1 + (cutoff / D)**(2*order))
    F = np.fft.fft2(img)
    Fshift = np.fft.fftshift(F)
    Gshift = H * Fshift
    G = np.fft.ifftshift(Gshift)
    g = np.fft.ifft2(G).real
    return g

# Gaussian highpass filter
def gaussian_highpass_filter(img, cutoff):
    M, N = img.shape
    center = (M//2, N//2)
    u, v = np.meshgrid(np.arange(N)-center[1], np.arange(M)-center[0])
    D = np.sqrt(u**2 + v**2)
    H = 1 - np.exp(-0.5*(D/cutoff)**2)
    F = np.fft.fft2(img)
    Fshift = np.fft.fftshift(F)
    Gshift = H * Fshift
    G = np.fft.ifftshift(Gshift)
    g = np.fft.ifft2(G).real
    return g

# Apply Butterworth highpass filter
cutoff = 50
order = 4
img_butterworth = butterworth_highpass_filter(img, cutoff, order)

# Apply Gaussian highpass filter
cutoff = 50
img_gaussian = gaussian_highpass_filter(img, cutoff)

# Display the original and filtered images
plt.subplot(1, 3, 1)
plt.imshow(img, cmap='gray')
plt.title('Original Image')

plt.subplot(1, 3, 2)
plt.imshow(img_butterworth, cmap='gray')
plt.title('Butterworth Highpass Filtered Image')

plt.subplot(1, 3, 3)
plt.imshow(img_gaussian, cmap='gray')
plt.title('Gaussian Highpass Filtered Image')

plt.show()