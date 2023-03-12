# Import Libary : NumPy and OpenCV.
import cv2
import numpy as np

# Muat gambar dan ubah menjadi skala abu-abu.
img = cv2.imread('image/FFT.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Terapkan FFT 2D ke gambar skala abu-abu menggunakan fungsi np.fft.fft2().
f = np.fft.fft2(gray)

# Geser komponen frekuensi nol ke tengah spektrum menggunakan np.fft.fftshift().
fshift = np.fft.fftshift(f)

# Hitung spektrum magnitudo menggunakan nilai absolut dari nilai kompleks.
magnitude_spectrum = 20*np.log(np.abs(fshift))

# Terapkan filter yang diinginkan ke spektrum besarnya. Misalnya, filter high-pass dapat diterapkan dengan menyetel komponen frekuensi rendah ke nol.
rows, cols = gray.shape
crow, ccol = rows/2 , cols/2
fshift[int(crow-30):int(crow+30), int(ccol-30):int(ccol+30)] = 0

#Geser komponen frekuensi nol kembali ke pojok kiri atas spektrum menggunakan np.fft.ifftshift().
f_ishift = np.fft.ifftshift(fshift)

# Terapkan FFT 2D terbalik menggunakan fungsi np.fft.ifft2().
img_back = np.fft.ifft2(f_ishift)

# Ubah nilai kompleks menjadi besaran dan tampilkan hasilnya.
img_back = np.abs(img_back)
cv2.imshow('Original Image', gray)
cv2.imshow('Magnitude Spectrum', magnitude_spectrum)
cv2.imshow('Filtered Image', img_back)
cv2.waitKey(0)
cv2.destroyAllWindows()
