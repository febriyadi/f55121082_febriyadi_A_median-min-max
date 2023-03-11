import cv2
import numpy as np

# Load the image
img = cv2.imread('image/max.jpg')

# Define the kernel size for maximum filtering
kernel_size = 5

# Define the kernel for maximum filtering
kernel = np.ones((kernel_size,kernel_size),np.uint8)

# Apply the maximum filter to the image
filtered_img = cv2.dilate(img, kernel, iterations = 1)

# Display the original and filtered images
cv2.imshow('Original Image',img)
cv2.imshow('Filtered Image',filtered_img)

cv2.waitKey(0)
cv2.destroyAllWindows()
