import numpy as np
from scipy.ndimage import minimum_filter

# Load an image
from skimage import data
camera = data.camera()

# Apply minimum filter
filtered_camera = minimum_filter(camera, size=3)

# Display the results
import matplotlib.pyplot as plt
fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(10, 5))
ax0.imshow(camera, cmap='gray')
ax0.set_title('Original')
ax1.imshow(filtered_camera, cmap='gray')
ax1.set_title('Filtered (min)')
plt.show()
