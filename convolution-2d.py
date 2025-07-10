from scipy.signal import convolve2d
import numpy as np

# Create a sample 2D image (represented as a NumPy array)
image = np.array([
    [10, 20, 30, 40],
    [50, 60, 70, 80],
    [90, 100, 110, 120]
])

# Define a 2D kernel (e.g., a simple blurring kernel)
kernel = np.array([
    [1/9, 1/9, 1/9],
    [1/9, 1/9, 1/9],
    [1/9, 1/9, 1/9]
])

print(f"Image:", image)
print(image.shape)
print(f"Kernel:", kernel)
print(kernel.shape)

# Perform 2D convolution
# 'valid' mode is often used in image processing to avoid border effects.
convolved_image = convolve2d(image, kernel, mode='valid')
print("Convolved Image (valid mode):\n", convolved_image)

# 'same' mode can also be used to maintain the original image dimensions.
convolved_image_same = convolve2d(image, kernel, mode='same')
print("Convolved Image (same mode):\n", convolved_image_same)

print(image.shape)
print(kernel.shape)
print(convolved_image_same.shape)