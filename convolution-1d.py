import numpy as np

# Define two 1D arrays
signal = np.array([1, 2, 3, 4])
kernel = np.array([0.5, 1, 0.5])

print(f"Signal:", signal)
print(signal.shape)
print(f"Kernel:", kernel)
print(kernel.shape)

# Perform convolution with 'full' mode (default)
# The output size is len(signal) + len(kernel) - 1
convolution_full = np.convolve(signal, kernel, mode='full')
print("Full convolution:", convolution_full)
print(convolution_full.shape)

# Perform convolution with 'valid' mode
# The output only includes positions where the kernel fully overlaps with the signal
convolution_valid = np.convolve(signal, kernel, mode='valid')
print("Valid convolution:", convolution_valid)

# Perform convolution with 'same' mode
# The output size is the same as the first input array (signal)
convolution_same = np.convolve(signal, kernel, mode='same')
print("Same convolution:", convolution_same)
