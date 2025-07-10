import tensorflow as tf
import numpy as np

# Define input parameters
batch_size = 1
channels = 1  # e.g., for grayscale 3D data like medical scans
depth = 10
height = 20
width = 20

# Create a sample 3D input volume
# Shape: (batch_size, depth, height, width, channels)
input_volume = np.random.rand(batch_size, depth, height, width, channels).astype(np.float32)

# Define the 3D convolutional layer
# filters: number of output filters (feature maps)
# kernel_size: dimensions of the convolution kernel (depth, height, width)
# activation: activation function to apply after convolution
conv3d_layer = tf.keras.layers.Conv3D(
    filters=32,
    kernel_size=(3, 3, 3),
    activation='relu',
    padding='same'  # 'same' padding adds zeros so output size matches input
)

# Apply the 3D convolution
output_volume = conv3d_layer(input_volume)

# Print the shapes of input and output
print(f"Input volume shape: {input_volume.shape}")
print(f"Output volume shape: {output_volume.shape}")

# Input volume shape: (1, 10, 20, 20, 1)
# Output volume shape: (1, 10, 20, 20, 32)
