import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.datasets import mnist

# Load the trained model
model = load_model("models/mnist_model.h5")

# Load a sample digit from the MNIST test set
(_, _), (x_test, y_test) = mnist.load_data()
img = x_test[0]
img_input = img.reshape(1, 28, 28, 1) / 255.0  # normalize and shape for model

# Show the input image
plt.figure(figsize=(2, 2))
plt.title(f"Input Image (Label: {y_test[0]})")
plt.imshow(img, cmap="gray")
plt.axis("off")
plt.show()

# Extract the outputs of the convolutional layers
layer_outputs = [layer.output for layer in model.layers if "conv" in layer.name]
activation_model = Model(inputs=model.input, outputs=layer_outputs)

# Run the image through the model to get the activations
feature_maps = activation_model.predict(img_input)

# Visualize feature maps from the first convolutional layer
first_layer_maps = feature_maps[0]
num_filters = first_layer_maps.shape[-1]

plt.figure(figsize=(15, 8))
for i in range(num_filters):
    ax = plt.subplot(4, 8, i + 1)  # adjust grid if needed
    plt.imshow(first_layer_maps[0, :, :, i], cmap="viridis")
    plt.axis("off")
    plt.title(f"F{i}")
plt.suptitle("Feature Maps - First Convolutional Layer")
plt.tight_layout()
plt.show()
