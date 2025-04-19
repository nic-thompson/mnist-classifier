from tensorflow.keras.datasets import mnist

from PIL import Image
import os

# Load the dataset
(x_train, y_train), _ = mnist.load_data()

# Pick the first digit and label
digit = x_train[0]
label = y_train[0]

# Save the image
output_dir = "samples"
os.makedirs(output_dir, exist_ok=True)

image_path = os.path.join(output_dir, f"digit_{label}.png")
Image.fromarray(digit).save(image_path)

print(f"Digit saved as {image_path}")
