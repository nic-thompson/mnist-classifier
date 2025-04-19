# scripts/save_sample_digit.py

import os
import numpy as np
from keras.datasets import mnist
from PIL import Image

# Load sample data
(x_train, y_train), _ = mnist.load_data()

# Pick a sample index and label
index = 0  # You can change this to 1, 2, etc.
digit = x_train[index]
label = y_train[index]

# Convert to PIL image and save
img = Image.fromarray(digit)
os.makedirs("images", exist_ok=True)
img_path = f"images/{label}.png"
img.save(img_path)

print(f"Saved digit '{label}' to {img_path}")
