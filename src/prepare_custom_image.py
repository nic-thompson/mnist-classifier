# src/prepare_custom_image.py
import sys
from PIL import Image
import numpy as np
import os

def prepare_image(image_path):
    image = Image.open(image_path).convert("L")  # Convert to grayscale
    image = image.resize((28, 28))  # Resize to 28x28
    image_array = np.array(image).astype("float32") / 255.0  # Normalize
    return (image_array * 255).astype("uint8")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python src/prepare_custom_image.py <input_image_path> <output_image_path>")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2]

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    processed = prepare_image(input_path)
    Image.fromarray(processed).save(output_path)
    print(f"Saved processed image to {output_path}")
