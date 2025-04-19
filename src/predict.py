import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import matplotlib.pyplot as plt
import os

def preprocess_image(image_path):
    """Load an image file and convert it to a 28x28 grayscale array."""
    image = Image.open(image_path).convert('L')  # grayscale
    image = image.resize((28, 28))
    image_array = np.array(image)
    image_array = 255 - image_array  # invert (white background, black digit)
    image_array = image_array / 255.0  # normalize
    image_array = image_array.reshape(1, 28, 28, 1)  # model expects 4D
    return image_array

def predict(image_path, model_path):
    model = load_model(model_path)
    image = preprocess_image(image_path)
    prediction = model.predict(image)
    predicted_label = np.argmax(prediction)
    confidence = np.max(prediction)

    print(f"Predicted digit: {predicted_label} (confidence: {confidence:.2f})")

    # Show the image for reference
    plt.imshow(image.reshape(28, 28), cmap="gray")
    plt.title(f"Prediction: {predicted_label}")
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict a digit using the trained model.")
    parser.add_argument("image_path", help="Path to the image file (28x28, or any size)")
    parser.add_argument("--model", default="models/mnist_model.h5", help="Path to the saved model file")
    args = parser.parse_args()

    if not os.path.exists(args.image_path):
        print(f"Error: Image not found at {args.image_path}")
        exit(1)

    predict(args.image_path, args.model)
