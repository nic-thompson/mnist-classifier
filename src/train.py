import tensorflow as tf
from tensorflow import keras
mnist = keras.datasets.mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical
import os


def load_data():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0  # normalize to [0, 1]
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)
    return x_train, y_train, x_test, y_test


def build_model():
    model = Sequential([
        Flatten(input_shape=(28, 28)),
        Dense(128, activation='relu'),
        Dense(10, activation='softmax'),
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def train():
    x_train, y_train, x_test, y_test = load_data()
    model = build_model()

    print("Training the model...")
    model.fit(x_train, y_train, epochs=5, batch_size=32)

    print("Evaluating the model...")
    loss, accuracy = model.evaluate(x_test, y_test)
    print(f"Test accuracy: {accuracy:.4f}")

    # Save model directory
    os.makedirs("models", exist_ok=True)
    model.save("models/mnist_model.h5")
    print("Model saved to models/mnist_model.h5")


if __name__ == "__main__":
    train()
