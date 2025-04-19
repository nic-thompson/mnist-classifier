import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout
import os

# Load and preprocess data
def load_data():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    # Normalize the pixel values to between 0 and 1
    x_train, x_test = x_train / 255.0, x_test / 255.0
    return x_train, y_train, x_test, y_test

# Build the model
def build_model():
    model = Sequential([
        Flatten(input_shape=(28, 28)),        # Flatten image to 1D vector
        Dense(128, activation='relu'),        # Hidden layer with ReLU
        Dropout(0.2),                         # Dropout to prevent overfitting
        Dense(10, activation='softmax')       # Output layer for 10 classes
    ])
    return model

# Train and evaluate the model
def train():
    x_train, y_train, x_test, y_test = load_data()
    model = build_model()
    
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    print("Training the model...")
    model.fit(x_train, y_train, epochs=5)

    print("Evaluating the model...")
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
    print(f"Test accuracy: {test_acc:.4f}")

    # Save model to disk
    os.makedirs('models', exist_ok=True)
    model.save('models/mnist_model.h5')
    print("Model saved to models/mnist_model.h5")

if __name__ == "__main__":
    train()
