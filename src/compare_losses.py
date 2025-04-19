import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

# Load MNIST data
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0  # Normalize

# --- Model 1: SparseCategoricalCrossentropy (integer labels) ---
model_sparse = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax'),
])

model_sparse.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

history_sparse = model_sparse.fit(
    x_train, y_train,
    epochs=5,
    validation_split=0.1,
    verbose=0
)

# --- Model 2: CategoricalCrossentropy (one-hot labels) ---
y_train_cat = to_categorical(y_train, 10)
y_test_cat = to_categorical(y_test, 10)

model_cat = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax'),
])

model_cat.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

history_cat = model_cat.fit(
    x_train, y_train_cat,
    epochs=5,
    validation_split=0.1,
    verbose=0
)

# --- Plot Comparison ---
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history_sparse.history['loss'], label='Sparse Loss')
plt.plot(history_cat.history['loss'], label='Categorical Loss')
plt.title('Training Loss Comparison')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history_sparse.history['accuracy'], label='Sparse Accuracy')
plt.plot(history_cat.history['accuracy'], label='Categorical Accuracy')
plt.title('Training Accuracy Comparison')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()
