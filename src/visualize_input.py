import sys
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt

# Paths
image_path = sys.argv[1] if len(sys.argv) > 1 else "images/processed/sun.png"
model_path = "models/mnist_model.h5"

# Load and process image
img = image.load_img(image_path, color_mode="grayscale", target_size=(28, 28))
img_array = image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)  # (1, 28, 28, 1)

# Predict
model = load_model(model_path)
pred = model.predict(img_array)
predicted_class = np.argmax(pred)
confidence = pred[0][predicted_class]

# Show what the model "sees"
plt.imshow(img_array.squeeze(), cmap="gray")
plt.title(f"Model sees this as: {predicted_class} (confidence: {confidence:.2f})")
plt.axis("off")
plt.show()
