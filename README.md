# 🧠 MNIST Image Classifier

This is a simple image classifier trained on the [MNIST handwritten digits dataset](http://yann.lecun.com/exdb/mnist/), built using TensorFlow and Keras.

---

## 📦 Project Structure

mnist_classifier/
├── src/
│ ├── train.py # Train and evaluate the model
│ ├── predict.py # Predict using a trained model
│ ├── prepare_custom_image.py # Preprocess custom images for inference
├── models/
│ └── mnist_model.h5 # Trained model (saved after training)
├── images/
│ ├── raw/ # Input images for prediction
│ └── processed/ # Preprocessed grayscale images
├── samples/ # Sample MNIST digits
├── .gitignore
├── README.md
└── requirements.txt

---

## 🚀 How to Run

### 1. Set up the environment

\`\`\`bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
\`\`\`

### 2. Train the model

\`\`\`bash
python src/train.py
\`\`\`

This will:
- Download the MNIST dataset
- Train a simple neural network
- Save the model to `models/mnist_model.h5`

---

Predict using a custom image
Preprocess your image (convert to 28x28 grayscale and normalize):

python src/prepare_custom_image.py images/raw/your_image.png images/processed/your_image.png
Run prediction:
python src/predict.py images/processed/your_image.png

---

## 📈 Results

After 5 epochs, the model typically achieves:

- ✅ Training accuracy: ~97.5%  
- ✅ Test accuracy: ~97.6%

---

## 🛠 Tech Stack

- Python 3.11  
- TensorFlow 2  
- Keras  
- CLI & terminal tools for training and running
- Pillow (image handling)
- Matplotlib (for optional visual debugging)

---

## 💡 Next Steps (Coming Soon?)

- [ ] Add a prediction script for testing with custom digits  
- [ ] Convert to Keras SavedModel format  
- [ ] Web app using Flask or FastAPI  
- [ ] Deploy to HuggingFace Spaces or Streamlit

---

Loss Function
This project uses sparse_categorical_crossentropy, which is ideal for multi-class classification tasks where labels are integers rather than one-hot encoded vectors. It measures how well the predicted probability distribution aligns with the true label.

---

💡 Future Enhancements
 Predict digits from custom images

 Image preprocessing pipeline

 Convert to Keras .keras or SavedModel format

 Web app using Flask or FastAPI

 Deploy to HuggingFace Spaces or Streamlit

---

## 📜 License

MIT — feel free to use, modify, and share!

---

## 🙌 Acknowledgements

Thanks to the TensorFlow/Keras team for making deep learning accessible!
