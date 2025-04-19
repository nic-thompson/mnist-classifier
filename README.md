# 🧠 MNIST Image Classifier

This is a simple image classifier trained on the [MNIST handwritten digits dataset](http://yann.lecun.com/exdb/mnist/), built using TensorFlow and Keras.

---

## 📦 Project Structure

mnist_classifier/  
├── src/  
│   └── train.py         # Trains and evaluates the model  
├── models/  
│   └── mnist_model.h5   # Trained model (saved after training)  
├── .gitignore  
├── README.md  
└── requirements.txt     # (Optional) Add your dependencies here

---

## 🚀 How to Run

### 1. Set up the environment

\`\`\`bash
python3 -m venv venv
source venv/bin/activate
pip install tensorflow
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

---

## 💡 Next Steps (Coming Soon?)

- [ ] Add a prediction script for testing with custom digits  
- [ ] Convert to Keras SavedModel format  
- [ ] Web app using Flask or FastAPI  
- [ ] Deploy to HuggingFace Spaces or Streamlit

---

Loss Function
This project uses the sparse_categorical_crossentropy loss function, which is well-suited for multi-class classification problems

---

## 📜 License

MIT — feel free to use, modify, and share!

---

## 🙌 Acknowledgements

Thanks to the TensorFlow/Keras team for making deep learning accessible!
