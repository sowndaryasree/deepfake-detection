# Deepfake Detection AI using CNN

## 📌 Project Overview

Deepfake technology can generate highly realistic fake images and videos using artificial intelligence. Detecting such manipulated media is important to prevent misinformation and digital fraud.

This project builds a **Deepfake Detection AI system** that classifies a face image as **REAL or FAKE** using a Convolutional Neural Network (CNN). The model is trained on a large dataset of real and fake faces and deployed as an interactive web application.

Users can upload a face image, and the system predicts whether it is real or AI-generated along with a confidence score.

---

## 🚀 Live Demo

Try the deployed application here:

🔗 https://huggingface.co/spaces/sowndarya18/deepfake-detection

---

## 🧠 Technologies Used

* Python
* TensorFlow / Keras
* MobileNetV2 (Transfer Learning)
* Gradio (Web Interface)
* Kaggle Dataset
* Google Colab
* Hugging Face Spaces
* GitHub

---

## 📂 Dataset

The model was trained using the **140K Real and Fake Faces Dataset** available on Kaggle.

Dataset features:

* Real human face images
* AI-generated fake face images
* Large dataset suitable for training deep learning models

---

## ⚙️ Model Architecture

The system uses **MobileNetV2** as a base model with transfer learning.

Architecture pipeline:

1. Pretrained MobileNetV2 (ImageNet weights)
2. Global Average Pooling Layer
3. Dropout Layer
4. Dense Layer with Sigmoid Activation

This architecture enables efficient feature extraction and accurate binary classification.

---

## 🔄 Workflow

1. Download dataset from Kaggle
2. Perform image preprocessing and augmentation
3. Train CNN model using transfer learning
4. Evaluate model performance
5. Save trained model as `.h5`
6. Build interactive UI using Gradio
7. Deploy the application on Hugging Face Spaces

---

## 🖥️ Application Interface

The web application allows users to:

* Upload a face image
* Run deepfake detection
* View prediction results
* See confidence score of the prediction

Output example:

🟢 REAL FACE
Confidence: 93.45%

or

🔴 FAKE FACE
Confidence: 87.12%

---

## 📊 Features

* Deep learning based image classification
* Real vs Fake face detection
* Confidence score output
* Interactive web interface
* Cloud deployment
* Lightweight model using transfer learning

---

## 📁 Project Structure

```
deepfake-detection
│
├── app.py
├── deepfake_model.h5
├── requirements.txt
├── README.md
```

---

## ▶️ Running the Project Locally

Install dependencies:

```
pip install -r requirements.txt
```

Run the application:

```
python app.py
```

The app will launch in your browser.

---

## 🌐 Deployment

The project is deployed using:

* Hugging Face Spaces for hosting
* Gradio for the web interface

---

## ⭐ Future Improvements

* Deepfake video detection
* Face detection preprocessing
* Improved accuracy using EfficientNet
* Real-time webcam detection

---

## 📜 License

This project is developed for educational and research purposes.
