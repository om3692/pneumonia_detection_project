# 🩺 Intelligent Diagnosis Using ML and Transfer Learning for Pneumonia Detection

## 📌 Project Overview
This repository contains a complete, production-ready machine learning pipeline for the automated diagnosis of pneumonia from chest X-ray images. Utilizing a deep learning approach—specifically Transfer Learning with a pre-trained ResNet50 architecture—the model achieves highly sensitive, empirical results designed to act as a diagnostic assistant for medical professionals.

## 🏗️ Architecture & Tech Stack
* **Language:** Python
* **Deep Learning Framework:** TensorFlow / Keras
* **Feature Extractor:** ResNet50 (Frozen Convolutional Base)
* **Computer Vision:** OpenCV / PIL
* **Data Ingestion:** Autonomous Kaggle API Integration
* **User Interface:** Streamlit (Interactive Doctor's Dashboard)

## ⚙️ How It Works
1. **Autonomous Data Pipeline:** The `data_loader.py` script securely connects to Kaggle, downloads the ~1.2GB dataset, and restructures the folders into a unified directory for dynamic partitioning (80% Train, 10% Validation, 10% Test).
2. **Transfer Learning:** By freezing the ResNet50 base layers, the model bypasses the vanishing gradient problem and utilizes pre-learned feature extraction. A custom classification head (GlobalAveragePooling2D, Dense, Dropout) maps these features to a binary output (Normal vs. Pneumonia).
3. **Interactive UI:** The Streamlit application abstracts the mathematical complexities into a clean, clinical dashboard for real-time inference.

## 🚀 Installation & Usage

**1. Clone the repository and install dependencies:**

git clone [https://github.com/YOUR_USERNAME/pneumonia-detection-ai.git](https://github.com/YOUR_USERNAME/pneumonia-detection-ai.git)
cd pneumonia-detection-ai
pip install -r requirements.txt

2. Kaggle API Setup:
Ensure your kaggle.json authentication token is placed in your system's global directory (~/.kaggle/ or C:\Users\Your-Username\.kaggle\). The data loader relies on this for autonomous ingestion.

3. Train the Model:

Bash
python train.py
4. Evaluate Performance:
Generate the confusion matrix and loss curves.

Bash
python evaluate.py
5. Launch the Doctor's Dashboard:

Bash
streamlit run app.py
