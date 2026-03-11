import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import cv2

# Configure the web page
st.set_page_config(page_title="Pneumonia Diagnosis Assistant", layout="centered")

@st.cache_resource
def load_pneumonia_model():
    """Loads the model once and caches it in memory for fast inference."""
    return tf.keras.models.load_model('pneumonia_resnet50_model.h5')

def preprocess_image(image: Image.Image):
    """Processes the doctor's uploaded image to match our training data pipeline."""
    img_array = np.array(image.convert('RGB'))
    img_resized = cv2.resize(img_array, (224, 224))
    img_normalized = img_resized / 255.0
    # Expand dimensions to (1, 224, 224, 3) because the model expects a batch
    img_batch = np.expand_dims(img_normalized, axis=0)
    return img_batch

# UI Design
st.title("🩺 Intelligent Pneumonia Diagnosis Assistant")
st.markdown("Upload a Chest X-Ray to receive an automated prediction based on our ResNet50 Deep Learning model.")

model = load_pneumonia_model()

# File uploader widget
uploaded_file = st.file_uploader("Choose a Chest X-Ray image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded X-Ray
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded X-Ray', use_column_width=True)
    
    st.write("Processing inference...")
    
    # Run the image through the pipeline
    processed_image = preprocess_image(image)
    prediction_prob = model.predict(processed_image)[0][0]
    
    # Translate the mathematical probability into clinical results
    if prediction_prob > 0.5:
        prediction_label = "PNEUMONIA DETECTED"
        confidence = prediction_prob * 100
        st.error(f"**Diagnosis Prediction:** {prediction_label}")
    else:
        prediction_label = "NORMAL (HEALTHY)"
        confidence = (1 - prediction_prob) * 100
        st.success(f"**Diagnosis Prediction:** {prediction_label}")
        
    st.info(f"**AI Confidence Score:** {confidence:.2f}%")