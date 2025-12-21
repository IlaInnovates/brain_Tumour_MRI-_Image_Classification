# ================================
# Brain Tumor Classification App
# ================================

import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os

# --------------------------------
# App Configuration
# --------------------------------
st.set_page_config(
    page_title="Brain Tumor Classification",
    page_icon="🧠",
    layout="centered"
)

st.title("🧠 Brain Tumor Classification")
st.markdown(
    """
    Upload a **Brain MRI image** and the model will classify it into one of the following categories:
    - **Glioma**
    - **Meningioma**
    - **No Tumor**
    - **Pituitary**
    """
)

# --------------------------------
# Constants (MUST MATCH TRAINING)
# --------------------------------
IMG_SIZE = (224, 224)
MODEL_PATH = "transfer_learning_best.keras"

# ⚠️ MUST match train_ds.class_names exactly
CLASS_NAMES = ['glioma', 'meningioma', 'no_tumor', 'pituitary']

# --------------------------------
# Load Model (Cached)
# --------------------------------
@st.cache_resource
def load_trained_model():
    if not os.path.exists(MODEL_PATH):
        st.error("❌ Model file not found. Please place 'transfer_learning_best.keras' in this folder.")
        st.stop()

    model = tf.keras.models.load_model(MODEL_PATH)
    return model

model = load_trained_model()

# --------------------------------
# Image Preprocessing (MATCH TRAINING)
# --------------------------------
def preprocess_image(image: Image.Image):
    image = image.convert("RGB")
    image = image.resize(IMG_SIZE)
    image = np.array(image, dtype=np.float32)
    image = image / 255.0   # single normalization
    image = np.expand_dims(image, axis=0)
    return image

# --------------------------------
# File Uploader
# --------------------------------
uploaded_file = st.file_uploader(
    "📤 Upload Brain MRI Image",
    type=["jpg", "jpeg", "png"]
)

# --------------------------------
# Prediction
# --------------------------------
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded MRI Image", use_container_width=True)

    with st.spinner("🔍 Analyzing image..."):
        processed_image = preprocess_image(image)
        predictions = model.predict(processed_image)

        predicted_class_index = np.argmax(predictions[0])
        confidence = predictions[0][predicted_class_index]

    predicted_label = CLASS_NAMES[predicted_class_index]

    # --------------------------------
    # Display Results
    # --------------------------------
    st.subheader("🧪 Prediction Result")
    st.success(f"**Tumor Type:** {predicted_label.upper()}")
    st.info(f"**Confidence:** {confidence * 100:.2f}%")

    # --------------------------------
    # Confidence Breakdown
    # --------------------------------
    st.subheader("📊 Confidence Scores")
    for i, class_name in enumerate(CLASS_NAMES):
        st.write(f"**{class_name.capitalize()}**: {predictions[0][i] * 100:.2f}%")

# --------------------------------
# Footer
# --------------------------------
st.markdown("---")
st.caption("Brain Tumor Classification using CNN & Transfer Learning (EfficientNetB0)")
