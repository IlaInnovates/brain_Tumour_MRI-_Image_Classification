import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os

# -----------------------------------
# CONFIG
# -----------------------------------
IMG_SIZE = 224
MODEL_PATH = "custom_cnn_best.keras"   # 🔴 MUST EXIST

CLASS_NAMES = [
    "glioma",
    "meningioma",
    "no_tumor",
    "pituitary"
]

st.set_page_config(page_title="Brain Tumor Classification", layout="centered")
st.title("🧠 Brain Tumor MRI Classification")

# -----------------------------------
# LOAD MODEL SAFELY
# -----------------------------------
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"❌ Model file '{MODEL_PATH}' not found.")
        st.stop()
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

# -----------------------------------
# IMAGE PREPROCESSING
# -----------------------------------
def preprocess_image(image):
    image = image.convert("RGB")
    image = image.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(image, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# -----------------------------------
# FILE UPLOADER
# -----------------------------------
uploaded_file = st.file_uploader(
    "Upload Brain MRI Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded MRI Image", use_column_width=True)

    img_tensor = preprocess_image(image)

    predictions = model.predict(img_tensor)[0]

    top_indices = np.argsort(predictions)[::-1]
    top1, top2 = top_indices[:2]

    st.subheader("🧪 Prediction Result")

    st.success(
        f"Primary Prediction: {CLASS_NAMES[top1].upper()} "
        f"({predictions[top1]*100:.2f}%)"
    )

    st.info(
        f"Second Possible: {CLASS_NAMES[top2].upper()} "
        f"({predictions[top2]*100:.2f}%)"
    )

    if predictions[top1] < 0.60:
        st.warning("⚠️ Low confidence prediction — MRI features are ambiguous.")

    st.subheader("📊 All Class Probabilities")
    for cls, prob in zip(CLASS_NAMES, predictions):
        st.write(f"{cls}: {prob*100:.2f}%")
