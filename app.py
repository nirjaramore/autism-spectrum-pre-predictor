import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from PIL import Image

st.title("🧠 Autism Detection System")

# Load model
model = load_model("models/best_model.h5")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image")

    img = np.array(image)
    img = cv2.resize(img, (224, 224)) / 255.0
    img = np.expand_dims(img, axis=0)

    pred = model.predict(img)[0][0]

    if pred < 0.5:
        st.error(f"Autistic ({1 - pred:.2f})")
    else:
        st.success(f"Non-Autistic ({pred:.2f})")