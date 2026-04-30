import streamlit as st
import cv2
import numpy as np
from predict import predict_image

# Page settings
st.set_page_config(page_title="Autism Detection", layout="centered")

# Custom styling
st.markdown("""
<style>
.title {
    text-align: center;
    font-size: 42px;
    font-weight: bold;
    color: #2c3e50;
}
.subtitle {
    text-align: center;
    font-size: 18px;
    color: #7f8c8d;
}
.result {
    padding: 20px;
    border-radius: 10px;
    text-align: center;
    font-size: 22px;
    font-weight: bold;
}
.autistic {
    background-color: #ffe6e6;
    color: #c0392b;
}
.non-autistic {
    background-color: #eafaf1;
    color: #27ae60;
}
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<div class="title">Autism Detection System</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">CNN Model (From Scratch)</div>', unsafe_allow_html=True)

st.write("")

# Upload section
uploaded_file = st.file_uploader("📤 Upload a Face Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)

    st.image(img, width=300, caption="Uploaded Image")

    if st.button("🔍 Predict"):
        with st.spinner("Analyzing Image..."):
            label, confidence = predict_image(img)

        st.write("")

        # Show result
        if label == "Autistic":
            st.markdown(
                f'<div class="result autistic">Prediction: {label}<br>Confidence: {confidence:.2f}%</div>',
                unsafe_allow_html=True
            )
        elif label == "Non-Autistic":
            st.markdown(
                f'<div class="result non-autistic">Prediction: {label}<br>Confidence: {confidence:.2f}%</div>',
                unsafe_allow_html=True
            )
        else:
            st.warning("⚠️ No face detected")

        # Progress bar
        st.progress(int(confidence))

# Footer
st.markdown("---")
st.info("⚠️ This system is for educational purposes only and not a medical diagnostic tool.")