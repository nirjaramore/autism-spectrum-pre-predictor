import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from src.gradcam_utils import generate_suggestions
from PIL import Image

from src.gradcam_utils import get_gradcam_heatmap, overlay_heatmap, generate_explanation

# 🔹 Page config
st.set_page_config(
    page_title="Autism Detection System",
    page_icon="🧠",
    layout="wide"
)

# 🔹 Custom CSS
st.markdown("""
<style>
.result-box {
    padding: 20px;
    border-radius: 12px;
    font-size: 20px;
    text-align: center;
    font-weight: bold;
}
.autistic {
    background-color: #3a1f1f;
    color: #ff6b6b;
}
.non-autistic {
    background-color: #1f3a2b;
    color: #4ade80;
}
</style>
""", unsafe_allow_html=True)

# 🔹 Header
st.markdown("<h1 style='text-align: center;'>🧠 Autism Detection System</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>AI-powered screening using facial analysis</p>", unsafe_allow_html=True)

# 🔹 Load model
model = load_model("models/best_model.h5")

# 🔹 Upload
uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:

    col1, col2 = st.columns(2)

    image = Image.open(uploaded_file).convert("RGB")
    img = np.array(image)

    with col1:
        st.image(image, caption="Uploaded Image", width='stretch')

    # 🔹 Preprocess
    img_resized = cv2.resize(img, (224, 224)) / 255.0
    img_input = np.expand_dims(img_resized, axis=0)

    # 🔹 Prediction
    pred = model.predict(img_input)[0][0]

    with col2:
        st.subheader("Prediction Result")

        if pred < 0.5:
            st.markdown(f"<div class='result-box autistic'>Autistic ({1 - pred:.2f})</div>", unsafe_allow_html=True)
            confidence = (1 - pred)
        else:
            st.markdown(f"<div class='result-box non-autistic'>Non-Autistic ({pred:.2f})</div>", unsafe_allow_html=True)
            confidence = pred

        # 🔥 Confidence bar
        st.write("Confidence")
        st.progress(int(confidence * 100))

        st.write(f"Raw score: {pred:.4f}")

    # 🔥 Grad-CAM
    st.markdown("---")
    st.subheader("🔥 Model Attention (Grad-CAM)")

    heatmap = get_gradcam_heatmap(model, img_input)
    cam_image = overlay_heatmap(heatmap, img)

    st.image(cam_image, caption="Grad-CAM Heatmap", width='stretch')

    # 🔹 Explanation
    st.subheader("🧠 Model Explanation")

    explanations = generate_explanation(heatmap)
    for exp in explanations:
        st.write(f"• {exp}")

    # 🔹 Suggestions
    st.subheader("📌 Guidance & Next Steps")

    suggestions = generate_suggestions(pred, explanations)

    for s in suggestions:
       st.write(f"• {s}")

    # 🔹 Warning
    st.warning("This is an AI-based screening tool, not a medical diagnosis.")
