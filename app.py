import streamlit as st
import cv2
import numpy as np
from predict import predict_image

# ---------------- PAGE CONFIG ----------------

st.set_page_config(
    page_title="Autism Detection System",
    page_icon="🧠",
    layout="centered"
)

# ---------------- CUSTOM CSS ----------------

st.markdown("""
<style>

body {
    background-color: #0f172a;
}

.title {
    text-align: center;
    font-size: 44px;
    font-weight: bold;
    color: #f8fafc;
    margin-bottom: 5px;
}

.subtitle {
    text-align: center;
    font-size: 18px;
    color: #94a3b8;
    margin-bottom: 30px;
}

.result {
    padding: 25px;
    border-radius: 16px;
    text-align: center;
    font-size: 24px;
    font-weight: bold;
    margin-top: 20px;
    margin-bottom: 20px;
    box-shadow: 0px 0px 20px rgba(0,0,0,0.15);
}

.autistic {
    background-color: #ffe6e6;
    color: #c0392b;
}

.non-autistic {
    background-color: #eafaf1;
    color: #27ae60;
}

.uncertain {
    background-color: #fff4e6;
    color: #e67e22;
}

.model-box {
    background-color: #172b45;
    padding: 20px;
    border-radius: 14px;
    margin-top: 15px;
    color: #e2e8f0;
}

.footer {
    text-align: center;
    color: #94a3b8;
    margin-top: 40px;
}

</style>
""", unsafe_allow_html=True)

# ---------------- TITLE ----------------

st.markdown(
    '<div class="title">🧠 Autism Detection System</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Hybrid CNN Ensemble Based Autism Detection</div>',
    unsafe_allow_html=True
)

st.write("")

# ---------------- INPUT METHOD ----------------

input_method = st.radio(
    "Choose Input Method",
    ["📤 Upload Image", "📷 Capture Image"]
)

img = None

# ---------------- IMAGE UPLOAD ----------------

if input_method == "📤 Upload Image":

    uploaded_file = st.file_uploader(
        "📤 Upload a Face Image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:

        file_bytes = np.asarray(
            bytearray(uploaded_file.read()),
            dtype=np.uint8
        )

        img = cv2.imdecode(file_bytes, 1)

# ---------------- CAMERA INPUT ----------------

elif input_method == "📷 Capture Image":

    camera_image = st.camera_input("📷 Capture Face Image")

    if camera_image is not None:

        file_bytes = np.asarray(
            bytearray(camera_image.read()),
            dtype=np.uint8
        )

        img = cv2.imdecode(file_bytes, 1)

# ---------------- PROCESS IMAGE ----------------

if img is not None:

    st.image(
        img,
        width=320,
        caption="Selected Image"
    )

    st.write("")

    # ---------------- PREDICT BUTTON ----------------

    if st.button("🔍 Predict"):

        with st.spinner("Analyzing Facial Features..."):

            label, confidence, details = predict_image(img)

        st.write("")

        # ---------------- NO FACE DETECTED ----------------

        if label == "No Face Detected":

            st.warning(
                "⚠️ No face detected. Please upload/capture a clear face image."
            )

            st.stop()

        # ---------------- UNCERTAIN ----------------

        elif label == "Uncertain":

            st.markdown(
                f'''
                <div class="result uncertain">
                    Prediction: {label}<br>
                    Confidence: {confidence:.2f}%
                </div>
                ''',
                unsafe_allow_html=True
            )

            st.info(
                "The ensemble models produced mixed confidence values. "
                "Try using a clearer front-facing image."
            )

        # ---------------- AUTISTIC ----------------

        elif label == "Autistic":

            st.markdown(
                f'''
                <div class="result autistic">
                    Prediction: {label}<br>
                    Confidence: {confidence:.2f}%
                </div>
                ''',
                unsafe_allow_html=True
            )

            st.markdown(f"""
            <div class="model-box">

            <h4>Best Model Selected: {details['best_model']}</h4>

            <br>

            <h4>Possible observed traits:</h4>

            • Social communication differences<br>
            • Reduced eye contact<br>
            • Repetitive behavioral tendencies<br>
            • Facial attention irregularities

            </div>
            """, unsafe_allow_html=True)

        # ---------------- NON AUTISTIC ----------------

        elif label == "Non-Autistic":

            st.markdown(
                f'''
                <div class="result non-autistic">
                    Prediction: {label}<br>
                    Confidence: {confidence:.2f}%
                </div>
                ''',
                unsafe_allow_html=True
            )

            st.markdown(f"""
            <div class="model-box">

            <h4>Best Model Selected: {details['best_model']}</h4>

            <br>

            Facial characteristics appear consistent with non-autistic patterns.

            </div>
            """, unsafe_allow_html=True)

        # ---------------- CONFIDENCE BAR ----------------

        st.write("### Confidence Score")

        st.progress(int(confidence))

        # ---------------- ADVANCED DETAILS ----------------

        with st.expander("🔍 Model Details (Advanced)"):

            st.write(
                f"Basic CNN Output: {details['basic']:.4f}"
            )

            st.write(
                f"Deep CNN Output: {details['deep']:.4f}"
            )

            st.write(
                f"Residual CNN Output: {details['residual']:.4f}"
            )

            st.write("")

            st.write(
                f"Best Model Used: {details['best_model']}"
            )

            st.write(
                f"Final Prediction Score: {details['ensemble']:.4f}"
            )

# ---------------- FOOTER ----------------

st.markdown("---")

st.markdown(
    """
    <div class="footer">
    ⚠️ This system is intended for educational and research purposes only.<br>
    It is not a medical diagnostic tool.
    </div>
    """,
    unsafe_allow_html=True
)