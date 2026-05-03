import streamlit as st
import cv2
import numpy as np
from predict import predict_image
from datetime import datetime

# =============================
# PAGE CONFIG
# =============================
st.set_page_config(page_title="Autism Screening System", layout="wide")

# =============================
# CSS (Hospital UI)
# =============================
st.markdown("""
<style>
.stApp { background-color: #f4f7fb; }

.header {
    background-color: #0d6efd;
    padding: 18px;
    border-radius: 10px;
    color: white;
    text-align: center;
    font-size: 26px;
    font-weight: bold;
}

.card {
    background: white;
    padding: 20px;
    border-radius: 10px;
    box-shadow: 0px 3px 10px rgba(0,0,0,0.08);
    margin-bottom: 20px;
}

.label {
    font-weight: bold;
    color: #34495e;
}

.result-good { background:#e8f8f5; color:#1abc9c; padding:12px; border-radius:8px; }
.result-bad { background:#fdecea; color:#c0392b; padding:12px; border-radius:8px; }
.result-mid { background:#fff3cd; color:#856404; padding:12px; border-radius:8px; }

</style>
""", unsafe_allow_html=True)

# =============================
# HEADER
# =============================
st.markdown('<div class="header">Autism Early Screening System</div>', unsafe_allow_html=True)

# =============================
# PATIENT FORM
# =============================
st.markdown("### Patient Information")

colA, colB, colC = st.columns(3)

with colA:
    name = st.text_input("Patient Name")

with colB:
    age = st.number_input("Age", 1, 100)

with colC:
    gender = st.selectbox("Gender", ["Male", "Female"])

st.write("---")

# =============================
# MAIN LAYOUT
# =============================
col1, col2 = st.columns([1,1])

# =============================
# LEFT SIDE (UPLOAD)
# =============================
with col1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("#### Upload Image")

    uploaded_file = st.file_uploader("", type=["jpg","png","jpeg"])

    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)

        st.image(img, caption="Patient Image", width=300)

    st.markdown('</div>', unsafe_allow_html=True)

# =============================
# RIGHT SIDE (RESULT)
# =============================
with col2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("#### Screening Result")

    if uploaded_file is not None:
        if st.button("Run Screening"):
            with st.spinner("Processing..."):
                label, confidence = predict_image(img)

            st.write("")

            if label == "Autistic":
                st.markdown(f'<div class="result-bad">High Risk<br>{confidence}% Confidence</div>', unsafe_allow_html=True)

            elif label == "Non-Autistic":
                st.markdown(f'<div class="result-good">Low Risk<br>{confidence}% Confidence</div>', unsafe_allow_html=True)

            elif label == "Uncertain":
                st.markdown(f'<div class="result-mid">Uncertain Case<br>{confidence}% Confidence</div>', unsafe_allow_html=True)

            else:
                st.warning("No face detected")

            # Confidence bar
            st.progress(int(confidence))

            # Save session result
            result_data = {
                "Name": name,
                "Age": age,
                "Gender": gender,
                "Result": label,
                "Confidence": confidence,
                "Time": datetime.now().strftime("%H:%M:%S")
            }

            if "history" not in st.session_state:
                st.session_state.history = []

            st.session_state.history.append(result_data)

            # Download report
            report = f"""
            Autism Screening Report
            -------------------------
            Name: {name}
            Age: {age}
            Gender: {gender}
            Result: {label}
            Confidence: {confidence}%
            Time: {result_data['Time']}
            """

            st.download_button("Download Report", report, file_name="report.txt")

    st.markdown('</div>', unsafe_allow_html=True)

# =============================
# HISTORY PANEL
# =============================
st.write("### Screening History")

if "history" in st.session_state:
    st.dataframe(st.session_state.history)
else:
    st.info("No history available")

# =============================
# FOOTER
# =============================
st.write("---")
st.warning("⚠️ This system is for early screening only. Not a medical diagnosis.")