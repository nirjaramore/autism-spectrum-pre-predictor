# predict.py

import cv2
import numpy as np
from tensorflow.keras.models import load_model
from face_detect import detect_face

model = load_model("model/cnn_model.h5")

def predict_image(img):

    face = detect_face(img)

    if face is None:
        return "No Face Detected", 0

    # Preprocessing
    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    face = cv2.resize(face, (128, 128))
    face = face / 255.0
    face = np.expand_dims(face, axis=0)

    # Prediction
    pred = model.predict(face)[0][0]

    print("Raw prediction:", pred)

    # ✅ Proper classification logic
    if 0.45 < pred < 0.6:
        return "Uncertain", round(pred * 100, 2)

    elif pred >= 0.6:
        label = "Non-Autistic"
        confidence = pred * 100

    else:
        label = "Autistic"
        confidence = (1 - pred) * 100

    return label, round(confidence, 2)