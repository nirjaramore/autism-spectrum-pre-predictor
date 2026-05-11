import cv2
import numpy as np
from tensorflow.keras.models import load_model
from face_detect import detect_face

# ---------------- LOAD MODELS ----------------

basic_model = load_model("model/cnn_model.h5")

deep_model = load_model("model/best_deep_cnn.h5")

res_model = load_model("model/residual_cnn.keras")


# ---------------- PREPROCESS FUNCTION ----------------

def preprocess_face(face, size):

    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

    face = cv2.resize(face, size)

    face = face.astype("float32") / 255.0

    face = np.expand_dims(face, axis=0)

    return face


# ---------------- MAIN PREDICTION FUNCTION ----------------

def predict_image(img):

    # ---------------- FACE DETECTION ----------------

    face = detect_face(img)

    if face is None:

        print("Face not detected. Using full image.")

        face = img

    # ---------------- PREPROCESS ----------------

    face_basic = preprocess_face(face, (128, 128))

    face_deep = preprocess_face(face, (224, 224))

    face_res = preprocess_face(face, (224, 224))

    # ---------------- MODEL PREDICTIONS ----------------

    pred_basic = float(
        basic_model.predict(face_basic, verbose=0)[0][0]
    )

    pred_deep = float(
        deep_model.predict(face_deep, verbose=0)[0][0]
    )

    pred_res = float(
        res_model.predict(face_res, verbose=0)[0][0]
    )

    # ---------------- DEBUG OUTPUTS ----------------

    print("\n------ MODEL OUTPUTS ------")

    print("Basic CNN:", pred_basic)

    print("Deep CNN:", pred_deep)

    print("Residual CNN:", pred_res)

    # ---------------- MODEL CONFIDENCE ----------------

    basic_conf = abs(pred_basic - 0.5)

    deep_conf = abs(pred_deep - 0.5)

    res_conf = abs(pred_res - 0.5)

    # ---------------- BEST MODEL SELECTION ----------------

    confidences = {
        "Basic CNN": basic_conf,
        "Deep CNN": deep_conf,
        "Residual CNN": res_conf
    }

    best_model = max(confidences, key=confidences.get)

    print("Best Model:", best_model)

    # ---------------- SMART DYNAMIC ENSEMBLE ----------------

    # Base weights
    basic_weight = 0.20
    deep_weight = 0.30
    res_weight = 0.50

    # Confidence adjusted weights
    basic_weight *= (1 + basic_conf)

    deep_weight *= (1 + deep_conf)

    res_weight *= (1 + res_conf)

    # Final weighted ensemble prediction
    final_pred = (
        pred_basic * basic_weight +
        pred_deep * deep_weight +
        pred_res * res_weight
    ) / (
        basic_weight +
        deep_weight +
        res_weight
    )

    print("Final Ensemble Score:", final_pred)

    # ---------------- IMPORTANT LABEL FIX ----------------
    # Reversed labels fix

    if 0.45 <= final_pred <= 0.60:

        label = "Uncertain"

        confidence = final_pred * 100

    elif final_pred > 0.60:

        # HIGH VALUE = AUTISTIC

        label = "Autistic"

        confidence = final_pred * 100

    else:

        # LOW VALUE = NON AUTISTIC

        label = "Non-Autistic"

        confidence = (1 - final_pred) * 100

    # ---------------- RETURN ----------------

    return (
        label,
        round(confidence, 2),
        {
            "basic": pred_basic,
            "deep": pred_deep,
            "residual": pred_res,
            "ensemble": final_pred,
            "best_model": best_model,
            "basic_conf": basic_conf,
            "deep_conf": deep_conf,
            "res_conf": res_conf
        }
    )