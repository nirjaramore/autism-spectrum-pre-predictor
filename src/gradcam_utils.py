import numpy as np
import tensorflow as tf
import cv2


def get_gradcam_heatmap(model, img_array, last_conv_layer_name="Conv_1"):
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, 0]

    grads = tape.gradient(loss, conv_outputs)

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0) / tf.reduce_max(heatmap)

    return heatmap.numpy()


def overlay_heatmap(heatmap, image):
    heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    heatmap = np.uint8(255 * heatmap)

    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    superimposed_img = cv2.addWeighted(image, 0.6, heatmap, 0.4, 0)

    return superimposed_img

def generate_explanation(heatmap):
    explanation = []

    h, w = heatmap.shape

    eye_region = heatmap[int(h*0.2):int(h*0.5), :]
    mouth_region = heatmap[int(h*0.6):int(h*0.9), :]
    center_region = heatmap[int(h*0.3):int(h*0.7), int(w*0.3):int(w*0.7)]

    if np.mean(eye_region) > 0.3:
        explanation.append("👀 Focus on eye region (eye contact behavior)")

    if np.mean(mouth_region) > 0.3:
        explanation.append("🙂 Focus on mouth region (facial expressions)")

    if np.mean(center_region) > 0.3:
        explanation.append("🧠 Focus on central facial alignment")

    if not explanation:
        explanation.append("General facial features analyzed")

    return explanation

def generate_suggestions(pred, explanations):
    suggestions = []

    # Determine class
    is_autistic = pred < 0.5
    confidence = (1 - pred) if is_autistic else pred

    # 🔹 Base suggestions
    if is_autistic:
        suggestions.append("Consider consulting a pediatric specialist for further evaluation.")
        suggestions.append("Observe social behaviors like eye contact and response to name.")
    else:
        suggestions.append("No strong indicators detected. Continue normal developmental monitoring.")

    # 🔹 Confidence-based suggestions
    if confidence < 0.6:
        suggestions.append("Result confidence is low. Try uploading a clearer, front-facing image.")
    elif confidence > 0.85:
        suggestions.append("Model is highly confident in this prediction.")

    # 🔹 Region-based suggestions
    for exp in explanations:
        if "eye" in exp:
            suggestions.append("Eye contact patterns may be relevant in this analysis.")
        if "mouth" in exp:
            suggestions.append("Facial expression cues were important for this prediction.")

    return list(set(suggestions))  # remove duplicates