import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
import os
from tensorflow.keras.models import load_model


# ── Load model ───────────────────────────────────────────────────────────────
model = load_model("models/best_model.h5")   # 🔥 use best model
_ = model(tf.zeros((1, 224, 224, 3)), training=False)


# ── Find last conv layer ─────────────────────────────────────────────────────
def find_last_conv(m):
    for layer in reversed(m.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer, m
        if isinstance(layer, tf.keras.Model):
            result = find_last_conv(layer)
            if result is not None:
                return result
    return None


conv_layer, owner_model = find_last_conv(model)
print("Last conv layer :", conv_layer.name)
print("Owner sub-model :", owner_model.name)


# ── Grad model ───────────────────────────────────────────────────────────────
grad_model = tf.keras.Model(
    inputs=owner_model.inputs,
    outputs=[conv_layer.output, owner_model.outputs[0]],
)


def get_head_layers(outer_model, sub_model):
    head, found = [], False
    for layer in outer_model.layers:
        if layer is sub_model:
            found = True
            continue
        if found:
            head.append(layer)
    return head


nested = owner_model is not model
head_layers = get_head_layers(model, owner_model) if nested else []
print("Head layers:", [l.name for l in head_layers])


# ── Load image ───────────────────────────────────────────────────────────────
def load_image(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (224, 224)).astype(np.float32) / 255.0
    return img


# ── Grad-CAM ─────────────────────────────────────────────────────────────────
def make_gradcam_heatmap(img_array):
    inputs = tf.cast(img_array, tf.float32)

    with tf.GradientTape() as tape:
        conv_outputs, sub_out = grad_model(inputs, training=False)
        tape.watch(conv_outputs)

        x = sub_out
        for layer in head_layers:
            x = layer(x, training=False)

        loss = x[:, 0]

    grads = tape.gradient(loss, conv_outputs)

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    heatmap = tf.reduce_sum(conv_outputs[0] * pooled_grads, axis=-1)

    heatmap = tf.maximum(heatmap, 0)
    heatmap /= tf.reduce_max(heatmap) + 1e-8

    return heatmap.numpy()


# ── Overlay ───────────────────────────────────────────────────────────────────
def overlay_heatmap(img_path, heatmap):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (224, 224))
    h = cv2.resize(heatmap, (224, 224))
    h = cv2.applyColorMap(np.uint8(255 * h), cv2.COLORMAP_JET)
    return cv2.addWeighted(img, 0.6, h, 0.4, 0)


# ── Normalize heatmap ────────────────────────────────────────────────────────
def normalize_heatmap(heatmap):
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap) + 1e-8
    return heatmap


# ── Region detection ─────────────────────────────────────────────────────────
def get_focus_regions(heatmap):
    h, w = heatmap.shape

    regions = {
        "eyes": np.mean(heatmap[0:int(h*0.33), :]),
        "nose": np.mean(heatmap[int(h*0.33):int(h*0.66), :]),
        "mouth": np.mean(heatmap[int(h*0.66):h, :])
    }

    return sorted(regions.items(), key=lambda x: x[1], reverse=True)


# ── Explanation ──────────────────────────────────────────────────────────────
def generate_explanation(sorted_regions):
    explanations = []

    for region, _ in sorted_regions[:2]:
        if region == "eyes":
            explanations.append("The model focused on the eye region, related to eye contact behavior.")
        elif region == "nose":
            explanations.append("The model analyzed central facial alignment patterns.")
        elif region == "mouth":
            explanations.append("The model analyzed the mouth region, related to facial expressions.")

    return explanations


# ── MAIN ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":

    folder = "dataset/test/autistic"
    threshold = 0.5

    # 🔥 LIMIT NUMBER OF IMAGES (VERY IMPORTANT)
    images = os.listdir(folder)[:3]   # change 3 → 1 or 5 if needed

    for img_name in images:

        img_path = os.path.join(folder, img_name)
        print("\nUsing image:", img_path)

        img = load_image(img_path)
        img_array = np.expand_dims(img, axis=0)

        # Prediction
        pred = model.predict(img_array, verbose=0)[0][0]
        label = "Autistic" if pred < threshold else "Non-Autistic"

        print(f"Prediction score : {pred:.4f}")
        print(f"Predicted class  : {label}")

        # Confidence
        if pred > 0.8 or pred < 0.2:
            print("Confidence: High")
        elif pred > 0.6 or pred < 0.4:
            print("Confidence: Medium")
        else:
            print("Confidence: Low")

        # Grad-CAM
        heatmap = make_gradcam_heatmap(img_array)
        heatmap = normalize_heatmap(heatmap)

        regions = get_focus_regions(heatmap)
        explanations = generate_explanation(regions)

        print("\n🧠 Explanation:")
        for exp in explanations:
            print("•", exp)

        result = overlay_heatmap(img_path, heatmap)

        # Display
        plt.figure(figsize=(9, 4))

        plt.subplot(1, 2, 1)
        orig = cv2.cvtColor(cv2.resize(cv2.imread(img_path), (224, 224)),
                            cv2.COLOR_BGR2RGB)
        plt.imshow(orig)
        plt.title("Original")
        plt.axis("off")

        plt.subplot(1, 2, 2)
        plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
        plt.title(f"Grad-CAM [{label} {pred:.2f}]")
        plt.axis("off")

        plt.tight_layout()
        plt.show()

    print("\n⚠️ Note: This is AI-based screening, not a medical diagnosis.")