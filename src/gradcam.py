import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
import os
from tensorflow.keras.models import load_model


# ── Load model ───────────────────────────────────────────────────────────────
model = load_model("models/autism_model.h5")
_ = model(tf.zeros((1, 224, 224, 3)), training=False)   # materialise graph


# ── Helpers ──────────────────────────────────────────────────────────────────
def find_last_conv(m):
    """
    Return (conv_layer, owner_model) where owner_model is the Keras Model
    object whose graph actually contains that Conv2D node.
    Walks the layer list in reverse; recurses into nested Model layers.
    """
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


# ── Build grad_model entirely within owner_model's graph ─────────────────────
#
#  conv_layer.output is a valid tensor ONLY inside owner_model's graph.
#  So we build:
#      grad_model : owner_model.input → [conv_layer.output, owner_model.output]
#
#  If owner_model == model  → we're done.
#  If owner_model is nested → we run the outer model's remaining head layers
#                             manually on top of owner_model's output.
#
grad_model = tf.keras.Model(
    inputs=owner_model.inputs,
    outputs=[conv_layer.output, owner_model.outputs[0]],
)


def get_head_layers(outer_model, sub_model):
    """Layers in outer_model that come after sub_model in the forward pass."""
    head, found = [], False
    for layer in outer_model.layers:
        if layer is sub_model:
            found = True
            continue
        if found:
            head.append(layer)
    return head


nested    = owner_model is not model
head_layers = get_head_layers(model, owner_model) if nested else []
print("Head layers after sub-model:", [l.name for l in head_layers])


# ── Image loader ─────────────────────────────────────────────────────────────
def load_image(img_path):
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Image not found: {img_path}")
    img = cv2.resize(img, (224, 224)).astype(np.float32) / 255.0
    return img


# ── Grad-CAM ─────────────────────────────────────────────────────────────────
def make_gradcam_heatmap(img_array):
    inputs = tf.cast(img_array, tf.float32)

    with tf.GradientTape() as tape:
        conv_outputs, sub_out = grad_model(inputs, training=False)
        tape.watch(conv_outputs)          # watch intermediate tensor explicitly

        # Run head layers (empty list when owner_model IS the outer model)
        x = sub_out
        for layer in head_layers:
            x = layer(x, training=False)
        final_pred = x

        loss = final_pred[:, 0]

    grads = tape.gradient(loss, conv_outputs)

    if grads is None:
        raise ValueError(
            "Gradients are None. The conv layer may not be connected to the "
            "output. Re-save the model with model.save() after rebuilding."
        )

    # Global-average-pool over spatial dims → shape (C,)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Weighted sum over channels → heatmap (H, W)
    heatmap = tf.reduce_sum(conv_outputs[0] * pooled_grads, axis=-1)

    # ReLU + normalise to [0, 1]
    heatmap = tf.maximum(heatmap, 0)
    max_val = tf.reduce_max(heatmap)
    if max_val > 0:
        heatmap = heatmap / max_val

    return heatmap.numpy()


# ── Overlay ───────────────────────────────────────────────────────────────────
def overlay_heatmap(img_path, heatmap, alpha=0.4):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (224, 224))
    h   = cv2.resize(heatmap, (224, 224))
    h   = cv2.applyColorMap(np.uint8(255 * h), cv2.COLORMAP_JET)
    return cv2.addWeighted(img, 1 - alpha, h, alpha, 0)


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":

    folder   = "dataset/test/autistic"
    img_name = os.listdir(folder)[0]
    img_path = os.path.join(folder, img_name)
    print("Using image:", img_path)

    img       = load_image(img_path)
    img_array = np.expand_dims(img, axis=0)          # (1, 224, 224, 3)

    # Prediction
    pred  = model.predict(img_array, verbose=0)[0][0]
    label = "Autistic" if pred < 0.5 else "Non-Autistic"
    print(f"Prediction score : {pred:.6e}")
    print(f"Predicted class  : {label}")

    # Grad-CAM
    heatmap = make_gradcam_heatmap(img_array)
    result  = overlay_heatmap(img_path, heatmap)

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
    plt.title(f"Grad-CAM  [{label}  {pred:.4f}]")
    plt.axis("off")

    plt.tight_layout()
    plt.savefig("gradcam_output.png", dpi=150)
    plt.show()
    print("Saved → gradcam_output.png")