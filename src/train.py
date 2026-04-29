import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.utils.class_weight import compute_class_weight

from preprocessing import get_data_generators
from model import build_model


# ── Load data ────────────────────────────────────────────────────────────────
train_data, test_data = get_data_generators()


# ── Compute Class Weights (VERY IMPORTANT) ───────────────────────────────────
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_data.classes),
    y=train_data.classes
)

class_weights = dict(enumerate(class_weights))
print("\n🔥 Class Weights:", class_weights)


# ── Callbacks ────────────────────────────────────────────────────────────────
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True
)

os.makedirs("models", exist_ok=True)

checkpoint = ModelCheckpoint(
    "models/best_model.h5",
    monitor='val_loss',
    save_best_only=True,
    verbose=1
)


# ════════════════════════════════════════════════════════════════════════════
# 🔥 TRAINING (ONLY PHASE 1 - BEST STRATEGY)
# ════════════════════════════════════════════════════════════════════════════

print("\n🔹 Training model (Frozen base model)\n")

model = build_model(fine_tune=False)

history = model.fit(
    train_data,
    validation_data=test_data,
    epochs=15,   # 🔥 increased epochs
    callbacks=[early_stop, checkpoint],
    class_weight=class_weights   # 🔥 KEY FIX
)


# ── Save final model ─────────────────────────────────────────────────────────
model.save("models/autism_model.h5")

print("✅ Model training completed and saved!")


# ── Plot Accuracy ────────────────────────────────────────────────────────────
plt.figure()
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.title("Accuracy Graph")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.grid()
plt.show()


# ── Plot Loss ────────────────────────────────────────────────────────────────
plt.figure()
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.title("Loss Graph")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.grid()
plt.show()