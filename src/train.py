import os
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from preprocessing import get_data_generators
from model import build_model


# ── Load data ────────────────────────────────────────────────────────────────
train_data, test_data = get_data_generators()


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
# 🔥 PHASE 1: Train only top layers (freeze base model)
# ════════════════════════════════════════════════════════════════════════════

print("\n🔹 Phase 1: Training classifier (Frozen base model)\n")

model = build_model(fine_tune=False)

history_phase1 = model.fit(
    train_data,
    validation_data=test_data,
    epochs=7,   # 🔥 short training
    callbacks=[early_stop, checkpoint]
)


# ════════════════════════════════════════════════════════════════════════════
# 🔥 PHASE 2: Fine-tuning last layers
# ════════════════════════════════════════════════════════════════════════════

print("\n🔹 Phase 2: Fine-tuning model\n")

model = build_model(fine_tune=True)

history_phase2 = model.fit(
    train_data,
    validation_data=test_data,
    epochs=13,   # remaining epochs
    callbacks=[early_stop, checkpoint]
)


# ── Save final model ─────────────────────────────────────────────────────────
model.save("models/autism_model.h5")

print("✅ Model training completed and saved!")


# ════════════════════════════════════════════════════════════════════════════
# 🔹 Combine histories for plotting
# ════════════════════════════════════════════════════════════════════════════

acc = history_phase1.history['accuracy'] + history_phase2.history['accuracy']
val_acc = history_phase1.history['val_accuracy'] + history_phase2.history['val_accuracy']

loss = history_phase1.history['loss'] + history_phase2.history['loss']
val_loss = history_phase1.history['val_loss'] + history_phase2.history['val_loss']


# ── Plot Accuracy ────────────────────────────────────────────────────────────
plt.figure()
plt.plot(acc, label='Train Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.legend()
plt.title("Accuracy Graph")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.grid()
plt.show()


# ── Plot Loss ────────────────────────────────────────────────────────────────
plt.figure()
plt.plot(loss, label='Train Loss')
plt.plot(val_loss, label='Validation Loss')
plt.legend()
plt.title("Loss Graph")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.grid()
plt.show()