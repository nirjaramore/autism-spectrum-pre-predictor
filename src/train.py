from preprocessing import get_data_generators
from model import build_model
import matplotlib.pyplot as plt
import os

# 🔹 Load data
train_data, test_data = get_data_generators()

# 🔹 Build model
model = build_model()

# 🔹 Train model
history = model.fit(
    train_data,
    validation_data=test_data,
    epochs=15
)

# 🔹 Create models folder if not exists
os.makedirs("models", exist_ok=True)

# 🔹 Save model
model.save("models/autism_model.h5")

print("✅ Model training completed and saved!")

# 🔹 Plot Accuracy
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.title("Accuracy Graph")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.show()

# 🔹 Plot Loss
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.title("Loss Graph")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.show()