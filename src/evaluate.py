from preprocessing import get_data_generators
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

# 🔹 Load data
train_data, test_data = get_data_generators()

# 🔹 Load trained model
model = load_model("models/autism_model.h5")

# 🔹 Get predictions
predictions = model.predict(test_data)
predicted_classes = (predictions > 0.5).astype("int32")

# 🔹 True labels
true_classes = test_data.classes

# 🔹 Confusion Matrix
cm = confusion_matrix(true_classes, predicted_classes)

print("\nConfusion Matrix:")
print(cm)

# 🔹 Classification Report
print("\nClassification Report:")
print(classification_report(true_classes, predicted_classes, target_names=["Autistic", "Non-Autistic"]))