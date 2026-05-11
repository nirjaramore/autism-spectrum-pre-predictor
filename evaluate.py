import numpy as np
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

model = load_model("model/cnn_model.h5")

IMG_SIZE = 128

def predict(img_path):
    img = load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
    img = img_to_array(img) / 255.0
    img = np.expand_dims(img, axis=0)

    pred = model.predict(img)[0][0]
    return 1 if pred > 0.5 else 0   # 1 = non_autistic, 0 = autistic

y_true = []
y_pred = []

for label, folder in enumerate(["autistic", "non_autistic"]):
    path = f"dataset/test/{folder}"
    for file in os.listdir(path):
        img_path = os.path.join(path, file)
        y_true.append(label)
        y_pred.append(predict(img_path))

from sklearn.metrics import classification_report, confusion_matrix

print(classification_report(y_true, y_pred))
print(confusion_matrix(y_true, y_pred))