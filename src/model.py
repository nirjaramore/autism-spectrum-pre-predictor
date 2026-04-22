from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam

# Image size
IMG_SIZE = (224, 224)

def build_model():
    # 🔹 Load pretrained MobileNetV2
    base_model = MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights='imagenet'
    )

    # 🔹 Freeze most layers
    for layer in base_model.layers[:-30]:   # freeze all except last 30 layers
        layer.trainable = False

    # 🔹 Unfreeze last layers (fine-tuning)
    for layer in base_model.layers[-30:]:
        layer.trainable = True

    # 🔹 Build model
    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),

        Dense(256, activation='relu'),
        Dropout(0.5),   # prevents overfitting

        Dense(128, activation='relu'),
        Dropout(0.3),

        Dense(1, activation='sigmoid')  # Binary classification
    ])

    # 🔹 Compile model
    model.compile(
        optimizer=Adam(learning_rate=0.0001),  # lower LR for fine-tuning
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    return model


if __name__ == "__main__":
    model = build_model()
    model.summary()