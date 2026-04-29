from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization, Input
from tensorflow.keras.optimizers import Adam

# Image size
IMG_SIZE = (224, 224)


def build_model(fine_tune=False):

    # 🔹 Input layer
    inputs = Input(shape=(224, 224, 3))

    # 🔹 Load pretrained MobileNetV2
    base_model = MobileNetV2(
        include_top=False,
        weights='imagenet',
        input_tensor=inputs
    )

    # 🔥 PHASE CONTROL (FIXED)
    if not fine_tune:
        # Phase 1 → Freeze ALL layers
        for layer in base_model.layers:
            layer.trainable = False
        learning_rate = 0.0001   # faster learning for head
    else:
        # Phase 2 → Unfreeze ONLY last 10 layers (reduced from 20)
        for layer in base_model.layers[:-10]:
            layer.trainable = False
        for layer in base_model.layers[-10:]:
            layer.trainable = True
        learning_rate = 0.000005   # 🔥 very low LR for safe fine-tuning

    # 🔹 Custom head
    x = base_model.output
    x = GlobalAveragePooling2D()(x)

    x = BatchNormalization()(x)

    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)

    x = Dense(128, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)

    outputs = Dense(1, activation='sigmoid')(x)

    # 🔹 Final model
    model = Model(inputs=inputs, outputs=outputs)

    # 🔹 Compile model
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    return model


# ── Test model ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("🔹 Phase 1 Model (Frozen Base)")
    model = build_model(fine_tune=False)
    model.summary()

    print("\n🔹 Phase 2 Model (Fine-Tuning)")
    model = build_model(fine_tune=True)
    model.summary()