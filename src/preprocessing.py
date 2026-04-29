from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# 🔹 Paths
TRAIN_DIR = "dataset/train"
TEST_DIR = "dataset/test"

# 🔹 Image size and batch
IMG_SIZE = (224, 224)
BATCH_SIZE = 32


def get_data_generators():

    # 🔥 BALANCED AUGMENTATION (not too aggressive)
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,

        rotation_range=15,
        zoom_range=0.15,

        width_shift_range=0.1,
        height_shift_range=0.1,

        shear_range=0.1,

        horizontal_flip=True,
        fill_mode='nearest'
    )

    # 🔹 Testing Data Generator (NO augmentation)
    test_datagen = ImageDataGenerator(
        rescale=1.0 / 255
    )

    # 🔥 FORCE CLASS ORDER (VERY IMPORTANT FIX)
    class_names = ['autistic', 'non_autistic']

    # 🔹 Load training data
    train_data = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='binary',
        classes=class_names,   # ✅ FORCE ORDER
        shuffle=True
    )

    # 🔹 Load testing data
    test_data = test_datagen.flow_from_directory(
        TEST_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='binary',
        classes=class_names,   # ✅ SAME ORDER
        shuffle=False
    )

    # 🔥 DEBUG PRINTS (VERY IMPORTANT)
    print("\n📊 CLASS MAPPING:")
    print(train_data.class_indices)

    print("\n📊 DATA INFO:")
    print("Training samples:", train_data.samples)
    print("Testing samples:", test_data.samples)

    return train_data, test_data


# 🔹 Run only when file is executed directly
if __name__ == "__main__":
    train_data, test_data = get_data_generators()