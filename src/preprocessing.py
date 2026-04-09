from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Paths
TRAIN_DIR = "dataset/train"
TEST_DIR = "dataset/test"

# Image size and batch
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# 🔹 Training Data Generator (with augmentation)
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True
)

# 🔹 Testing Data Generator (NO augmentation)
test_datagen = ImageDataGenerator(
    rescale=1./255
)

# 🔹 Load training data
train_data = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary'
)

# 🔹 Load testing data
test_data = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary'
)

# Debug output
print("Class indices:", train_data.class_indices)
print("Training samples:", train_data.samples)
print("Testing samples:", test_data.samples)