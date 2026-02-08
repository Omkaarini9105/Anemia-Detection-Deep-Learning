# ===========================
# Anemia Detection from Blood Cell Images
# ===========================

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# ===========================
# PARAMETERS
# ===========================
IMG_SIZE = (150, 150)
BATCH_SIZE = 32
EPOCHS = 10
USE_TRANSFER_LEARNING = False  # Set True to use MobileNetV2

# ===========================
# DATASET PATH (Kaggle input)
# ===========================
DATASET_PATH = "/kaggle/input/anemia-dataset"  # Replace with your Kaggle dataset path
TRAIN_DIR = os.path.join(DATASET_PATH, "train")
VAL_DIR   = os.path.join(DATASET_PATH, "val")
TEST_DIR  = os.path.join(DATASET_PATH, "test")

# ===========================
# DATA GENERATORS
# ===========================
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    zoom_range=0.1,
    horizontal_flip=True
)

val_test_datagen = ImageDataGenerator(rescale=1./255)

train_data = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary'
)

val_data = val_test_datagen.flow_from_directory(
    VAL_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary'
)

test_data = val_test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=False
)

# ===========================
# MODEL DEFINITION
# ===========================
if USE_TRANSFER_LEARNING:
    base_model = tf.keras.applications.MobileNetV2(weights='imagenet', include_top=False, input_shape=(150,150,3))
    base_model.trainable = False

    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
else:
    model = Sequential([
        Conv2D(32, (3,3), activation='relu', input_shape=(150,150,3)),
        MaxPooling2D(2,2),

        Conv2D(64, (3,3), activation='relu'),
        MaxPooling2D(2,2),

        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.summary()

# ===========================
# TRAIN MODEL
# ===========================
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS
)

# ===========================
# EVALUATE MODEL
# ===========================
test_loss, test_acc = model.evaluate(test_data)
print("Test Accuracy:", test_acc)

# ===========================
# PLOT ACCURACY & LOSS
# ===========================
plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.legend()
plt.title("Accuracy")

plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.legend()
plt.title("Loss")

plt.show()

# ===========================
# CONFUSION MATRIX & REPORT
# ===========================
y_pred = model.predict(test_data)
y_pred_classes = (y_pred > 0.5).astype(int).reshape(-1)
y_true = test_data.classes

cm = confusion_matrix(y_true, y_pred_classes)
print("Confusion Matrix:\n", cm)
print("\nClassification Report:\n", classification_report(y_true, y_pred_classes))

plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
            xticklabels=test_data.class_indices, yticklabels=test_data.class_indices)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# ===========================
# PREDICT SINGLE IMAGE
# ===========================
def predict_single_image(img_path):
    img = load_img(img_path, target_size=IMG_SIZE)
    img_array = img_to_array(img)/255.0
    img_array = np.expand_dims(img_array, axis=0)
    pred = model.predict(img_array)
    if pred[0][0] > 0.5:
        print(f"{img_path} -> Prediction: Anemic")
    else:
        print(f"{img_path} -> Prediction: Normal")

# Example usage (replace with any image path)
# predict_single_image("/kaggle/input/anemia-dataset/test/Normal/sample1.jpg")
