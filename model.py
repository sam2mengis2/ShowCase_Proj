#Core
import os

# import cv2

# Visualization
# import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# import seaborn as sns

# TensorFlow / Keras
import tensorflow as tf
from PIL import Image

# Evaluation
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow import keras
from tensorflow.keras import layers, models
#
# Transfer learning backbones
from tensorflow.keras.applications import DenseNet121, EfficientNetB0, ResNet50
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import (
    BatchNormalization,
    Conv2D,
    Dense,
    Dropout,
    Flatten,
    GlobalAveragePooling2D,
    MaxPooling2D,
)
from tensorflow.keras.losses import BinaryCrossentropy, CategoricalCrossentropy
from tensorflow.keras.metrics import AUC, Precision, Recall
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam

# Data preprocessing & augmentation
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ── Dataset paths ──────────────────────────────────────────────────────────────
train_dir = "/home/ygg/projects/showcase/Training"
test_dir = "/home/ygg/projects/showcase/Testing"


def build_dataframe(data_dir):
    filepaths = []
    labels = []
    for label in os.listdir(data_dir):
        label_folder = os.path.join(data_dir, label)
        if os.path.isdir(label_folder):
            for image in os.listdir(label_folder):
                if image.endswith((".jpg", ".jpeg", ".png")):
                    filepaths.append(os.path.join(label_folder, image))
                    labels.append(label)
    return pd.DataFrame({"filepath": filepaths, "label": labels})


train_df = build_dataframe(train_dir)
test_df = build_dataframe(test_dir)

print("Training set:")
print(train_df["label"].value_counts())
print(f"\nTotal training images: {len(train_df)}")

print("\nTesting set:")
print(test_df["label"].value_counts())
print(f"\nTotal testing images: {len(test_df)}")


# ── Data generators ────────────────────────────────────────────────────────────
# Augmentation for training, just normalisation for testing
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    zoom_range=0.1,
)

test_datagen = ImageDataGenerator(rescale=1.0 / 255)

train_generator = train_datagen.flow_from_dataframe(
    dataframe=train_df,
    x_col="filepath",
    y_col="label",
    target_size=(224, 224),
    batch_size=16,
    class_mode="categorical",
)

test_generator = test_datagen.flow_from_dataframe(
    dataframe=test_df,
    x_col="filepath",
    y_col="label",
    target_size=(224, 224),
    batch_size=16,
    class_mode="categorical",
    shuffle=False,
)

print(train_generator.class_indices)


# ── Model definition ───────────────────────────────────────────────────────────
def generate_model():
    model = tf.keras.Sequential(
        [
            # 32 filters, 3×3 kernel, ReLU activation
            tf.keras.layers.Conv2D(32, kernel_size=3, activation="relu"),
            tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(1, 1)),
            tf.keras.layers.Conv2D(64, kernel_size=3, activation="relu"),
            tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(1024, activation="relu"),
            tf.keras.layers.Dense(4, activation="softmax"),
        ]
    )
    return model


# ── Training ───────────────────────────────────────────────────────────────────
model = generate_model()

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

model.fit(train_generator, validation_data=test_generator, epochs=5)

model.save("models/brain_tumor_model.keras")
