import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Paths and parameters
DATASET_PATH = r"C:\Users\Weron\Desktop\AIML\dataset"  # główny folder zawierający 'train' i 'valid'
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 5  # Na potrzeby testowe

# Data preparation – bez podziału na subsety
datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True,
)

# Treningowy generator
train_gen = datagen.flow_from_directory(
    os.path.join(DATASET_PATH, "train"),
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

# Walidacyjny generator
val_gen = datagen.flow_from_directory(
    os.path.join(DATASET_PATH, "valid"),
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

# Helper: plot confusion matrix
def plot_confusion_matrix(cm, class_names, title):
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names, cmap='Blues')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(f"{title.replace(' ', '_').lower()}.png")
    plt.close()

# Helper: train model
def train_model(model, name):
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(train_gen, validation_data=val_gen, epochs=EPOCHS, verbose=1)
    val_gen.reset()
    preds = model.predict(val_gen)
    y_pred = np.argmax(preds, axis=1)
    y_true = val_gen.classes
    print(f"\n{name} Classification Report:\n")
    print(classification_report(y_true, y_pred, target_names=val_gen.class_indices.keys()))
    cm = confusion_matrix(y_true, y_pred)
    plot_confusion_matrix(cm, list(val_gen.class_indices.keys()), f"{name} Confusion Matrix")
    return model

# 1. CNN from scratch
cnn_model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(train_gen.num_classes, activation='softmax')
])
train_model(cnn_model, "CNN From Scratch")

# 2. MobileNetV2
mobilenet_base = tf.keras.applications.MobileNetV2(input_shape=(IMG_SIZE, IMG_SIZE, 3),
                                                    include_top=False,
                                                    weights='imagenet')
mobilenet_base.trainable = False
mobilenet_model = tf.keras.Sequential([
    mobilenet_base,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(train_gen.num_classes, activation='softmax')
])
train_model(mobilenet_model, "MobileNetV2")

# 3. ResNet50
resnet_base = tf.keras.applications.ResNet50(input_shape=(IMG_SIZE, IMG_SIZE, 3),
                                             include_top=False,
                                             weights='imagenet')
resnet_base.trainable = False
resnet_model = tf.keras.Sequential([
    resnet_base,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(train_gen.num_classes, activation='softmax')
])
train_model(resnet_model, "ResNet50")
