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
DATASET_PATH = r"C:\Users\Weron\Desktop\AIML\dataset"
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 2  # mniejsze na czas testów
USE_PARTIAL_DATA = True  # ustaw na False do pełnego treningu

# Data preparation
datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True,
)

# Funkcja: wczytuje tylko część danych z generatora
def shrink_generator(generator, limit):
    x_list, y_list = [], []
    count = 0
    for x_batch, y_batch in generator:
        x_list.append(x_batch)
        y_list.append(y_batch)
        count += len(x_batch)
        if count >= limit:
            break
    x_array = np.concatenate(x_list)[:limit]
    y_array = np.concatenate(y_list)[:limit]
    return x_array, y_array

# Generatory
train_gen_full = datagen.flow_from_directory(
    os.path.join(DATASET_PATH, "train"),
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=True
)

val_gen_full = datagen.flow_from_directory(
    os.path.join(DATASET_PATH, "valid"),
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

# Użyj tylko części danych
if USE_PARTIAL_DATA:
    train_data, train_labels = shrink_generator(train_gen_full, train_gen_full.samples // 5)
    val_data, val_labels = shrink_generator(val_gen_full, val_gen_full.samples // 5)
else:
    train_data = train_gen_full
    val_data = val_gen_full

# Rysuj confusion matrix
def plot_confusion_matrix(cm, class_names, title):
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names, cmap='Blues')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(f"{title.replace(' ', '_').lower()}.png")
    plt.close()

# Rysuj wykresy uczenia się
def plot_training_history(history, name):
    plt.figure(figsize=(12, 5))

    # Loss
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    if 'val_loss' in history.history:
        plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title(f'{name} Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Train Acc')
    if 'val_accuracy' in history.history:
        plt.plot(history.history['val_accuracy'], label='Val Acc')
    plt.title(f'{name} Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.savefig(f"{name.replace(' ', '_').lower()}_training.png")
    plt.close()

# Trenuj model i twórz raport
def train_model(model, name):
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    if USE_PARTIAL_DATA:
        history = model.fit(train_data, train_labels, validation_data=(val_data, val_labels), epochs=EPOCHS, verbose=1)
        preds = model.predict(val_data)
        y_true = np.argmax(val_labels, axis=1)
    else:
        history = model.fit(train_data, validation_data=val_data, epochs=EPOCHS, verbose=1)
        val_data.reset()
        preds = model.predict(val_data)
        y_true = val_data.classes

    y_pred = np.argmax(preds, axis=1)
    print(f"\n{name} Classification Report:\n")
    print(classification_report(y_true, y_pred, target_names=val_gen_full.class_indices.keys()))
    cm = confusion_matrix(y_true, y_pred)
    plot_confusion_matrix(cm, list(val_gen_full.class_indices.keys()), f"{name} Confusion Matrix")
    plot_training_history(history, name)
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
    tf.keras.layers.Dense(train_gen_full.num_classes, activation='softmax')
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
    tf.keras.layers.Dense(train_gen_full.num_classes, activation='softmax')
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
    tf.keras.layers.Dense(train_gen_full.num_classes, activation='softmax')
])
train_model(resnet_model, "ResNet50")
