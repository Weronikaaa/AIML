import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras import regularizers
from PIL import ImageFile
from collections import Counter

# Pozwól na ładowanie uszkodzonych obrazów
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Ścieżki do danych
train_dir = r'C:\Users\Weron\Desktop\AIML\train'
test_dir = r'C:\Users\Weron\Desktop\AIML\valid'

# Parametry
IMG_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = 30  # Zwiększona liczba epok

# Rozszerzona augmentacja danych
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.3,
    height_shift_range=0.3,
    shear_range=0.3,
    zoom_range=0.3,
    brightness_range=[0.6,1.4],
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='nearest',
    channel_shift_range=0.2  # Nowy parametr
)

test_datagen = ImageDataGenerator(rescale=1./255)

# Ładowanie danych
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=True
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

# Analiza rozkładu klas
print("Rozkład klas treningowych:", Counter(train_generator.classes))
print("Rozkład klas testowych:", Counter(test_generator.classes))

# Pobierz liczbę klas
num_classes = len(train_generator.class_indices)
class_names = list(train_generator.class_indices.keys())
print(f"Liczba klas: {num_classes}")
print(f"Nazwy klas: {class_names}")

# Optymalizowana architektura modelu
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
          kernel_regularizer=regularizers.l2(0.001)),
    BatchNormalization(),
    MaxPooling2D(2, 2),
    Dropout(0.4),
    
    Conv2D(64, (3, 3), activation='relu', kernel_regularizer=regularizers.l2(0.001)),
    BatchNormalization(),
    MaxPooling2D(2, 2),
    Dropout(0.4),
    
    Conv2D(128, (3, 3), activation='relu', kernel_regularizer=regularizers.l2(0.001)),
    BatchNormalization(),
    MaxPooling2D(2, 2),
    Dropout(0.5),
    
    Conv2D(256, (3, 3), activation='relu', kernel_regularizer=regularizers.l2(0.001)),
    BatchNormalization(),
    MaxPooling2D(2, 2),
    Dropout(0.5),
    
    Flatten(),
    Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
    BatchNormalization(),
    Dropout(0.6),
    
    Dense(num_classes, activation='softmax')
])

# Optymalizacja hiperparametrów
optimizer = Adam(learning_rate=0.0001)  # Zmniejszone learning rate
model.compile(
    optimizer=optimizer,
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Callbacks
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=7,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.2,
    patience=3,
    min_lr=1e-7,
    verbose=1
)

# Trenowanie
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=test_generator,
    validation_steps=test_generator.samples // BATCH_SIZE,
    callbacks=[early_stopping, reduce_lr],
    verbose=1
)

# Ewaluacja
test_loss, test_acc = model.evaluate(test_generator)
print(f"\nFinal Test Accuracy: {test_acc:.4f}")
print(f"Best Validation Accuracy: {max(history.history['val_accuracy']):.4f}")

# Wykresy
plt.figure(figsize=(15, 6))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy', pad=10)
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss', pad=10)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()

# Zapisz model
model.save('optimized_tomato_model.h5')
print("Model zapisany jako 'optimized_tomato_model.h5'")