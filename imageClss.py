import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define image size and batch size
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# Image data generator with augmentation
datagen = ImageDataGenerator(
    rescale=1./255,   # Normalize pixel values
    validation_split=0.2,  # 80% train, 20% validation
    rotation_range=20,  # Augmentations
    zoom_range=0.2,
    horizontal_flip=True
)

# Load training data
train_data = datagen.flow_from_directory(
    "dataset/",
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    subset="training"
)

# Load validation data
val_data = datagen.flow_from_directory(
    "dataset/",
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    subset="validation"
)


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# Define CNN Model
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(224, 224, 3)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')  # Binary Classification
])

# Compile Model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the Model
model.fit(train_data, validation_data=val_data, epochs=10)

# Save the Model
model.save("flatfoot_model.h5")


import numpy as np
import cv2

# Load trained model
model = tf.keras.models.load_model("flatfoot_model.h5")

# Function to preprocess a new image
def preprocess_image(image_path, img_size=(224, 224)):
    img = cv2.imread(image_path)
    img = cv2.resize(img, img_size)
    img = img / 255.0  # Normalize
    return np.expand_dims(img, axis=0)  # Add batch dimension

# Test with a new image
img_path = "test_image2.jpg"  # Change to your test image path
img = preprocess_image(img_path)
prediction = model.predict(img)[0][0]

# Output result
if prediction > 0.5:
    print("Flatfoot Detected")
else:
    print("Normal Foot")
