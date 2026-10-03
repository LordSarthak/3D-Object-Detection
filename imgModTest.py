import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Load trained model
model = tf.keras.models.load_model("flatfoot_model.h5")

# Function to preprocess a new image
def preprocess_image(image_path, img_size=(224, 224)):
    img = cv2.imread(image_path)
    img = cv2.resize(img, img_size)
    img = img / 255.0  # Normalize
    return np.expand_dims(img, axis=0)  # Add batch dimension

# Test with a new image
img_path = "xray_test.jpg"  # Change to your test image path
img = preprocess_image(img_path)
prediction = model.predict(img)[0][0]

# Output result
if prediction > 0.5:
    print("Flatfoot Detected")
else:
    print("Normal Foot")