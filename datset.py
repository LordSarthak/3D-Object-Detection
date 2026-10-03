import cv2
import os
import numpy as np
from tensorflow.keras.utils import to_categorical

IMG_SIZE = 64  # Resize all images to 64x64 pixels

data = []
labels = []

for category in ["Flatfoot", "Normal"]:
    path = os.path.join("dataset", category)
    label = 0 if category == "Flatfoot" else 1  # Assign labels: 0=Flatfoot, 1=Normal
    
    for img_name in os.listdir(path):
        img_path = os.path.join(path, img_name)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Read as grayscale
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))  # Resize
        data.append(img)
        labels.append(label)

# Convert to NumPy arrays
data = np.array(data).reshape(-1, IMG_SIZE, IMG_SIZE, 1)  # Reshape for CNN input
data = data / 255.0  # Normalize to [0,1]
labels = to_categorical(labels, num_classes=2)  # One-hot encoding

# Save dataset
np.save("dataset_x.npy", data)
np.save("dataset_y.npy", labels)

print("Dataset prepared successfully!")
