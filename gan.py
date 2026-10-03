import tensorflow as tf
from tensorflow.keras.layers import Dense, Reshape, Flatten, LeakyReLU, Conv2D, Conv2DTranspose, BatchNormalization
from tensorflow.keras.models import Sequential
import numpy as np
import matplotlib.pyplot as plt

# Image shape (adjust as per dataset)
IMG_SHAPE = (64, 64, 1)  # Grayscale images of 64x64
LATENT_DIM = 100  # Size of the random noise vector

# Generator Model
def build_generator():
    model = Sequential([
        Dense(8*8*256, input_dim=LATENT_DIM),
        LeakyReLU(alpha=0.2),
        Reshape((8, 8, 256)),
        Conv2DTranspose(128, kernel_size=4, strides=2, padding="same"),
        BatchNormalization(),
        LeakyReLU(alpha=0.2),
        Conv2DTranspose(64, kernel_size=4, strides=2, padding="same"),
        BatchNormalization(),
        LeakyReLU(alpha=0.2),
        Conv2DTranspose(1, kernel_size=4, strides=2, padding="same", activation="tanh")  # Output grayscale image
    ])
    return model

# Discriminator Model
def build_discriminator():
    model = Sequential([
        Conv2D(64, kernel_size=4, strides=2, padding="same", input_shape=IMG_SHAPE),
        LeakyReLU(alpha=0.2),
        Conv2D(128, kernel_size=4, strides=2, padding="same"),
        LeakyReLU(alpha=0.2),
        Flatten(),
        Dense(1, activation="sigmoid")  # Binary classification (real/fake)
    ])
    return model

# Compile Discriminator
discriminator = build_discriminator()
discriminator.compile(loss="binary_crossentropy", optimizer=tf.keras.optimizers.Adam(0.0002, 0.5), metrics=["accuracy"])

# Build GAN
generator = build_generator()
discriminator.trainable = False  # Freeze discriminator in GAN training
gan_input = tf.keras.Input(shape=(LATENT_DIM,))
gan_output = discriminator(generator(gan_input))
gan = tf.keras.models.Model(gan_input, gan_output)
gan.compile(loss="binary_crossentropy", optimizer=tf.keras.optimizers.Adam(0.0002, 0.5))

# Training Function
def train_gan(dataset, epochs=5000, batch_size=32):
    real_labels = np.ones((batch_size, 1))
    fake_labels = np.zeros((batch_size, 1))

    for epoch in range(epochs):
        # Train Discriminator
        idx = np.random.randint(0, dataset.shape[0], batch_size)
        real_images = dataset[idx]
        noise = np.random.normal(0, 1, (batch_size, LATENT_DIM))
        fake_images = generator.predict(noise)
        
        d_loss_real = discriminator.train_on_batch(real_images, real_labels)
        d_loss_fake = discriminator.train_on_batch(fake_images, fake_labels)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        # Train Generator
        noise = np.random.normal(0, 1, (batch_size, LATENT_DIM))
        g_loss = gan.train_on_batch(noise, real_labels)

        # Print progress
        if epoch % 100 == 0:
            print(f"Epoch {epoch}: D Loss = {d_loss[0]:.4f}, G Loss = {g_loss:.4f}")
            sample_images(generator)

# Function to generate and display sample images
def sample_images(generator, n=5):
    noise = np.random.normal(0, 1, (n, LATENT_DIM))
    gen_images = generator.predict(noise)
    gen_images = (gen_images + 1) / 2.0  # Rescale to [0,1]
    
    fig, axes = plt.subplots(1, n, figsize=(10, 2))
    for i in range(n):
        axes[i].imshow(gen_images[i].squeeze(), cmap='gray')
        axes[i].axis("off")
    plt.show()

# Load dataset and train (Replace 'your_dataset.npy' with actual dataset)
dataset = np.load("dataset_x.npy")  # Assuming shape (num_samples, 64, 64, 1)
dataset = (dataset - 127.5) / 127.5  # Normalize to [-1, 1]
train_gan(dataset)
