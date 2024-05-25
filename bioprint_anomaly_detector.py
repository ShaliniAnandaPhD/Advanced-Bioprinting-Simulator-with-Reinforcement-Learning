# BioPrintAnomalyDetector: Unsupervised anomaly detection in bioprinted structures using deep learning
# Based on the paper "Unsupervised anomaly detection in bioprinted structures using deep learning" (2023)

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Lambda, Reshape, Conv2D, MaxPooling2D, UpSampling2D
from tensorflow.keras.models import Model
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler

### Data Preparation ###

# Load bioprinted structure images
# Data format: 3D grayscale images of bioprinted structures (e.g., .npy files)
# Data size: 1000 images, each with dimensions (100, 100, 100)
data_folder = "bioprinted_structures/"
image_files = os.listdir(data_folder)
images = []
for file in image_files:
    image = np.load(os.path.join(data_folder, file))
    images.append(image)

# Convert images to numpy array
images = np.array(images)

# Preprocess the images
images = images.astype("float32") / 255.0
images = np.reshape(images, (len(images), 100, 100, 100, 1))

# Split data into training and testing sets
train_size = int(0.8 * len(images))
train_images = images[:train_size]
test_images = images[train_size:]

### Variational Autoencoder ###

# Define the encoder network
def create_encoder(input_shape):
    inputs = Input(shape=input_shape)
    x = Conv2D(32, (3, 3, 3), activation="relu", padding="same")(inputs)
    x = MaxPooling2D((2, 2, 2), padding="same")(x)
    x = Conv2D(64, (3, 3, 3), activation="relu", padding="same")(x)
    x = MaxPooling2D((2, 2, 2), padding="same")(x)
    x = Conv2D(128, (3, 3, 3), activation="relu", padding="same")(x)
    x = MaxPooling2D((2, 2, 2), padding="same")(x)
    x = Reshape(target_shape=(8*8*8*128,))(x)
    x = Dense(256, activation="relu")(x)
    z_mean = Dense(64, name="z_mean")(x)
    z_log_var = Dense(64, name="z_log_var")(x)
    z = Lambda(sampling, name="z")([z_mean, z_log_var])
    encoder = Model(inputs, [z_mean, z_log_var, z], name="encoder")
    return encoder

# Define the decoder network
def create_decoder(latent_dim):
    latent_inputs = Input(shape=(latent_dim,))
    x = Dense(8*8*8*128, activation="relu")(latent_inputs)
    x = Reshape((8, 8, 8, 128))(x)
    x = Conv2D(128, (3, 3, 3), activation="relu", padding="same")(x)
    x = UpSampling2D((2, 2, 2))(x)
    x = Conv2D(64, (3, 3, 3), activation="relu", padding="same")(x)
    x = UpSampling2D((2, 2, 2))(x)
    x = Conv2D(32, (3, 3, 3), activation="relu", padding="same")(x)
    x = UpSampling2D((2, 2, 2))(x)
    outputs = Conv2D(1, (3, 3, 3), activation="sigmoid", padding="same")(x)
    decoder = Model(latent_inputs, outputs, name="decoder")
    return decoder

# Define the variational autoencoder
def create_vae(input_shape, latent_dim):
    encoder = create_encoder(input_shape)
    decoder = create_decoder(latent_dim)
    inputs = Input(shape=input_shape)
    z_mean, z_log_var, z = encoder(inputs)
    outputs = decoder(z)
    vae = Model(inputs, outputs, name="vae")
    
    reconstruction_loss = tf.reduce_mean(tf.square(inputs - outputs), axis=(1, 2, 3))
    kl_loss = -0.5 * tf.reduce_mean(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=-1)
    vae_loss = tf.reduce_mean(reconstruction_loss + kl_loss)
    vae.add_loss(vae_loss)
    
    return vae

# Define the sampling function
def sampling(args):
    z_mean, z_log_var = args
    epsilon = tf.keras.backend.random_normal(shape=(tf.keras.backend.shape(z_mean)[0], 64))
    return z_mean + tf.exp(0.5 * z_log_var) * epsilon

# Create and train the variational autoencoder
input_shape = (100, 100, 100, 1)
latent_dim = 64
vae = create_vae(input_shape, latent_dim)
vae.compile(optimizer="adam")
vae.fit(train_images, epochs=50, batch_size=32, validation_data=(test_images, None))

# Get the encoder model
encoder = vae.get_layer("encoder")

### One-Class SVM ###

# Extract latent features using the encoder
train_features = encoder.predict(train_images)[0]
test_features = encoder.predict(test_images)[0]

# Scale the features
scaler = StandardScaler()
train_features_scaled = scaler.fit_transform(train_features)
test_features_scaled = scaler.transform(test_features)

# Train the one-class SVM
svm = OneClassSVM(kernel="rbf", gamma="scale", nu=0.05)
svm.fit(train_features_scaled)

# Predict anomalies
train_anomalies = svm.predict(train_features_scaled)
test_anomalies = svm.predict(test_features_scaled)

### Anomaly Visualization ###

# Visualize anomalies in the test set
anomaly_indices = np.where(test_anomalies == -1)[0]
normal_indices = np.where(test_anomalies == 1)[0]

# Visualize anomalous structures
for idx in anomaly_indices[:5]:
    anomaly_image = test_images[idx]
    plt.imshow(np.squeeze(anomaly_image[:, :, 50]), cmap="gray")
    plt.title(f"Anomalous Structure {idx}")
    plt.show()

# Visualize normal structures
for idx in normal_indices[:5]:
    normal_image = test_images[idx]
    plt.imshow(np.squeeze(normal_image[:, :, 50]), cmap="gray")
    plt.title(f"Normal Structure {idx}")
    plt.show()
