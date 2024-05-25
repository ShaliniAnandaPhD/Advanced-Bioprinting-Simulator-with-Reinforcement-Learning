# CellVitalityMonitor: Real-time monitoring of cell viability in bioprinting using deep learning
# Based on the paper "Real-time monitoring of cell viability in bioprinting using deep learning" (2023)

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from sklearn.model_selection import train_test_split
from skimage import io, transform

### Data Preparation ###

# Load fluorescence images and corresponding labels
# Data format: Images are stored in a directory with filenames indicating the label (viable or non-viable)
#   Example: "image_viable_001.tif", "image_nonviable_002.tif"
# Data size: 1000 images (500 viable, 500 non-viable)
image_dir = "fluorescence_images/"
image_files = os.listdir(image_dir)

images = []
labels = []
for file in image_files:
    image = io.imread(image_dir + file)
    images.append(image)
    if "viable" in file:
        labels.append(1)
    else:
        labels.append(0)

# Convert images and labels to numpy arrays
images = np.array(images)
labels = np.array(labels)

# Preprocess the images
images = images / 255.0  # Normalize pixel values to [0, 1]
images = transform.resize(images, (100, 100, 3))  # Resize images to a consistent size

# Split the data into training and testing sets
train_images, test_images, train_labels, test_labels = train_test_split(
    images, labels, test_size=0.2, random_state=42)

### Model Training ###

# Define the CNN architecture
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(100, 100, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(train_images, train_labels, epochs=10, batch_size=32, validation_split=0.1)

# Evaluate the model on the test set
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f"Test accuracy: {test_acc:.4f}")

# Save the trained model
model.save("cell_viability_model.h5")

### Real-time Monitoring ###

# Load the trained model
model = tf.keras.models.load_model("cell_viability_model.h5")

# Set up the fluorescence microscope and camera
microscope = FluorescenceMicroscope()
camera = microscope.get_camera()

# Set the monitoring parameters
monitoring_interval = 5  # Capture images every 5 seconds
monitoring_duration = 3600  # Monitor for 1 hour (3600 seconds)

# Start the bioprinting process
bioprinter = BioPrinter()
bioprinter.start_printing()

# Perform real-time monitoring
start_time = time.time()
while time.time() - start_time < monitoring_duration:
    # Capture a fluorescence image
    image = camera.capture_image()
    
    # Preprocess the image
    image = image / 255.0
    image = transform.resize(image, (100, 100, 3))
    image = np.expand_dims(image, axis=0)
    
    # Predict the cell viability
    viability = model.predict(image)[0][0]
    
    # Check if the viability is below the threshold
    if viability < 0.5:
        print(f"Warning: Low cell viability detected at {time.time() - start_time:.2f} seconds")
        
        # Perform corrective actions
        bioprinter.adjust_parameters()  # Adjust bioprinting parameters (e.g., temperature, pressure)
        bioprinter.add_nutrients()  # Add nutrients to the bioink
    
    # Wait for the next monitoring interval
    time.sleep(monitoring_interval)

# Stop the bioprinting process
bioprinter.stop_printing()

# Evaluate the overall cell viability
final_image = camera.capture_image()
final_image = final_image / 255.0
final_image = transform.resize(final_image, (100, 100, 3))
final_image = np.expand_dims(final_image, axis=0)
final_viability = model.predict(final_image)[0][0]
print(f"Final cell viability: {final_viability:.4f}")
