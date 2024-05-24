# BioPrintQC: Real-time quality control in bioprinting using deep learning
# Based on the paper "Real-time quality control in bioprinting using deep learning" (2023)

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from skimage import io, transform

### Data Preparation ###

# Load labeled image dataset
# Data format: folder of images with filename indicating quality score (0-10)
# e.g. "print_quality_7.jpg" for an image with quality score 7
# Data size: 10,000 labeled images
image_folder = "bioprint_images/"
image_files = os.listdir(image_folder)

# Load and preprocess images
images = []
labels = []
for file in image_files:
    # Load image
    image = io.imread(image_folder + file)
    
    # Resize to 256x256
    image = transform.resize(image, (256, 256))
    
    # Convert to RGB if grayscale
    if len(image.shape) == 2:
        image = np.stack((image,)*3, axis=-1)
    
    images.append(image)
    
    # Extract label from filename
    label = int(file.split("_")[-1].split(".")[0])
    labels.append(label)

# Convert to numpy arrays
X = np.array(images)
y = np.array(labels)

# Normalize pixel values to [0, 1]
X = X / 255.0

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

### Model Training ###

# Define CNN architecture
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(256, 256, 3)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1)
])

# Compile model with mean squared error loss and Adam optimizer
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Train model for 50 epochs with 20% validation split
history = model.fit(X_train, y_train, epochs=50, validation_split=0.2)

# Evaluate model on test set
test_loss, test_mae = model.evaluate(X_test, y_test)
print(f"Test loss: {test_loss:.4f}, Test MAE: {test_mae:.4f}")

# Save trained model
model.save("bioprint_qc_model.h5")

### Real-time Quality Assessment ###

# Load trained model
model = load_model("bioprint_qc_model.h5")

# Initialize camera for real-time imaging
camera = cv2.VideoCapture(0)

while True:
    # Capture frame from camera
    ret, frame = camera.read()
    
    # Preprocess frame
    frame = cv2.resize(frame, (256, 256))
    frame = frame / 255.0
    frame = np.expand_dims(frame, axis=0)
    
    # Predict quality score
    quality_score = model.predict(frame)[0][0]
    
    # Display frame with quality score
    cv2.putText(frame, f"Quality Score: {quality_score:.2f}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("BioPrint Quality Assessment", frame)
    
    # Break loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
        
# Release camera and close windows
camera.release()
cv2.destroyAllWindows()
