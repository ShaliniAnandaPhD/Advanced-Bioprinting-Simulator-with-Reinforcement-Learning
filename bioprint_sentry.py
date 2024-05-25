# BioPrintSentry: Error detection and recovery in bioprinting using deep learning
# Based on the paper "Error detection and recovery in bioprinting using deep learning" (2023)

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Flatten, Dense, Reshape
from sklearn.model_selection import train_test_split

### Data Preparation ###

# Load bioprinting sensor data from CSV files
# Data format: Each file represents a single bioprinting session
#   Columns: Timestamp, Sensor readings (e.g., pressure, temperature), Error label (0 or 1)
# Data size: 1000 bioprinting sessions, each with 1000 time steps
data = []
labels = []
for i in range(1000):
    file_path = f"bioprint_data_{i+1}.csv"
    session_data = np.genfromtxt(file_path, delimiter=',')
    data.append(session_data[:, :-1])
    labels.append(session_data[:, -1])

# Convert data and labels to numpy arrays
data = np.array(data)
labels = np.array(labels)

# Normalize sensor readings
max_values = np.max(data, axis=(0, 1))
min_values = np.min(data, axis=(0, 1))
data = (data - min_values) / (max_values - min_values)

# Reshape data for input to autoencoder
data = data.reshape(data.shape[0], data.shape[1], data.shape[2], 1)

# Split data into training and testing sets
train_data, test_data, train_labels, test_labels = train_test_split(data, labels, test_size=0.2, random_state=42)

### Anomaly Detection Model ###

# Define the autoencoder architecture
input_shape = (train_data.shape[1], train_data.shape[2], 1)
input_layer = Input(shape=input_shape)

# Encoder
x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_layer)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
encoded = MaxPooling2D((2, 2), padding='same')(x)

# Decoder
x = Conv2D(128, (3, 3), activation='relu', padding='same')(encoded)
x = UpSampling2D((2, 2))(x)
x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

# Autoencoder model
autoencoder = Model(input_layer, decoded)
autoencoder.compile(optimizer='adam', loss='mse')

# Train the autoencoder
autoencoder.fit(train_data, train_data, epochs=50, batch_size=32, shuffle=True, validation_split=0.1)

# Detect anomalies using reconstruction error
reconstructions = autoencoder.predict(test_data)
mse = np.mean(np.power(test_data - reconstructions, 2), axis=(1, 2, 3))
threshold = np.percentile(mse, 95)
anomalies = mse > threshold

### Error Recovery Strategies ###

# Define error recovery strategies based on detected anomalies
def recover_pressure_error(pressure_readings):
    # Adjust pressure settings based on anomalous readings
    adjusted_pressure = np.mean(pressure_readings[-10:])  # Example: Use average of last 10 readings
    return adjusted_pressure

def recover_temperature_error(temperature_readings):
    # Adjust temperature settings based on anomalous readings
    adjusted_temperature = np.median(temperature_readings[-10:])  # Example: Use median of last 10 readings
    return adjusted_temperature

def recover_clogged_nozzle(pressure_readings):
    # Perform nozzle cleaning procedure
    # Example: Increase pressure temporarily to clear the clog
    cleaning_pressure = 1.5 * np.max(pressure_readings)
    cleaning_duration = 5  # Example: Clean for 5 time steps
    return cleaning_pressure, cleaning_duration

# Implement error recovery based on detected anomalies
def recover_errors(session_data, anomalies):
    recovered_data = session_data.copy()
    
    for i in range(len(anomalies)):
        if anomalies[i]:
            if i < len(anomalies) - 10:
                if np.all(anomalies[i:i+10]):
                    # Continuous anomaly detected
                    if np.mean(session_data[i:i+10, 0]) > 0.8:  # Example: Check if pressure is high
                        # Recover from clogged nozzle
                        cleaning_pressure, cleaning_duration = recover_clogged_nozzle(session_data[i:i+10, 0])
                        recovered_data[i:i+cleaning_duration, 0] = cleaning_pressure
                    else:
                        # Recover from pressure error
                        adjusted_pressure = recover_pressure_error(session_data[i:i+10, 0])
                        recovered_data[i:i+10, 0] = adjusted_pressure
                        
                    if np.mean(session_data[i:i+10, 1]) > 0.7:  # Example: Check if temperature is high
                        # Recover from temperature error
                        adjusted_temperature = recover_temperature_error(session_data[i:i+10, 1])
                        recovered_data[i:i+10, 1] = adjusted_temperature
                        
    return recovered_data

### Error Recovery Evaluation ###

# Evaluate error recovery on test data
recovered_data = []
for i in range(len(test_data)):
    session_data = test_data[i]
    session_anomalies = anomalies[i]
    recovered_session = recover_errors(session_data, session_anomalies)
    recovered_data.append(recovered_session)

recovered_data = np.array(recovered_data)

# Evaluate recovery performance using mean squared error
recovery_mse = np.mean(np.power(test_data - recovered_data, 2))
print(f"Recovery MSE: {recovery_mse:.4f}")

# Visualize original and recovered data for a sample session
sample_index = 0
original_data = test_data[sample_index]
recovered_sample = recovered_data[sample_index]

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(original_data[:, :, 0])
plt.title("Original Pressure")
plt.subplot(1, 2, 2)
plt.plot(recovered_sample[:, :, 0])
plt.title("Recovered Pressure")
plt.tight_layout()
plt.show()
