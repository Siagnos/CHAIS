# run_chais_keras.py
import tensorflow as tf
import numpy as np
import pandas as pd

# Load the Keras model
model = tf.keras.models.load_model("models/finalnet_savedmodel")
print("Model loaded successfully.")

# Load ECG data
ecg_data = pd.read_csv("data/example_ecg.csv", header=None).values
ecg_data = (ecg_data - np.mean(ecg_data)) / np.std(ecg_data)  # Normalize

# Add batch dimension
input_tensor = np.expand_dims(ecg_data, axis=0)

# Run inference
output = model.predict(input_tensor)
print("Model output:")
print(output)
