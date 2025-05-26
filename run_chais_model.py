!git clone https://github.com/Siagnos/CHAIS.git
!pip install tensorflow pandas requests

import tensorflow as tf
import numpy as np
import pandas as pd
import keras # Import keras explicitly
import requests # Import requests
from io import StringIO # Import StringIO

# Load the SavedModel as a TFSMLayer
# Assuming the default serving endpoint is 'serving_default'
model_layer = keras.layers.TFSMLayer("CHAIS/models/finalnet_savedmodel", call_endpoint='serving_default') # Corrected path to the saved model

# Use the raw GitHub URL to download the file content
url = "https://raw.githubusercontent.com/Siagnos/CHAIS/main/example_ecg.txt"

try:
    # Fetch the content from the raw URL
    response = requests.get(url)
    response.raise_for_status() # Raise an HTTPError for bad responses (4xx or 5xx)

    # Decode the content and load it using numpy from a stringIO object
    # Load data from the text content and ensure it's a 2D array
    ecg_data = np.loadtxt(StringIO(response.text)).reshape(-1, 1)

except requests.exceptions.RequestException as e:
    print(f"Error fetching the file from URL: {e}")
    # You might want to add a more robust fallback or exit here
except ValueError as e:
    print(f"Error processing the data from the file: {e}")
    # Handle potential issues if the loaded data is not in the expected format

ecg_data = (ecg_data - ecg_data.mean()) / ecg_data.std()

# Model expects input shape (None, 5000, 12).
# The example data has shape (100, 1).
# We need to pad and expand features to match the expected shape.
# This is a simple padding with zeros and replicating the single feature 12 times.
# A more appropriate method depends on the actual data and model design.
required_length = 5000
current_length = ecg_data.shape[0]
padding_length = required_length - current_length

if padding_length > 0:
    # Pad with zeros at the end
    padded_ecg_data = np.pad(ecg_data, ((0, padding_length), (0, 0)), mode='constant')
else:
    # If data is longer, truncate it
    padded_ecg_data = ecg_data[:required_length, :]

# Replicate the single feature 12 times to match the expected 12 features
input_tensor_reshaped = np.repeat(padded_ecg_data, 12, axis=1)

# Add batch dimension and ensure float32 dtype
input_tensor = np.expand_dims(input_tensor_reshaped, axis=0).astype(np.float32)

# Call the TFSMLayer with the correct keyword argument name 'inputs'
# as indicated by the traceback's signature for the TFSMLayer object.
output = model_layer(inputs=input_tensor)
print(output)

# Use tf.squeeze to remove the batch dimension, resulting in a 1D tensor
# Then convert the tensor to a NumPy array using .numpy() to use standard Python/NumPy operations
probs = tf.squeeze(output['output_1']).numpy()

# Format and print
labels = [
    "PCWP > 15 mmHg",
    "mPAP > 20 mmHg",
    "PVR > 3 Wood Units",
    "CO > 4 L/min"
]

for prob, label in zip(probs, labels):
    print(f"A {prob * 100:.1f}% probability that {label}")
