import torch
import pandas as pd
import numpy as np

# Path to model (update if needed)
model_path = 'models/finalnet_savedmodel.pt'

# Try loading the full model
try:
    model = torch.load(model_path, map_location=torch.device('cpu'))
    print("Model loaded successfully.")
except Exception as e:
    print("Failed to load model. You may need to define the model class.")
    print(e)
    exit()

# Create or load a CSV file like data/example_ecg.csv
ecg_csv_path = 'data/example_ecg.csv'  # <-- replace with your path

# Load ECG data
try:
    ecg_data = pd.read_csv(ecg_csv_path, header=None).values
    print(f"ECG data loaded. Shape: {ecg_data.shape}")
except Exception as e:
    print("Error loading ECG data")
    print(e)
    exit()

# Normalize input
ecg_data = (ecg_data - np.mean(ecg_data)) / np.std(ecg_data)

# Convert to PyTorch tensor
input_tensor = torch.tensor(ecg_data, dtype=torch.float32).unsqueeze(0)

# Run inference
model.eval()
with torch.no_grad():
    output = model(input_tensor)

print("Model output:")
print(output)
