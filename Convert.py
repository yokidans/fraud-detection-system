import numpy as np
import pandas as pd

# File paths
npy_path = r"C:\Users\tefer\Desktop\fraud-detection-system\data\processed\ecommerce_target.npy"
csv_path = r"C:\Users\tefer\Desktop\fraud-detection-system\data\processed\ecommerce_target.csv"

# 1. Load the .npy file
array = np.load(npy_path)

# 2. Save as CSV
pd.DataFrame(array).to_csv(csv_path, index=False)

print(f"CSV saved to {csv_path}")