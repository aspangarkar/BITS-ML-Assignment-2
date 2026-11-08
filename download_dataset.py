
# ============================================================
# BITS Pilani M.Tech AIML
# Machine Learning - Assignment 2
# Step 1: Download and explore Dry Bean Dataset
# ============================================================

import pandas as pd
from ucimlrepo import fetch_ucirepo

print("Downloading Dry Bean Dataset from UCI...")

# Fetch dataset
dry_bean = fetch_ucirepo(id=602)

# Extract input features and target
X = dry_bean.data.features
y = dry_bean.data.targets

# Combine features and target
df = pd.concat([X, y], axis=1)

# Create data directory if required
import os
os.makedirs("data", exist_ok=True)

# Save dataset
file_path = "data/dry_bean_dataset.csv"
df.to_csv(file_path, index=False)

# Display dataset information
print("\nDataset downloaded successfully!")

print("\nDataset Shape:")
print(df.shape)

print("\nFirst 5 Records:")
print(df.head())

print("\nColumn Names:")
print(df.columns.tolist())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nClass Distribution:")
print(df["Class"].value_counts())

print("\nDataset Information:")
print(df.info())

print("\nDataset saved at:", file_path)
