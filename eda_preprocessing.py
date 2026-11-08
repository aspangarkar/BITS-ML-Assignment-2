
# ============================================================
# BITS Pilani M.Tech AIML
# Machine Learning - Assignment 2
# Step 2: EDA and Data Preprocessing
# ============================================================

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split

# ------------------------------------------------------------
# 1. Load dataset
# ------------------------------------------------------------

df = pd.read_csv("data/dry_bean_dataset.csv")

os.makedirs("results", exist_ok=True)
os.makedirs("data", exist_ok=True)

TARGET = "Class"

print("=" * 60)
print("DRY BEAN DATASET - EXPLORATORY DATA ANALYSIS")
print("=" * 60)

# ------------------------------------------------------------
# 2. Dataset validation
# ------------------------------------------------------------

print("\n1. DATASET DIMENSIONS")
print("Rows:", df.shape[0])
print("Columns:", df.shape[1])

assert TARGET in df.columns, "Target column not found!"
assert df.shape[1] - 1 >= 12, "Minimum 12 features required!"
assert df.shape[0] >= 500, "Minimum 500 instances required!"

X = df.drop(columns=[TARGET])
y = df[TARGET]

print("\nNumber of input features:", X.shape[1])
print("Number of classes:", y.nunique())

# ------------------------------------------------------------
# 3. Missing value analysis
# ------------------------------------------------------------

print("\n2. MISSING VALUES")
print(df.isnull().sum())

# ------------------------------------------------------------
# 4. Duplicate analysis
# ------------------------------------------------------------

print("\n3. DUPLICATE RECORDS")
print("Duplicate rows:", df.duplicated().sum())

# We will report duplicates, not silently delete them.

# ------------------------------------------------------------
# 5. Statistical summary
# ------------------------------------------------------------

print("\n4. STATISTICAL SUMMARY")
print(X.describe().round(2).to_string())

# ------------------------------------------------------------
# 6. Class distribution
# ------------------------------------------------------------

print("\n5. CLASS DISTRIBUTION")
print(y.value_counts())

plt.figure(figsize=(10, 5))
sns.countplot(data=df, x=TARGET, order=y.value_counts().index)
plt.title("Distribution of Dry Bean Classes")
plt.xlabel("Bean Class")
plt.ylabel("Number of Samples")
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig("results/class_distribution.png", dpi=150)
plt.show()

# ------------------------------------------------------------
# 7. Feature correlation analysis
# ------------------------------------------------------------

plt.figure(figsize=(14, 10))

correlation = X.corr(numeric_only=True)

sns.heatmap(
    correlation,
    cmap="coolwarm",
    annot=False,
    linewidths=0.3
)

plt.title("Correlation Heatmap - Dry Bean Features")
plt.tight_layout()
plt.savefig("results/correlation_heatmap.png", dpi=150)
plt.show()

# ------------------------------------------------------------
# 8. Train-Test Split
# ------------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\n6. TRAIN-TEST SPLIT")
print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))

# ------------------------------------------------------------
# 9. Save training and testing datasets
# ------------------------------------------------------------

train_df = X_train.copy()
train_df[TARGET] = y_train

test_df = X_test.copy()
test_df[TARGET] = y_test

train_df.to_csv("data/train_data.csv", index=False)
test_df.to_csv("data/test_data.csv", index=False)

print("\nFiles saved successfully:")
print("data/train_data.csv")
print("data/test_data.csv")

# ------------------------------------------------------------
# 10. Final summary
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("EDA AND PREPROCESSING COMPLETED SUCCESSFULLY")
print("=" * 60)
