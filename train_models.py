
# ============================================================
# BITS Pilani M.Tech AIML
# Machine Learning - Assignment 2
# Step 3: Train Five Classification Models
# ============================================================

import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report
)

# ------------------------------------------------------------
# 1. Create output directories
# ------------------------------------------------------------

os.makedirs("model", exist_ok=True)
os.makedirs("results/confusion_matrices", exist_ok=True)
os.makedirs("results/classification_reports", exist_ok=True)

# ------------------------------------------------------------
# 2. Load training and testing datasets
# ------------------------------------------------------------

print("=" * 65)
print("DRY BEAN CLASSIFICATION - MODEL TRAINING")
print("=" * 65)

train_df = pd.read_csv("data/train_data.csv")
test_df = pd.read_csv("data/test_data.csv")

TARGET = "Class"

X_train = train_df.drop(columns=[TARGET])
y_train = train_df[TARGET]

X_test = test_df.drop(columns=[TARGET])
y_test = test_df[TARGET]

print("\nTraining records:", len(X_train))
print("Testing records:", len(X_test))
print("Number of features:", X_train.shape[1])
print("Number of classes:", y_train.nunique())

# ------------------------------------------------------------
# 3. Initialize five classification models
# ------------------------------------------------------------

models = {

    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            max_iter=2000,
            random_state=42
        ))
    ]),

    "Decision Tree": DecisionTreeClassifier(
        random_state=42
    ),

    "KNN": Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", KNeighborsClassifier(
            n_neighbors=5
        ))
    ]),

    "Naive Bayes": GaussianNB(),

    "Support Vector Machine": Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", SVC(
            kernel="rbf",
            probability=True,
            random_state=42
        ))
    ])
}

# ------------------------------------------------------------
# 4. Train models and evaluate performance
# ------------------------------------------------------------

results = []

for name, model in models.items():

    print("\n" + "-" * 65)
    print("Training:", name)
    print("-" * 65)

    # Train model using training data only
    model.fit(X_train, y_train)

    # Predict test data
    y_pred = model.predict(X_test)

    # Predict class probabilities for AUC
    y_prob = model.predict_proba(X_test)

    # Calculate evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)

    auc = roc_auc_score(
        y_test,
        y_prob,
        labels=model.classes_,
        multi_class="ovr",
        average="weighted"
    )

    precision = precision_score(
        y_test, y_pred,
        average="weighted",
        zero_division=0
    )

    recall = recall_score(
        y_test, y_pred,
        average="weighted",
        zero_division=0
    )

    f1 = f1_score(
        y_test, y_pred,
        average="weighted",
        zero_division=0
    )

    mcc = matthews_corrcoef(y_test, y_pred)

    # Display metrics
    print(f"Accuracy  : {accuracy:.4f}")
    print(f"AUC Score : {auc:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1 Score  : {f1:.4f}")
    print(f"MCC Score : {mcc:.4f}")

    # Store results
    results.append({
        "ML Model Name": name,
        "Accuracy": accuracy,
        "AUC": auc,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "MCC": mcc
    })

    # --------------------------------------------------------
    # 5. Save trained model
    # --------------------------------------------------------

    filename = (
        name.lower()
        .replace(" ", "_")
        + ".joblib"
    )

    joblib.dump(model, f"model/{filename}")

    print("Saved model:", f"model/{filename}")

    # --------------------------------------------------------
    # 6. Generate confusion matrix
    # --------------------------------------------------------

    cm = confusion_matrix(
        y_test,
        y_pred,
        labels=model.classes_
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=model.classes_
    )

    fig, ax = plt.subplots(figsize=(10, 8))

    display.plot(
        ax=ax,
        cmap="Blues",
        xticks_rotation=45,
        colorbar=False
    )

    plt.title(f"Confusion Matrix - {name}")
    plt.tight_layout()

    image_name = filename.replace(".joblib", ".png")

    plt.savefig(
        f"results/confusion_matrices/{image_name}",
        dpi=150
    )

    plt.close(fig)

    # --------------------------------------------------------
    # 7. Generate classification report
    # --------------------------------------------------------

    report = classification_report(
        y_test,
        y_pred,
        zero_division=0
    )

    report_path = (
        "results/classification_reports/"
        + filename.replace(".joblib", ".txt")
    )

    with open(report_path, "w") as file:
        file.write(report)

    print("Classification report saved.")

# ------------------------------------------------------------
# 8. Create model comparison table
# ------------------------------------------------------------

comparison = pd.DataFrame(results)

comparison.to_csv(
    "results/model_comparison.csv",
    index=False
)

print("\n" + "=" * 65)
print("FINAL MODEL COMPARISON")
print("=" * 65)

print(comparison.round(4).to_string(index=False))

# ------------------------------------------------------------
# 9. Identify overall winner
# ------------------------------------------------------------

winner = comparison.loc[
    comparison["F1"].idxmax()
]

print("\nOVERALL WINNER BASED ON WEIGHTED F1 SCORE:")
print(winner["ML Model Name"])

print("\nPerformance:")

for metric in [
    "Accuracy",
    "AUC",
    "Precision",
    "Recall",
    "F1",
    "MCC"
]:
    print(f"{metric}: {winner[metric]:.4f}")



# ------------------------------------------------------------
# 10. Completion message
# ------------------------------------------------------------

print("\n" + "=" * 65)
print("ALL FIVE MODELS TRAINED SUCCESSFULLY!")
print("=" * 65)

print("\nGenerated files:")
print("model/                  - Five saved models")
print("results/model_comparison.csv")
print("results/confusion_matrices/")
print("results/classification_reports/")
