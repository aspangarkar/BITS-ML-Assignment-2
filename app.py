
# ============================================================
# BITS Pilani M.Tech AIML
# Machine Learning - Assignment 2
# Streamlit Classification Dashboard
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)

# ------------------------------------------------------------
# Page configuration
# ------------------------------------------------------------

st.set_page_config(
    page_title="Dry Bean Classification Dashboard",
    page_icon="🌱",
    layout="wide"
)

st.title("🌱 Dry Bean Classification Dashboard")

st.markdown(
    """
    **BITS Pilani | M.Tech Artificial Intelligence and Machine Learning**

    Machine Learning Assignment 2

    Compare five machine learning classification algorithms
    using the Dry Bean dataset.
    """
)

st.divider()

# ------------------------------------------------------------
# Model configuration
# ------------------------------------------------------------

MODEL_FILES = {
    "Logistic Regression": "model/logistic_regression.joblib",
    "Decision Tree": "model/decision_tree.joblib",
    "KNN": "model/knn.joblib",
    "Naive Bayes": "model/naive_bayes.joblib",
    "Support Vector Machine": "model/support_vector_machine.joblib"
}


@st.cache_resource
def load_models():

    loaded = {}

    for name, path in MODEL_FILES.items():

        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Model file missing: {path}"
            )

        loaded[name] = joblib.load(path)

    return loaded


# ------------------------------------------------------------
# Dataset upload
# ------------------------------------------------------------

st.header("1. Upload Test Dataset")

uploaded_file = st.file_uploader(
    "Upload your test dataset (CSV)",
    type=["csv"]
)

st.info(
    "The uploaded CSV must contain all 16 input features "
    "and the target column named 'Class'."
)

if uploaded_file is not None:

    try:

        test_df = pd.read_csv(uploaded_file)

        st.success("Dataset uploaded successfully!")

        st.subheader("Dataset Preview")

        st.dataframe(
            test_df.head(10),
            use_container_width=True
        )

        if "Class" not in test_df.columns:
            st.error("Target column 'Class' is missing.")
            st.stop()

        models = load_models()

        # Get expected input feature names
        feature_names = list(
            models["Logistic Regression"].feature_names_in_
        )

        missing_features = [
            feature for feature in feature_names
            if feature not in test_df.columns
        ]

        if missing_features:
            st.error(
                f"Missing input features: {missing_features}"
            )
            st.stop()

        # Prepare test features and target
        X_test = test_df[feature_names].apply(
            pd.to_numeric, errors="coerce"
        )

        y_test = test_df["Class"]

        if X_test.isnull().any().any() or y_test.isnull().any():
            st.error(
                "Dataset contains missing or non-numeric values. "
                "Please upload a valid test CSV."
            )
            st.stop()

        if len(X_test) == 0:
            st.error("Uploaded dataset is empty.")
            st.stop()

        # ----------------------------------------------------
        # Evaluate all models
        # ----------------------------------------------------

        st.divider()

        st.header("2. Model Performance Comparison")

        results = {}
        predictions = {}

        for name, model in models.items():

            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)

            predictions[name] = y_pred

            metrics = {
                "Accuracy": accuracy_score(y_test, y_pred),

                "AUC": roc_auc_score(
                    y_test,
                    y_prob,
                    labels=model.classes_,
                    multi_class="ovr",
                    average="weighted"
                ),

                "Precision": precision_score(
                    y_test,
                    y_pred,
                    average="weighted",
                    zero_division=0
                ),

                "Recall": recall_score(
                    y_test,
                    y_pred,
                    average="weighted",
                    zero_division=0
                ),

                "F1": f1_score(
                    y_test,
                    y_pred,
                    average="weighted",
                    zero_division=0
                ),

                "MCC": matthews_corrcoef(
                    y_test,
                    y_pred
                )
            }

            results[name] = metrics

        comparison_df = pd.DataFrame(results).T
        comparison_df.index.name = "ML Model Name"

        st.dataframe(
            comparison_df.style.format("{:.4f}"),
            use_container_width=True
        )

        # ----------------------------------------------------
        # Best model
        # ----------------------------------------------------

        winner = comparison_df["F1"].idxmax()

        st.success(
            f"Best performing model based on weighted F1: {winner}"
        )

        # ----------------------------------------------------
        # Model selection
        # ----------------------------------------------------

        st.divider()

        st.header("3. Select Classification Model")

        selected_model = st.selectbox(
            "Choose a machine learning algorithm",
            list(MODEL_FILES.keys())
        )

        selected_metrics = results[selected_model]

        st.subheader(
            f"Evaluation Metrics - {selected_model}"
        )

        columns = st.columns(3)

        metric_names = [
            "Accuracy", "AUC", "Precision",
            "Recall", "F1", "MCC"
        ]

        for index, metric in enumerate(metric_names):

            columns[index % 3].metric(
                metric,
                f"{selected_metrics[metric]:.4f}"
            )

        # ----------------------------------------------------
        # Confusion Matrix
        # ----------------------------------------------------

        st.divider()

        st.header("4. Confusion Matrix")

        selected_predictions = predictions[selected_model]

        selected_classes = models[selected_model].classes_

        cm = confusion_matrix(
            y_test,
            selected_predictions,
            labels=selected_classes
        )

        fig, ax = plt.subplots(figsize=(10, 7))

        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=selected_classes,
            yticklabels=selected_classes,
            ax=ax
        )

        ax.set_xlabel("Predicted Class")
        ax.set_ylabel("Actual Class")
        ax.set_title(f"Confusion Matrix - {selected_model}")

        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()

        st.pyplot(fig)
        plt.close(fig)

        # ----------------------------------------------------
        # Classification Report
        # ----------------------------------------------------

        st.divider()

        st.header("5. Classification Report")

        report = classification_report(
            y_test,
            selected_predictions,
            zero_division=0,
            output_dict=True
        )

        report_df = pd.DataFrame(report).transpose()

        st.dataframe(
            report_df.style.format("{:.4f}"),
            use_container_width=True
        )

        # ----------------------------------------------------
        # Completion
        # ----------------------------------------------------

        st.divider()

        st.caption(
            "BITS Pilani M.Tech AIML | Machine Learning Assignment 2"
        )

    except Exception as error:

        st.error(f"Unable to process the uploaded dataset: {error}")

else:

    st.warning(
        "Please upload test_data.csv to begin model evaluation."
    )

    st.markdown(
        """
        ### Available classification models

        1. Logistic Regression
        2. Decision Tree
        3. K-Nearest Neighbors
        4. Gaussian Naive Bayes
        5. Support Vector Machine
        """
    )
