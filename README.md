# Machine Learning Assignment 2
## Dry Bean Classification Using Five Machine Learning Algorithms

**BITS Pilani – Work Integrated Learning Programmes Division**  
**Programme:** M.Tech Artificial Intelligence and Machine Learning  
**Subject:** Machine Learning  
**Assignment:** Assignment 2
**Name: Ashutosh Subhash Pangarkar
**BITS ID: 2026AJ05272

---

## 1. Problem Statement

The objective of this assignment is to implement and evaluate five machine learning classification algorithms using a common classification dataset.

The project uses the Dry Bean Dataset to classify bean varieties based on their physical characteristics.

The five classification algorithms implemented are:

1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbors (KNN)
4. Gaussian Naive Bayes
5. Support Vector Machine (SVM)

The performance of each model is evaluated using six metrics:

- Accuracy
- Area Under the ROC Curve (AUC)
- Precision
- Recall
- F1 Score
- Matthews Correlation Coefficient (MCC)

An interactive Streamlit application is developed to demonstrate and compare the performance of the trained models.

---

## 2. Dataset Description

**Dataset Name:** Dry Bean Dataset  
**Source:** UCI Machine Learning Repository

**Dataset Link:** https://archive.ics.uci.edu/dataset/602/dry+bean+dataset

### Dataset characteristics

| Parameter | Description |
|---|---|
| Total instances | 13,611 |
| Input features | 16 |
| Target variable | Class |
| Number of classes | 7 |
| Feature type | Numerical |
| Problem type | Multiclass Classification |

### Bean categories

The dataset contains seven varieties of dry beans:

1. Seker
2. Barbunya
3. Bombay
4. Cali
5. Dermosan
6. Horoz
7. Sira

### Input features

The 16 numerical features describe the physical characteristics of beans:

1. Area
2. Perimeter
3. MajorAxisLength
4. MinorAxisLength
5. AspectRation
6. Eccentricity
7. ConvexArea
8. EquivDiameter
9. Extent
10. Solidity
11. roundness
12. Compactness
13. ShapeFactor1
14. ShapeFactor2
15. ShapeFactor3
16. ShapeFactor4

The target variable is `Class`, representing the bean variety.

### Data preprocessing

The following preprocessing steps were performed:

- Loaded and examined the dataset.
- Analyzed missing values and duplicate records.
- Generated statistical summaries.
- Visualized class distribution.
- Generated a feature correlation heatmap.
- Divided the dataset into training and testing sets using an 80:20 split.
- Used stratified sampling to maintain class proportions.

The dataset was divided into:

| Dataset | Number of records |
|---|---:|
| Training | 10,888 |
| Testing | 2,723 |
| Total | 13,611 |

Random state: 42.

StandardScaler was applied within the pipelines for Logistic Regression, KNN and SVM.

---

## 3. GitHub Repository Link

**GitHub Repository:**  
[ADD YOUR GITHUB REPOSITORY URL HERE]

The repository contains:

- Complete Python source code.
- Trained models for all five classifiers.
- requirements.txt.
- README.md.
- Test dataset.
- Model evaluation results.
- Confusion matrices.
- Classification reports.
- Streamlit application.

---

## 4. Models Used and Performance Evaluation

All five classification models were trained using the same training dataset and evaluated using the same testing dataset.

Weighted averaging was used for Precision, Recall and F1 Score. Multiclass AUC was calculated using the One-vs-Rest (OvR) strategy with weighted averaging.

### 4.1 Model Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9207 | 0.9934 | 0.9215 | 0.9207 | 0.9209 | 0.9042 |
| Decision Tree | 0.8917 | 0.9330 | 0.8912 | 0.8917 | 0.8913 | 0.8691 |
| KNN | 0.9166 | 0.9812 | 0.9174 | 0.9166 | 0.9168 | 0.8992 |
| Naive Bayes | 0.7639 | 0.9644 | 0.7654 | 0.7639 | 0.7615 | 0.7154 |
| Support Vector Machine | **0.9221** | **0.9937** | **0.9224** | **0.9221** | **0.9222** | **0.9059** |

### 4.2 Observations on Model Performance

**Logistic Regression**

Logistic Regression achieved an accuracy of 92.07% and an AUC score of 0.9934. It demonstrated strong classification performance and good separation between bean classes. Its performance was very close to that of the Support Vector Machine.

**Decision Tree Classifier**

The Decision Tree achieved an accuracy of 89.17% and an AUC score of 0.9330. Its performance was lower than the other major classification models, indicating comparatively weaker generalization on the testing dataset.

**K-Nearest Neighbors (KNN)**

KNN achieved an accuracy of 91.66% and an AUC score of 0.9812. It demonstrated good classification performance, although its results were slightly below Logistic Regression and Support Vector Machine.

**Gaussian Naive Bayes**

Naive Bayes achieved an accuracy of 76.39% and an AUC score of 0.9644. It recorded the lowest accuracy and F1 score among the five models. This suggests that its predicted class labels were comparatively less effective on this dataset.

**Support Vector Machine**

Support Vector Machine achieved the highest accuracy of 92.21%, AUC of 0.9937, Precision of 0.9224, Recall of 0.9221, F1 Score of 0.9222 and MCC of 0.9059.

It demonstrated the best overall classification performance among the five implemented algorithms.

### 4.3 Overall Winner

**Support Vector Machine (SVM)** is the overall winner for the Dry Bean Dataset.

It achieved the highest weighted F1 score, along with the highest accuracy, AUC and MCC.

Logistic Regression also performed competitively, with only a small difference in accuracy compared with SVM.

Based on the experimental results, SVM is the preferred classification model for this dataset.

---

## 5. Streamlit Application

An interactive web application was developed using Streamlit to demonstrate the trained classification models.

### Application features

1. Dataset upload option supporting CSV files.
2. Model selection dropdown.
3. Performance comparison of all five classification models.
4. Display of six evaluation metrics.
5. Confusion matrix visualization.
6. Classification report.
7. Identification of the best-performing model.

The application loads the previously trained models using Joblib. It does not retrain models using the uploaded test dataset.

**Live Streamlit Application:**  
[ADD YOUR STREAMLIT COMMUNITY CLOUD URL HERE]

---

## 6. Project Structure

```text
ML_Assignment_2/
│
├── app.py
├── download_dataset.py
├── eda_preprocessing.py
├── train_models.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── data/
│   ├── dry_bean_dataset.csv
│   ├── train_data.csv
│   └── test_data.csv
│
├── model/
│   ├── logistic_regression.joblib
│   ├── decision_tree.joblib
│   ├── knn.joblib
│   ├── naive_bayes.joblib
│   └── support_vector_machine.joblib
│
├── results/
│   ├── model_comparison.csv
│   ├── class_distribution.png
│   ├── correlation_heatmap.png
│   │
│   ├── confusion_matrices/
│   └── classification_reports/
│
└── screenshots/
```

---

## 7. Installation and Execution Instructions

### Prerequisites

- Python 3.11 or later
- pip
- Git

### Step 1: Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

### Step 2: Navigate to the project directory

```bash
cd ML_Assignment_2
```

### Step 3: Create virtual environment

Windows:

```powershell
python -m venv venv
venv\Scripts\activate
```

Linux:

```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 4: Install dependencies

```bash
pip install -r requirements.txt
```

### Step 5: Download the dataset (if required)

```bash
python download_dataset.py
```

### Step 6: Perform EDA and preprocessing

```bash
python eda_preprocessing.py
```

### Step 7: Train all five classification models

```bash
python train_models.py
```

### Step 8: Launch Streamlit application

```bash
streamlit run app.py
```

The application will be accessible at:

```text
http://localhost:8501
```

Upload `data/test_data.csv` to evaluate the trained models.

---

## 8. Technologies Used

| Technology | Purpose |
|---|---|
| Python | Programming language |
| Pandas | Data manipulation |
| NumPy | Numerical computations |
| Scikit-learn | Machine learning algorithms |
| Matplotlib | Data visualization |
| Seaborn | Statistical visualization |
| Streamlit | Interactive web application |
| Joblib | Model serialization |
| Git and GitHub | Version control and source code hosting |

---

## 9. Conclusion

This assignment demonstrates an end-to-end machine learning classification workflow, starting from dataset selection and exploratory data analysis through model training, performance evaluation and interactive application development.

Five classification algorithms were implemented and evaluated using six standard performance metrics.

The experimental results show that Support Vector Machine achieved the best overall performance, with 92.21% accuracy and a weighted F1 score of 0.9222.

The Streamlit application provides an interactive interface for uploading test datasets, selecting classification algorithms and examining their performance.

The project provides practical experience in implementing machine learning algorithms, evaluating classification performance and deploying machine learning models through a web application.

---

**BITS Pilani – M.Tech AIML**  
**Machine Learning – Assignment 2**
