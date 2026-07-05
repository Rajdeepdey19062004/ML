# Breast Cancer Classification using Machine Learning

## Overview

This project applies multiple machine learning algorithms to classify breast cancer tumors as **malignant** or **benign**.
The dataset is processed using Python data science libraries and evaluated using several classification models.

The objective is to **compare the performance of different machine learning algorithms** using accuracy score, confusion matrix, and classification report.

---

# Algorithms Used

The following machine learning models are implemented:

1. K-Nearest Neighbors (KNN)
2. Logistic Regression
3. Naive Bayes
4. Support Vector Machine (SVM)
5. Decision Tree
6. Random Forest

Each model is trained and evaluated using the same dataset and testing split.

---

# Dataset

The dataset used is a **Breast Cancer dataset** containing tumor measurements.

Selected features:

* diagnosis
* radius_mean
* texture_mean

Target variable:

* **diagnosis**

  * M → Malignant
  * B → Benign

---

# Project Workflow

## 1. Data Loading

The dataset is loaded using Pandas.

```python
data = pd.read_csv("cancer_data.csv")
```

Selected columns:

```python
data_final = data[["diagnosis", "radius_mean", "texture_mean"]]
```

---

## 2. Data Preparation

Features and target variables are separated.

```python
x = data_final.iloc[:, 1:].values
y = data_final.iloc[:, 0].values
```

Dataset is split into training and testing sets.

```python
train_test_split(x, y, test_size=0.3)
```

---

## 3. Feature Scaling

Standardization is applied using **StandardScaler** to normalize input features.

```python
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
```

---

# Model Training and Evaluation

Each model follows the same process:

1. Train the model
2. Predict test data
3. Calculate accuracy
4. Generate confusion matrix
5. Display classification report

Example (KNN):

```python
model = KNeighborsClassifier(n_neighbors=5)
model.fit(x_train, y_train)
```

Prediction:

```python
y_pred = model.predict(x_test)
```

---

# Visualization

Confusion matrices for all models are visualized using **Seaborn heatmaps**.

Example:

```python
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True)
```

This allows visual comparison of model performance.

---

# Evaluation Metrics

The following metrics are used to evaluate each model:

* Accuracy Score
* Precision
* Recall
* F1 Score
* Confusion Matrix

---

# Output

The program prints:

* Model accuracy
* Classification report
* Confusion matrix plots

Example output:

```
KNN ACCURACY: 95.3 %

LOGISTIC ACCURACY: 96.4 %

NAIVE BAYES ACCURACY: 94.7 %

SVM ACCURACY: 97.1 %
```

---

# Technologies Used

* Python
* NumPy
* Pandas
* Matplotlib
* Seaborn
* Scikit-learn

---

# Project Structure

```
breast-cancer-ml
│
├── cancer_data.csv
├── model.py
├── requirements.txt
└── README.md
```

---

# Future Improvements

Possible improvements for the project:

* Use more input features
* Apply feature selection techniques
* Perform hyperparameter tuning
* Add cross-validation
* Deploy as a web application

---

# Conclusion

This project demonstrates how multiple machine learning algorithms can be applied to medical datasets to predict breast cancer diagnosis.
By comparing different models, we can identify the most accurate algorithm for classification tasks.

**Requirements for this project**
numpy
pandas
matplotlib
seaborn
scikit-learn
