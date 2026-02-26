[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/q-zKsdy6)
# Linear & Logistic Regression

This lab focuses ONLY on:
- Linear Regression (Diabetes dataset)
- Logistic Regression (Breast Cancer dataset)

You will implement full supervised learning workflows
and analyze model performance, stability, and overfitting.

------------------------------------------------------------
DATASETS USED
------------------------------------------------------------
1. sklearn.datasets.load_diabetes()
   → Regression task

2. sklearn.datasets.load_breast_cancer()
   → Binary classification task

------------------------------------------------------------
ALLOWED LIBRARIES
------------------------------------------------------------
- numpy
- scikit-learn

DO NOT use:
- Deep learning libraries
- Any model other than LinearRegression and LogisticRegression

------------------------------------------------------------
GENERAL REQUIREMENTS
------------------------------------------------------------
- Follow function signatures EXACTLY.
- Do NOT modify function names.
- All functions must return values.
- Use StandardScaler for feature scaling.
- Split data using 80-20 train-test split.
- Use random_state=42 for reproducibility.
- Comment clearly where explanation is required.

------------------------------------------------------------
QUESTION 1 (Linear Regression Pipeline)
------------------------------------------------------------
Using diabetes dataset:

1. Load dataset.
2. Split into train and test (80-20).
3. Standardize features (fit only on train).
4. Train LinearRegression model.
5. Compute:
   - Train MSE
   - Test MSE
   - Train R²
   - Test R²
6. Identify top 3 features with largest absolute coefficients.
7. In comments:
   - Does the model overfit?
   - Why is feature scaling important?

------------------------------------------------------------
QUESTION 2 (Cross-Validation – Linear Regression)
------------------------------------------------------------
1. Perform 5-fold cross-validation on LinearRegression.
2. Compute mean and standard deviation of R².
3. Compare CV mean with test R² from Q1.
4. In comments:
   - What does standard deviation represent?
   - How does CV reduce variance risk?

------------------------------------------------------------
QUESTION 3 (Logistic Regression Pipeline)
------------------------------------------------------------
Using breast cancer dataset:

1. Split into train-test (80-20).
2. Standardize features.
3. Train LogisticRegression (max_iter=5000).
4. Compute:
   - Train Accuracy
   - Test Accuracy
   - Precision
   - Recall
   - F1-score
   - Confusion matrix
5. In comments:
   - What does a False Negative mean in medical context?

------------------------------------------------------------
QUESTION 4 (Regularization in Logistic Regression)
------------------------------------------------------------
Train LogisticRegression for:

C = [0.01, 0.1, 1, 10, 100]

For each C:
- Compute train accuracy
- Compute test accuracy

Return dictionary:
{C: (train_acc, test_acc)}

In comments:
- What happens when C is very small?
- What happens when C is very large?
- Which case leads to overfitting?

------------------------------------------------------------
QUESTION 5 (Cross-Validation – Logistic Regression)
------------------------------------------------------------
1. Perform 5-fold cross-validation (C=1).
2. Compute:
   - Mean accuracy
   - Std accuracy
3. Compare CV mean with test accuracy.
4. In comments:
   - Why is cross-validation critical in medical diagnosis?

