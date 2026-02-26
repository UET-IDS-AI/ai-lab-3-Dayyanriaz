import numpy as np
from sklearn.datasets import load_diabetes, load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    mean_squared_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# =========================================================
# QUESTION 1 – Linear Regression Pipeline (Diabetes)
# =========================================================

def diabetes_linear_pipeline():
    # 1️⃣ Load dataset
    data = load_diabetes()
    X, y = data.data, data.target

    # 2️⃣ Split into train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 3️⃣ Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 4️⃣ Train model
    model = LinearRegression()
    model.fit(X_train_scaled, y_train)

    # 5️⃣ Compute metrics
    y_train_pred = model.predict(X_train_scaled)
    y_test_pred = model.predict(X_test_scaled)

    train_mse = mean_squared_error(y_train, y_train_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)

    # 6️⃣ Top 3 features with largest absolute coefficients
    coefs = np.abs(model.coef_)
    top_3_feature_indices = np.argsort(coefs)[-3:][::-1].tolist()

    # 💬 COMMENTS:
    # - If train R² >> test R² → overfitting.
    # - Scaling ensures all features have equal influence on regression weights.

    return train_mse, test_mse, train_r2, test_r2, top_3_feature_indices


# =========================================================
# QUESTION 2 – Cross-Validation (Linear Regression)
# =========================================================

def diabetes_cross_validation():
    data = load_diabetes()
    X, y = data.data, data.target

    # Standardize full dataset
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = LinearRegression()

    # 5-fold cross-validation
    scores = cross_val_score(model, X_scaled, y, cv=5, scoring="r2")

    mean_r2 = np.mean(scores)
    std_r2 = np.std(scores)

    # 💬 COMMENTS:
    # - std_r2 shows stability of model performance.
    # - Cross-validation reduces variance risk by averaging over folds.

    return mean_r2, std_r2


# =========================================================
# QUESTION 3 – Logistic Regression Pipeline (Cancer)
# =========================================================

def cancer_logistic_pipeline():
    data = load_breast_cancer()
    X, y = data.data, data.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LogisticRegression(max_iter=5000)
    model.fit(X_train_scaled, y_train)

    y_train_pred = model.predict(X_train_scaled)
    y_test_pred = model.predict(X_test_scaled)

    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    precision = precision_score(y_test, y_test_pred)
    recall = recall_score(y_test, y_test_pred)
    f1 = f1_score(y_test, y_test_pred)

    # Optionally: confusion matrix (not returned)
    cm = confusion_matrix(y_test, y_test_pred)

    # 💬 COMMENT:
    # A False Negative means: The model predicts "no cancer" when cancer is actually present.
    # → Dangerous in medical context because it misses a real patient needing treatment.

    return train_accuracy, test_accuracy, precision, recall, f1


# =========================================================
# QUESTION 4 – Logistic Regularization Path
# =========================================================

def cancer_logistic_regularization():
    data = load_breast_cancer()
    X, y = data.data, data.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    results = {}
    for C in [0.01, 0.1, 1, 10, 100]:
        model = LogisticRegression(max_iter=5000, C=C)
        model.fit(X_train_scaled, y_train)

        train_acc = model.score(X_train_scaled, y_train)
        test_acc = model.score(X_test_scaled, y_test)

        results[C] = (train_acc, test_acc)

    # 💬 COMMENT:
    # - Small C → Strong regularization → Underfitting (weights too small).
    # - Large C → Weak regularization → Overfitting (model fits noise).
    # - Overfitting occurs at very large C.

    return results


# =========================================================
# QUESTION 5 – Cross-Validation (Logistic Regression)
# =========================================================

def cancer_cross_validation():
    data = load_breast_cancer()
    X, y = data.data, data.target

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = LogisticRegression(C=1, max_iter=5000)

    scores = cross_val_score(model, X_scaled, y, cv=5, scoring="accuracy")

    mean_accuracy = np.mean(scores)
    std_accuracy = np.std(scores)

    # 💬 COMMENT:
    # Cross-validation is vital in medical diagnosis because data imbalance or small samples
    # could make one split misleading. CV ensures model performs reliably across multiple subsets.

    return mean_accuracy, std_accuracy
