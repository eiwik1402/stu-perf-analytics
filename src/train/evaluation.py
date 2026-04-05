import os
import pandas as pd
import pickle
from sklearn.metrics import (
    # Regression
    r2_score,
    mean_absolute_error,
    mean_squared_error,
    # Classification
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "processed_data")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")

# ── Load test data ────────────────────────────────────────────────────────────
X_test = pd.read_csv(os.path.join(DATA_DIR, "X_test.csv"))
y_test_reg = pd.read_csv(os.path.join(DATA_DIR, "y_test.csv"))["overall_score"]
y_test_clf = pd.read_csv(os.path.join(DATA_DIR, "y_test.csv"))["final_grade"]


# ── Evaluate Regression ───────────────────────────────────────────────────────
def evaluate_regression():
    model_path = os.path.join(MODEL_DIR, "linear_regression.pkl")

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    y_pred = model.predict(X_test)

    print("=" * 45)
    print("  LINEAR REGRESSION — Test Set Evaluation")
    print("=" * 45)
    print(f"  R² Score : {r2_score(y_test_reg, y_pred):.4f}")
    print(f"  MAE      : {mean_absolute_error(y_test_reg, y_pred):.4f}")
    print(f"  RMSE     : {mean_squared_error(y_test_reg, y_pred) ** 0.5:.4f}")
    print()


# ── Evaluate Classification ───────────────────────────────────────────────────
def evaluate_classification():
    model_path = os.path.join(MODEL_DIR, "logistic_regression.pkl")

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    y_pred = model.predict(X_test)

    print("=" * 45)
    print("  LOGISTIC REGRESSION — Test Set Evaluation")
    print("=" * 45)
    print(f"  Accuracy  : {accuracy_score(y_test_clf, y_pred):.4f}")
    print(f"  Precision : {precision_score(y_test_clf, y_pred, average='weighted', zero_division=0):.4f}")
    print(f"  Recall    : {recall_score(y_test_clf, y_pred, average='weighted', zero_division=0):.4f}")
    print(f"  F1 Score  : {f1_score(y_test_clf, y_pred, average='weighted', zero_division=0):.4f}")
    print()
    print("  Classification Report:")
    print(classification_report(y_test_clf, y_pred, zero_division=0))


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    evaluate_regression()
    evaluate_classification()