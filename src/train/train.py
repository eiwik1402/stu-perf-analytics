import os
import json
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import GridSearchCV

DATA_DIR   = os.path.join(os.path.dirname(__file__), "..", "..", "data", "processed_data")
MODEL_DIR  = os.path.join(os.path.dirname(__file__), "..", "models")
PARAMS_PATH = os.path.join(MODEL_DIR, "best_params.json")

os.makedirs(MODEL_DIR, exist_ok=True)

X_train = pd.read_csv(os.path.join(DATA_DIR, "X_train.csv"))
X_val   = pd.read_csv(os.path.join(DATA_DIR, "X_val.csv"))
y_train_clf = pd.read_csv(os.path.join(DATA_DIR, "y_train.csv"))["final_grade"]
y_val_clf   = pd.read_csv(os.path.join(DATA_DIR, "y_val.csv"))["final_grade"]
y_train_reg = pd.read_csv(os.path.join(DATA_DIR, "y_train.csv"))["overall_score"]
y_val_reg   = pd.read_csv(os.path.join(DATA_DIR, "y_val.csv"))["overall_score"]

X_full     = np.vstack([X_train, X_val])
y_clf_full = pd.concat([y_train_clf, y_val_clf], ignore_index=True)
y_reg_full = pd.concat([y_train_reg, y_val_reg], ignore_index=True)

best_params = {}

print("=" * 50)
print("  TUNING: Logistic Regression")
print("=" * 50)

clf_grid = GridSearchCV(
    LogisticRegression(
        solver="saga",
        max_iter=2000,
        class_weight="balanced",
        random_state=42,
    ),
    param_grid={"C": [0.01, 0.1, 1, 10, 100]},
    cv=5,
    scoring="precision_weighted",
    n_jobs=-1,
    verbose=1,
)
clf_grid.fit(X_full, y_clf_full)

best_params["logistic_regression"] = {
    "C": clf_grid.best_params_["C"],
    "cv_precision": round(clf_grid.best_score_, 4),
}
print(f"Best C          : {clf_grid.best_params_['C']}")
print(f"Best CV precision: {clf_grid.best_score_:.4f}\n")

print("=" * 50)
print("  TUNING: Ridge Regression")
print("=" * 50)

reg_grid = GridSearchCV(
    Ridge(random_state=42),
    param_grid={"alpha": [0.01, 0.1, 1, 10, 50, 100]},
    cv=5,
    scoring="r2",
    n_jobs=-1,
    verbose=1,
)
reg_grid.fit(X_full, y_reg_full)

best_params["ridge_regression"] = {
    "alpha": reg_grid.best_params_["alpha"],
    "cv_r2": round(reg_grid.best_score_, 4),
}
print(f"Best alpha  : {reg_grid.best_params_['alpha']}")
print(f"Best CV R²  : {reg_grid.best_score_:.4f}\n")

with open(PARAMS_PATH, "w") as f:
    json.dump(best_params, f, indent=4)

print(f"Best params saved to {PARAMS_PATH}")
print(json.dumps(best_params, indent=4))