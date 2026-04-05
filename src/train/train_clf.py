import os
import pandas as pd
import pickle
from sklearn.linear_model import LogisticRegression

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "processed_data")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
MODEL_PATH = os.path.join(MODEL_DIR, "logistic_regression.pkl")

os.makedirs(MODEL_DIR, exist_ok=True)

# Data đã được scale + encode đầy đủ từ preprocess.py
X_train = pd.read_csv(os.path.join(DATA_DIR, "X_train.csv"))
y_train = pd.read_csv(os.path.join(DATA_DIR, "y_train.csv"))["final_grade"]

# class_weight="balanced" như một lớp bảo vệ thêm dù data đã cân bằng
model = LogisticRegression(
    C=0.1,
    max_iter=1000,
    class_weight="balanced",
    random_state=42,
)
model.fit(X_train, y_train)

train_acc = model.score(X_train, y_train)
print(f"Train Accuracy: {train_acc:.4f}")

with open(MODEL_PATH, "wb") as f:
    pickle.dump(model, f)

print(f"Logistic Regression model saved to {MODEL_PATH}")