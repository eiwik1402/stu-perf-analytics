import os
import pandas as pd
import pickle
from sklearn.linear_model import Ridge

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "processed_data")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
MODEL_PATH = os.path.join(MODEL_DIR, "linear_regression.pkl")

os.makedirs(MODEL_DIR, exist_ok=True)

# Data đã được scale + encode đầy đủ từ preprocess.py
X_train = pd.read_csv(os.path.join(DATA_DIR, "X_train.csv"))
y_train = pd.read_csv(os.path.join(DATA_DIR, "y_train.csv"))["overall_score"]

# Ridge thay LinearRegression: tránh overfitting, cải thiện precision
model = Ridge(alpha=1.0)
model.fit(X_train, y_train)

train_score = model.score(X_train, y_train)
print(f"Train R² score: {train_score:.4f}")

with open(MODEL_PATH, "wb") as f:
    pickle.dump(model, f)

print(f"Ridge Regression model saved to {MODEL_PATH}")