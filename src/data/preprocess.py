import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

RAW_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "data", "raw_data", "raw_data.csv")
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "processed_data")
TEST_SIZE      = 0.2   # 20% tổng → test
VAL_SIZE       = 0.25  # 25% của 80% còn lại → 20% tổng → val
RANDOM_STATE   = 42
 
TARGET = "final_grade"
 
DROP_COLS = ["student_id"]
 
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"[load]  Shape: {df.shape}")
    print(f"[load]  Columns: {df.columns.tolist()}")
    return df
 
def handle_missing(df: pd.DataFrame) -> pd.DataFrame:
    before = df.shape[0]
 
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
 
    for col in num_cols:
        if df[col].isna().any():
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            print(f"[missing] '{col}': fill median = {median_val:.2f}")
 
    for col in cat_cols:
        if df[col].isna().any():
            mode_val = df[col].mode()[0]
            df[col] = df[col].fillna(mode_val)
            print(f"[missing] '{col}': fill mode = '{mode_val}'")
 
    df = df.dropna()
    print(f"[missing] Dropped {before - df.shape[0]} rows còn NaN. Remaining: {df.shape[0]}")
    return df
 
def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    # Điểm trung bình 3 môn
    df["avg_score"] = (
        df["math_score"] + df["science_score"] + df["english_score"]
    ) / 3

    # Tỉ lệ study_hours / attendance
    df["study_efficiency"] = df["study_hours"] / (df["attendance_percentage"] + 1e-6)

    # Fix: xử lý extra_activities dạng string hoặc số
    if "extra_activities" in df.columns:
        col = df["extra_activities"].astype(str).str.strip().str.lower()
        # Nếu là "yes"/"no" hoặc "true"/"false"
        if col.isin(["yes", "no", "true", "false"]).any():
            df["has_extra"] = col.isin(["yes", "true"]).astype(int)
        else:
            # Nếu là số dạng string ("0", "1", "2",...)
            df["has_extra"] = (pd.to_numeric(col, errors="coerce").fillna(0) > 0).astype(int)

    print(f"[fe]    Features sau engineering: {df.shape[1]} cột")
    return df
 
BINARY_COLS  = ["gender", "internet_access"]          # chỉ 2 giá trị → 0/1
ORDINAL_COLS = ["parent_education"]                   # có thứ tự ngầm định
NOMINAL_COLS = ["school_type", "study_method"]        # không có thứ tự → one-hot
EDUCATION_ORDER = ["none", "primary", "secondary", "bachelor", "master", "phd"]
 
 
def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    le = LabelEncoder()
    for col in BINARY_COLS:
        if col in df.columns:
            df[col] = le.fit_transform(df[col].astype(str))
            print(f"[encode] '{col}': label-encode → {df[col].unique()}")
 
    if "parent_education" in df.columns:
        order_map = {v: i for i, v in enumerate(EDUCATION_ORDER)}
        df["parent_education"] = (
            df["parent_education"]
            .str.lower()
            .map(order_map)
            .fillna(-1)           # giá trị chưa biết → -1
            .astype(int)
        )
        print(f"[encode] 'parent_education': ordinal-encode")
 
    existing_nominal = [c for c in NOMINAL_COLS if c in df.columns]
    if existing_nominal:
        df = pd.get_dummies(df, columns=existing_nominal, drop_first=True, dtype=int)
        print(f"[encode] One-hot: {existing_nominal}")
 
    return df
 
def split_data(df: pd.DataFrame):
    drop = [c for c in DROP_COLS + [TARGET] if c in df.columns]
    X = df.drop(columns=drop)
    y = df[TARGET]

    # Bước 1: tách test (20% tổng)
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=None
    )

    # Bước 2: tách val từ phần còn lại (25% của 80% = 20% tổng)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=VAL_SIZE, random_state=RANDOM_STATE, stratify=None
    )

    print(f"[split]  Train: {X_train.shape} | Val: {X_val.shape} | Test: {X_test.shape}")
    return X_train, X_val, X_test, y_train, y_val, y_test
 
 
def scale_features(X_train: pd.DataFrame, X_val: pd.DataFrame, X_test: pd.DataFrame):
    num_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()

    scaler = StandardScaler()
    X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
    X_val[num_cols]   = scaler.transform(X_val[num_cols])
    X_test[num_cols]  = scaler.transform(X_test[num_cols])

    print(f"[scale]  StandardScaler fit trên {len(num_cols)} cột số")
    return X_train, X_val, X_test, scaler
 
def save_processed(X_train, X_val, X_test, y_train, y_val, y_test):
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    X_train.to_csv(os.path.join(PROCESSED_DIR, "X_train.csv"), index=False)
    X_val.to_csv(os.path.join(PROCESSED_DIR,   "X_val.csv"),   index=False)
    X_test.to_csv(os.path.join(PROCESSED_DIR,  "X_test.csv"),  index=False)
    y_train.to_csv(os.path.join(PROCESSED_DIR, "y_train.csv"), index=False, header=True)
    y_val.to_csv(os.path.join(PROCESSED_DIR,   "y_val.csv"),   index=False, header=True)
    y_test.to_csv(os.path.join(PROCESSED_DIR,  "y_test.csv"),  index=False, header=True)
    print(f"[save]   Đã lưu vào '{PROCESSED_DIR}/' (train / val / test)")

def run_pipeline():
    df = load_data(RAW_PATH)
    df = handle_missing(df)
    df = feature_engineering(df)
    df = encode_features(df)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df)
    X_train, X_val, X_test, scaler = scale_features(X_train, X_val, X_test)
    save_processed(X_train, X_val, X_test, y_train, y_val, y_test)
    print("\n✅ Preprocessing hoàn tất!")
    return X_train, X_val, X_test, y_train, y_val, y_test, scaler
 
if __name__ == "__main__":
    run_pipeline()