import os
import pickle
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay,
    precision_score, recall_score, f1_score, accuracy_score,
    r2_score, mean_absolute_error, mean_squared_error,
)

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR     = os.path.dirname(__file__)
MODEL_DIR    = os.path.join(BASE_DIR, "..", "src", "models")
DATA_DIR     = os.path.join(BASE_DIR, "..", "data", "processed_data")
CLF_PATH     = os.path.join(MODEL_DIR, "logistic_regression.pkl")
REG_PATH     = os.path.join(MODEL_DIR, "linear_regression.pkl")
SCALER_PATH  = os.path.join(MODEL_DIR, "scaler.pkl")

EDUCATION_ORDER   = ["none", "primary", "secondary", "bachelor", "master", "phd"]
TRAVEL_TIME_ORDER = ["<15 min", "15-30 min", "30-60 min", ">60 min"]

# ── Load models ───────────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    with open(CLF_PATH, "rb") as f:
        clf = pickle.load(f)
    with open(REG_PATH, "rb") as f:
        reg = pickle.load(f)
    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)
    return clf, reg, scaler

@st.cache_data
def load_test_data():
    X_test  = pd.read_csv(os.path.join(DATA_DIR, "X_test.csv"))
    y_test  = pd.read_csv(os.path.join(DATA_DIR, "y_test.csv"))
    return X_test, y_test

clf, reg, scaler = load_models()
X_test, y_test   = load_test_data()

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.title("📊 Student Performance")
page = st.sidebar.radio("Chọn trang", ["🔮 Dự đoán", "📈 Đánh giá mô hình"])

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 1: DỰ ĐOÁN
# ═══════════════════════════════════════════════════════════════════════════════
if page == "🔮 Dự đoán":
    st.title("🔮 Dự đoán kết quả học sinh")
    st.markdown("Nhập thông tin học sinh để dự đoán **điểm tổng** và **xếp loại**.")

    col1, col2 = st.columns(2)

    with col1:
        age                  = st.number_input("Tuổi", min_value=10, max_value=25, value=16)
        study_hours          = st.number_input("Giờ học/ngày", min_value=0.0, max_value=24.0, value=3.0, step=0.5)
        attendance_pct       = st.slider("Tỉ lệ đi học (%)", 0, 100, 80)
        math_score           = st.number_input("Điểm Toán", 0.0, 100.0, 70.0)
        science_score        = st.number_input("Điểm Khoa học", 0.0, 100.0, 70.0)
        english_score        = st.number_input("Điểm Tiếng Anh", 0.0, 100.0, 70.0)

    with col2:
        internet_access      = st.selectbox("Có internet?", ["yes", "no"])
        travel_time          = st.selectbox("Thời gian đi học", TRAVEL_TIME_ORDER)
        parent_education     = st.selectbox("Trình độ phụ huynh", EDUCATION_ORDER)
        school_type          = st.selectbox("Loại trường", ["public", "private"])
        study_method         = st.selectbox("Phương pháp học", ["solo", "group study", "mixed", "notes", "online videos", "textbook"])
        has_extra            = st.selectbox("Hoạt động ngoại khóa?", ["yes", "no"])

    if st.button("🚀 Dự đoán", use_container_width=True):
        # Feature engineering
        avg_score        = (math_score + science_score + english_score) / 3
        study_efficiency = study_hours / (attendance_pct + 1e-6)

        # Encode
        internet_val  = 1 if internet_access == "yes" else 0
        travel_val    = TRAVEL_TIME_ORDER.index(travel_time)
        edu_val       = EDUCATION_ORDER.index(parent_education)
        has_extra_val = 1 if has_extra == "yes" else 0

        # One-hot: school_type (ref: private), study_method (ref: solo)
        school_public        = 1 if school_type == "public" else 0
        sm_group             = 1 if study_method == "group study" else 0
        sm_mixed             = 1 if study_method == "mixed" else 0
        sm_notes             = 1 if study_method == "notes" else 0
        sm_online            = 1 if study_method == "online videos" else 0
        sm_textbook          = 1 if study_method == "textbook" else 0

        # Assemble & scale
        features = np.array([[
            age, edu_val, study_hours, attendance_pct, internet_val,
            travel_val, math_score, science_score, english_score,
            avg_score, study_efficiency, has_extra_val,
            school_public, sm_group, sm_mixed, sm_notes, sm_online, sm_textbook
        ]])
        features_scaled = scaler.transform(features)

        # Predict
        score = reg.predict(features_scaled)[0]
        grade = clf.predict(features_scaled)[0]

        st.markdown("---")
        c1, c2 = st.columns(2)
        c1.metric("📝 Điểm tổng dự đoán", f"{score:.1f} / 100")
        c2.metric("🏅 Xếp loại dự đoán", grade)

        # Proba chart
        proba = clf.predict_proba(features_scaled)[0]
        classes = clf.classes_
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.bar(classes, proba, color="steelblue")
        ax.set_xlabel("Xếp loại")
        ax.set_ylabel("Xác suất")
        ax.set_title("Phân phối xác suất xếp loại")
        st.pyplot(fig)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 2: ĐÁNH GIÁ MÔ HÌNH
# ═══════════════════════════════════════════════════════════════════════════════
else:
    st.title("📈 Đánh giá mô hình")

    tab1, tab2 = st.tabs(["📐 Regression", "🏷️ Classification"])

    # ── Regression ────────────────────────────────────────────────────────────
    with tab1:
        st.subheader("Ridge Regression — overall_score")
        y_true_reg = y_test["overall_score"]
        y_pred_reg = reg.predict(X_test)

        c1, c2, c3 = st.columns(3)
        c1.metric("R² Score", f"{r2_score(y_true_reg, y_pred_reg):.4f}")
        c2.metric("MAE",      f"{mean_absolute_error(y_true_reg, y_pred_reg):.4f}")
        c3.metric("RMSE",     f"{mean_squared_error(y_true_reg, y_pred_reg)**0.5:.4f}")

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.scatter(y_true_reg, y_pred_reg, alpha=0.4, s=15, color="steelblue")
        ax.plot([y_true_reg.min(), y_true_reg.max()],
                [y_true_reg.min(), y_true_reg.max()], "r--", linewidth=1)
        ax.set_xlabel("Thực tế")
        ax.set_ylabel("Dự đoán")
        ax.set_title("Actual vs Predicted")
        st.pyplot(fig)

    # ── Classification ────────────────────────────────────────────────────────
    with tab2:
        st.subheader("Logistic Regression — final_grade")
        y_true_clf = y_test["final_grade"]
        y_pred_clf = clf.predict(X_test)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Accuracy",  f"{accuracy_score(y_true_clf, y_pred_clf):.4f}")
        c2.metric("Precision", f"{precision_score(y_true_clf, y_pred_clf, average='weighted', zero_division=0):.4f}")
        c3.metric("Recall",    f"{recall_score(y_true_clf, y_pred_clf, average='weighted', zero_division=0):.4f}")
        c4.metric("F1",        f"{f1_score(y_true_clf, y_pred_clf, average='weighted', zero_division=0):.4f}")

        fig, ax = plt.subplots(figsize=(6, 5))
        cm = confusion_matrix(y_true_clf, y_pred_clf, labels=clf.classes_)
        disp = ConfusionMatrixDisplay(cm, display_labels=clf.classes_)
        disp.plot(ax=ax, colorbar=False, cmap="Blues")
        ax.set_title("Confusion Matrix")
        st.pyplot(fig)