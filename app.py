"""
no-show-predictor - Streamlit app
Author: Kokouvi Ferdinand DJATA - Ferdio Ai Vision
"""
import streamlit as st
import joblib
import pandas as pd
import pathlib

BASE = pathlib.Path(__file__).parent
BEST_PATH = BASE / "models" / "best_model.joblib"
MODEL_PATH = BASE / "models" / "model_DecisionTree.joblib"

st.set_page_config(
    page_title="no-show-predictor",
    layout="wide"
)

@st.cache_resource
def load_model():
    if BEST_PATH.exists():
        return joblib.load(BEST_PATH)
    return joblib.load(MODEL_PATH)

model = load_model()
clf = model.named_steps["clf"]
preprocessor = model.named_steps["pre"]

st.sidebar.header("Patient and Appointment")

age = st.sidebar.slider("Age", 0, 110, 54)
gender = st.sidebar.selectbox("Gender", ["F", "M"])
neighbourhood = st.sidebar.text_input("Neighbourhood", "Jardim da Penha")
scholarship = st.sidebar.selectbox("Scholarship", [0, 1])
hipertension = st.sidebar.selectbox("Hipertension", [0, 1])
diabetes = st.sidebar.selectbox("Diabetes", [0, 1])
alcoholism = st.sidebar.selectbox("Alcoholism", [0, 1])
handicap_bin = st.sidebar.selectbox("Handicap_bin", [0, 1])
sms_received = st.sidebar.selectbox("SMS_received", [0, 1])
delay_days = st.sidebar.slider("Delay_days", 0, 100, 5)
hour_scheduled = st.sidebar.slider("Hour_scheduled", 0, 23, 10)
weekday = st.sidebar.selectbox("Weekday_appointment", ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"])
threshold = st.sidebar.slider("Operational threshold", 0.1, 0.9, 0.35, 0.05)

st.title("no-show-predictor")
st.caption("Predicting medical appointment no-shows - Binary classification decision support system | Author: Ferdio Ai Vision")

st.success("Model loaded")

if st.button("Predict No-show Probability"):
    X = pd.DataFrame([{
        "Gender": gender,
        "Neighbourhood": neighbourhood,
        "Weekday_appointment": weekday,
        "Age": age,
        "Scholarship": scholarship,
        "Hipertension": hipertension,
        "Diabetes": diabetes,
        "Alcoholism": alcoholism,
        "Handicap_bin": handicap_bin,
        "SMS_received": sms_received,
        "Delay_days": delay_days,
        "Hour_scheduled": hour_scheduled,
        "Delay_negative_flag": 0
    }])

    proba = model.predict_proba(X)[0, 1]
    pred = int(proba >= threshold)
    leaf_id = clf.apply(preprocessor.transform(X))[0]

    col1, col2 = st.columns([2, 1])

    with col1:
        st.metric("P(No-show=1 | X)", f"{proba:.3f}")
        st.metric("Prediction at threshold", "No-show" if pred == 1 else "Show")

        if pred == 1:
            st.warning("High risk: Recommendation = targeted phone call + personalized SMS, consider overbooking mitigation")
        else:
            st.info("Low risk: Standard reminder")

        with st.expander("Model Details and Interpretability"):
            st.write(f"Leaf reached: {leaf_id}")
            st.write(f"Maximum depth: {clf.max_depth}")
            st.write(f"Number of leaves: {clf.get_n_leaves()}")
            st.write("Decision path learned from 110521 appointments. See notebooks/EDA_and_Modeling.ipynb for SHAP and feature importance.")

    with col2:
        with st.container(border=True):
            st.subheader("About the Model")
            st.write("Model: DecisionTree")
            st.write("Parameters: max_depth=7, min_samples_split=10, class_weight=balanced")
            st.write("5-fold CV ROC-AUC: 0.72529")
            st.write("Test ROC-AUC: 0.72462")
            st.write("Test Recall: 0.8776")
            st.write("Dataset: Kaggle Medical Appointment No Shows - 110521 rows")

        with st.expander("Methodology"):
            st.write("Stratified split 80/20 seed 42, 5-fold CV ROC-AUC")
            st.write("Imbalance handling: class_weight balanced + threshold 0.35 optimized for recall")
            st.write("Top predictors: Delay_days, Scholarship, Age, Neighbourhood")

st.divider()
st.caption("Operational threshold 0.35 balances precision vs recall. Best model ROC-AUC 0.72462 > 0.70 required, Recall 0.8776 > 0.65 required. Decision support only - not for denying appointments. Requires local recalibration. No patient data stored. MIT License - Ferdio Ai Vision")