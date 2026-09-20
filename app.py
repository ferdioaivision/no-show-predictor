"""
no-show-predictor - Streamlit app
Author: Kokouvi Ferdinand DJATA - Ferdio Ai Vision
"""
import json
import pathlib

import joblib
import pandas as pd
import streamlit as st

from src.train import fit_deployed_pipeline, load_data

BASE = pathlib.Path(__file__).parent
MODEL_PATH = BASE / "models" / "model_DecisionTree.joblib"
META_PATH = BASE / "models" / "metadata.json"

st.set_page_config(page_title="no-show-predictor", layout="wide")


@st.cache_data
def load_metadata():
    return json.loads(META_PATH.read_text())


@st.cache_resource
def load_model():
    try:
        return joblib.load(MODEL_PATH)
    except Exception:  # e.g. scikit-learn version different from the one that saved the file
        st.warning("The saved model could not be loaded (likely a scikit-learn version mismatch). Refitting it from the dataset.")
        return fit_deployed_pipeline(load_data(), load_metadata()["deployed_params"])


meta = load_metadata()
model = load_model()
clf = model.named_steps["clf"]
preprocessor = model.named_steps["pre"]
neighbourhoods = sorted(preprocessor.named_transformers_["cat"].categories_[1])

base_rate = meta["split"]["no_show_rate_train"]
default_threshold = round(meta["threshold"], 2)

st.sidebar.header("Patient and appointment")
age = st.sidebar.slider("Age", 0, 110, 40)
gender = st.sidebar.selectbox("Gender", ["F", "M"])
neighbourhood = st.sidebar.selectbox("Neighbourhood", neighbourhoods, index=neighbourhoods.index("JARDIM DA PENHA"))
scholarship = st.sidebar.checkbox("Scholarship (social programme)")
hipertension = st.sidebar.checkbox("Hypertension")
diabetes = st.sidebar.checkbox("Diabetes")
alcoholism = st.sidebar.checkbox("Alcoholism")
handicap = st.sidebar.checkbox("Handicap")
sms_received = st.sidebar.checkbox("SMS reminder received")
delay_days = st.sidebar.slider("Days between scheduling and appointment", 0, 180, 5)
hour_scheduled = st.sidebar.slider("Hour of scheduling", 0, 23, 10)
weekday = st.sidebar.selectbox("Appointment weekday", ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"])
threshold = st.sidebar.slider("Decision threshold", 0.05, 0.80, default_threshold, 0.01,
                              help="Patients with a predicted risk at or above this value are flagged for a reminder call.")

st.title("no-show-predictor")
st.caption("Predicting medical appointment no-shows | decision support, not a diagnostic or scheduling rule | Author: Ferdio Ai Vision")

if st.button("Predict no-show probability"):
    X = pd.DataFrame([{
        "Gender": gender, "Neighbourhood": neighbourhood, "Weekday_appointment": weekday,
        "Age": age, "Scholarship": int(scholarship), "Hipertension": int(hipertension), "Diabetes": int(diabetes),
        "Alcoholism": int(alcoholism), "Handicap_bin": int(handicap), "SMS_received": int(sms_received),
        "Delay_days": delay_days, "Same_day": int(delay_days == 0), "Hour_scheduled": hour_scheduled,
    }])
    proba = float(model.predict_proba(X)[0, 1])
    flagged = proba >= threshold
    leaf_id = int(clf.apply(preprocessor.transform(X))[0])

    col1, col2 = st.columns([2, 1])
    with col1:
        st.metric("Predicted no-show probability", f"{proba:.1%}", f"{proba / base_rate:.1f}x the average rate ({base_rate:.1%})", delta_color="off")
        st.metric("Decision at threshold", "Flag for reminder call" if flagged else "Standard reminder")
        if flagged:
            st.warning("Above the threshold: suggested action is a targeted reminder call. This must not be used to refuse or cancel appointments.")
        else:
            st.info("Below the threshold: standard reminder.")
        with st.expander("Model details"):
            st.write(f"Leaf reached: {leaf_id} | maximum depth: {clf.max_depth} | number of leaves: {clf.get_n_leaves()}")
            top = [(k, v) for k, v in meta["permutation_importance_auc_drop"].items() if v >= 0.002]
            st.write("Most influential features (drop in test ROC-AUC when shuffled): "
                     + ", ".join(f"{k} ({v:.3f})" for k, v in top))
    with col2:
        with st.container(border=True):
            st.subheader("About the model")
            st.write(f"Decision tree, {meta['deployed_params']}")
            st.write(f"Test ROC-AUC: {meta['test']['roc_auc']:.3f} | PR-AUC: {meta['test']['pr_auc']:.3f} (random: {meta['split']['no_show_rate_test']:.3f})")
            st.write(f"Data: Kaggle Medical Appointment No Shows, {meta['cleaning']['clean_rows']:,} appointments, Brazil 2016")

st.divider()
st.subheader("Operating points on the held-out test set")
ops = pd.DataFrame(meta["operating_points"]).T[["target_recall_oof", "threshold", "precision", "recall", "flagged_rate"]]
ops.columns = ["Target recall (training CV)", "Threshold", "Test precision", "Test recall", "Share of appointments flagged"]
st.dataframe(ops.round(3).reset_index(drop=True))
st.caption("Lower thresholds catch more no-shows but flag more appointments. The model gives a probability calibrated to this dataset; "
           "it needs local recalibration and validation before any use in another health system. No patient data is stored.")
