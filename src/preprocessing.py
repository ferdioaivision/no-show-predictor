"""
no-show-predictor - preprocessing
Author: Kokouvi Ferdinand DJATA - Ferdio Ai Vision
Rigorous cleaning per spec
"""
import pandas as pd

def load_and_clean(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    if "Handcap" in df.columns:
        df = df.rename(columns={"Handcap": "Handicap"})
    df = df.drop(columns=[c for c in ["PatientId", "AppointmentID"] if c in df.columns])
    df["No-show"] = df["No-show"].map({"Yes": 1, "No": 0}).astype(int)
    invalid = ((df["Age"] < 0) | (df["Age"] > 110)).sum()
    print(f"Invalid Age removed: {invalid}")
    df = df[(df["Age"] >= 0) & (df["Age"] <= 110)].copy()
    df["ScheduledDay"] = pd.to_datetime(df["ScheduledDay"], utc=True)
    df["AppointmentDay"] = pd.to_datetime(df["AppointmentDay"], utc=True)
    df["Delay_days"] = (df["AppointmentDay"] - df["ScheduledDay"]).dt.total_seconds() / 86400.0
    df["Delay_negative_flag"] = (df["Delay_days"] < 0).astype(int)
    df["Delay_days"] = df["Delay_days"].clip(lower=0)
    df["Weekday_appointment"] = df["AppointmentDay"].dt.day_name()
    df["Hour_scheduled"] = df["ScheduledDay"].dt.hour
    df["Handicap_bin"] = (df["Handicap"] > 0).astype(int)
    return df

def get_feature_lists():
    cat_cols = ["Gender", "Neighbourhood", "Weekday_appointment"]
    num_cols = ["Age", "Scholarship", "Hipertension", "Diabetes", "Alcoholism", "Handicap_bin", "SMS_received", "Delay_days", "Hour_scheduled", "Delay_negative_flag"]
    return cat_cols, num_cols
