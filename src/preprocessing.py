"""
no-show-predictor - preprocessing

Cleaning rules (all counts are stored in df.attrs["cleaning"]):
  * Handcap -> Handicap (typo in the source file).
  * Rows with Age < 0 or Age > 110 are removed.
  * Delay_days is the number of CALENDAR days between the scheduling date and
    the appointment date. The source file stores the appointment as a date at
    00:00 but the scheduling as a full timestamp, so a difference in
    fractional days would make every same-day appointment look negative.
  * Rows whose appointment date is before the scheduling date are impossible
    and are removed.
  * Same_day = 1 when Delay_days == 0. Same-day appointments have a very low
    no-show rate (4.7 % vs 21 % for a 1-day delay), a jump that a linear model
    cannot represent with Delay_days alone (test AUC 0.66 without it, 0.72 with it).
  * PatientId is kept as `patient_id` only to build patient-grouped splits; it
    is never a model feature.
"""
import pandas as pd

NUMERIC_FEATURES = ["Age", "Scholarship", "Hipertension", "Diabetes", "Alcoholism",
                    "Handicap_bin", "SMS_received", "Delay_days", "Same_day", "Hour_scheduled"]
CATEGORICAL_FEATURES = ["Gender", "Neighbourhood", "Weekday_appointment"]
TARGET = "No-show"


def load_and_clean(path) -> pd.DataFrame:
    df = pd.read_csv(path)
    report = {"raw_rows": int(len(df))}

    df.columns = [c.strip() for c in df.columns]
    df = df.rename(columns={"Handcap": "Handicap", "PatientId": "patient_id"})
    df = df.drop(columns=["AppointmentID"])
    df[TARGET] = df[TARGET].map({"Yes": 1, "No": 0}).astype(int)

    invalid_age = (df["Age"] < 0) | (df["Age"] > 110)
    report["invalid_age_removed"] = int(invalid_age.sum())
    df = df[~invalid_age].copy()

    scheduled = pd.to_datetime(df["ScheduledDay"], utc=True)
    appointment = pd.to_datetime(df["AppointmentDay"], utc=True)
    df["Delay_days"] = (appointment.dt.normalize() - scheduled.dt.normalize()).dt.days
    impossible = df["Delay_days"] < 0
    report["appointment_before_scheduling_removed"] = int(impossible.sum())
    df = df[~impossible].copy()

    df["Same_day"] = (df["Delay_days"] == 0).astype(int)
    df["Weekday_appointment"] = appointment[df.index].dt.day_name()
    df["Hour_scheduled"] = scheduled[df.index].dt.hour
    df["Handicap_bin"] = (df["Handicap"] > 0).astype(int)
    df = df.reset_index(drop=True)

    report["clean_rows"] = int(len(df))
    report["no_show_rate"] = float(df[TARGET].mean())
    report["unique_patients"] = int(df["patient_id"].nunique())
    report["neighbourhoods"] = int(df["Neighbourhood"].nunique())
    df.attrs["cleaning"] = report
    return df


def get_feature_lists():
    return list(CATEGORICAL_FEATURES), list(NUMERIC_FEATURES)
