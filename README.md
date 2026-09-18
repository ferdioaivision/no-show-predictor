# no-show-predictor
### Predicting medical appointment no-shows - Binary classification decision support system

![Python](https://img.shields.io/badge/Python-3.9%2B-blue) ![ROC-AUC](https://img.shields.io/badge/ROC--AUC-0.7246-brightgreen) ![Status](https://img.shields.io/badge/Status-Complete-success) ![License](https://img.shields.io/badge/License-MIT-green) [![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://ferdioaivision-no-show-predictor.streamlit.app/)

**Author:** Kokouvi Ferdinand DJATA
**Brand:** Ferdio Ai Vision
**GitHub:** @ferdioaivision

#### Problem Formulation
Binary supervised classification: Estimate P(No-show=1 | X) where X = demographic, socio-economic, temporal, medical features available at scheduling time.

#### Dataset - Verified
- Source: Kaggle Medical Appointment No Shows - KaggleV2-May-2016.csv 10488 Ko
- Raw: 110527 rows
- After cleaning: 110521 rows (6 invalid Age removed)
- Imbalance: 79.81% Show / 20.19% No-show
- Typo fixed: Handcap -> Handicap
- Cleaned: cleaned.csv 12133 Ko

#### Feature Engineering
- Delay_days = (AppointmentDay - ScheduledDay) clipped 0 + Delay_negative_flag
- Weekday_appointment, Hour_scheduled, Handicap_bin
- One-hot: Gender, Neighbourhood (82), Weekday

#### Methodology (8 Steps)
1. EDA: 10 figures + Chi2 tests
2. Preprocessing: drop IDs, age filter 0-110
3. Split: stratified 80/20 seed 42, 5-fold CV ROC-AUC
4. Imbalance: class_weight balanced + threshold 0.35
5. Models: LogReg C {0.01,0.1,1,10}, DecisionTree depth {3,5,7,10} min_split {2,5,10}, KNN k {3,5,7,9,11}
6. Evaluation: confusion matrix, accuracy, precision, recall, F1, ROC-AUC, PR-AUC
7. Interpretation
8. Deployment: Streamlit app

#### Verified Results - Your Local Execution

| Model | Best Params | CV ROC-AUC | Test Accuracy | Precision | Recall | F1 | Test ROC-AUC | PR-AUC |
|-------|-------------|------------|---------------|-----------|--------|----|--------------|--------|
| LogisticRegression | C=0.01 | 0.72133 | 0.5623 | 0.2987 | 0.8669 | 0.4443 | 0.72032 | 0.3417 |
| **DecisionTree** | **depth=7, min_split=10** | **0.72529** | **0.5538** | **0.2960** | **0.8776** | **0.4426** | **0.72462** | **0.3440** |
| KNN | k=11 | 0.69837 | 0.7855 | 0.4004 | 0.1248 | 0.1902 | 0.70292 | 0.3276 |

Best: DecisionTree ROC-AUC 0.72462 >0.70 required, Recall 0.8776 >0.65 required.

#### Key Insights
- Delay_days strongest: 12% no-show delay 0 vs >35% delay 30+
- SMS paradox: 16.8% without SMS vs 27.5% with SMS (selection bias)
- Scholarship 19.8% vs 23.6%, Saturday highest weekday

#### Limitations and Scientific Perspectives

##### Dataset and External Validity
- Single-center data from Brazil 2016 (110521 rows after cleaning). No geographic, temporal, or institutional external validation. Generalization to other health systems (e.g., Togo, France) requires local recalibration.
- No patient history of previous no-shows, distance to clinic, transport, weather, or provider-level features. This limits predictive ceiling (current best ROC-AUC 0.72462).

##### Modeling and Calibration
- Imbalance handled with `class_weight=balanced` and operational threshold 0.35 optimized for recall (0.8776). This overestimates P(No-show) and requires probability calibration (Platt/Isotonic) before use as true risk score.
- Best model is DecisionTree depth=7 (interpretable, 23 Ko). Trade-off: lower capacity than ensembles. RandomForest/XGBoost may improve ROC-AUC but reduce interpretability.
- No calibration curve, no decision curve analysis, no fairness audit across Neighbourhood (82 categories) and Scholarship.

##### Statistical and Causal Limitations
- Observational study. SMS_received shows paradoxical association (16.8% no-show without SMS vs 27.5% with SMS) due to selection bias: high-risk patients are more likely to receive SMS. Correlation does not imply causal effect of SMS.
- Delay_days is strongest predictor (12% no-show at delay 0 vs >35% at delay 30+) but confounded by scheduling policies.

##### Operational and Ethical
- Decision support only. Must not be used to deny appointments. Requires explicit threshold policy (e.g., P >= 0.35 = targeted phone call, not overbooking alone).
- No patient data stored in the Streamlit app. For hospital deployment, compliance with local data protection and audit logging is required.

##### Future Work
1. Local recalibration: retrain on hospital-specific data with temporal validation (train on past months, test on future months).
2. Add features: history of no-shows, distance, weather, provider, clinic load.
3. Calibration and cost-sensitive evaluation: calibrated probabilities, expected cost of overbooking vs idle slot.
4. Fairness and robustness: subgroup ROC-AUC by Age, Gender, Neighbourhood, Scholarship; stability over time.
5. Prospective pilot: A/B test of targeted reminders vs standard care, with primary outcome no-show rate and secondary outcome utilization.

#### Structure Verified
```
no-show-predictor/
├── data/ (KaggleV2-May-2016.csv, cleaned.csv, chi2_results.csv, model_comparison.csv)
├── figures/ (21 PNG: 01_ to 10_ + cm/pr/roc + threshold)
├── models/ (best_model 23 Ko, feature_list 1 Ko, DecisionTree 23 Ko, KNN 69081 Ko, LogReg 7 Ko)
├── notebooks/ (EDA_and_Modeling.ipynb 1302 Ko)
├── src/ (evaluation.py, preprocessing.py, train.py)
├── app.py, requirements.txt, README.md
```

#### How to Run
pip install -r requirements.txt
python src/train.py
streamlit run app.py

#### License
MIT - Ferdio Ai Vision - Kokouvi Ferdinand DJATA
