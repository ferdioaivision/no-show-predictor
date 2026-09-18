"""
no-show-predictor - training pipeline
Seed 42, stratified split 80/20, 5-fold CV, ROC-AUC scoring
"""
import pandas as pd, pathlib, joblib
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from preprocessing import load_and_clean, get_feature_lists

SEED = 42
BASE = pathlib.Path(__file__).resolve().parent.parent
DATA_PATH = BASE / "data" / "KaggleV2-May-2016.csv"
CLEAN_PATH = BASE / "data" / "cleaned.csv"

def main():
    if CLEAN_PATH.exists():
        df = pd.read_csv(CLEAN_PATH)
    else:
        df = load_and_clean(str(DATA_PATH))
        df.to_csv(CLEAN_PATH, index=False)

    cat_cols, num_cols = get_feature_lists()
    X = df[cat_cols + num_cols]
    y = df["No-show"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=SEED, stratify=y)
    print(f"Train {X_train.shape} Test {X_test.shape} Positive rate train {y_train.mean():.3f}")

    pre_scaled = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
        ("num", StandardScaler(), num_cols)
    ])
    pre_tree = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
        ("num", "passthrough", num_cols)
    ])

    models = {
        "LogisticRegression": (
            Pipeline([("pre", pre_scaled), ("clf", LogisticRegression(class_weight="balanced", max_iter=1000, random_state=SEED))]),
            {"clf__C": [0.01, 0.1, 1, 10]}
        ),
        "DecisionTree": (
            Pipeline([("pre", pre_tree), ("clf", DecisionTreeClassifier(class_weight="balanced", random_state=SEED))]),
            {"clf__max_depth": [3,5,7,10], "clf__min_samples_split": [2,5,10]}
        ),
        "KNN": (
            Pipeline([("pre", pre_scaled), ("clf", KNeighborsClassifier())]),
            {"clf__n_neighbors": [3,5,7,9,11]}
        )
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    best_overall = None
    best_score = -1

    for name, (pipe, grid) in models.items():
        print(f"\nTraining {name} with grid {grid}")
        gs = GridSearchCV(pipe, grid, cv=cv, scoring="roc_auc", n_jobs=-1)
        gs.fit(X_train, y_train)
        y_pred = gs.best_estimator_.predict(X_test)
        y_proba = gs.best_estimator_.predict_proba(X_test)[:,1]
        print(f"Best params {gs.best_params_} CV ROC-AUC {gs.best_score_:.4f} Test ROC-AUC {roc_auc_score(y_test, y_proba):.4f}")
        print(classification_report(y_test, y_pred, digits=4))
        joblib.dump(gs.best_estimator_, BASE / f"models/model_{name}.joblib")
        if roc_auc_score(y_test, y_proba) > best_score:
            best_score = roc_auc_score(y_test, y_proba)
            best_overall = gs.best_estimator_

    joblib.dump(best_overall, BASE / "models/best_model.joblib")
    joblib.dump(cat_cols+num_cols, BASE / "models/feature_list.joblib")
    print(f"\nBest overall model saved with ROC-AUC {best_score:.4f}")

if __name__ == "__main__":
    main()
