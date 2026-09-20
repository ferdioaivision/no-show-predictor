"""
no-show-predictor - training pipeline

Design choices (each one fixes a weakness of a plain random split + grid search):
  * Split by PATIENT. 60 % of the appointments of a random test set belong to a
    patient who also appears in the training set; a grouped split removes this
    leak. StratifiedGroupKFold keeps the no-show rate equal in both parts.
  * Hyper-parameters are tuned with patient-grouped 5-fold CV on the training
    set only; models are compared on CV ROC-AUC, never on the test set.
  * No class_weight: it inflates predicted probabilities (mean 0.44 for a 20 %
    base rate) without improving AUC. Ranking and decision are separated: the
    operating threshold is chosen afterwards, on out-of-fold predictions of the
    training set, to reach a target recall.
  * Grids are wide enough to bracket the optimum (a k-NN grid stopping at k=11
    hid the fact that AUC keeps improving up to k~100).

Run:  python src/train.py
"""
import json
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold, cross_val_predict
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

try:  # imported as part of the "src" package (Streamlit app)
    from .evaluation import metrics_at_threshold, ranking_metrics, select_threshold
    from .preprocessing import TARGET, get_feature_lists, load_and_clean
except ImportError:  # run as a script
    from evaluation import metrics_at_threshold, ranking_metrics, select_threshold
    from preprocessing import TARGET, get_feature_lists, load_and_clean

SEED = 42
TARGET_RECALL = 0.65                      # policy choice: flag enough appointments to catch 65 % of no-shows
POLICY_RECALLS = (0.50, 0.65, 0.80)       # operating points reported for the deployed model
DEPLOYED_MODEL = "DecisionTree"           # interpretable; CV AUC is within noise of the other models

BASE = Path(__file__).resolve().parent.parent
DATA_PATH = BASE / "data" / "KaggleV2-May-2016.csv"
MODELS_DIR = BASE / "models"
COMPARISON_PATH = BASE / "data" / "model_comparison.csv"
METADATA_PATH = MODELS_DIR / "metadata.json"


def load_data() -> pd.DataFrame:
    return load_and_clean(DATA_PATH)


def make_cv() -> StratifiedGroupKFold:
    return StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)


def make_split(df: pd.DataFrame):
    """80/20 split by patient (first fold of a 5-fold stratified grouped split)."""
    train_idx, test_idx = next(make_cv().split(df, df[TARGET], df["patient_id"]))
    return train_idx, test_idx


def _preprocessors():
    cat_cols, num_cols = get_feature_lists()
    onehot = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    scaled = ColumnTransformer([("cat", clone(onehot), cat_cols), ("num", StandardScaler(), num_cols)])
    tree = ColumnTransformer([("cat", clone(onehot), cat_cols), ("num", "passthrough", num_cols)])
    return scaled, tree


def build_models():
    scaled, tree = _preprocessors()
    return {
        "LogisticRegression": (
            Pipeline([("pre", scaled), ("clf", LogisticRegression(max_iter=1000, random_state=SEED))]),
            {"clf__C": [0.001, 0.003, 0.01, 0.03, 0.1, 1]},
        ),
        "DecisionTree": (
            Pipeline([("pre", tree), ("clf", DecisionTreeClassifier(random_state=SEED))]),
            {"clf__max_depth": [3, 5, 6, 7, 8, 10], "clf__min_samples_leaf": [1, 50, 100, 300, 600, 1000]},
        ),
        "KNN": (
            Pipeline([("pre", scaled), ("clf", KNeighborsClassifier())]),
            {"clf__n_neighbors": [51, 101, 201, 301]},
        ),
    }


def fit_deployed_pipeline(df: pd.DataFrame, params: dict) -> Pipeline:
    """Refits the deployed decision tree on the training rows (used by the app if the saved file cannot be loaded)."""
    cat_cols, num_cols = get_feature_lists()
    _, tree = _preprocessors()
    pipe = Pipeline([("pre", tree), ("clf", DecisionTreeClassifier(random_state=SEED))]).set_params(**params)
    train_idx, _ = make_split(df)
    return pipe.fit(df.iloc[train_idx][cat_cols + num_cols], df.iloc[train_idx][TARGET])


def main():
    warnings.filterwarnings("ignore")
    df = load_data()
    cleaning = df.attrs["cleaning"]
    print("Cleaning:", cleaning)

    cat_cols, num_cols = get_feature_lists()
    features = cat_cols + num_cols
    train_idx, test_idx = make_split(df)
    X_train, y_train, g_train = df.iloc[train_idx][features], df.iloc[train_idx][TARGET], df.iloc[train_idx]["patient_id"]
    X_test, y_test = df.iloc[test_idx][features], df.iloc[test_idx][TARGET]
    shared = len(set(g_train) & set(df.iloc[test_idx]["patient_id"]))
    print(f"Train {X_train.shape} Test {X_test.shape} | no-show rate train {y_train.mean():.4f} test {y_test.mean():.4f} | patients shared: {shared}")

    cv = make_cv()
    rows, fitted = [], {}
    for name, (pipe, grid) in build_models().items():
        t0 = time.time()
        gs = GridSearchCV(pipe, grid, cv=cv, scoring="roc_auc", n_jobs=-1, refit=True)
        gs.fit(X_train, y_train, groups=g_train)
        proba = gs.best_estimator_.predict_proba(X_test)[:, 1]
        m = ranking_metrics(y_test, proba)
        std = float(gs.cv_results_["std_test_score"][gs.best_index_])
        edge = [k for k, v in grid.items() if gs.best_params_[k] in (min(v), max(v)) and len(v) > 1]
        print(f"{name}: best {gs.best_params_} | CV AUC {gs.best_score_:.4f} +/- {std:.4f} | test AUC {m['roc_auc']:.4f} "
              f"PR-AUC {m['pr_auc']:.4f} Brier {m['brier']:.4f} | {time.time() - t0:.0f}s"
              + (f" | WARNING: best value at the edge of the grid for {edge}" if edge else ""))
        rows.append({"Model": name, "best_params": str(gs.best_params_), "cv_roc_auc": gs.best_score_, "cv_roc_auc_std": std,
                     "test_roc_auc": m["roc_auc"], "test_pr_auc": m["pr_auc"], "test_brier": m["brier"]})
        fitted[name] = gs

    # Reference points: no model at all
    delay = X_test["Delay_days"].to_numpy()
    from sklearn.metrics import average_precision_score, roc_auc_score
    rows.append({"Model": "Delay_days alone (no model)", "best_params": "", "cv_roc_auc": np.nan, "cv_roc_auc_std": np.nan,
                 "test_roc_auc": roc_auc_score(y_test, delay), "test_pr_auc": average_precision_score(y_test, delay), "test_brier": np.nan})
    rows.append({"Model": "Random ranking (prevalence)", "best_params": "", "cv_roc_auc": np.nan, "cv_roc_auc_std": np.nan,
                 "test_roc_auc": 0.5, "test_pr_auc": float(y_test.mean()), "test_brier": np.nan})
    comparison = pd.DataFrame(rows)
    comparison.to_csv(COMPARISON_PATH, index=False, float_format="%.5f")
    print("\n", comparison.round(4).to_string(index=False))

    # Deployed model: threshold chosen on out-of-fold predictions of the training set
    best_pipe = fitted[DEPLOYED_MODEL].best_estimator_
    oof = cross_val_predict(clone(best_pipe), X_train, y_train, groups=g_train, cv=cv, method="predict_proba")[:, 1]
    test_proba = best_pipe.predict_proba(X_test)[:, 1]
    points = {}
    for target in POLICY_RECALLS:
        thr = select_threshold(y_train, oof, target)
        points[str(target)] = {"target_recall_oof": target, **metrics_at_threshold(y_test, test_proba, thr)}
    threshold = points[str(TARGET_RECALL)]["threshold"]
    print(f"\nDeployed {DEPLOYED_MODEL}: threshold {threshold:.4f} (target recall {TARGET_RECALL} on out-of-fold predictions)")
    for k, v in points.items():
        print(f"  target {k}: threshold {v['threshold']:.3f} -> test precision {v['precision']:.3f}, recall {v['recall']:.3f}, flagged {v['flagged_rate']:.1%}")

    pi = permutation_importance(best_pipe, X_test, y_test, scoring="roc_auc", n_repeats=5, random_state=0)
    importance = pd.Series(pi.importances_mean, index=features).sort_values(ascending=False)

    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(fitted["LogisticRegression"].best_estimator_, MODELS_DIR / "model_LogisticRegression.joblib")
    joblib.dump(best_pipe, MODELS_DIR / "model_DecisionTree.joblib")   # k-NN is not saved: it stores the whole training set (~70 MB)

    metadata = {
        "trained_with": {"scikit-learn": sklearn.__version__, "pandas": pd.__version__, "numpy": np.__version__},
        "seed": SEED,
        "cleaning": cleaning,
        "split": {"train_rows": int(len(train_idx)), "test_rows": int(len(test_idx)), "shared_patients": int(shared),
                  "no_show_rate_train": float(y_train.mean()), "no_show_rate_test": float(y_test.mean())},
        "deployed_model": DEPLOYED_MODEL,
        "deployed_params": {k: (v.item() if hasattr(v, "item") else v) for k, v in fitted[DEPLOYED_MODEL].best_params_.items()},
        "cv_roc_auc": float(fitted[DEPLOYED_MODEL].best_score_),
        "test": ranking_metrics(y_test, test_proba),
        "target_recall": TARGET_RECALL,
        "threshold": threshold,
        "operating_points": points,
        "permutation_importance_auc_drop": {k: float(v) for k, v in importance.items()},
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2))
    print(f"\nSaved models to {MODELS_DIR} and comparison to {COMPARISON_PATH}")


if __name__ == "__main__":
    main()
