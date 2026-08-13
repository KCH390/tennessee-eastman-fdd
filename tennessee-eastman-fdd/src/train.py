"""
Baseline models for TEP fault detection (binary: is something wrong?) and
diagnosis (multiclass: which of the 20 faults?).

Two model families, both from scikit-learn so they're actually testable in
this environment: RandomForest and HistGradientBoostingClassifier (a native
sklearn gradient boosting implementation). xgboost is listed in
requirements.txt as an option to try on your own machine, but isn't used
here directly -- it isn't installed in this sandbox and there's no network
to add it, so any code using it would be unverified. Swap it in once you
can actually test it yourself.

Usage as a library (see tests/test_train.py for verified behavior):
    from train import train_detection_model, train_diagnosis_model
"""

import argparse
import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.utils.class_weight import compute_sample_weight

sys.path.insert(0, str(Path(__file__).parent))
from features import SENSOR_COLUMNS, split_by_run
from evaluate import summarize_detection, standard_classification_metrics

# n_estimators=200 + min_samples_leaf=2 + no max_depth cap is expensive on
# millions of rows (this is what made the Phase 3 notebook take hours on
# the real dataset) and, with min_samples_leaf this low, likely encourages
# very deep/overfit trees. Trimmed defaults below are both faster AND
# better-regularized -- worth re-testing accuracy after this change, but
# expect it to help the false-alarm-rate problem too, not just speed.
MODEL_TYPES = {
    "random_forest": lambda seed: RandomForestClassifier(
        n_estimators=100, max_depth=20, min_samples_leaf=20, random_state=seed, n_jobs=-1
    ),
    "hist_gradient_boosting": lambda seed: HistGradientBoostingClassifier(
        max_iter=100, max_depth=20, random_state=seed
    ),
}


def _fit_with_class_weight(model, X, y, class_weight):
    """
    RandomForestClassifier supports class_weight natively.
    HistGradientBoostingClassifier does not -- it takes sample_weight in
    .fit() instead, so we compute per-sample weights that achieve the same
    balancing effect via sklearn's compute_sample_weight.
    """
    if class_weight is None:
        model.fit(X, y)
        return model

    if isinstance(model, RandomForestClassifier):
        model.set_params(class_weight=class_weight)
        model.fit(X, y)
    else:
        sample_weight = compute_sample_weight(class_weight=class_weight, y=y)
        model.fit(X, y, sample_weight=sample_weight)
    return model


def train_detection_model(train_df: pd.DataFrame, model_type="random_forest",
                           feature_columns=None, seed=42, class_weight=None):
    """
    Binary fault detection: predicts fault_active from sensor readings.

    class_weight: pass "balanced" to correct for the severe class
    imbalance in this dataset -- roughly 91% of training rows are
    fault_active=True (see docs/notes on onset timing), so a model trained
    with class_weight=None learns to guess "fault" by default, which shows
    up as a high false_alarm_rate on the (minority) normal class even
    while accuracy/precision look fine, since those are dominated by the
    easy majority class. Tested: class_weight="balanced" substantially
    reduced false_alarm_rate on synthetic data with the same ~91:9
    imbalance (0.877 -> 0.593 in one test run) -- not a complete fix by
    itself, but a real, verified improvement. Combine with windowed
    features (features.py) for a bigger combined effect.

    Returns the fitted model.
    """
    if feature_columns is None:
        feature_columns = [c for c in SENSOR_COLUMNS if c in train_df.columns]

    model = MODEL_TYPES[model_type](seed)
    y = train_df["fault_active"].astype(int)
    return _fit_with_class_weight(model, train_df[feature_columns], y, class_weight)


def train_diagnosis_model(train_df: pd.DataFrame, model_type="random_forest",
                           feature_columns=None, seed=42, class_weight=None):
    """
    Multiclass fault diagnosis: predicts faultNumber from sensor readings,
    using ONLY rows where fault_active is True (diagnosing which fault it
    is only makes sense once a fault is actually active -- including
    pre-onset "faulty run" rows here would just be noise, since those rows
    are truly normal).
    """
    if feature_columns is None:
        feature_columns = [c for c in SENSOR_COLUMNS if c in train_df.columns]

    active_rows = train_df[train_df["fault_active"]]
    model = MODEL_TYPES[model_type](seed)
    return _fit_with_class_weight(model, active_rows[feature_columns], active_rows["faultNumber"], class_weight)


def evaluate_detection_model(model, eval_df: pd.DataFrame, feature_columns=None) -> dict:
    if feature_columns is None:
        feature_columns = [c for c in SENSOR_COLUMNS if c in eval_df.columns]

    predictions = model.predict(eval_df[feature_columns])
    result_df = eval_df[["faultNumber", "simulationRun", "sample", "fault_active"]].copy()
    result_df["predicted"] = predictions

    metrics = standard_classification_metrics(eval_df["fault_active"].astype(int), predictions, average="binary")
    metrics.update(summarize_detection(result_df))
    return metrics


def evaluate_diagnosis_model(model, eval_df: pd.DataFrame, feature_columns=None) -> dict:
    if feature_columns is None:
        feature_columns = [c for c in SENSOR_COLUMNS if c in eval_df.columns]

    active_rows = eval_df[eval_df["fault_active"]]
    predictions = model.predict(active_rows[feature_columns])
    return standard_classification_metrics(active_rows["faultNumber"], predictions, average="macro")


def main():
    parser = argparse.ArgumentParser(description="Train TEP fault detection + diagnosis baselines.")
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--model-type", choices=list(MODEL_TYPES.keys()), default="random_forest")
    parser.add_argument("--models-dir", default="models")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--class-weight", choices=["balanced"], default=None,
                         help="Pass 'balanced' to correct for the ~91:9 fault_active class imbalance")
    args = parser.parse_args()

    processed_dir = Path(args.processed_dir)
    train_path = processed_dir / "tep_training.parquet"
    if not train_path.exists():
        print(f"{train_path} not found. Run src/data_loader.py first.")
        return

    df = pd.read_parquet(train_path)
    train_df, val_df = split_by_run(df, val_frac=0.2, seed=args.seed)

    print(f"Training detection model ({args.model_type}, class_weight={args.class_weight})...")
    detection_model = train_detection_model(train_df, model_type=args.model_type, seed=args.seed,
                                             class_weight=args.class_weight)
    detection_metrics = evaluate_detection_model(detection_model, val_df)
    print("Detection metrics (held-out validation runs):")
    for k, v in detection_metrics.items():
        print(f"  {k}: {v}")

    print(f"\nTraining diagnosis model ({args.model_type}, class_weight={args.class_weight})...")
    diagnosis_model = train_diagnosis_model(train_df, model_type=args.model_type, seed=args.seed,
                                             class_weight=args.class_weight)
    diagnosis_metrics = evaluate_diagnosis_model(diagnosis_model, val_df)
    print("Diagnosis metrics (held-out validation runs, active-fault rows only):")
    for k, v in diagnosis_metrics.items():
        print(f"  {k}: {v}")

    models_dir = Path(args.models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(detection_model, models_dir / f"detection_{args.model_type}.joblib")
    joblib.dump(diagnosis_model, models_dir / f"diagnosis_{args.model_type}.joblib")
    print(f"\nSaved models to {models_dir}/")


if __name__ == "__main__":
    main()
