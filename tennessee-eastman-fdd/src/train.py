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

sys.path.insert(0, str(Path(__file__).parent))
from features import SENSOR_COLUMNS, split_by_run
from evaluate import summarize_detection, standard_classification_metrics

MODEL_TYPES = {
    "random_forest": lambda seed: RandomForestClassifier(
        n_estimators=200, min_samples_leaf=2, random_state=seed, n_jobs=-1
    ),
    "hist_gradient_boosting": lambda seed: HistGradientBoostingClassifier(random_state=seed),
}


def train_detection_model(train_df: pd.DataFrame, model_type="random_forest",
                           feature_columns=None, seed=42):
    """
    Binary fault detection: predicts fault_active from sensor readings.
    Returns the fitted model.
    """
    if feature_columns is None:
        feature_columns = [c for c in SENSOR_COLUMNS if c in train_df.columns]

    model = MODEL_TYPES[model_type](seed)
    model.fit(train_df[feature_columns], train_df["fault_active"].astype(int))
    return model


def train_diagnosis_model(train_df: pd.DataFrame, model_type="random_forest",
                           feature_columns=None, seed=42):
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
    model.fit(active_rows[feature_columns], active_rows["faultNumber"])
    return model


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
    args = parser.parse_args()

    processed_dir = Path(args.processed_dir)
    train_path = processed_dir / "tep_training.parquet"
    if not train_path.exists():
        print(f"{train_path} not found. Run src/data_loader.py first.")
        return

    df = pd.read_parquet(train_path)
    train_df, val_df = split_by_run(df, val_frac=0.2, seed=args.seed)

    print(f"Training detection model ({args.model_type})...")
    detection_model = train_detection_model(train_df, model_type=args.model_type, seed=args.seed)
    detection_metrics = evaluate_detection_model(detection_model, val_df)
    print("Detection metrics (held-out validation runs):")
    for k, v in detection_metrics.items():
        print(f"  {k}: {v}")

    print(f"\nTraining diagnosis model ({args.model_type})...")
    diagnosis_model = train_diagnosis_model(train_df, model_type=args.model_type, seed=args.seed)
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
