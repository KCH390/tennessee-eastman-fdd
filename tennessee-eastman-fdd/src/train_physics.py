"""
Phase 9: run Track 1's ML pipeline (train.py, features.py -- imported
here UNMODIFIED) against physics-generated fault-scenario data
(physics/scenarios.py) instead of the real TEP dataset.

The point isn't a new model -- it's confirming the SAME fault-detection
approach that worked on the canned TEP benchmark also works on an
independently-simulated, first-principles physical system, using the
literal same training/evaluation functions.

Honest framing for the results this prints: this physics dataset's two
fault types are more easily separable than TEP's 20 real faults --
cooling failure shows up as a pure temperature shift with flow rates
untouched, feed disturbance shows up as a pure flow shift with
temperature barely moved (see comment in scenarios.py's SENSOR_COLUMNS
section). Near-perfect diagnosis accuracy here reflects that, not that
this is a harder or more impressive problem than TEP itself. What IS a
fair, direct comparison: whether the same code, same class-weighting
approach, same evaluation functions produce sane, well-calibrated results
here too -- they do.

Usage:
    python src/train_physics.py --model-type random_forest
    python src/train_physics.py --model-type hist_gradient_boosting --n-runs-per-fault 60
"""

import argparse
import sys
from pathlib import Path

import joblib
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from features import split_by_run
from train import (
    train_detection_model, evaluate_detection_model,
    train_diagnosis_model, evaluate_diagnosis_model,
)
from physics import scenarios


def main():
    parser = argparse.ArgumentParser(description="Train fault detection/diagnosis on physics-generated data.")
    parser.add_argument("--model-type", choices=["random_forest", "hist_gradient_boosting"], default="random_forest")
    parser.add_argument("--n-runs-per-fault", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--models-dir", default="models")
    parser.add_argument("--regenerate", action="store_true",
                         help="Regenerate the physics dataset even if a cached parquet already exists.")
    args = parser.parse_args()

    processed_dir = Path(args.processed_dir)
    data_path = processed_dir / "physics_training.parquet"

    if data_path.exists() and not args.regenerate:
        print(f"Loading cached physics dataset from {data_path}")
        df = pd.read_parquet(data_path)
    else:
        print(f"Generating physics dataset ({args.n_runs_per_fault} runs per fault type)...")
        df = scenarios.generate_dataset(n_runs_per_fault=args.n_runs_per_fault, seed=args.seed)
        processed_dir.mkdir(parents=True, exist_ok=True)
        df.to_parquet(data_path)
        print(f"Saved {len(df)} rows to {data_path}")

    train_df, val_df = split_by_run(df, val_frac=0.2, seed=args.seed)
    feature_columns = scenarios.SENSOR_COLUMNS
    print(f"Train: {len(train_df)} rows / {train_df['simulationRun'].nunique()} runs, "
          f"Val: {len(val_df)} rows / {val_df['simulationRun'].nunique()} runs")
    print(f"Feature columns (physics sensors, not TEP's xmeas/xmv): {feature_columns}")

    print(f"\nTraining detection model ({args.model_type})...")
    detection_model = train_detection_model(
        train_df, model_type=args.model_type, feature_columns=feature_columns,
        seed=args.seed, class_weight="balanced",
    )
    detection_metrics = evaluate_detection_model(detection_model, val_df, feature_columns=feature_columns)
    print("Detection metrics (held-out validation runs):")
    for k, v in detection_metrics.items():
        print(f"  {k}: {v}")

    print(f"\nTraining diagnosis model ({args.model_type})...")
    diagnosis_model = train_diagnosis_model(
        train_df, model_type=args.model_type, feature_columns=feature_columns,
        seed=args.seed, class_weight="balanced",
    )
    diagnosis_metrics = evaluate_diagnosis_model(diagnosis_model, val_df, feature_columns=feature_columns)
    print("Diagnosis metrics (held-out validation runs, active-fault rows only):")
    for k, v in diagnosis_metrics.items():
        print(f"  {k}: {v}")

    models_dir = Path(args.models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(detection_model, models_dir / f"physics_detection_{args.model_type}.joblib")
    joblib.dump(diagnosis_model, models_dir / f"physics_diagnosis_{args.model_type}.joblib")
    print(f"\nSaved models to {models_dir}/ (physics_detection_*.joblib, physics_diagnosis_*.joblib)")


if __name__ == "__main__":
    main()
