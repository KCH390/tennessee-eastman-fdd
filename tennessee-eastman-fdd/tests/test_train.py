"""
End-to-end test of the training pipeline in src/train.py, using synthetic
data with a DELIBERATELY learnable pattern (not real TEP data -- that's
untestable here, see module docstrings elsewhere for why). This validates
the pipeline mechanics (data in -> model out -> predictions -> metrics all
line up correctly), not real-world model performance.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from train import (
    train_detection_model, train_diagnosis_model,
    evaluate_detection_model, evaluate_diagnosis_model,
)
from features import SENSOR_COLUMNS


def make_learnable_dataset(n_runs_per_fault=20, n_samples=40, onset=20, seed=0):
    """
    faultNumber 0: fault-free, xmeas_1 hovers near 0 the whole run.
    faultNumber 1: xmeas_1 jumps to ~10 after onset (clearly separable).
    faultNumber 2: xmeas_1 jumps to ~-10 after onset (separable from both
                   fault-free AND fault 1, for the diagnosis task).
    """
    rng = np.random.RandomState(seed)
    rows = []

    for fault_number, shift in [(0, 0.0), (1, 10.0), (2, -10.0)]:
        n_runs = n_runs_per_fault if fault_number > 0 else n_runs_per_fault * 2
        for run in range(1, n_runs + 1):
            for s in range(1, n_samples + 1):
                is_active = fault_number > 0 and s > onset
                row = {
                    "faultNumber": fault_number, "simulationRun": run, "sample": s,
                    "fault_active": is_active,
                }
                base = shift if is_active else 0.0
                for col in SENSOR_COLUMNS:
                    row[col] = base + rng.normal(0, 0.5)
                rows.append(row)

    return pd.DataFrame(rows)


def test_detection_model_learns_the_separable_pattern():
    df = make_learnable_dataset(seed=1)
    train_df = df[df.simulationRun <= 15]
    val_df = df[df.simulationRun > 15]

    model = train_detection_model(train_df, model_type="random_forest", seed=42)
    metrics = evaluate_detection_model(model, val_df)

    assert metrics["accuracy"] > 0.9, f"expected high accuracy on a clearly separable pattern, got {metrics['accuracy']}"
    assert metrics["false_alarm_rate"] < 0.1, f"expected a low false alarm rate, got {metrics['false_alarm_rate']}"
    assert metrics["detection_rate"] > 0.9, f"expected most faults detected, got {metrics['detection_rate']}"


def test_diagnosis_model_distinguishes_fault_types():
    df = make_learnable_dataset(seed=2)
    train_df = df[df.simulationRun <= 15]
    val_df = df[df.simulationRun > 15]

    model = train_diagnosis_model(train_df, model_type="random_forest", seed=42)
    metrics = evaluate_diagnosis_model(model, val_df)

    assert metrics["accuracy"] > 0.9, f"expected high accuracy distinguishing +10 vs -10 shift, got {metrics['accuracy']}"


def test_diagnosis_model_only_trains_on_active_rows():
    # if pre-onset rows leaked into diagnosis training, the model would see
    # near-zero features labeled as faultNumber 1 or 2, corrupting it
    df = make_learnable_dataset(n_samples=40, onset=20, seed=3)
    train_df = df[df.simulationRun <= 15]

    active_only = train_df[train_df["fault_active"]]
    assert (active_only["faultNumber"] > 0).all(), "sanity check on the fixture itself"

    model = train_diagnosis_model(train_df, model_type="random_forest", seed=42)
    # predicting on a clean, held-out active-only sample should still work well
    val_df = df[df.simulationRun > 15]
    metrics = evaluate_diagnosis_model(model, val_df)
    assert metrics["accuracy"] > 0.9


def test_hist_gradient_boosting_model_type_also_works():
    df = make_learnable_dataset(seed=4)
    train_df = df[df.simulationRun <= 15]
    val_df = df[df.simulationRun > 15]

    model = train_detection_model(train_df, model_type="hist_gradient_boosting", seed=42)
    metrics = evaluate_detection_model(model, val_df)
    assert metrics["accuracy"] > 0.9


def make_imbalanced_dataset(seed, pos_ratio=0.914, n_normal=600, signal=0.6):
    """
    Mimics the real dataset's ~91.4% fault_active imbalance, with a
    modest-but-real signal in both classes (not a trivially separable
    toy case) -- tests whether class_weight recovers minority-class
    performance, not whether the problem is solvable at all.
    """
    rng = np.random.RandomState(seed)
    n_pos = int(n_normal * pos_ratio / (1 - pos_ratio))
    rows = []
    sample_counter = 0
    for is_active, n in [(False, n_normal), (True, n_pos)]:
        shift = signal if is_active else 0.0
        for i in range(n):
            sample_counter += 1
            row = {
                "faultNumber": 1 if is_active else 0,
                "simulationRun": 1,
                "sample": sample_counter,
                "fault_active": is_active,
            }
            for col in SENSOR_COLUMNS:
                row[col] = shift + rng.normal(0, 1.0)
            rows.append(row)
    df = pd.DataFrame(rows)
    return df.sample(frac=1, random_state=seed).reset_index(drop=True)


def test_class_weight_balanced_reduces_false_alarm_rate_on_imbalanced_data():
    train_df = make_imbalanced_dataset(seed=1)
    val_df = make_imbalanced_dataset(seed=2)

    model_unweighted = train_detection_model(train_df, model_type="random_forest", seed=42, class_weight=None)
    model_weighted = train_detection_model(train_df, model_type="random_forest", seed=42, class_weight="balanced")

    metrics_unweighted = evaluate_detection_model(model_unweighted, val_df)
    metrics_weighted = evaluate_detection_model(model_weighted, val_df)

    assert metrics_weighted["false_alarm_rate"] < metrics_unweighted["false_alarm_rate"], (
        f"expected class_weight='balanced' to reduce false_alarm_rate, got "
        f"unweighted={metrics_unweighted['false_alarm_rate']:.3f} "
        f"weighted={metrics_weighted['false_alarm_rate']:.3f}"
    )


def test_class_weight_works_for_hist_gradient_boosting_too():
    # HistGradientBoostingClassifier doesn't support class_weight natively --
    # this exercises the compute_sample_weight fallback path specifically
    train_df = make_imbalanced_dataset(seed=3)
    val_df = make_imbalanced_dataset(seed=4)

    model_unweighted = train_detection_model(train_df, model_type="hist_gradient_boosting", seed=42, class_weight=None)
    model_weighted = train_detection_model(train_df, model_type="hist_gradient_boosting", seed=42, class_weight="balanced")

    metrics_unweighted = evaluate_detection_model(model_unweighted, val_df)
    metrics_weighted = evaluate_detection_model(model_weighted, val_df)

    assert metrics_weighted["false_alarm_rate"] < metrics_unweighted["false_alarm_rate"]


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
