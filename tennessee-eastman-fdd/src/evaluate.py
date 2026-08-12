"""
Evaluation for TEP fault detection/diagnosis -- goes beyond plain accuracy,
which hides what actually matters for a monitoring system: how FAST does it
catch a real fault (detection delay), and how often does it cry wolf during
normal operation (false alarm rate)?

Both are pure functions operating on a DataFrame with columns:
    faultNumber, simulationRun, sample, fault_active (true label), predicted
so they're testable without a trained model.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
)


def detection_delay(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each (faultNumber, simulationRun) where faultNumber > 0 (an actual
    fault scenario), finds the first sample AFTER the true onset where the
    model's prediction is also positive.

    Returns one row per faulty run: faultNumber, simulationRun, onset_sample,
    detected_sample (NaN if never detected in this run), delay_samples
    (NaN if never detected).
    """
    results = []
    faulty = df[df["faultNumber"] > 0]

    for (fault_number, run), group in faulty.groupby(["faultNumber", "simulationRun"]):
        group = group.sort_values("sample")
        active = group[group["fault_active"]]
        if len(active) == 0:
            continue  # no post-onset samples in this slice of data at all
        onset_sample = active["sample"].min()

        detected = active[active["predicted"] == 1]
        if len(detected) == 0:
            results.append({
                "faultNumber": fault_number, "simulationRun": run,
                "onset_sample": onset_sample, "detected_sample": np.nan,
                "delay_samples": np.nan,
            })
        else:
            detected_sample = detected["sample"].min()
            results.append({
                "faultNumber": fault_number, "simulationRun": run,
                "onset_sample": onset_sample, "detected_sample": detected_sample,
                "delay_samples": detected_sample - onset_sample,
            })

    return pd.DataFrame(results)


def false_alarm_rate(df: pd.DataFrame) -> float:
    """
    Fraction of samples where the model predicts positive but the true
    fault_active is False -- includes both fully fault-free runs AND the
    pre-onset samples of faulty runs, both of which should score as normal.
    """
    normal = df[~df["fault_active"]]
    if len(normal) == 0:
        return float("nan")
    return (normal["predicted"] == 1).mean()


def summarize_detection(df: pd.DataFrame) -> dict:
    """High-level summary combining delay and false alarm rate."""
    delays = detection_delay(df)
    n_runs = len(delays)
    n_detected = delays["delay_samples"].notna().sum()

    return {
        "n_faulty_runs": n_runs,
        "detection_rate": n_detected / n_runs if n_runs else float("nan"),
        "mean_delay_samples": delays["delay_samples"].mean(),
        "median_delay_samples": delays["delay_samples"].median(),
        "false_alarm_rate": false_alarm_rate(df),
    }


def standard_classification_metrics(y_true, y_pred, average="macro") -> dict:
    """Wraps the usual sklearn metrics for consistent reporting alongside the domain ones above."""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average=average, zero_division=0),
        "recall": recall_score(y_true, y_pred, average=average, zero_division=0),
        "f1": f1_score(y_true, y_pred, average=average, zero_division=0),
    }
