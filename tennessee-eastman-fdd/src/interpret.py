"""
Root-cause analysis: maps model feature importance from raw column names
(xmeas_7, xmv_3, ...) to their actual physical meaning (Reactor Pressure,
A Feed Flow, ...) using data/external/tep_variable_reference.csv.

This is the piece that turns "xmeas_7 was important" into "Reactor
Pressure was the most important indicator for Fault 4" -- the whole point
of doing root-cause analysis on a real chemical process rather than an
anonymized tabular dataset.
"""

from pathlib import Path

import numpy as np
import pandas as pd

REFERENCE_PATH = Path(__file__).parent.parent / "data" / "external" / "tep_variable_reference.csv"


def load_variable_reference(path: Path = REFERENCE_PATH) -> pd.DataFrame:
    return pd.read_csv(path)


def global_feature_importance(model, feature_columns, reference_df=None) -> pd.DataFrame:
    """
    Feature importance for the whole model, joined to physical tag names.
    Works with any model exposing .feature_importances_ (RandomForest,
    HistGradientBoostingClassifier does NOT expose this directly -- see
    note in per_fault_importance below).
    """
    if reference_df is None:
        reference_df = load_variable_reference()

    if not hasattr(model, "feature_importances_"):
        raise ValueError(
            f"{type(model).__name__} doesn't expose .feature_importances_ directly "
            "(true of HistGradientBoostingClassifier -- use permutation_importance "
            "from sklearn.inspection instead for that model type)."
        )

    importances = pd.DataFrame({
        "column_name": feature_columns,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)

    result = importances.merge(reference_df[["column_name", "description", "units"]],
                                on="column_name", how="left")
    return result


def per_fault_importance(train_df: pd.DataFrame, feature_columns, fault_numbers=None,
                          reference_df=None, seed=42, n_estimators=100):
    """
    Trains a small one-vs-rest RandomForest PER FAULT (this fault's active
    rows vs. everything else) and returns each fault's top features mapped
    to physical names. This is more informative than one multiclass
    model's importances for root-cause purposes -- it directly answers
    "what does THIS fault look like" rather than "what matters on average
    across all faults."
    """
    from sklearn.ensemble import RandomForestClassifier

    if reference_df is None:
        reference_df = load_variable_reference()
    if fault_numbers is None:
        fault_numbers = sorted(f for f in train_df["faultNumber"].unique() if f > 0)

    results = {}
    for fault_number in fault_numbers:
        subset = train_df[
            (train_df["faultNumber"] == fault_number) & train_df["fault_active"]
            | (train_df["faultNumber"] == 0)
        ]
        y = (subset["faultNumber"] == fault_number).astype(int)
        if y.nunique() < 2:
            continue  # no positive or no negative examples, can't train

        model = RandomForestClassifier(n_estimators=n_estimators, min_samples_leaf=2, random_state=seed)
        model.fit(subset[feature_columns], y)

        importances = global_feature_importance(model, feature_columns, reference_df)
        results[fault_number] = importances

    return results


def top_features_summary(per_fault_results: dict, top_n=5) -> pd.DataFrame:
    """Flattens per_fault_importance()'s output into one readable table: top N features per fault."""
    rows = []
    for fault_number, importances in per_fault_results.items():
        for rank, (_, row) in enumerate(importances.head(top_n).iterrows(), start=1):
            rows.append({
                "faultNumber": fault_number,
                "rank": rank,
                "column_name": row["column_name"],
                "description": row["description"],
                "importance": row["importance"],
            })
    return pd.DataFrame(rows)
