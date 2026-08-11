"""
Feature engineering for the TEP dataset.

Two framings are supported, per the Phase 2 plan:
  1. Per-timestep (baseline) -- use xmeas_1..41, xmv_1..11 directly. Simple,
     fast, what most baseline papers do first.
  2. Windowed -- add rolling mean/std and rate-of-change features computed
     WITHIN each (faultNumber, simulationRun) group. This captures process
     dynamics a single timestep can't, at the cost of losing the first
     `window` samples of each run (no history yet to compute a window).

CRITICAL: rolling/windowed features must be computed per-run, never across
runs -- concatenating all data and rolling over the whole DataFrame would
leak information across unrelated simulation runs (and even across
different fault scenarios) into each other. get_windowed_features() groups
by (faultNumber, simulationRun) specifically to prevent this; tests/
test_features.py checks this doesn't leak across a run boundary.

Also critical: train/validation splitting must be done BY RUN, not by row.
Splitting by row would put samples from the same simulation run in both
train and validation, which massively overstates validation performance
(adjacent timesteps of the same run are highly correlated). The real
train/test split is already safe (Rieth et al. used non-overlapping RNG
seeds between the training and testing files) -- split_by_run() is for
carving a validation set OUT OF the training split during development.
"""

import numpy as np
import pandas as pd

MEASUREMENT_COLUMNS = [f"xmeas_{i}" for i in range(1, 42)]
MANIPULATED_COLUMNS = [f"xmv_{i}" for i in range(1, 12)]
SENSOR_COLUMNS = MEASUREMENT_COLUMNS + MANIPULATED_COLUMNS


def get_windowed_features(df: pd.DataFrame, window: int = 10, sensor_columns=None) -> pd.DataFrame:
    """
    Adds rolling mean, rolling std, and rate-of-change (diff from `window`
    samples ago) for each sensor column, computed within each
    (faultNumber, simulationRun) group. Rows without enough history for a
    full window (the first `window` samples of each run) are dropped, since
    a partial/short window isn't comparable to the rest.
    """
    if sensor_columns is None:
        sensor_columns = SENSOR_COLUMNS

    df = df.sort_values(["faultNumber", "simulationRun", "sample"]).reset_index(drop=True)
    grouped = df.groupby(["faultNumber", "simulationRun"], group_keys=False)

    feature_frames = [df]
    for col in sensor_columns:
        roll = grouped[col].rolling(window=window, min_periods=window)
        rolling_mean = roll.mean().reset_index(level=[0, 1], drop=True)
        rolling_std = roll.std().reset_index(level=[0, 1], drop=True)
        rate_of_change = grouped[col].diff(periods=window)

        feature_frames.append(pd.DataFrame({
            f"{col}_roll_mean": rolling_mean,
            f"{col}_roll_std": rolling_std,
            f"{col}_roc": rate_of_change,
        }))

    result = pd.concat(feature_frames, axis=1)
    # first `window` samples of each run have no full window yet -- drop them
    result = result.dropna(subset=[f"{sensor_columns[0]}_roll_mean"]).reset_index(drop=True)
    return result


def split_by_run(df: pd.DataFrame, val_frac: float = 0.2, seed: int = 42):
    """
    Splits into train/validation BY simulationRun (all rows from a given
    run go entirely to one side), stratified by faultNumber so each fault
    scenario is represented in both splits. Never split this data by row.
    """
    rng = np.random.RandomState(seed)
    train_parts, val_parts = [], []

    for fault_number, group in df.groupby("faultNumber"):
        runs = group["simulationRun"].unique()
        rng.shuffle(runs)
        n_val = max(1, int(len(runs) * val_frac)) if len(runs) > 1 else 0
        val_runs = set(runs[:n_val])
        train_parts.append(group[~group["simulationRun"].isin(val_runs)])
        val_parts.append(group[group["simulationRun"].isin(val_runs)])

    train = pd.concat(train_parts, ignore_index=True)
    val = pd.concat(val_parts, ignore_index=True)
    return train, val
