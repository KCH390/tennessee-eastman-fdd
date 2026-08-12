"""Tests for src/tune.py -- the critical property is that GroupKFold never splits a run across folds."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from features import SENSOR_COLUMNS
from tune import make_run_groups, tune_model


def test_make_run_groups_distinguishes_same_run_number_different_fault():
    df = pd.DataFrame({"faultNumber": [0, 0, 1, 1], "simulationRun": [1, 1, 1, 1]})
    groups = make_run_groups(df)
    assert groups[0] == groups[1]
    assert groups[0] != groups[2]


def test_group_kfold_never_splits_a_run_across_folds():
    rng = np.random.RandomState(0)
    rows = []
    for fault_number in [0, 1, 2]:
        for run in range(1, 11):
            for s in range(1, 6):
                row = {"faultNumber": fault_number, "simulationRun": run, "sample": s}
                for col in SENSOR_COLUMNS:
                    row[col] = rng.normal(0, 1)
                rows.append(row)
    df = pd.DataFrame(rows)
    groups = make_run_groups(df)

    gkf = GroupKFold(n_splits=5)
    for train_idx, val_idx in gkf.split(df, groups=groups):
        train_groups = set(groups[train_idx])
        val_groups = set(groups[val_idx])
        assert not (train_groups & val_groups)


def test_tune_model_finds_a_good_model_on_learnable_data():
    rng = np.random.RandomState(5)
    rows = []
    for fault_number, shift in [(0, 0.0), (1, 6.0)]:
        for run in range(1, 21):
            for s in range(1, 21):
                row = {"faultNumber": fault_number, "simulationRun": run, "sample": s,
                       "fault_active": fault_number > 0}
                for col in SENSOR_COLUMNS:
                    row[col] = shift + rng.normal(0, 1)
                rows.append(row)
    train_df = pd.DataFrame(rows)

    search = tune_model(train_df, target_column="fault_active", feature_columns=SENSOR_COLUMNS,
                         model_type="random_forest", n_splits=3, n_iter=3, seed=42)
    assert search.best_score_ > 0.8


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
