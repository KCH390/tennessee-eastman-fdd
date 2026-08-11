"""
Tests for src/features.py. Focused specifically on the two ways this kind
of code silently leaks information if the grouping is wrong:
  1. Rolling/windowed features computed across a run boundary (using data
     from the END of one run to compute "history" at the START of another)
  2. A train/validation split that puts rows from the same simulationRun on
     both sides
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from features import get_windowed_features, split_by_run


def make_run(fault_number, run, values, sample_start=1):
    """One synthetic run where xmeas_1 follows a known, distinct sequence."""
    n = len(values)
    df = pd.DataFrame({
        "faultNumber": fault_number,
        "simulationRun": run,
        "sample": range(sample_start, sample_start + n),
    })
    for i in range(1, 42):
        df[f"xmeas_{i}"] = values if i == 1 else 0.0
    for i in range(1, 12):
        df[f"xmv_{i}"] = 0.0
    return df


def test_windowed_features_do_not_leak_across_run_boundary():
    # Run 1: xmeas_1 is constant 1000 for its whole (short) length
    # Run 2: xmeas_1 is constant 5 for its whole length
    # If rolling incorrectly spans the boundary, run 2's early rolling_mean
    # would be pulled toward 1000. It must not be.
    run1 = make_run(fault_number=0, run=1, values=[1000.0] * 15)
    run2 = make_run(fault_number=0, run=2, values=[5.0] * 15)
    combined = pd.concat([run1, run2], ignore_index=True)

    result = get_windowed_features(combined, window=10)

    run2_features = result[result.simulationRun == 2]
    assert len(run2_features) > 0, "run 2 should have rows with a full window"
    # every rolling mean for run 2 must be close to 5, never contaminated by run 1's 1000
    assert (run2_features["xmeas_1_roll_mean"] - 5.0).abs().max() < 1e-9, (
        "run 2's rolling mean is contaminated by run 1's data -- leakage across run boundary"
    )


def test_windowed_features_do_not_leak_across_fault_boundary_same_run_number():
    # Same simulationRun number (1) but different faultNumber -- these are
    # DIFFERENT simulations that happen to share a run index, must not mix.
    fault0_run1 = make_run(fault_number=0, run=1, values=[1000.0] * 15)
    fault1_run1 = make_run(fault_number=1, run=1, values=[5.0] * 15)
    combined = pd.concat([fault0_run1, fault1_run1], ignore_index=True)

    result = get_windowed_features(combined, window=10)

    fault1_features = result[result.faultNumber == 1]
    assert len(fault1_features) > 0
    assert (fault1_features["xmeas_1_roll_mean"] - 5.0).abs().max() < 1e-9, (
        "faultNumber=1 run=1 contaminated by faultNumber=0 run=1 -- grouping by "
        "simulationRun alone (without faultNumber) would cause exactly this bug"
    )


def test_windowed_features_drop_incomplete_window_rows():
    run1 = make_run(fault_number=0, run=1, values=list(range(15)))
    result = get_windowed_features(pd.DataFrame(run1), window=10)
    # first 9 rows (samples 1-9) don't have a full window of 10 -> dropped
    assert result["sample"].min() == 10


def test_split_by_run_never_splits_a_run_across_both_sides():
    rows = []
    for fault_number in [0, 1, 2]:
        for run in range(1, 11):
            rows.append(make_run(fault_number, run, values=[0.0] * 5))
    df = pd.concat(rows, ignore_index=True)

    train, val = split_by_run(df, val_frac=0.2, seed=42)

    train_run_keys = set(zip(train.faultNumber, train.simulationRun))
    val_run_keys = set(zip(val.faultNumber, val.simulationRun))
    overlap = train_run_keys & val_run_keys
    assert not overlap, f"these (faultNumber, run) pairs appear in BOTH train and val: {overlap}"


def test_split_by_run_stratifies_by_fault():
    rows = []
    for fault_number in [0, 1, 2]:
        for run in range(1, 11):
            rows.append(make_run(fault_number, run, values=[0.0] * 5))
    df = pd.concat(rows, ignore_index=True)

    train, val = split_by_run(df, val_frac=0.2, seed=42)

    # every fault scenario present in the input should appear in both splits
    assert set(train.faultNumber.unique()) == {0, 1, 2}
    assert set(val.faultNumber.unique()) == {0, 1, 2}


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
