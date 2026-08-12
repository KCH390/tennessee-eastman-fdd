"""Tests for src/interpret.py -- validated against synthetic data with a KNOWN causal feature."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from features import SENSOR_COLUMNS
from interpret import load_variable_reference, per_fault_importance, top_features_summary


def test_reference_table_shape_and_spot_check():
    ref = load_variable_reference()
    assert len(ref) == 52
    assert set(ref.columns) == {"tag", "column_name", "description", "units", "source_note"}
    assert ref[ref.column_name == "xmeas_7"]["description"].iloc[0] == "Reactor Pressure"
    assert ref[ref.column_name == "xmv_10"]["description"].iloc[0] == "Reactor Cooling Water Flow"


def test_per_fault_importance_finds_the_known_causal_feature():
    rng = np.random.RandomState(0)
    rows = []
    for fault_number, driver_col, shift in [(0, None, 0.0), (4, "xmeas_9", 15.0)]:
        for run in range(1, 21):
            for s in range(1, 41):
                is_active = fault_number > 0 and s > 20
                row = {"faultNumber": fault_number, "simulationRun": run, "sample": s,
                       "fault_active": is_active}
                for col in SENSOR_COLUMNS:
                    base = shift if (is_active and col == driver_col) else 0.0
                    row[col] = base + rng.normal(0, 0.5)
                rows.append(row)
    df = pd.DataFrame(rows)

    results = per_fault_importance(df, SENSOR_COLUMNS, fault_numbers=[4], seed=42)
    top_row = results[4].iloc[0]
    assert top_row["column_name"] == "xmeas_9"
    assert top_row["description"] == "Reactor Temperature"


def test_top_features_summary_flattens_correctly():
    rng = np.random.RandomState(1)
    rows = []
    for fault_number, driver_col, shift in [(0, None, 0.0), (7, "xmeas_7", 20.0)]:
        for run in range(1, 21):
            for s in range(1, 41):
                is_active = fault_number > 0 and s > 20
                row = {"faultNumber": fault_number, "simulationRun": run, "sample": s,
                       "fault_active": is_active}
                for col in SENSOR_COLUMNS:
                    base = shift if (is_active and col == driver_col) else 0.0
                    row[col] = base + rng.normal(0, 0.5)
                rows.append(row)
    df = pd.DataFrame(rows)

    results = per_fault_importance(df, SENSOR_COLUMNS, fault_numbers=[7], seed=42)
    summary = top_features_summary(results, top_n=3)
    assert len(summary) == 3
    assert summary.iloc[0]["faultNumber"] == 7
    assert summary.iloc[0]["column_name"] == "xmeas_7"


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
