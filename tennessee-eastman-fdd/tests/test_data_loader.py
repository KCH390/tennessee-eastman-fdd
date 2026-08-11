"""
Tests for the onset-aware fault labeling logic in src/data_loader.py.

These run against synthetic data matching the documented Rieth et al.
schema -- no real TEP data or pyreadr required, since combine_and_label()
is pure and doesn't touch files. This is the highest-risk logic in the
loader (getting the onset boundary wrong silently mislabels data), so it's
covered here permanently rather than just checked once ad hoc.
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from data_loader import combine_and_label, TRAIN_ONSET_SAMPLE, TEST_ONSET_SAMPLE


def make_mock(fault_numbers, runs, samples_per_run, n_meas=41, n_mv=11):
    rows = []
    for fn in fault_numbers:
        for run in runs:
            for s in samples_per_run:
                row = {"faultNumber": fn, "simulationRun": run, "sample": s}
                for i in range(1, n_meas + 1):
                    row[f"xmeas_{i}"] = 0.0
                for i in range(1, n_mv + 1):
                    row[f"xmv_{i}"] = 0.0
                rows.append(row)
    return pd.DataFrame(rows)


def test_training_onset_boundary():
    fault_free = make_mock([0], runs=[1], samples_per_run=range(1, 26))
    faulty = make_mock([1], runs=[1], samples_per_run=range(1, 26))
    combined = combine_and_label(fault_free, faulty, onset_sample=TRAIN_ONSET_SAMPLE)

    faulty_rows = combined[combined.faultNumber == 1].set_index("sample")
    assert faulty_rows.loc[TRAIN_ONSET_SAMPLE, "fault_active"] == False
    assert faulty_rows.loc[TRAIN_ONSET_SAMPLE + 1, "fault_active"] == True


def test_testing_onset_boundary():
    fault_free = make_mock([0], runs=[1], samples_per_run=range(155, 166))
    faulty = make_mock([2], runs=[1], samples_per_run=range(155, 166))
    combined = combine_and_label(fault_free, faulty, onset_sample=TEST_ONSET_SAMPLE)

    faulty_rows = combined[combined.faultNumber == 2].set_index("sample")
    assert faulty_rows.loc[TEST_ONSET_SAMPLE, "fault_active"] == False
    assert faulty_rows.loc[TEST_ONSET_SAMPLE + 1, "fault_active"] == True


def test_fault_free_rows_never_active():
    fault_free = make_mock([0], runs=[1], samples_per_run=range(1, 50))
    faulty = make_mock([1], runs=[1], samples_per_run=range(1, 50))
    combined = combine_and_label(fault_free, faulty, onset_sample=TRAIN_ONSET_SAMPLE)
    assert not combined[combined.faultNumber == 0]["fault_active"].any()


def test_faults_filter():
    fault_free = make_mock([0], runs=[1, 2], samples_per_run=[1, 2])
    faulty = make_mock([1, 2, 3], runs=[1, 2], samples_per_run=[1, 2])
    filtered = combine_and_label(fault_free, faulty, onset_sample=20, faults=[1, 3])
    assert set(filtered["faultNumber"].unique()) == {0, 1, 3}


def test_runs_filter_applies_to_both_fault_free_and_faulty():
    fault_free = make_mock([0], runs=[1, 2], samples_per_run=[1, 2])
    faulty = make_mock([1, 2], runs=[1, 2], samples_per_run=[1, 2])
    filtered = combine_and_label(fault_free, faulty, onset_sample=20, runs=[1])
    assert set(filtered["simulationRun"].unique()) == {1}
    # both fault-free and faulty rows should be restricted, not just one
    assert set(filtered[filtered.faultNumber == 0]["simulationRun"].unique()) == {1}
    assert set(filtered[filtered.faultNumber == 1]["simulationRun"].unique()) == {1}


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
