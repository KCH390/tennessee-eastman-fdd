"""
Tests for src/evaluate.py's domain-specific metrics. Each test constructs a
tiny synthetic prediction sequence where the correct answer is known by
hand construction, not just "does it run without crashing."
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from evaluate import detection_delay, false_alarm_rate, summarize_detection


def test_detection_delay_immediate_detection():
    # onset at sample 21 (fault_active starts True there), model detects
    # correctly starting exactly at sample 21 -> delay should be 0
    samples = list(range(1, 31))
    fault_active = [s > 20 for s in samples]
    predicted = [1 if s > 20 else 0 for s in samples]  # model matches truth exactly
    df = pd.DataFrame({
        "faultNumber": 1, "simulationRun": 1, "sample": samples,
        "fault_active": fault_active, "predicted": predicted,
    })
    result = detection_delay(df)
    assert len(result) == 1
    assert result.iloc[0]["onset_sample"] == 21
    assert result.iloc[0]["detected_sample"] == 21
    assert result.iloc[0]["delay_samples"] == 0


def test_detection_delay_late_detection():
    # onset at 21, model doesn't flag positive until sample 25 -> delay = 4
    samples = list(range(1, 31))
    fault_active = [s > 20 for s in samples]
    predicted = [1 if s >= 25 else 0 for s in samples]
    df = pd.DataFrame({
        "faultNumber": 1, "simulationRun": 1, "sample": samples,
        "fault_active": fault_active, "predicted": predicted,
    })
    result = detection_delay(df)
    assert result.iloc[0]["delay_samples"] == 4


def test_detection_delay_never_detected():
    samples = list(range(1, 31))
    fault_active = [s > 20 for s in samples]
    predicted = [0] * 30  # model never flags anything
    df = pd.DataFrame({
        "faultNumber": 1, "simulationRun": 1, "sample": samples,
        "fault_active": fault_active, "predicted": predicted,
    })
    result = detection_delay(df)
    assert len(result) == 1
    assert pd.isna(result.iloc[0]["delay_samples"])
    assert pd.isna(result.iloc[0]["detected_sample"])


def test_detection_delay_multiple_runs_independent():
    # two runs of the same fault -- one detected fast, one slow. Each
    # should get its OWN delay, not be mixed together.
    rows = []
    for run, detect_at in [(1, 21), (2, 26)]:
        for s in range(1, 31):
            rows.append({
                "faultNumber": 1, "simulationRun": run, "sample": s,
                "fault_active": s > 20, "predicted": 1 if s >= detect_at else 0,
            })
    df = pd.DataFrame(rows)
    result = detection_delay(df).set_index("simulationRun")
    assert result.loc[1, "delay_samples"] == 0
    assert result.loc[2, "delay_samples"] == 5


def test_false_alarm_rate_counts_pre_onset_samples_as_normal():
    # a faulty run's PRE-onset samples (fault_active=False) that get a
    # positive prediction should count as false alarms
    samples = list(range(1, 31))
    fault_active = [s > 20 for s in samples]
    # model incorrectly flags samples 15-20 (pre-onset) as positive
    predicted = [1 if 15 <= s <= 20 else (1 if s > 20 else 0) for s in samples]
    df = pd.DataFrame({
        "faultNumber": 1, "simulationRun": 1, "sample": samples,
        "fault_active": fault_active, "predicted": predicted,
    })
    # 20 normal samples (1-20), 6 of them (15-20) got a false positive
    rate = false_alarm_rate(df)
    assert abs(rate - 6 / 20) < 1e-9


def test_false_alarm_rate_zero_when_no_false_positives():
    samples = list(range(1, 21))
    df = pd.DataFrame({
        "faultNumber": 0, "simulationRun": 1, "sample": samples,
        "fault_active": [False] * 20, "predicted": [0] * 20,
    })
    assert false_alarm_rate(df) == 0.0


def test_summarize_detection_combines_metrics_correctly():
    rows = []
    # run 1: detected immediately
    for s in range(1, 31):
        rows.append({"faultNumber": 1, "simulationRun": 1, "sample": s,
                      "fault_active": s > 20, "predicted": 1 if s > 20 else 0})
    # run 2: never detected
    for s in range(1, 31):
        rows.append({"faultNumber": 1, "simulationRun": 2, "sample": s,
                      "fault_active": s > 20, "predicted": 0})
    df = pd.DataFrame(rows)
    summary = summarize_detection(df)
    assert summary["n_faulty_runs"] == 2
    assert summary["detection_rate"] == 0.5  # 1 of 2 runs detected
    assert summary["mean_delay_samples"] == 0.0  # only the detected run contributes (NaN excluded by pandas mean)


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
