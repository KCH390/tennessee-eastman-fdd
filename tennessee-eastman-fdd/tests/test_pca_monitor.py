"""Tests for src/pca_monitor.py -- control limit calibration, false alarm rate, detection power."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from features import SENSOR_COLUMNS
from pca_monitor import PCAMonitor


def make_correlated_normal_data(n_samples, seed):
    r = np.random.RandomState(seed)
    latent = r.normal(0, 1, size=n_samples)
    rows = []
    for i in range(n_samples):
        row = {"faultNumber": 0, "simulationRun": 1, "sample": i + 1, "fault_active": False}
        for j, col in enumerate(SENSOR_COLUMNS):
            row[col] = latent[i] * (1 + 0.05 * j) + r.normal(0, 0.3)
        rows.append(row)
    return pd.DataFrame(rows)


def make_faulty_data(n_samples, seed, shift=8.0):
    r = np.random.RandomState(seed)
    latent = r.normal(0, 1, size=n_samples)
    rows = []
    for i in range(n_samples):
        row = {"faultNumber": 1, "simulationRun": 1, "sample": i + 1, "fault_active": True}
        for j, col in enumerate(SENSOR_COLUMNS):
            row[col] = latent[i] * (1 + 0.05 * j) + shift + r.normal(0, 0.3)
        rows.append(row)
    return pd.DataFrame(rows)


def test_control_limits_calibrated_sensibly():
    train_normal = make_correlated_normal_data(2000, seed=1)
    monitor = PCAMonitor(variance_threshold=0.9, alpha=0.01).fit(train_normal, SENSOR_COLUMNS)
    train_scores = monitor.score(train_normal)
    assert train_scores["predicted"].mean() < 0.05


def test_false_alarm_rate_low_on_held_out_normal_data():
    train_normal = make_correlated_normal_data(2000, seed=1)
    monitor = PCAMonitor(variance_threshold=0.9, alpha=0.01).fit(train_normal, SENSOR_COLUMNS)
    held_out_normal = make_correlated_normal_data(1000, seed=2)
    scores = monitor.score(held_out_normal)
    assert scores["predicted"].mean() < 0.10


def test_detects_obvious_fault_signal():
    train_normal = make_correlated_normal_data(2000, seed=1)
    monitor = PCAMonitor(variance_threshold=0.9, alpha=0.01).fit(train_normal, SENSOR_COLUMNS)
    faulty = make_faulty_data(500, seed=3, shift=8.0)
    scores = monitor.score(faulty)
    assert scores["predicted"].mean() > 0.8


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
