"""
Tests for physics/scenarios.py.

Uses the same fake-cantera pattern as test_physics_cstr.py (see that
file's docstring) since scenarios.py imports physics.cstr, which needs
Cantera unless it's already installed for real.

Run directly with:  python tests/test_scenarios.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # for _test_support
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))  # for physics/

import types

import numpy as np

from _test_support import approx, run_tests  # noqa: E402


def _install_fake_cantera_if_needed():
    try:
        import cantera  # noqa: F401
        return
    except ImportError:
        pass

    fake_ct = types.ModuleType("cantera")

    class _FakeSolution:
        standard_cp_R = np.array([81.1, 123.3, 143.5, 75.3]) / 8.314462618

    fake_ct.Solution = lambda **kwargs: _FakeSolution()
    fake_ct._is_fake = True
    sys.modules["cantera"] = fake_ct


_install_fake_cantera_if_needed()

from physics import scenarios  # noqa: E402


def test_normal_run_has_no_active_fault():
    run = scenarios.generate_run(0, 0, seed=1)
    assert not run.df["fault_active"].any()
    assert (run.df["faultNumber"] == 0).all()


def test_faulty_run_onset_matches_configured_onset_minute():
    run = scenarios.generate_run(101, 1, seed=1)
    onset_idx = int(scenarios._ONSET_MINUTE / scenarios._SAMPLE_INTERVAL_MINUTES)
    assert not run.df["fault_active"].iloc[:onset_idx].any()
    assert run.df["fault_active"].iloc[onset_idx:].all()


def test_cooling_failure_raises_temperature_after_onset():
    run = scenarios.generate_run(101, 1, seed=1)
    onset_idx = int(scenarios._ONSET_MINUTE / scenarios._SAMPLE_INTERVAL_MINUTES)
    T_before = run.df["T"].iloc[:onset_idx].mean()
    T_after = run.df["T"].iloc[-20:].mean()  # late in the run, after re-settling
    assert T_after > T_before


def test_feed_disturbance_shifts_methanol_flow_but_not_temperature_much():
    # Direction/magnitude are randomized per run -- check the flow rate
    # actually moved away from nominal (1.0 L/min) post-onset, and that
    # temperature moved much less than the cooling-failure case does.
    run = scenarios.generate_run(102, 1, seed=1)
    onset_idx = int(scenarios._ONSET_MINUTE / scenarios._SAMPLE_INTERVAL_MINUTES)
    Q_meoh_after = run.df["Q_meoh"].iloc[-20:].mean()
    assert abs(Q_meoh_after - 1.0) > 0.1  # meaningfully off nominal

    T_before = run.df["T"].iloc[:onset_idx].mean()
    T_after = run.df["T"].iloc[-20:].mean()
    cooling_run = scenarios.generate_run(101, 2, seed=1)
    cooling_T_before = cooling_run.df["T"].iloc[:onset_idx].mean()
    cooling_T_after = cooling_run.df["T"].iloc[-20:].mean()
    assert abs(T_after - T_before) < abs(cooling_T_after - cooling_T_before)


def test_generate_dataset_has_expected_shape_and_fault_numbers():
    df = scenarios.generate_dataset(n_runs_per_fault=3, seed=1)
    assert set(df["faultNumber"].unique()) == {0, 101, 102}
    assert df["simulationRun"].nunique() == 9  # 3 runs x 3 fault types
    for col in scenarios.SENSOR_COLUMNS:
        assert col in df.columns
    assert "fault_active" in df.columns


def test_different_seeds_give_different_severity():
    # Confirms severity randomization is actually wired up, not a no-op.
    run_a = scenarios.generate_run(101, 1, seed=1)
    run_b = scenarios.generate_run(101, 1, seed=999)
    T_a = run_a.df["T"].iloc[-20:].mean()
    T_b = run_b.df["T"].iloc[-20:].mean()
    assert T_a != approx(T_b, abs=0.05)


if __name__ == "__main__":
    raise SystemExit(run_tests(globals()))
