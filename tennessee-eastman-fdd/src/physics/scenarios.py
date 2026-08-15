"""
Generates fault-scenario time series from the CSTR physics model
(physics/cstr.py), shaped into the SAME schema as the TEP dataset
(faultNumber, simulationRun, sample, sensor columns, fault_active) --
specifically so Track 1's ML pipeline (src/train.py, src/features.py) can
be reused UNMODIFIED on this physics-generated data. That reuse is the
whole point of Phase 9: "does the fault-detection approach that worked on
the canned TEP benchmark also work on an independently-simulated,
first-principles physical system?"

Fault types (Phase 8 scope: CSTR only, no column faults yet):
    faultNumber 0   -- normal operation, no fault
    faultNumber 101 -- cooling jacket failure (UA drops partway through the run)
    faultNumber 102 -- feed ratio disturbance (methanol feed rate shifts)

(101/102 rather than 1/2, deliberately, so these never collide with real
TEP fault numbers 1-20 if this data is ever combined with TEP data in the
same table.)

Sensor columns (the physics analog of TEP's xmeas/xmv):
    C_MeOH, C_AcOH, C_MeOAc, C_H2O  -- reactor concentrations, mol/L
    T                                -- reactor temperature, K
    Q_acid, Q_meoh                  -- actual feed flow rates, L/min

Q_acid/Q_meoh are included specifically because they're realistic
"measured" flow-meter readings -- a feed disturbance shows up directly in
them (much like TEP's xmv columns directly reflect manipulated-variable
faults), while UA itself (equipment fouling/failure) is NOT included as a
sensor, since that's not something a flow/temperature-instrumented plant
actually measures directly -- a cooling failure has to be INFERRED from
its effect on T, not read off a UA sensor. This intentionally creates the
same kind of per-fault detectability asymmetry Track 1 found in the real
TEP faults (13/18/20 showing significant detection lag vs. the rest).

fault_active semantics: True from the exact simulated onset time onward,
for every faulty run. This is a cleaner ground-truth signal than TEP's
(which has its own onset-timing quirks -- see features.py's docstring) --
here we know exactly when the fault started because we caused it. Model
detection LAG is then a genuine, honestly-measured property of the
model+data, not conflated with any ambiguity in the labels themselves.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import cstr

SENSOR_COLUMNS = ["C_MeOH", "C_AcOH", "C_MeOAc", "C_H2O", "T", "Q_acid", "Q_meoh"]

_NORMAL_PARAMS = dict(
    V=100.0, Q_acid=1.0, Q_meoh=1.0,
    T_acid_in=298.15, T_meoh_in=298.15,
    UA=2000.0, T_jacket=298.15,
)

_ONSET_MINUTE = 30.0
_RUN_DURATION_MINUTES = 300.0
_SAMPLE_INTERVAL_MINUTES = 1.0

# Measurement noise, applied only to the sampled/reported values (not fed
# back into the ODE integration) -- standard deviations chosen as small,
# representative fractions of typical operating values, same spirit as
# the "representative literature values" caveat used throughout physics/
# for constants that weren't independently verified for this project.
_NOISE_SIGMA = {
    "C_MeOH": 0.05, "C_AcOH": 0.05, "C_MeOAc": 0.05, "C_H2O": 0.05,  # mol/L
    "T": 0.15,       # K
    "Q_acid": 0.01, "Q_meoh": 0.01,  # L/min
}


@dataclass
class ScenarioRun:
    fault_number: int
    simulation_run: int
    df: pd.DataFrame  # columns: faultNumber, simulationRun, sample, *SENSOR_COLUMNS, fault_active


def _sample_times():
    return np.arange(0.0, _RUN_DURATION_MINUTES + 1e-9, _SAMPLE_INTERVAL_MINUTES)


def _add_noise(values: np.ndarray, sensor: str, rng: np.random.Generator) -> np.ndarray:
    sigma = _NOISE_SIGMA[sensor]
    return values + rng.normal(0.0, sigma, size=values.shape)


def generate_run(fault_number: int, simulation_run: int, seed: int) -> ScenarioRun:
    """
    fault_number: 0 (normal), 101 (cooling failure), or 102 (feed disturbance).
    seed: controls both the fault severity (for faulty runs) and the
    measurement noise realization, so every run is a distinct, reproducible
    scenario -- same role TEP's many simulationRuns-per-fault play.
    """
    rng = np.random.default_rng(seed)
    t = _sample_times()
    onset_idx = int(_ONSET_MINUTE / _SAMPLE_INTERVAL_MINUTES)

    normal_reactor = cstr.CSTR(cstr.CSTRParams(**_NORMAL_PARAMS))
    C0, T0 = normal_reactor.steady_state()
    y0 = np.concatenate([C0, [T0]])

    if fault_number == 0:
        sol = normal_reactor.simulate(y0=y0, t_span=(t[0], t[-1]), t_eval=t)
        Q_acid_trace = np.full_like(t, _NORMAL_PARAMS["Q_acid"])
        Q_meoh_trace = np.full_like(t, _NORMAL_PARAMS["Q_meoh"])
        y = sol.y

    elif fault_number in (101, 102):
        # Pre-onset: normal params. Post-onset: faulted params, continuing
        # from wherever the pre-onset segment left off (piecewise
        # integration -- solve_ivp doesn't support a mid-run parameter
        # change directly, so this is two calls stitched together).
        t_pre = t[: onset_idx + 1]
        sol_pre = normal_reactor.simulate(y0=y0, t_span=(t_pre[0], t_pre[-1]), t_eval=t_pre)

        faulted_params = dict(_NORMAL_PARAMS)
        if fault_number == 101:
            severity = rng.uniform(0.05, 0.3)  # UA drops to 5-30% of nominal
            faulted_params["UA"] = _NORMAL_PARAMS["UA"] * severity
        else:  # 102
            direction = rng.choice([-1.0, 1.0])
            magnitude = rng.uniform(0.3, 0.7)
            faulted_params["Q_meoh"] = _NORMAL_PARAMS["Q_meoh"] * (1.0 + direction * magnitude)

        faulted_reactor = cstr.CSTR(cstr.CSTRParams(**faulted_params))
        t_post = t[onset_idx:]
        y_post0 = sol_pre.y[:, -1]
        sol_post = faulted_reactor.simulate(y0=y_post0, t_span=(t_post[0], t_post[-1]), t_eval=t_post)

        y = np.concatenate([sol_pre.y[:, :-1], sol_post.y], axis=1)  # drop duplicated onset sample from pre
        Q_acid_trace = np.full_like(t, faulted_params["Q_acid"])
        Q_meoh_trace = np.full_like(t, faulted_params["Q_meoh"])
        Q_acid_trace[:onset_idx] = _NORMAL_PARAMS["Q_acid"]
        Q_meoh_trace[:onset_idx] = _NORMAL_PARAMS["Q_meoh"]

    else:
        raise ValueError(f"unknown fault_number {fault_number}")

    raw = {
        "C_MeOH": y[0], "C_AcOH": y[1], "C_MeOAc": y[2], "C_H2O": y[3], "T": y[4],
        "Q_acid": Q_acid_trace, "Q_meoh": Q_meoh_trace,
    }
    noisy = {sensor: _add_noise(values, sensor, rng) for sensor, values in raw.items()}

    fault_active = np.zeros(len(t), dtype=bool)
    if fault_number != 0:
        fault_active[onset_idx:] = True

    df = pd.DataFrame({
        "faultNumber": fault_number,
        "simulationRun": simulation_run,
        "sample": np.arange(len(t)),
        **noisy,
        "fault_active": fault_active,
    })
    return ScenarioRun(fault_number, simulation_run, df)


def generate_dataset(n_runs_per_fault: int = 40, seed: int = 42) -> pd.DataFrame:
    """
    n_runs_per_fault normal + n_runs_per_fault cooling-failure +
    n_runs_per_fault feed-disturbance runs, concatenated into one
    DataFrame with the same shape TEP's tep_training.parquet has.
    """
    rng = np.random.default_rng(seed)
    frames = []
    run_id = 0
    for fault_number in (0, 101, 102):
        for _ in range(n_runs_per_fault):
            run_seed = int(rng.integers(0, 2**31 - 1))
            frames.append(generate_run(fault_number, run_id, run_seed).df)
            run_id += 1
    return pd.concat(frames, ignore_index=True)
