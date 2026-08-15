# Tennessee Eastman Process: Fault Detection & Diagnosis

Machine learning fault detection and diagnosis on the Tennessee Eastman
Process (TEP) — the standard chemical-process benchmark for evaluating
process monitoring and fault diagnosis methods, introduced by Downs and
Vogel (1993) from Eastman Chemical Company. Paired with a complementary
physics-based reactor simulation (Cantera + NRTL) demonstrating
first-principles process modeling alongside the data-driven ML work —
and, in Phase 9, confirming the same fault-detection approach generalizes
to that independently-simulated physical system.

## Problem Statement

Chemical and manufacturing processes generate continuous streams of sensor
data. Two related but distinct questions matter for plant operators:

1. **Fault detection** — is something wrong right now?
2. **Fault diagnosis** — if so, which of the known fault modes is it?

Both matter for different reasons: detection needs to be fast and have a
low false-alarm rate; diagnosis needs to be accurate enough that an
operator trusts the system's answer over their own judgment.

## Architecture

```
Harvard Dataverse (.RData)
        │  src/data_loader.py  (pyreadr)
        ▼
data/processed/tep_training.parquet, tep_testing.parquet
        │  src/features.py
        │    - per-run rolling mean/std + rate-of-change (get_windowed_features)
        │    - by-run, fault-stratified train/val split (split_by_run)
        ▼
src/train.py
        - detection: binary fault_active classifier (RandomForest or
          HistGradientBoosting), class_weight="balanced" to correct the
          ~91:9 fault/normal imbalance in training data
        - diagnosis: multiclass faultNumber classifier, fit only on
          fault_active rows
        │
        ▼
models/detection_<type>.joblib, models/diagnosis_<type>.joblib
        │  src/evaluate.py (standard_classification_metrics, summarize_detection)
        ▼
app.py — Streamlit dashboard (detection timeline, confusion matrix,
feature importance, simulated live monitoring)
```

Two correctness constraints run through the whole pipeline and are covered
by `tests/`:
- **No leakage across runs.** `simulationRun` numbers repeat across
  different `faultNumber` scenarios, and rolling/windowed features must
  never be computed across a run boundary — both `get_windowed_features`
  and `split_by_run` group by `(faultNumber, simulationRun)` for this.
- **Onset-aware labeling.** Early samples in "faulty" runs (training
  samples 1–20, testing samples 1–160) are still normal; `fault_active`
  already encodes this rather than trusting `faultNumber != 0` alone.

### Track 2: physics-based reactor + column, and the ML generalization test

```
src/physics/
  thermo.py      Cantera species thermo (liquid Cp only -- see module docstring
                  for why Cantera's Reactor/ReactorNet classes aren't used here)
  reaction.py     Published kinetics + heat of reaction (methyl acetate esterification)
  cstr.py         Non-isothermal CSTR, two mixing feed streams, cooling jacket
  nrtl.py         General N-component NRTL activity-coefficient engine
  vle_params.py   NRTL binary parameters (2 of 6 pairs sourced from real data;
                  4 acetic-acid pairs deliberately left ideal, not fabricated)
  properties.py   Antoine vapor-pressure data
  column.py       Dynamic tray-by-tray distillation column (CMO + rigorous NRTL VLE)
  scenarios.py    Generates TEP-schema fault data (cooling failure, feed
                  disturbance) from the CSTR, for reuse by Track 1's ML code
        │
        ▼
src/train_physics.py -- imports train.py's train_detection_model /
        evaluate_detection_model / train_diagnosis_model / evaluate_diagnosis_model
        UNMODIFIED, points them at physics/scenarios.py's data instead of TEP's
        │
        ▼
models/physics_detection_<type>.joblib, models/physics_diagnosis_<type>.joblib
```

This is the real system behind Eastman Chemical's landmark methyl acetate
reactive-distillation process (Agreda, Partin & Heise, 1990) — same
Eastman as the TEP benchmark. `demo_track2.py` runs the CSTR through a
cooling-failure fault and the NRTL module through an azeotrope prediction,
end to end, producing `track2_cooling_failure.png` and
`track2_azeotrope.png`.

## Tech Stack

- **Python (pandas, numpy)** — data loading and feature engineering
- **pyreadr** — reading the TEP dataset's native .RData format
- **scikit-learn** — fault detection (binary) and diagnosis (multiclass)
  models: `RandomForestClassifier` and `HistGradientBoostingClassifier`
- **xgboost** — listed as an alternative model backend to try; not
  currently wired into `train.py` or benchmarked here
- **Streamlit, Plotly** — monitoring dashboard
- **Cantera** — species thermodynamics for the CSTR (Track 2); see
  `src/physics/thermo.py` for exactly what it is/isn't used for
- **scipy, NRTL (hand-implemented)** — CSTR/column ODE integration and
  non-ideal VLE (Track 2) — see `src/physics/nrtl.py`

## Project Structure

```
tep-fault-diagnosis/
├── data/
│   ├── raw/                # downloaded source data (gitignored -- see data/README.md)
│   ├── processed/          # cleaned / feature-engineered data (gitignored;
│   │                       #   includes physics_training.parquet, Track 2)
│   └── external/           # reference material, small lookup files (checked in)
├── models/                 # trained model artifacts (gitignored)
├── notebooks/
│   └── exploration.ipynb   # EDA and modeling notebook
├── src/
│   ├── data_loader.py      # .RData -> processed parquet (pyreadr)
│   ├── features.py         # windowed features, by-run/fault-stratified split
│   ├── train.py            # detection + diagnosis model training (Track 1)
│   ├── evaluate.py         # shared metrics helpers
│   ├── train_physics.py    # runs train.py's ML pipeline on Track 2 data (Phase 9)
│   └── physics/            # Track 2: CSTR + distillation column (see above)
├── app.py                  # Streamlit dashboard (Phase 5)
├── demo_track2.py          # runnable Track 2 demo (CSTR fault + azeotrope plots)
├── tests/
│   ├── test_features.py
│   ├── test_train.py
│   ├── test_physics_cstr.py
│   ├── test_nrtl.py
│   ├── test_column.py
│   ├── test_scenarios.py
│   └── _test_support.py    # dependency-free pytest stand-in (no install needed)
├── requirements.txt
├── .gitignore
└── README.md
```

## Setup

```bash
git clone https://github.com/KCH390/tep-fault-diagnosis.git
cd tep-fault-diagnosis
python -m venv venv
source venv/bin/activate  # venv\Scripts\activate on Windows
pip install -r requirements.txt
```

**Data:** download the TEP `.RData` files from the
[Harvard Dataverse](https://doi.org/10.7910/DVN/6C3JR1) (Rieth et al.,
2017) into `data/raw/` — see `data/README.md` for the exact filenames
expected.

**Run the pipeline:**

```bash
python src/data_loader.py                                          # .RData -> processed parquet
python src/train.py --model-type random_forest --class-weight balanced   # train + evaluate
```

`--model-type` also accepts `hist_gradient_boosting`. Run both if you want
the dashboard's model-type selector to have something to compare.

**Run the dashboard:**

```bash
streamlit run app.py
```

## Results

Findings from running the pipeline against the real Rieth et al. (2017)
dataset:

- **HistGradientBoosting slightly outperforms RandomForest** on both the
  detection and diagnosis tasks.
- **Faults 3, 9, and 15 underperform**, consistent with the literature —
  these are documented as statistically near-indistinguishable from normal
  operation in TEP, not a modeling gap.
- **Detection is fast enough to be operationally useful overall, but not
  uniformly** — faults 13, 18, and 20 show significant detection lag
  compared to the rest, so a deployment would need fault-specific latency
  expectations rather than a single SLA.
- Class-weighting the detection model against the ~91:9 fault/normal
  imbalance in training meaningfully reduced the false-alarm rate (see
  `src/train.py` docstring for the controlled test showing the effect in
  isolation).

*(See the Exploration notebook for some plots and numbers)*

### Does it generalize beyond the canned benchmark? (Phase 9)

`src/train_physics.py` runs the exact same `train_detection_model` /
`evaluate_detection_model` / `train_diagnosis_model` / `evaluate_diagnosis_model`
functions from `src/train.py` — unmodified — against fault-scenario data
generated from the Track 2 CSTR physics model (`src/physics/scenarios.py`)
instead of the real TEP dataset. Two fault types (cooling-jacket failure,
feed-ratio disturbance), 7 physics sensor columns instead of TEP's 52:

| | RandomForest | HistGradientBoosting |
|---|---|---|
| Detection accuracy | 98.9% | 98.8% |
| Detection false-alarm rate | 0.2% | 0.3% |
| Mean detection delay | 3.6 samples | 4.0 samples |
| Diagnosis accuracy | 100% | 99.8% |

**Honest read of this, not just the headline numbers**: the near-perfect
diagnosis accuracy reflects that this system's two fault types have
physically orthogonal, directly-measured signatures — cooling failure
shows up as a pure temperature shift with feed flows untouched; feed
disturbance shows up as a pure flow-rate shift with temperature barely
moved. That's a much easier separation problem than TEP's 20 real,
overlapping fault classes (including 3 that are near-undetectable by
design). The fair takeaway isn't "this problem is easier than TEP, so the
method is better here" — it's that the same code, same class-weighting
approach, and same evaluation functions produced sane, well-calibrated
results on data TEP's benchmark never touched, which is the actual claim
Phase 9 set out to test.

Run it yourself: `python src/train_physics.py --model-type random_forest`

## Background

Built by Kerry Hall, applying a chemical engineering background and
manufacturing quality engineering experience (Goodyear, Morgan Advanced
Materials) to the standard academic benchmark for process fault detection
in the field he studied.

## License

MIT
