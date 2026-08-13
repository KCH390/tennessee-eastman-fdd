# Tennessee Eastman Process: Fault Detection & Diagnosis

Machine learning fault detection and diagnosis on the Tennessee Eastman
Process (TEP) — the standard chemical-process benchmark for evaluating
process monitoring and fault diagnosis methods, introduced by Downs and
Vogel (1993) from Eastman Chemical Company. Paired with a complementary
physics-based reactor simulation (Cantera) demonstrating first-principles
process modeling alongside the data-driven ML work.

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

## Tech Stack

- **Python (pandas, numpy)** — data loading and feature engineering
- **pyreadr** — reading the TEP dataset's native .RData format
- **scikit-learn** — fault detection (binary) and diagnosis (multiclass)
  models: `RandomForestClassifier` and `HistGradientBoostingClassifier`
- **xgboost** — listed as an alternative model backend to try; not
  currently wired into `train.py` or benchmarked here
- **Streamlit, Plotly** — monitoring dashboard
- **Cantera** *(Track 2, not yet integrated)* — physics-based CSTR reactor
  simulation with real, verifiable chemistry, structurally inspired by TEP

## Project Structure

```
tep-fault-diagnosis/
├── data/
│   ├── raw/                # downloaded source data (gitignored -- see data/README.md)
│   ├── processed/          # cleaned / feature-engineered data (gitignored)
│   └── external/           # reference material, small lookup files (checked in)
├── models/                 # trained model artifacts (gitignored)
├── notebooks/
│   └── exploration.ipynb   # EDA and modeling notebook
├── src/
│   ├── data_loader.py      # .RData -> processed parquet (pyreadr)
│   ├── features.py         # windowed features, by-run/fault-stratified split
│   ├── train.py            # detection + diagnosis model training
│   └── evaluate.py         # shared metrics helpers
├── app.py                  # Streamlit dashboard (Phase 5)
├── tests/
│   ├── test_features.py
│   └── test_train.py
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


## Background

Built by Kerry Hall, applying a chemical engineering background and
manufacturing quality engineering experience (Goodyear, Morgan Advanced
Materials) to the standard academic benchmark for process fault detection
in the field he studied.

## License

MIT