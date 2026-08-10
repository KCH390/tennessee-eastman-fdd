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

*(to be filled in once the data pipeline and modeling approach are settled
in Phase 1-2)*

## Tech Stack

- **Python (pandas, numpy)** — data loading and feature engineering
- **pyreadr** — reading the TEP dataset's native .RData format
- **scikit-learn, xgboost** — fault detection (binary) and diagnosis
  (multiclass) models
- **Streamlit** — monitoring dashboard
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
│   └── exploration.ipynb   # EDA and modeling notebook (added in Phase 1+)
├── src/
│   └── (data loading, feature engineering, training, evaluation -- added Phase 1+)
├── tests/
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

Data acquisition and the rest of the pipeline will be documented here as
each phase lands.

## Background

Built by Kerry Hall, applying a chemical engineering background and
manufacturing quality engineering experience (Goodyear, Morgan Advanced
Materials) to the standard academic benchmark for process fault detection
in the field he studied.

## License

MIT
