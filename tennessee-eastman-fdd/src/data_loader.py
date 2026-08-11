"""
Data loading for the Tennessee Eastman Process (Rieth et al. 2017) dataset.

Source: Rieth, C.A., Amsel, B.D., Tran, R., & Cook, M.B. (2017). "Additional
Tennessee Eastman Process Simulation Data for Anomaly Detection Evaluation."
Harvard Dataverse, V1. https://doi.org/10.7910/DVN/6C3JR1

Download the four files manually (no automated download is offered by
Harvard Dataverse -- robots disallow automated fetching, and the files are
~1.3GB combined) and place them in data/raw/:
    TEP_FaultFree_Training.RData
    TEP_Faulty_Training.RData
    TEP_FaultFree_Testing.RData
    TEP_Faulty_Testing.RData

SCHEMA (55 columns per Rieth et al.'s documentation):
    faultNumber   -- 0 for fault-free runs; 1-20 for faulty runs (which
                     fault SCENARIO was simulated -- see note below on
                     onset timing)
    simulationRun -- 1-500, a distinct RNG seed per run (train/test seeds
                     are non-overlapping)
    sample        -- time index within the run. Training: 1-500 (25 hours
                     at 3-min sampling). Testing: 1-960 (48 hours).
    xmeas_1..xmeas_41 -- 41 measured process variables
    xmv_1..xmv_11      -- 11 manipulated variables

CRITICAL LABELING DETAIL, sourced from multiple independent papers
describing this exact dataset (see notebooks/exploration.ipynb Phase 1
section for citations): faultNumber marks which fault SCENARIO a run
belongs to, but the fault is not active for the entire run --
    - Training runs: the fault is introduced after sample 20 (1 hour in).
      Samples 1-20 are still normal operation.
    - Testing runs: the fault is introduced after sample 160 (8 hours in).
      Samples 1-160 are still normal operation.
Treating every row in a "faulty" run as a positive fault label (a common
mistake) mislabels the pre-onset samples as faulty when they're actually
normal, which would corrupt both training and evaluation. This loader
adds a `fault_active` column that accounts for this correctly.

Also worth knowing before modeling (see fault_descriptions.csv): faults
3, 9, and 15 are widely reported in the literature as statistically very
difficult or impossible to detect from this data -- not a bug in this
loader, an established property of the process itself.
"""

import argparse
from pathlib import Path

import pandas as pd
# pyreadr is only needed for _load_rdata (actual file reading), imported lazily
# there -- keeps combine_and_label() usable/testable without pyreadr installed.

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")

TRAIN_ONSET_SAMPLE = 20   # fault becomes active AFTER this sample, in training runs
TEST_ONSET_SAMPLE = 160   # fault becomes active AFTER this sample, in testing runs

FILES = {
    ("training", False): "TEP_FaultFree_Training.RData",
    ("training", True): "TEP_Faulty_Training.RData",
    ("testing", False): "TEP_FaultFree_Testing.RData",
    ("testing", True): "TEP_Faulty_Testing.RData",
}

RDATA_OBJECT_NAMES = {
    ("training", False): "fault_free_training",
    ("training", True): "faulty_training",
    ("testing", False): "fault_free_testing",
    ("testing", True): "faulty_testing",
}


def _load_rdata(path: Path, object_name: str) -> pd.DataFrame:
    import pyreadr

    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Download it from "
            "https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/6C3JR1 "
            f"and place it at {path}."
        )
    result = pyreadr.read_r(str(path))
    if object_name not in result:
        raise ValueError(
            f"Expected object '{object_name}' in {path}, found: {list(result.keys())}. "
            "The file may not match the documented Rieth et al. format."
        )
    return result[object_name]


def combine_and_label(fault_free: pd.DataFrame, faulty: pd.DataFrame, onset_sample: int,
                       faults=None, runs=None) -> pd.DataFrame:
    """
    Pure function, no file I/O -- combines fault-free + faulty data and adds
    the onset-aware fault_active label. Separated from load_split() so this
    critical logic is testable independently of pyreadr/file availability.
    """
    if faults is not None:
        faulty = faulty[faulty["faultNumber"].isin(faults)]
    if runs is not None:
        fault_free = fault_free[fault_free["simulationRun"].isin(runs)]
        faulty = faulty[faulty["simulationRun"].isin(runs)]

    combined = pd.concat([fault_free, faulty], ignore_index=True)
    combined["fault_active"] = (combined["faultNumber"] > 0) & (combined["sample"] > onset_sample)
    return combined


def load_split(split: str, raw_dir: Path = RAW_DIR, faults=None, runs=None) -> pd.DataFrame:
    """
    Loads and combines fault-free + faulty data for one split ('training'
    or 'testing'), with onset-aware fault_active labeling.

    faults: optional list of fault numbers to include from the faulty file
            (in addition to all fault-free data). None = all faults.
    runs:   optional list/range of simulationRun values to include, applied
            to BOTH fault-free and faulty data. None = all runs. Useful for
            fast iteration during development -- the full dataset is large
            (500 runs x 500-960 samples x 20 faults for the faulty side).
    """
    if split not in ("training", "testing"):
        raise ValueError("split must be 'training' or 'testing'")

    onset_sample = TRAIN_ONSET_SAMPLE if split == "training" else TEST_ONSET_SAMPLE

    fault_free_path = raw_dir / FILES[(split, False)]
    faulty_path = raw_dir / FILES[(split, True)]

    fault_free = _load_rdata(fault_free_path, RDATA_OBJECT_NAMES[(split, False)])
    faulty = _load_rdata(faulty_path, RDATA_OBJECT_NAMES[(split, True)])

    return combine_and_label(fault_free, faulty, onset_sample, faults=faults, runs=runs)


def main():
    parser = argparse.ArgumentParser(description="Load TEP data and cache a processed parquet file.")
    parser.add_argument("--split", choices=["training", "testing", "both"], default="both")
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--faults", type=int, nargs="*", default=None)
    parser.add_argument("--runs", type=int, nargs="*", default=None)
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    splits = ["training", "testing"] if args.split == "both" else [args.split]
    for split in splits:
        df = load_split(split, raw_dir=raw_dir, faults=args.faults, runs=args.runs)
        out_path = PROCESSED_DIR / f"tep_{split}.parquet"
        df.to_parquet(out_path, index=False)
        print(f"{split}: {len(df):,} rows, {df['simulationRun'].nunique()} runs, "
              f"{df['faultNumber'].nunique()} fault scenarios -> {out_path}")
        print(f"  fault_active: {df['fault_active'].sum():,} / {len(df):,} rows "
              f"({df['fault_active'].mean():.1%})")


if __name__ == "__main__":
    main()
