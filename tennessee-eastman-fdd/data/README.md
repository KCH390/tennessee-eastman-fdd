# Data

## raw/
Source data, downloaded not committed (see .gitignore) & main README. 

## processed/
Cleaned / feature-engineered versions derived from raw/, produced by
scripts in src/. Also gitignored -- regenerate by re-running the pipeline,
don't hand-edit anything here.

## external/
Small reference files worth checking into git directly (e.g. a variable/
tag name lookup table, fault-type descriptions). Nothing large or
regenerable goes here.
