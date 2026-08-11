# Data

## raw/
Source data, downloaded not committed (see .gitignore). Documented here
once Phase 1 settles which dataset variant we're using (original
Chiang/Downs & Vogel vs. Rieth et al. 2017) and exactly where to get it.

## processed/
Cleaned / feature-engineered versions derived from raw/, produced by
scripts in src/. Also gitignored -- regenerate by re-running the pipeline,
don't hand-edit anything here.

## external/
Small reference files worth checking into git directly (e.g. a variable/
tag name lookup table, fault-type descriptions). Nothing large or
regenerable goes here.
