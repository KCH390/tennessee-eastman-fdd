"""
Hyperparameter tuning for the detection/diagnosis models.

CRITICAL DETAIL: sklearn's GridSearchCV/RandomizedSearchCV default to
plain KFold, which splits by ROW. For this data that's the same leakage
problem features.py's split_by_run() exists to prevent -- adjacent
timesteps of the same simulation run are highly correlated, so a plain
KFold would put some rows of a run in the training fold and others in the
validation fold of the SAME CV split, overstating cross-validated
performance. This module uses GroupKFold with groups built from
(faultNumber, simulationRun) combined, so an entire run is always kept
together on one side of every fold.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold, RandomizedSearchCV

PARAM_DISTRIBUTIONS = {
    "random_forest": {
        "n_estimators": [100, 200, 300],
        "max_depth": [None, 10, 20, 30],
        "min_samples_leaf": [1, 2, 4, 8],
    },
    "hist_gradient_boosting": {
        "max_iter": [100, 200, 300],
        "max_depth": [None, 10, 20],
        "learning_rate": [0.03, 0.1, 0.3],
    },
}

BASE_MODELS = {
    "random_forest": lambda seed: RandomForestClassifier(random_state=seed, n_jobs=-1),
    "hist_gradient_boosting": lambda seed: HistGradientBoostingClassifier(random_state=seed),
}


def make_run_groups(df: pd.DataFrame) -> np.ndarray:
    """
    One group ID per distinct (faultNumber, simulationRun) pair -- since
    simulationRun numbers repeat across fault scenarios (see features.py),
    grouping by simulationRun alone would incorrectly treat different
    faults' runs as the same group.
    """
    return (df["faultNumber"].astype(str) + "_" + df["simulationRun"].astype(str)).values


def tune_model(train_df: pd.DataFrame, target_column: str, feature_columns,
               model_type="random_forest", n_splits=5, n_iter=10, seed=42, scoring="f1_macro"):
    """
    Randomized search over PARAM_DISTRIBUTIONS[model_type], using
    GroupKFold so no simulation run is ever split across a fold boundary.
    Returns the fitted RandomizedSearchCV object (best_estimator_,
    best_params_, cv_results_ all available on it).
    """
    groups = make_run_groups(train_df)
    cv = GroupKFold(n_splits=n_splits)

    base_model = BASE_MODELS[model_type](seed)
    search = RandomizedSearchCV(
        base_model,
        param_distributions=PARAM_DISTRIBUTIONS[model_type],
        n_iter=n_iter,
        cv=cv,
        scoring=scoring,
        random_state=seed,
        n_jobs=-1,
    )
    search.fit(train_df[feature_columns], train_df[target_column], groups=groups)
    return search
