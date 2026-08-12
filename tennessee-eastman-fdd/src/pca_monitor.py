"""
Classical PCA-based process monitoring: Hotelling's T^2 and SPE (Q)
statistics. This is the standard multivariate statistical process control
(MSPC) approach used throughout the actual TEP literature (Chiang,
Russell, and Braatz's book builds this exact method on this exact
dataset) -- included here as a complementary, more classical technique
alongside the tree-based ML models, not a replacement for them.

How it works: fit PCA on NORMAL operating data only, keeping enough
components to explain a target variance fraction. For a new sample:
  - T^2 measures how far the sample is from normal WITHIN the retained
    principal component subspace (captures unusual combinations of the
    variables PCA considers most important)
  - SPE (squared prediction error, aka Q) measures how far the sample is
    OUTSIDE that subspace (captures anomalies PCA's top components don't
    explain -- a genuinely different failure mode from T^2)
Control limits for both are set from the normal training data at a target
false-alarm rate, then flagged as an alarm if a new sample exceeds them.
"""

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


class PCAMonitor:
    def __init__(self, n_components=None, variance_threshold=0.9, alpha=0.01):
        """
        n_components: fix the number of components directly. If None, pick
            the smallest number that explains >= variance_threshold of
            total variance in the normal training data.
        alpha: false-alarm rate for the control limits (e.g. 0.01 -> the
            control limit is set so 1% of NORMAL training samples exceed it).
        """
        self.n_components = n_components
        self.variance_threshold = variance_threshold
        self.alpha = alpha
        self.scaler_ = None
        self.pca_ = None
        self.t2_limit_ = None
        self.spe_limit_ = None

    def fit(self, normal_df: pd.DataFrame, feature_columns):
        self.feature_columns_ = list(feature_columns)
        X = normal_df[self.feature_columns_].values

        self.scaler_ = StandardScaler()
        X_scaled = self.scaler_.fit_transform(X)

        if self.n_components is not None:
            self.pca_ = PCA(n_components=self.n_components)
        else:
            full_pca = PCA().fit(X_scaled)
            cumulative = np.cumsum(full_pca.explained_variance_ratio_)
            n = int(np.searchsorted(cumulative, self.variance_threshold) + 1)
            self.pca_ = PCA(n_components=n)

        self.pca_.fit(X_scaled)

        t2_train, spe_train = self._compute_statistics(X_scaled)
        # empirical control limits at the (1 - alpha) percentile of normal data
        self.t2_limit_ = np.quantile(t2_train, 1 - self.alpha)
        self.spe_limit_ = np.quantile(spe_train, 1 - self.alpha)
        return self

    def _compute_statistics(self, X_scaled):
        scores = self.pca_.transform(X_scaled)
        eigenvalues = self.pca_.explained_variance_
        # guard against near-zero eigenvalues causing a divide-by-near-zero blowup
        eigenvalues = np.where(eigenvalues < 1e-12, 1e-12, eigenvalues)
        t2 = np.sum((scores ** 2) / eigenvalues, axis=1)

        reconstructed = self.pca_.inverse_transform(scores)
        spe = np.sum((X_scaled - reconstructed) ** 2, axis=1)
        return t2, spe

    def score(self, df: pd.DataFrame) -> pd.DataFrame:
        X = df[self.feature_columns_].values
        X_scaled = self.scaler_.transform(X)
        t2, spe = self._compute_statistics(X_scaled)

        result = df[["faultNumber", "simulationRun", "sample"]].copy()
        if "fault_active" in df.columns:
            result["fault_active"] = df["fault_active"]
        result["t2"] = t2
        result["spe"] = spe
        result["t2_alarm"] = t2 > self.t2_limit_
        result["spe_alarm"] = spe > self.spe_limit_
        result["predicted"] = (result["t2_alarm"] | result["spe_alarm"]).astype(int)
        return result
