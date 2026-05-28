"""
calibration.py — translate epsilon from StandardScaler-normalized space
                 back to original feature units.

Why this matters for CCS:
    ε = 0.40 in scaled space is uninterpretable to practitioners.
    This module answers: "How many packets / bytes / milliseconds is that?"

Usage:
    from netadv.eval.calibration import epsilon_in_original_units, calibration_table

    table = calibration_table(
        pipeline,
        feature_names=NUM_FEATURES,
        epsilons=[0.10, 0.20, 0.40],
        representative_features=["dur", "spkts", "sbytes", "sttl"],
    )
    print(table.to_string(index=False))
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline


def epsilon_in_original_units(
    epsilon: float,
    pipeline: Pipeline,
    feature_names: list[str],
) -> dict[str, float]:
    """
    Convert a scalar epsilon (in normalized space) to per-feature original-
    domain magnitudes.

    For StandardScaler: x_scaled = (x - mean) / std
    So a perturbation of ε in scaled space = ε * std in original space.

    Parameters
    ----------
    epsilon : float
        Perturbation budget in the StandardScaler-normalized feature space.
    pipeline : fitted sklearn Pipeline
        Must contain a 'prep' step with a 'num' ColumnTransformer that
        includes a StandardScaler.
    feature_names : list of str
        Names of numeric features in the order the scaler saw them.

    Returns
    -------
    dict mapping feature_name → perturbation magnitude in original units
    """
    scaler = pipeline.named_steps["prep"].named_transformers_["num"].named_steps["scaler"]
    stds = scaler.scale_
    return {name: float(epsilon * std) for name, std in zip(feature_names, stds)}


def calibration_table(
    pipeline: Pipeline,
    feature_names: list[str],
    epsilons: list[float],
    representative_features: list[str] | None = None,
) -> pd.DataFrame:
    """
    Build a human-readable table showing ε in original-domain units for
    a subset of representative features.

    Parameters
    ----------
    representative_features : list of str, optional
        Subset of feature_names to show. If None, all features are included.

    Returns
    -------
    pd.DataFrame with columns: feature, unit_note, and one column per epsilon.
    """
    if representative_features is None:
        representative_features = feature_names

    rows = []
    for eps in epsilons:
        orig = epsilon_in_original_units(eps, pipeline, feature_names)
        for feat in representative_features:
            if feat in orig:
                rows.append({
                    "epsilon": eps,
                    "feature": feat,
                    "original_unit_magnitude": orig[feat],
                })

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    pivot = df.pivot_table(
        index="feature",
        columns="epsilon",
        values="original_unit_magnitude",
    ).reset_index()
    pivot.columns.name = None
    pivot.columns = ["feature"] + [f"ε={e}" for e in epsilons]
    return pivot


# Representative features for UNSW-NB15 calibration table in the paper
UNSW_REPRESENTATIVE = ["dur", "spkts", "dpkts", "sbytes", "dbytes", "sttl", "dttl"]

# Representative features for CICIDS2017 calibration table in the paper
CICIDS_REPRESENTATIVE = [
    "Flow Duration",
    "Total Fwd Packets",
    "Total Backward Packets",
    "Total Length of Fwd Packets",
    "Fwd Packet Length Max",
    "Flow IAT Mean",
]
