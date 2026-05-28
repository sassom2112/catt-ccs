from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass

from .spec import ConstraintSpec


@dataclass
class ConstraintBounds:
    """
    Per-feature lower/upper bounds in the preprocessed (standardized + OHE) space.

    Produced by :meth:`from_spec`; consumed by the attack projection step.

    Attributes
    ----------
    lb, ub : ndarray of shape (n_total,)
        Lower / upper bounds in the preprocessed feature space.
    n_num  : int  — number of numeric (StandardScaler) features.
    n_total: int  — total features (numeric + OHE).
    """
    lb: np.ndarray
    ub: np.ndarray
    n_num: int
    n_total: int

    @classmethod
    def from_spec(cls, spec: ConstraintSpec, pipeline) -> ConstraintBounds:
        """
        Compute bounds in the preprocessed feature space.

        Parameters
        ----------
        spec:
            Dataset-specific :class:`~netadv.constraints.spec.ConstraintSpec`.
        pipeline:
            Fitted sklearn pipeline.  Expected structure::

                pipeline.named_steps["prep"]
                    .named_transformers_["num"].named_steps["scaler"]   # StandardScaler
                    .named_transformers_["cat"].named_steps["encoder"]  # OneHotEncoder (optional)

            The ``"cat"`` branch is optional — datasets with no categorical features
            (e.g. CICIDS2017 / CICFlowMeter output) work without it.

        Returns
        -------
        ConstraintBounds
            Bounds in the (StandardScaler-transformed + OHE) feature space.
        """
        pre    = pipeline.named_steps["prep"]
        scaler = pre.named_transformers_["num"].named_steps["scaler"]

        n_num = scaler.n_features_in_

        if "cat" in pre.named_transformers_:
            ohe   = pre.named_transformers_["cat"].named_steps["encoder"]
            n_ohe = sum(len(cats) for cats in ohe.categories_)
        else:
            n_ohe = 0

        n_total = n_num + n_ohe

        lb = np.full(n_total, -np.inf, dtype=np.float32)
        ub = np.full(n_total,  np.inf, dtype=np.float32)

        mean_    = scaler.mean_
        scale_   = scaler.scale_
        feat_idx = {name: i for i, name in enumerate(spec.numeric_features)}

        for feat_name, bound in spec.bounds.items():
            if feat_name not in feat_idx:
                continue
            idx = feat_idx[feat_name]
            if bound.lo is not None:
                lb[idx] = (bound.lo - mean_[idx]) / scale_[idx]
            if bound.hi is not None:
                ub[idx] = (bound.hi - mean_[idx]) / scale_[idx]

        # OHE features: continuous relaxation to [0, 1]
        if n_ohe > 0:
            lb[n_num:] = 0.0
            ub[n_num:] = 1.0

        return cls(lb=lb, ub=ub, n_num=n_num, n_total=n_total)


def validity_report(X_transformed: np.ndarray, bounds: ConstraintBounds) -> pd.DataFrame:
    """
    Check how many samples in X_transformed violate each bound.

    Run on clean (unperturbed, preprocessed) data before launching attacks.
    An empty return DataFrame means all samples are domain-valid.  Any rows
    returned indicate bound violations in the clean data — the usual cause is
    a mismatch between the feature order in ``ConstraintSpec.numeric_features``
    and the column order the pipeline was fitted on.

    Parameters
    ----------
    X_transformed:
        Preprocessed feature matrix (StandardScaler-transformed + OHE), shape (n, d).
    bounds:
        :class:`ConstraintBounds` for the dataset.

    Returns
    -------
    DataFrame with columns: feature_idx, lb, ub, n_below_lb, n_above_ub,
    n_violations, violation_rate.  Sorted descending by n_violations.
    Empty DataFrame if no violations are found.
    """
    rows = []
    n_samples = len(X_transformed)

    for i in range(bounds.n_total):
        col  = X_transformed[:, i]
        lb_i = float(bounds.lb[i])
        ub_i = float(bounds.ub[i])

        n_lo = int((col < lb_i).sum()) if not np.isinf(lb_i) else 0
        n_hi = int((col > ub_i).sum()) if not np.isinf(ub_i) else 0
        n_viol = n_lo + n_hi

        if n_viol > 0:
            rows.append({
                "feature_idx":   i,
                "lb":            None if np.isinf(lb_i) else round(lb_i, 4),
                "ub":            None if np.isinf(ub_i) else round(ub_i, 4),
                "n_below_lb":    n_lo,
                "n_above_ub":    n_hi,
                "n_violations":  n_viol,
                "violation_rate": round(n_viol / n_samples, 6),
            })

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.sort_values("n_violations", ascending=False).reset_index(drop=True)
