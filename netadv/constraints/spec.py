from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass(frozen=True)
class FeatureBound:
    """
    Domain bounds for a single feature in the original (unscaled) space.

    lo=None means no lower bound; hi=None means no upper bound.
    In the preprocessed space, bounds are transformed via (val - mean) / scale.
    """
    lo: Optional[float] = None
    hi: Optional[float] = None


# Common presets ---------------------------------------------------------------
NONNEG  = FeatureBound(lo=0.0)
BOOLEAN = FeatureBound(lo=0.0, hi=1.0)
TTL     = FeatureBound(lo=0.0, hi=255.0)
PORT    = FeatureBound(lo=0.0, hi=65535.0)


@dataclass
class ConstraintSpec:
    """
    Dataset-level constraint specification for numeric features.

    Parameters
    ----------
    numeric_features:
        Ordered list of numeric feature names, matching the order expected by the
        fitted ColumnTransformer's 'num' branch.  Used to map names → column indices.
    bounds:
        Mapping from feature name → FeatureBound in the original domain.
        Features not listed are unconstrained (lb = -inf, ub = +inf).
    """
    numeric_features: list
    bounds: Dict[str, FeatureBound] = field(default_factory=dict)
