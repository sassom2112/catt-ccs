"""
sklearn_wrap.py — thin wrappers so RF/XGBoost can be used as black-box
                  target classifiers in transfer-attack evaluations.

The wrappers implement a minimal interface expected by transfer.py:
    predict(X: np.ndarray) -> np.ndarray  (int labels, shape [N])
    predict_proba(X: np.ndarray) -> np.ndarray  (shape [N, 2])

Any scikit-learn binary classifier already satisfies this; the wrappers
add a common .name attribute and a from_trained() constructor for
clarity in benchmark scripts.
"""

from __future__ import annotations

import numpy as np


class SklearnClassifier:
    """Wraps any fitted sklearn binary classifier for transfer evaluation."""

    def __init__(self, model, name: str = "sklearn"):
        self.model = model
        self.name  = name

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X).astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        return float((self.predict(X) == y).mean())


def train_random_forest(X_tr, y_tr, *, n_estimators=300, n_jobs=-1, seed=42):
    from sklearn.ensemble import RandomForestClassifier
    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        class_weight="balanced",
        random_state=seed,
        n_jobs=n_jobs,
    )
    clf.fit(X_tr, y_tr)
    return SklearnClassifier(clf, name="RandomForest")


def train_xgboost(X_tr, y_tr, *, seed=42):
    try:
        from xgboost import XGBClassifier
    except ImportError:
        raise ImportError("xgboost not installed — pip install netadv-ccs[xgb]")
    scale_pos_weight = float((y_tr == 0).sum()) / max((y_tr == 1).sum(), 1)
    clf = XGBClassifier(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.1,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        random_state=seed,
        n_jobs=-1,
        verbosity=0,
    )
    clf.fit(X_tr, y_tr)
    return SklearnClassifier(clf, name="XGBoost")
