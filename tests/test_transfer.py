"""
Tests for black-box transfer attack evaluation (transfer.py) and the
sklearn classifier wrappers (sklearn_wrap.py).

These tests verify that:
  1. transfer_evasion_rate correctly measures evasion on black-box targets.
  2. The wrapper interface (predict / predict_proba / name) is correct.
  3. transfer_table returns the expected row structure.
  4. train_random_forest and train_xgboost (if available) fit and predict.
"""

import numpy as np
import pytest
from sklearn.dummy import DummyClassifier

from netadv.attacks.transfer import transfer_evasion_rate, transfer_table
from netadv.classifiers.sklearn_wrap import SklearnClassifier, train_random_forest


# ── Helpers ───────────────────────────────────────────────────────────────────

def _always_benign_clf(name="AlwaysBenign"):
    """Sklearn classifier that always predicts 0 (benign)."""
    clf = DummyClassifier(strategy="constant", constant=0)
    clf.fit([[0], [1]], [0, 1])
    return SklearnClassifier(clf, name=name)


def _always_attack_clf(name="AlwaysAttack"):
    """Sklearn classifier that always predicts 1 (attack)."""
    clf = DummyClassifier(strategy="constant", constant=1)
    clf.fit([[0], [1]], [0, 1])
    return SklearnClassifier(clf, name=name)


def _make_data(n=50, d=5, n_attacks=30, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d)).astype(np.float32)
    y = np.zeros(n, dtype=int)
    y[:n_attacks] = 1
    X_adv = X + 0.1  # simple perturbation (doesn't need to fool anything here)
    return X, y, X_adv


# ── SklearnClassifier wrapper ─────────────────────────────────────────────────

class TestSklearnClassifier:
    def test_predict_returns_int_array(self):
        clf = _always_benign_clf()
        X = np.zeros((10, 1), dtype=np.float32)
        preds = clf.predict(X)
        assert preds.dtype in (np.int32, np.int64, int)
        assert preds.shape == (10,)

    def test_name_attribute(self):
        clf = _always_attack_clf("TestModel")
        assert clf.name == "TestModel"

    def test_score_is_accuracy(self):
        clf = _always_benign_clf()
        X = np.zeros((10, 1))
        y = np.zeros(10, dtype=int)
        assert clf.score(X, y) == pytest.approx(1.0)

    def test_score_zero_when_all_wrong(self):
        clf = _always_benign_clf()
        X = np.zeros((10, 1))
        y = np.ones(10, dtype=int)
        assert clf.score(X, y) == pytest.approx(0.0)


# ── transfer_evasion_rate ─────────────────────────────────────────────────────

class TestTransferEvasionRate:
    def test_always_benign_model_evades_all_attacks(self):
        X, y, X_adv = _make_data(n=50, n_attacks=30)
        clf = _always_benign_clf()
        result = transfer_evasion_rate(X_adv, y, targets=[clf])
        assert result["AlwaysBenign"] == pytest.approx(1.0)

    def test_always_attack_model_evades_nothing(self):
        X, y, X_adv = _make_data(n=50, n_attacks=30)
        clf = _always_attack_clf()
        result = transfer_evasion_rate(X_adv, y, targets=[clf])
        assert result["AlwaysAttack"] == pytest.approx(0.0)

    def test_denominator_is_attack_only(self):
        """Only y==1 samples count in the denominator."""
        n_attacks = 20
        X, y, X_adv = _make_data(n=40, n_attacks=n_attacks)
        # Target always predicts benign → all n_attacks evade
        clf = _always_benign_clf()
        result = transfer_evasion_rate(X_adv, y, targets=[clf])
        assert result["AlwaysBenign"] == pytest.approx(1.0)

    def test_no_attacks_returns_zero(self):
        X = np.zeros((10, 5), dtype=np.float32)
        y = np.zeros(10, dtype=int)   # all benign
        clf = _always_benign_clf()
        result = transfer_evasion_rate(X, y, targets=[clf])
        assert result["AlwaysBenign"] == 0.0

    def test_multiple_targets_reported_independently(self):
        X, y, X_adv = _make_data(n=50, n_attacks=25)
        targets = [_always_benign_clf("B"), _always_attack_clf("A")]
        result = transfer_evasion_rate(X_adv, y, targets=targets)
        assert "B" in result and "A" in result
        assert result["B"] == pytest.approx(1.0)
        assert result["A"] == pytest.approx(0.0)

    def test_rate_is_float_in_unit_interval(self):
        X, y, X_adv = _make_data()
        clf = _always_benign_clf()
        rate = transfer_evasion_rate(X_adv, y, targets=[clf])["AlwaysBenign"]
        assert isinstance(rate, float)
        assert 0.0 <= rate <= 1.0


# ── transfer_table ────────────────────────────────────────────────────────────

class TestTransferTable:
    def test_returns_list_of_dicts(self):
        X, y, X_adv = _make_data()
        targets = [_always_benign_clf()]
        rows = transfer_table(X, y, X_adv, X_adv, targets)
        assert isinstance(rows, list)
        assert all(isinstance(r, dict) for r in rows)

    def test_row_has_required_keys(self):
        X, y, X_adv = _make_data()
        targets = [_always_benign_clf()]
        rows = transfer_table(X, y, X_adv, X_adv, targets)
        for row in rows:
            assert "classifier" in row
            assert "variant" in row
            assert "evasion_rate" in row

    def test_variants_are_constrained_and_unconstrained(self):
        X, y, X_adv = _make_data()
        targets = [_always_benign_clf()]
        rows = transfer_table(X, y, X_adv, X_adv, targets)
        variants = {r["variant"] for r in rows}
        assert "constrained" in variants
        assert "unconstrained" in variants

    def test_two_targets_produce_four_rows(self):
        X, y, X_adv = _make_data()
        targets = [_always_benign_clf("B"), _always_attack_clf("A")]
        rows = transfer_table(X, y, X_adv, X_adv, targets)
        # 2 variants × 2 targets = 4 rows
        assert len(rows) == 4


# ── train_random_forest (integration) ────────────────────────────────────────

class TestTrainRandomForest:
    def test_rf_trains_and_predicts(self):
        rng = np.random.default_rng(0)
        X = rng.standard_normal((200, 10)).astype(np.float32)
        y = (X[:, 0] > 0).astype(int)
        rf = train_random_forest(X, y, n_estimators=10, seed=0)
        preds = rf.predict(X)
        assert preds.shape == (200,)
        assert set(preds).issubset({0, 1})

    def test_rf_name_attribute(self):
        rng = np.random.default_rng(0)
        X = rng.standard_normal((100, 5)).astype(np.float32)
        y = (X[:, 0] > 0).astype(int)
        rf = train_random_forest(X, y, n_estimators=5, seed=0)
        assert rf.name == "RandomForest"

    def test_rf_accuracy_above_chance(self):
        rng = np.random.default_rng(42)
        X = rng.standard_normal((500, 10)).astype(np.float32)
        y = (X[:, 0] > 0).astype(int)
        rf = train_random_forest(X, y, n_estimators=20, seed=42)
        acc = rf.score(X, y)
        assert acc > 0.6   # should be much better than chance (0.5)


# ── train_xgboost (optional) ──────────────────────────────────────────────────

class TestTrainXGBoost:
    def test_xgb_trains_if_available(self):
        pytest.importorskip("xgboost")
        from netadv.classifiers.sklearn_wrap import train_xgboost
        rng = np.random.default_rng(0)
        X = rng.standard_normal((200, 10)).astype(np.float32)
        y = (X[:, 0] > 0).astype(int)
        xgb = train_xgboost(X, y, seed=0)
        preds = xgb.predict(X)
        assert preds.shape == (200,)
        assert xgb.name == "XGBoost"

    def test_xgb_raises_import_error_when_absent(self, monkeypatch):
        import builtins
        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "xgboost":
                raise ImportError("xgboost not installed")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", mock_import)
        from netadv.classifiers import sklearn_wrap
        import importlib
        importlib.reload(sklearn_wrap)

        with pytest.raises(ImportError):
            sklearn_wrap.train_xgboost(
                np.zeros((10, 2)), np.zeros(10, dtype=int)
            )
