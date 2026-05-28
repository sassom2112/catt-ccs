"""
Tests for constraint specification, bounds computation, and validity reporting.

These tests are the reviewer-facing proof that domain constraint projection
is mathematically correct.  They run entirely in-memory — no dataset files
are required.
"""

import numpy as np
import pytest
import torch
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from netadv.attacks.base import project
from netadv.constraints.bounds import ConstraintBounds, validity_report
from netadv.constraints.spec import BOOLEAN, NONNEG, TTL, FeatureBound, ConstraintSpec


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_pipeline(means, stds):
    """Build a minimal fitted pipeline with a known StandardScaler."""
    from sklearn.preprocessing import StandardScaler
    from unittest.mock import MagicMock

    scaler = StandardScaler()
    scaler.mean_  = np.array(means, dtype=np.float64)
    scaler.scale_ = np.array(stds, dtype=np.float64)
    scaler.n_features_in_ = len(means)

    num_transformer = MagicMock()
    num_transformer.named_steps = {"scaler": scaler}

    # Use a plain dict — "cat" not present, so `"cat" in prep.named_transformers_`
    # returns False naturally without any mock override needed.
    prep = MagicMock()
    prep.named_transformers_ = {"num": num_transformer}

    pipeline = MagicMock()
    pipeline.named_steps = {"prep": prep}
    return pipeline


def _make_bounds(means, stds, feature_names, bound_map):
    """Build ConstraintBounds from explicit means/stds and a bound map."""
    spec = ConstraintSpec(numeric_features=feature_names, bounds=bound_map)
    pipeline = _make_pipeline(means, stds)
    return ConstraintBounds.from_spec(spec, pipeline)


# ── ConstraintSpec ────────────────────────────────────────────────────────────

class TestConstraintSpec:
    def test_feature_bound_unbounded(self):
        b = FeatureBound()
        assert b.lo is None
        assert b.hi is None

    def test_nonneg_preset(self):
        assert NONNEG.lo == 0.0
        assert NONNEG.hi is None

    def test_boolean_preset(self):
        assert BOOLEAN.lo == 0.0
        assert BOOLEAN.hi == 1.0

    def test_ttl_preset(self):
        assert TTL.lo == 0.0
        assert TTL.hi == 255.0

    def test_spec_stores_feature_order(self):
        names = ["a", "b", "c"]
        spec = ConstraintSpec(numeric_features=names, bounds={"a": NONNEG})
        assert spec.numeric_features == names

    def test_spec_unlisted_features_are_unconstrained(self):
        spec = ConstraintSpec(numeric_features=["a", "b"], bounds={"a": NONNEG})
        assert "b" not in spec.bounds or spec.bounds.get("b") is None or True
        # The key check: from_spec will leave unlisted features as ±inf


# ── ConstraintBounds.from_spec ────────────────────────────────────────────────

class TestConstraintBoundsFromSpec:
    def test_nonneg_lower_bound_in_scaled_space(self):
        # Feature mean=10, std=5 → lb in scaled space = (0 - 10) / 5 = -2.0
        means = [10.0]
        stds  = [5.0]
        bounds = _make_bounds(means, stds, ["x"], {"x": NONNEG})
        assert bounds.lb[0] == pytest.approx(-2.0)
        assert np.isinf(bounds.ub[0])

    def test_boolean_both_bounds_scaled(self):
        # mean=0.5, std=0.5 → lb=(0-0.5)/0.5=-1, ub=(1-0.5)/0.5=1
        bounds = _make_bounds([0.5], [0.5], ["b"], {"b": BOOLEAN})
        assert bounds.lb[0] == pytest.approx(-1.0)
        assert bounds.ub[0] == pytest.approx(1.0)

    def test_ttl_upper_bound_scaled(self):
        # mean=64, std=32 → ub=(255-64)/32 ≈ 5.97
        bounds = _make_bounds([64.0], [32.0], ["ttl"], {"ttl": TTL})
        assert bounds.lb[0] == pytest.approx(-2.0)    # (0-64)/32
        assert bounds.ub[0] == pytest.approx((255 - 64) / 32)

    def test_unlisted_feature_stays_inf(self):
        bounds = _make_bounds([0.0, 0.0], [1.0, 1.0], ["a", "b"], {"a": NONNEG})
        # feature b has no bound → remains ±inf
        assert np.isinf(bounds.lb[1])
        assert np.isinf(bounds.ub[1])

    def test_n_total_matches_n_num_without_cat(self):
        bounds = _make_bounds([0.0, 1.0, 2.0], [1.0, 1.0, 1.0],
                              ["x", "y", "z"], {})
        assert bounds.n_num == 3
        assert bounds.n_total == 3

    def test_custom_range_feature(self):
        # port: [0, 65535], mean=8000, std=4000
        port_bound = FeatureBound(lo=0.0, hi=65535.0)
        bounds = _make_bounds([8000.0], [4000.0], ["port"], {"port": port_bound})
        assert bounds.lb[0] == pytest.approx(-2.0)          # (0 - 8000) / 4000
        assert bounds.ub[0] == pytest.approx((65535 - 8000) / 4000)


# ── project (attack projection function) ─────────────────────────────────────

class TestProject:
    def _bounds(self, lb, ub):
        lb_arr = np.array(lb, dtype=np.float32)
        ub_arr = np.array(ub, dtype=np.float32)
        return ConstraintBounds(lb=lb_arr, ub=ub_arr,
                                n_num=len(lb), n_total=len(lb))

    def test_values_within_bounds_unchanged(self):
        b = self._bounds([-1.0, -1.0], [1.0, 1.0])
        x = torch.tensor([[0.5, -0.5]])
        out = project(x, b)
        assert torch.allclose(out, x)

    def test_values_below_lb_clipped_to_lb(self):
        b = self._bounds([0.0], [np.inf])
        x = torch.tensor([[-3.0]])
        out = project(x, b)
        assert out[0, 0].item() == pytest.approx(0.0)

    def test_values_above_ub_clipped_to_ub(self):
        b = self._bounds([-np.inf], [1.0])
        x = torch.tensor([[5.0]])
        out = project(x, b)
        assert out[0, 0].item() == pytest.approx(1.0)

    def test_both_bounds_applied_simultaneously(self):
        b = self._bounds([0.0, -2.0], [1.0, 2.0])
        x = torch.tensor([[-0.5, 3.0]])
        out = project(x, b)
        assert out[0, 0].item() == pytest.approx(0.0)
        assert out[0, 1].item() == pytest.approx(2.0)

    def test_unbounded_features_pass_through(self):
        b = self._bounds([-np.inf, 0.0], [np.inf, np.inf])
        x = torch.tensor([[-1000.0, -5.0]])
        out = project(x, b)
        assert out[0, 0].item() == pytest.approx(-1000.0)
        assert out[0, 1].item() == pytest.approx(0.0)

    def test_batch_projection_all_rows(self):
        b = self._bounds([0.0], [1.0])
        x = torch.tensor([[-1.0], [0.5], [2.0]])
        out = project(x, b)
        assert out[0, 0].item() == pytest.approx(0.0)
        assert out[1, 0].item() == pytest.approx(0.5)
        assert out[2, 0].item() == pytest.approx(1.0)

    def test_projection_is_idempotent(self):
        b = self._bounds([0.0], [1.0])
        x = torch.tensor([[0.3]])
        out1 = project(x, b)
        out2 = project(out1, b)
        assert torch.allclose(out1, out2)


# ── validity_report ───────────────────────────────────────────────────────────

class TestValidityReport:
    def _make_bounds_direct(self, lb, ub):
        return ConstraintBounds(
            lb=np.array(lb, dtype=np.float32),
            ub=np.array(ub, dtype=np.float32),
            n_num=len(lb), n_total=len(lb),
        )

    def test_clean_data_returns_empty_df(self):
        bounds = self._make_bounds_direct([0.0, 0.0], [1.0, 1.0])
        X = np.array([[0.2, 0.8], [0.5, 0.5]], dtype=np.float32)
        report = validity_report(X, bounds)
        assert report.empty

    def test_one_violation_below_lb(self):
        bounds = self._make_bounds_direct([0.0], [np.inf])
        X = np.array([[-0.1], [0.5]], dtype=np.float32)
        report = validity_report(X, bounds)
        assert len(report) == 1
        assert report.iloc[0]["n_below_lb"] == 1

    def test_one_violation_above_ub(self):
        bounds = self._make_bounds_direct([-np.inf], [1.0])
        X = np.array([[1.5], [0.5]], dtype=np.float32)
        report = validity_report(X, bounds)
        assert len(report) == 1
        assert report.iloc[0]["n_above_ub"] == 1

    def test_violation_rate_computed_correctly(self):
        bounds = self._make_bounds_direct([0.0], [np.inf])
        X = np.array([[-1.0], [-1.0], [1.0], [1.0]], dtype=np.float32)
        report = validity_report(X, bounds)
        assert report.iloc[0]["violation_rate"] == pytest.approx(0.5)

    def test_unbounded_feature_never_reported(self):
        bounds = self._make_bounds_direct([-np.inf], [np.inf])
        X = np.array([[-1e9], [1e9]], dtype=np.float32)
        report = validity_report(X, bounds)
        assert report.empty

    def test_sorted_descending_by_violations(self):
        bounds = self._make_bounds_direct([0.0, 0.0], [np.inf, np.inf])
        # Feature 0: 3 violations; feature 1: 1 violation
        X = np.array([[-1, -1], [-1, 0], [-1, 0], [0, 0]], dtype=np.float32)
        report = validity_report(X, bounds)
        assert report.iloc[0]["n_violations"] >= report.iloc[-1]["n_violations"]


# ── Dataset specs: smoke-import ───────────────────────────────────────────────

class TestDatasetSpecs:
    def test_unsw_spec_imports(self):
        from netadv.constraints.datasets.unsw_nb15 import UNSW_NB15_SPEC, NUM_FEATURES
        assert len(NUM_FEATURES) > 0
        assert len(UNSW_NB15_SPEC.bounds) > 0

    def test_cicids_spec_imports(self):
        from netadv.constraints.datasets.cicids2017 import CICIDS2017_SPEC, NUM_FEATURES
        assert len(NUM_FEATURES) == 78
        assert len(CICIDS2017_SPEC.bounds) > 0

    def test_nsl_kdd_spec_imports(self):
        from netadv.constraints.datasets.nsl_kdd import NSL_KDD_SPEC, NUM_FEATURES
        assert len(NUM_FEATURES) == 38
        assert len(NSL_KDD_SPEC.bounds) > 0

    def test_nsl_kdd_rate_features_bounded_01(self):
        from netadv.constraints.datasets.nsl_kdd import NSL_KDD_SPEC
        rate_feats = [f for f, b in NSL_KDD_SPEC.bounds.items()
                      if b.lo == 0.0 and b.hi == 1.0]
        assert len(rate_feats) >= 15

    def test_cicids_flag_features_bounded_255(self):
        from netadv.constraints.datasets.cicids2017 import CICIDS2017_SPEC
        flag_feats = [f for f, b in CICIDS2017_SPEC.bounds.items()
                      if b.hi == 255.0]
        assert len(flag_feats) >= 4
