"""
Tests for FGSM and PGD attack implementations, plus the evasion_rate metric.

All tests use a tiny synthetic model and random data so no dataset files
are needed.  The critical invariants verified:

  1. L∞ budget is always respected (adversarial perturbation ≤ epsilon).
  2. Constrained outputs always satisfy domain bounds.
  3. Unconstrained attacks can violate bounds (confirming constraints matter).
  4. PGD evasion ≥ FGSM evasion on this toy problem (stronger attack).
  5. evasion_rate counts only attack-class samples in the denominator.
"""

import numpy as np
import pytest
import torch
import torch.nn as nn

from netadv.attacks.fgsm import fgsm
from netadv.attacks.pgd import pgd
from netadv.constraints.bounds import ConstraintBounds
from netadv.eval.metrics import evasion_rate


# ── Fixtures ──────────────────────────────────────────────────────────────────

class _LinearModel(nn.Module):
    """Linear model with squeeze so output is shape [n], matching BCEWithLogitsLoss."""
    def __init__(self, input_dim):
        super().__init__()
        self.fc = nn.Linear(input_dim, 1, bias=False)

    def forward(self, x):
        return self.fc(x).squeeze(-1)


def _trivial_model(input_dim: int) -> nn.Module:
    """Linear model that classifies sample as attack (1) when mean feature > 0."""
    model = _LinearModel(input_dim)
    with torch.no_grad():
        model.fc.weight.fill_(1.0 / input_dim)
    model.eval()
    return model


def _make_bounds(n, lo=-np.inf, hi=np.inf):
    return ConstraintBounds(
        lb=np.full(n, lo, dtype=np.float32),
        ub=np.full(n, hi, dtype=np.float32),
        n_num=n, n_total=n,
    )


@pytest.fixture
def setup():
    torch.manual_seed(0)
    np.random.seed(0)
    n, d = 200, 10
    X = np.random.randn(n, d).astype(np.float32)
    y = (X.mean(axis=1) > 0).astype(int)
    model = _trivial_model(d)
    device = torch.device("cpu")
    return model, X, y, d, device


# ── FGSM ─────────────────────────────────────────────────────────────────────

class TestFGSM:
    def test_output_shape_matches_input(self, setup):
        model, X, y, d, device = setup
        bounds = _make_bounds(d)
        X_adv = fgsm(model, X, y, epsilon=0.1, bounds=bounds, device=device)
        assert X_adv.shape == X.shape

    def test_linf_budget_respected(self, setup):
        model, X, y, d, device = setup
        eps = 0.1
        bounds = _make_bounds(d)
        X_adv = fgsm(model, X, y, epsilon=eps, bounds=bounds, device=device)
        max_perturbation = np.abs(X_adv - X).max()
        assert max_perturbation <= eps + 1e-5

    def test_constrained_output_satisfies_bounds(self, setup):
        model, X, y, d, device = setup
        lo, hi = 0.0, 2.0
        bounds = _make_bounds(d, lo=lo, hi=hi)
        X_adv = fgsm(model, X, y, epsilon=0.5, bounds=bounds, device=device)
        assert X_adv.min() >= lo - 1e-5
        assert X_adv.max() <= hi + 1e-5

    def test_unconstrained_can_violate_nonneg(self, setup):
        model, X, y, d, device = setup
        X_pos = np.abs(X) + 1.0  # all positive
        bounds = _make_bounds(d)  # ±inf = no constraint
        X_adv = fgsm(model, X_pos, y, epsilon=2.0, bounds=bounds, device=device)
        # With no constraint, some values may go negative
        # (not guaranteed on all inputs, but very likely with eps=2.0 and mean=1)
        nonneg_bounds = _make_bounds(d, lo=0.0)
        constrained_adv = fgsm(model, X_pos, y, epsilon=2.0, bounds=nonneg_bounds, device=device)
        assert constrained_adv.min() >= -1e-5
        # The two outputs should differ (constraint has effect)
        assert not np.allclose(X_adv, constrained_adv)

    def test_output_dtype_is_float32(self, setup):
        model, X, y, d, device = setup
        bounds = _make_bounds(d)
        X_adv = fgsm(model, X, y, epsilon=0.1, bounds=bounds, device=device)
        assert X_adv.dtype == np.float32

    def test_zero_epsilon_returns_original(self, setup):
        model, X, y, d, device = setup
        bounds = _make_bounds(d)
        X_adv = fgsm(model, X, y, epsilon=0.0, bounds=bounds, device=device)
        assert np.allclose(X_adv, X, atol=1e-6)


# ── PGD ──────────────────────────────────────────────────────────────────────

class TestPGD:
    def test_output_shape_matches_input(self, setup):
        model, X, y, d, device = setup
        bounds = _make_bounds(d)
        X_adv = pgd(model, X, y, epsilon=0.1, alpha=0.01, n_steps=5,
                    bounds=bounds, device=device)
        assert X_adv.shape == X.shape

    def test_linf_budget_respected(self, setup):
        model, X, y, d, device = setup
        eps = 0.2
        bounds = _make_bounds(d)
        X_adv = pgd(model, X, y, epsilon=eps, alpha=eps/10, n_steps=10,
                    bounds=bounds, device=device)
        max_perturbation = np.abs(X_adv - X).max()
        assert max_perturbation <= eps + 1e-5

    def test_constrained_output_satisfies_bounds(self, setup):
        model, X, y, d, device = setup
        lo, hi = -0.5, 0.5
        bounds = _make_bounds(d, lo=lo, hi=hi)
        X_adv = pgd(model, X, y, epsilon=0.3, alpha=0.01, n_steps=10,
                    bounds=bounds, device=device)
        assert X_adv.min() >= lo - 1e-5
        assert X_adv.max() <= hi + 1e-5

    def test_pgd_at_least_as_effective_as_fgsm(self, setup):
        model, X, y, d, device = setup
        eps = 0.3
        bounds = _make_bounds(d)
        X_fgsm = fgsm(model, X, y, epsilon=eps, bounds=bounds, device=device)
        X_pgd  = pgd(model, X, y, epsilon=eps, alpha=eps/10, n_steps=20,
                     bounds=bounds, device=device)
        er_fgsm = evasion_rate(model, X_fgsm, y, device=device)
        er_pgd  = evasion_rate(model, X_pgd,  y, device=device)
        assert er_pgd >= er_fgsm - 0.05   # PGD should not be much worse

    def test_no_random_init_deterministic(self, setup):
        model, X, y, d, device = setup
        bounds = _make_bounds(d)
        kwargs = dict(epsilon=0.1, alpha=0.01, n_steps=5, bounds=bounds,
                      device=device, random_init=False)
        X1 = pgd(model, X, y, **kwargs)
        X2 = pgd(model, X, y, **kwargs)
        assert np.allclose(X1, X2)

    def test_output_dtype_is_float32(self, setup):
        model, X, y, d, device = setup
        bounds = _make_bounds(d)
        X_adv = pgd(model, X, y, epsilon=0.1, alpha=0.01, n_steps=3,
                    bounds=bounds, device=device)
        assert X_adv.dtype == np.float32

    def test_constraint_tightens_evasion_rate(self, setup):
        """Constrained PGD should have lower or equal evasion than unconstrained."""
        model, X, y, d, device = setup
        eps = 0.5
        tight_bounds  = _make_bounds(d, lo=0.0, hi=1.0)
        no_bounds     = _make_bounds(d)
        X_con  = pgd(model, X, y, epsilon=eps, alpha=0.05, n_steps=20,
                     bounds=tight_bounds, device=device)
        X_uncon = pgd(model, X, y, epsilon=eps, alpha=0.05, n_steps=20,
                      bounds=no_bounds, device=device)
        er_con  = evasion_rate(model, X_con,  y, device=device)
        er_uncon = evasion_rate(model, X_uncon, y, device=device)
        # Unconstrained should be at least as effective (the key paper claim)
        assert er_uncon >= er_con - 0.02


# ── evasion_rate ──────────────────────────────────────────────────────────────

class TestEvasionRate:
    def test_denominator_is_attack_class_only(self):
        """Evasion rate counts only y==1 samples in the denominator."""
        model = nn.Linear(4, 1, bias=False)
        # Model classifies everything as benign (weight=0)
        with torch.no_grad():
            model.weight.fill_(0.0)
        model.eval()
        X = np.random.randn(10, 4).astype(np.float32)
        y = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])  # 5 attacks, 5 benign
        rate = evasion_rate(model, X, y, device=torch.device("cpu"))
        # All attacks evade (model always predicts 0 = benign)
        assert rate == pytest.approx(1.0)

    def test_no_attacks_returns_zero(self):
        model = nn.Linear(2, 1, bias=False)
        model.eval()
        X = np.zeros((5, 2), dtype=np.float32)
        y = np.zeros(5, dtype=int)  # all benign
        rate = evasion_rate(model, X, y, device=torch.device("cpu"))
        assert rate == 0.0

    def test_perfect_detection_returns_zero(self):
        """Model that always predicts attack → no evasion."""
        model = nn.Linear(4, 1, bias=False)
        with torch.no_grad():
            model.weight.fill_(100.0)   # huge logit → always attack
        model.eval()
        X = np.ones((10, 4), dtype=np.float32)
        y = np.ones(10, dtype=int)
        rate = evasion_rate(model, X, y, device=torch.device("cpu"))
        assert rate == pytest.approx(0.0)

    def test_rate_between_zero_and_one(self, setup):
        model, X, y, d, device = setup
        bounds = _make_bounds(d)
        X_adv = fgsm(model, X, y, epsilon=0.2, bounds=bounds, device=device)
        rate = evasion_rate(model, X_adv, y, device=device)
        assert 0.0 <= rate <= 1.0
