"""
transfer.py — black-box transfer attack evaluation.

Workflow:
  1. Generate adversarial examples against a white-box surrogate (MLP).
  2. Evaluate those examples against one or more black-box target classifiers
     (RF, XGBoost) that were never seen during attack generation.

This is the key CCS extension: the workshop paper showed white-box MLP gaps;
this module measures whether those gaps persist when the adversary only has
surrogate access (the realistic threat model for deployed NIDS).

Usage:
    from netadv.attacks.transfer import transfer_evasion_rate

    adv_examples = pgd(surrogate_mlp, X_te, y_te, epsilon=0.2,
                       bounds=bounds, device=device)

    results = transfer_evasion_rate(adv_examples, y_te, targets)
    # results: dict  {classifier_name: evasion_rate}
"""

from __future__ import annotations

import numpy as np


def transfer_evasion_rate(
    X_adv: np.ndarray,
    y_true: np.ndarray,
    targets: list,
    attack_class: int = 1,
) -> dict[str, float]:
    """
    Evaluate transfer evasion rate of pre-generated adversarial examples
    against a list of black-box target classifiers.

    Parameters
    ----------
    X_adv : np.ndarray, shape [N, d]
        Adversarial examples generated against a surrogate model.
    y_true : np.ndarray, shape [N]
        True binary labels (1 = attack, 0 = benign).
    targets : list of SklearnClassifier (or any object with .predict())
        Black-box target classifiers to evaluate against.
    attack_class : int
        Class label that represents attacks (default 1). Only attack-class
        samples are included in the evasion rate denominator — we measure
        how many attacks are misclassified as benign.

    Returns
    -------
    dict mapping classifier name → evasion rate (float in [0, 1])
    """
    attack_mask = (y_true == attack_class)
    if attack_mask.sum() == 0:
        return {t.name: 0.0 for t in targets}

    X_attacks = X_adv[attack_mask]
    results = {}
    for clf in targets:
        preds = clf.predict(X_attacks)
        evaded = (preds != attack_class).sum()
        results[clf.name] = float(evaded) / len(X_attacks)
    return results


def transfer_table(
    X_te: np.ndarray,
    y_te: np.ndarray,
    adv_constrained: np.ndarray,
    adv_unconstrained: np.ndarray,
    targets: list,
) -> list[dict]:
    """
    Build a comparison table row for one (epsilon, attack_type) combination.

    Returns a list of dicts with keys:
        classifier, variant, evasion_rate
    """
    rows = []
    for variant, X_adv in [("constrained", adv_constrained),
                            ("unconstrained", adv_unconstrained)]:
        rates = transfer_evasion_rate(X_adv, y_te, targets)
        for clf_name, rate in rates.items():
            rows.append({
                "classifier": clf_name,
                "variant":    variant,
                "evasion_rate": rate,
            })
    return rows
