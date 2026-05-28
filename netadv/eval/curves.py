from typing import Callable, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import f1_score

from .metrics import _get_preds


def robustness_curve(
    model: nn.Module,
    X_clean: np.ndarray,
    y: np.ndarray,
    epsilons: List[float],
    attack_fn: Callable,
    device: torch.device = None,
) -> pd.DataFrame:
    """
    Sweep epsilon values and record F1, accuracy, and evasion rate at each level.

    Parameters
    ----------
    attack_fn : callable with signature ``attack_fn(epsilon: float) -> np.ndarray``
                returning adversarial examples at that budget.

    Returns
    -------
    DataFrame with columns: epsilon, clean_f1, clean_acc, adv_f1, adv_acc,
    evasion_rate, f1_drop.
    """
    if device is None:
        device = next(model.parameters()).device

    clean_preds = _get_preds(model, X_clean, device)
    clean_f1    = f1_score(y, clean_preds, zero_division=0)
    clean_acc   = float((clean_preds == y).mean())

    rows = []
    for eps in epsilons:
        X_adv      = attack_fn(epsilon=eps)
        adv_preds  = _get_preds(model, X_adv, device)
        attack_mask = y == 1
        evasion    = float((adv_preds[attack_mask] == 0).mean()) if attack_mask.sum() > 0 else 0.0
        adv_f1     = f1_score(y, adv_preds, zero_division=0)
        rows.append({
            "epsilon":      eps,
            "clean_f1":     clean_f1,
            "clean_acc":    clean_acc,
            "adv_f1":       adv_f1,
            "adv_acc":      float((adv_preds == y).mean()),
            "evasion_rate": evasion,
            "f1_drop":      clean_f1 - adv_f1,
        })

    return pd.DataFrame(rows)


def per_category_evasion(
    model: nn.Module,
    X_adv: np.ndarray,
    y: np.ndarray,
    attack_cat: np.ndarray,
    device: torch.device = None,
) -> pd.DataFrame:
    """
    Per attack category: sample count and evasion rate.

    Only evaluates rows where ``y == 1`` (actual attacks).

    Returns
    -------
    DataFrame with columns: attack_category, n_samples, evasion_rate.
    Sorted descending by evasion_rate.
    """
    if device is None:
        device = next(model.parameters()).device

    adv_preds = _get_preds(model, X_adv, device)
    rows = []
    for cat in sorted(np.unique(attack_cat[y == 1])):
        mask = (attack_cat == cat) & (y == 1)
        if mask.sum() == 0:
            continue
        rows.append({
            "attack_category": cat,
            "n_samples":       int(mask.sum()),
            "evasion_rate":    float((adv_preds[mask] == 0).mean()),
        })

    return pd.DataFrame(rows).sort_values("evasion_rate", ascending=False).reset_index(drop=True)


def compare_clean_vs_hardened(
    standard_model: nn.Module,
    hardened_model: nn.Module,
    X_clean: np.ndarray,
    X_adv: np.ndarray,
    y: np.ndarray,
    device: torch.device = None,
) -> pd.DataFrame:
    """
    Side-by-side metric comparison: standard vs. adversarially trained model.

    Returns
    -------
    DataFrame indexed by model name with columns:
    Clean F1, Adv F1, Clean Acc, Adv Acc, Evasion Rate, F1 Retained (%).
    """
    if device is None:
        device = next(standard_model.parameters()).device

    rows = []
    for label, model in [("Standard", standard_model), ("Adversarially Trained", hardened_model)]:
        c_preds = _get_preds(model, X_clean, device)
        a_preds = _get_preds(model, X_adv, device)
        attack_mask = y == 1
        rows.append({
            "Model":        label,
            "Clean F1":     f1_score(y, c_preds, zero_division=0),
            "Adv F1":       f1_score(y, a_preds, zero_division=0),
            "Clean Acc":    float((c_preds == y).mean()),
            "Adv Acc":      float((a_preds == y).mean()),
            "Evasion Rate": float((a_preds[attack_mask] == 0).mean()) if attack_mask.sum() > 0 else 0.0,
        })

    df = pd.DataFrame(rows).set_index("Model")
    df["F1 Retained (%)"] = (df["Adv F1"] / df["Clean F1"] * 100).round(1)
    return df
