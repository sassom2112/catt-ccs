import numpy as np
import torch
import torch.nn as nn


def _get_preds(
    model: nn.Module,
    X: np.ndarray,
    device: torch.device,
    batch_size: int = 4096,
) -> np.ndarray:
    model.eval()
    logits = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            xb = torch.tensor(X[i:i + batch_size], dtype=torch.float32).to(device)
            logits.append(model(xb).cpu().numpy())
    return (np.concatenate(logits) > 0).astype(int)


def evasion_rate(
    model: nn.Module,
    X_adv: np.ndarray,
    y: np.ndarray,
    device: torch.device = None,
) -> float:
    """
    Fraction of true-attack samples (y == 1) that the model misclassifies as
    normal after perturbation — the primary metric for adversarial IDS evaluation.
    """
    if device is None:
        device = next(model.parameters()).device
    preds = _get_preds(model, X_adv, device)
    attack_mask = y == 1
    if attack_mask.sum() == 0:
        return 0.0
    return float((preds[attack_mask] == 0).mean())


def accuracy_under_attack(
    model: nn.Module,
    X_adv: np.ndarray,
    y: np.ndarray,
    device: torch.device = None,
) -> float:
    """Overall accuracy on adversarial examples."""
    if device is None:
        device = next(model.parameters()).device
    preds = _get_preds(model, X_adv, device)
    return float((preds == y).mean())


def xgb_transfer_evasion(
    pipeline,
    X_adv_transformed: np.ndarray,
    y: np.ndarray,
) -> float:
    """
    Transfer-attack evasion rate: adversarial examples crafted against an MLP
    surrogate, evaluated on an XGBoost classifier.

    ``X_adv_transformed`` must already be in the preprocessed feature space
    (i.e. the pipeline's preprocessing step has already been applied).
    """
    xgb_clf = pipeline.named_steps["clf"]
    preds = xgb_clf.predict(X_adv_transformed)
    attack_mask = y == 1
    if attack_mask.sum() == 0:
        return 0.0
    return float((preds[attack_mask] == 0).mean())
