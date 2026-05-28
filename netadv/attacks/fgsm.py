import numpy as np
import torch
import torch.nn as nn

from ..constraints.bounds import ConstraintBounds
from .base import project


def fgsm(
    model: nn.Module,
    X: np.ndarray,
    y: np.ndarray,
    epsilon: float,
    bounds: ConstraintBounds,
    device: torch.device = None,
    batch_size: int = 4096,
) -> np.ndarray:
    """
    Fast Gradient Sign Method with domain constraint projection.

    Produces adversarial examples that remain physically plausible for
    network traffic (non-negative counts, TTL in [0,255], ports in [0,65535], etc.).

    Parameters
    ----------
    model     : differentiable binary classifier (output is a scalar logit).
    X         : preprocessed features, shape (n, d).
    y         : binary labels (0 = normal, 1 = attack), shape (n,).
    epsilon   : L∞ perturbation budget in the preprocessed space.
    bounds    : domain constraint bounds from :class:`~netadv.constraints.ConstraintBounds`.
    device    : torch device; inferred from model parameters if None.
    batch_size: number of samples per forward/backward pass.

    Returns
    -------
    X_adv : np.ndarray, same shape as X.
    """
    if device is None:
        device = next(model.parameters()).device

    criterion = nn.BCEWithLogitsLoss()
    model.eval()
    results = []

    for i in range(0, len(X), batch_size):
        xb = torch.tensor(X[i:i + batch_size], dtype=torch.float32).to(device)
        xb.requires_grad_(True)
        yb = torch.tensor(y[i:i + batch_size], dtype=torch.float32).to(device)

        loss = criterion(model(xb), yb)
        loss.backward()

        with torch.no_grad():
            x_adv = xb + epsilon * xb.grad.sign()
            x_adv = project(x_adv, bounds)
        results.append(x_adv.cpu().numpy())

    return np.concatenate(results, axis=0)
