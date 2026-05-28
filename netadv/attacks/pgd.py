import numpy as np
import torch
import torch.nn as nn

from ..constraints.bounds import ConstraintBounds
from .base import project


def pgd(
    model: nn.Module,
    X: np.ndarray,
    y: np.ndarray,
    epsilon: float,
    alpha: float,
    n_steps: int,
    bounds: ConstraintBounds,
    device: torch.device = None,
    random_init: bool = True,
    batch_size: int = 2048,
) -> np.ndarray:
    """
    Projected Gradient Descent attack (Madry et al., 2018) with domain constraint projection.

    At each step the iterate is projected onto both the L∞ epsilon-ball around the
    original point *and* the domain-validity region defined by ``bounds``.  This
    means adversarial examples are valid network flows, not just gradient noise.

    Parameters
    ----------
    model      : differentiable binary classifier (output is a scalar logit).
    X          : preprocessed features, shape (n, d).
    y          : binary labels (0 = normal, 1 = attack), shape (n,).
    epsilon    : L∞ perturbation budget in the preprocessed space.
    alpha      : per-step size (typically epsilon / n_steps * 2.5).
    n_steps    : number of PGD iterations.
    bounds     : domain constraint bounds from :class:`~netadv.constraints.ConstraintBounds`.
    device     : torch device; inferred from model parameters if None.
    random_init: start from a random point inside the epsilon-ball (RS-PGD).
    batch_size : number of samples per forward/backward pass.

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
        X_orig = torch.tensor(X[i:i + batch_size], dtype=torch.float32).to(device)
        y_b    = torch.tensor(y[i:i + batch_size], dtype=torch.float32).to(device)

        if random_init:
            delta = torch.zeros_like(X_orig).uniform_(-epsilon, epsilon)
            x_adv = project(X_orig + delta, bounds)
        else:
            x_adv = X_orig.clone()

        for _ in range(n_steps):
            x_adv = x_adv.detach().requires_grad_(True)
            loss = criterion(model(x_adv), y_b)
            loss.backward()

            with torch.no_grad():
                x_adv = x_adv + alpha * x_adv.grad.sign()
                # Stay inside the epsilon-ball of the original point…
                delta = torch.clamp(x_adv - X_orig, -epsilon, epsilon)
                # …and inside the domain-validity region
                x_adv = project(X_orig + delta, bounds)

        results.append(x_adv.detach().cpu().numpy())

    return np.concatenate(results, axis=0)
