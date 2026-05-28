import torch
from ..constraints.bounds import ConstraintBounds


def project(x: torch.Tensor, bounds: ConstraintBounds) -> torch.Tensor:
    """Clip x to the domain-valid bounds (in the preprocessed feature space)."""
    lb = torch.tensor(bounds.lb, dtype=x.dtype, device=x.device)
    ub = torch.tensor(bounds.ub, dtype=x.dtype, device=x.device)
    return torch.clamp(x, lb, ub)
