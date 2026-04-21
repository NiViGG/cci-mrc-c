import torch


def compute_cov(x: torch.Tensor, jitter: float = 0.0, symmetrize: bool = True) -> torch.Tensor:
    """
    Compute sample covariance with optional diagonal jitter.
    """
    if x.ndim == 1:
        x = x.unsqueeze(1)
    if x.ndim != 2:
        raise ValueError("x must be 1D or 2D tensor.")
    x = x - x.mean(dim=0, keepdim=True)
    if x.size(0) < 2:
        raise ValueError("Need at least two samples to compute covariance.")
    cov = (x.T @ x) / (x.size(0) - 1)
    if symmetrize:
        cov = 0.5 * (cov + cov.T)
    if jitter < 0:
        raise ValueError("jitter must be non-negative.")
    eye = torch.eye(cov.size(0), device=cov.device, dtype=cov.dtype)
    return cov + eye * jitter
