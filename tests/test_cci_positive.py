import torch

from metrics.cci import CCIEstimator


def test_cci_positive_dependency():
    x = torch.randn(400, 5)
    y = x + 0.1 * torch.randn(400, 5)
    z = torch.randn(400, 5)
    assert CCIEstimator().compute(x, y, z) > 0.0
