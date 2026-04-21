import torch

from metrics.cci import CCIEstimator


def test_cci_near_zero_for_independence():
    x = torch.randn(600, 5)
    y = torch.randn(600, 5)
    z = torch.randn(600, 5)
    assert CCIEstimator().compute(x, y, z) < 0.2
