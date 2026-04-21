import torch

from metrics.cci import CCIEstimator


def test_cci_near_zero_for_independence():
    torch.manual_seed(7)
    x = torch.randn(800, 5)
    y = torch.randn(800, 5)
    z = torch.randn(800, 5)
    value = CCIEstimator().compute(x, y, z)
    assert 0.0 <= value < 0.15
