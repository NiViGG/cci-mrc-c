import torch

from metrics.cci import CCIEstimator


def test_cci_positive_dependency():
    torch.manual_seed(11)
    x = torch.randn(600, 5)
    z = torch.randn(600, 5)
    independent = CCIEstimator().compute(x, torch.randn(600, 5), z)
    dependent = CCIEstimator().compute(x, x + 0.08 * torch.randn(600, 5), z)
    assert dependent > independent + 0.2
