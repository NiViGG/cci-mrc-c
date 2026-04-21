import torch

from metrics.cci import CCIEstimator
from models.baseline import FeedForwardBaseline
from models.mrc_c import MRCCore
from models.transformer import SimpleTransformer


def test_shape_mismatch_raises():
    h_t = torch.randn(100, 8)
    h_t1 = torch.randn(90, 8)
    e_t = torch.randn(100, 4)
    try:
        CCIEstimator().compute(h_t, h_t1, e_t)
        assert False, "Expected ValueError on mismatched sample counts"
    except ValueError:
        assert True


def test_model_output_shapes_match_contract():
    env_t = torch.randn(2, 5)
    ff = FeedForwardBaseline(env_dim=5, hidden_dim=32)
    h_ff, pred_ff = ff(env_t)
    assert h_ff.shape == (2, 32)
    assert pred_ff.shape == (2, 5)

    core = MRCCore(env_dim=5, hidden_dim=32)
    h0, c0 = core.init_state(batch_size=2)
    h1, c1, pred_core, recon = core(env_t, h0, c0)
    assert h1.shape == (2, 32)
    assert c1.shape == (2, 32)
    assert pred_core.shape == (2, 5)
    assert recon.shape == (2, 32)

    tr = SimpleTransformer(env_dim=5, hidden_dim=32, heads=4)
    h_tr, pred_tr = tr(torch.randn(2, 20, 5))
    assert h_tr.shape == (2, 20, 32)
    assert pred_tr.shape == (2, 20, 5)
