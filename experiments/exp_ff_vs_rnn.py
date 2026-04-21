import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from metrics.cci import CCIEstimator
from models.baseline import FeedForwardBaseline
from models.mrc_c import MRCCore


def _run_feedforward(env: torch.Tensor, hidden_dim: int = 64) -> tuple[float, float | None]:
    model = FeedForwardBaseline(env_dim=env.size(1), hidden_dim=hidden_dim)
    states = []
    for t in range(env.size(0) - 1):
        h_t, _ = model(env[t].unsqueeze(0))
        states.append(h_t.detach())
    h = torch.cat(states, dim=0)
    estimator = CCIEstimator()
    return estimator.compute(h[:-1], h[1:], env[:-2]), estimator.last_jitter


def _run_recurrent(env: torch.Tensor, hidden_dim: int = 64) -> tuple[float, float | None]:
    model = MRCCore(env_dim=env.size(1), hidden_dim=hidden_dim)
    opt = torch.optim.Adam(model.parameters(), lr=0.005)
    h_t, c_t = model.init_state(batch_size=1, device=env.device)
    states = []
    for t in range(env.size(0) - 1):
        x_t = env[t].unsqueeze(0)
        x_next = env[t + 1].unsqueeze(0)
        h_next, c_next, pred, h_recon = model(x_t, h_t, c_t)
        loss = nn.MSELoss()(pred, x_next) + 0.05 * nn.MSELoss()(h_recon, h_t)
        opt.zero_grad()
        loss.backward()
        opt.step()
        states.append(h_next.detach())
        h_t, c_t = h_next.detach(), c_next.detach()
    h = torch.cat(states, dim=0)
    estimator = CCIEstimator()
    return estimator.compute(h[:-1], h[1:], env[:-2]), estimator.last_jitter


def run(seeds: list[int] | None = None, seq_len: int = 700, env_dim: int = 5) -> dict:
    seeds = seeds or [42, 43, 44, 45, 46]
    ff_values = []
    ff_jitters = []
    rnn_values = []
    rnn_jitters = []

    for seed in seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)
        env = torch.randn(seq_len, env_dim)
        ff_val, ff_jitter = _run_feedforward(env)
        rnn_val, rnn_jitter = _run_recurrent(env)
        ff_values.append(float(ff_val))
        ff_jitters.append(ff_jitter)
        rnn_values.append(float(rnn_val))
        rnn_jitters.append(rnn_jitter)

    payload = {
        "experiment": "feedforward_vs_recurrent",
        "runs": len(seeds),
        "seeds": seeds,
        "results": {
            "feedforward_baseline": {
                "per_run": ff_values,
                "mean": float(np.mean(ff_values)),
                "std": float(np.std(ff_values)),
                "jitters": ff_jitters,
            },
            "mrc_c_core": {
                "per_run": rnn_values,
                "mean": float(np.mean(rnn_values)),
                "std": float(np.std(rnn_values)),
                "jitters": rnn_jitters,
            },
        },
        "unit": "bits (shannon)",
    }
    Path("results").mkdir(exist_ok=True)
    with open("results/cci_values.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return payload


if __name__ == "__main__":
    out = run()
    print("Feedforward mean:", out["results"]["feedforward_baseline"]["mean"])
    print("Recurrent mean:", out["results"]["mrc_c_core"]["mean"])
