import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from metrics.cci import CCIEstimator
from models.baseline import FeedForwardBaseline
from models.mrc_c import MRCCore


def _estimate_with_bias_control(
    h_t: torch.Tensor,
    h_t1: torch.Tensor,
    e_t: torch.Tensor,
    seed: int,
    n_perm: int = 20,
) -> dict:
    estimator = CCIEstimator()
    raw = estimator.compute(h_t, h_t1, e_t)
    used_jitter = estimator.last_jitter

    g = torch.Generator()
    g.manual_seed(seed + 10_000)
    null_values = []
    for _ in range(n_perm):
        idx = torch.randperm(h_t1.size(0), generator=g)
        null_values.append(estimator.compute(h_t, h_t1[idx], e_t))

    null_mean = float(np.mean(null_values))
    null_std = float(np.std(null_values))
    corrected = max(raw - null_mean, 0.0)
    return {
        "raw": float(raw),
        "null_mean": null_mean,
        "null_std": null_std,
        "bias_corrected": float(corrected),
        "used_jitter": used_jitter,
    }


def _run_feedforward(
    env: torch.Tensor, seed: int, hidden_dim: int = 64, n_perm: int = 20
) -> dict:
    model = FeedForwardBaseline(env_dim=env.size(1), hidden_dim=hidden_dim)
    states = []
    for t in range(env.size(0) - 1):
        h_t, _ = model(env[t].unsqueeze(0))
        states.append(h_t.detach())
    h = torch.cat(states, dim=0)
    return _estimate_with_bias_control(h[:-1], h[1:], env[:-2], seed=seed, n_perm=n_perm)


def _run_recurrent(
    env: torch.Tensor, seed: int, hidden_dim: int = 64, n_perm: int = 20
) -> dict:
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
    return _estimate_with_bias_control(h[:-1], h[1:], env[:-2], seed=seed, n_perm=n_perm)


def run(
    seeds: list[int] | None = None, seq_len: int = 700, env_dim: int = 5, n_perm: int = 20
) -> dict:
    seeds = seeds or [42, 43, 44, 45, 46]
    ff_runs = []
    rnn_runs = []

    for seed in seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)
        env = torch.randn(seq_len, env_dim)
        ff_runs.append(_run_feedforward(env, seed=seed, n_perm=n_perm))
        rnn_runs.append(_run_recurrent(env, seed=seed, n_perm=n_perm))

    ff_corrected = [r["bias_corrected"] for r in ff_runs]
    rnn_corrected = [r["bias_corrected"] for r in rnn_runs]
    ff_raw = [r["raw"] for r in ff_runs]
    rnn_raw = [r["raw"] for r in rnn_runs]

    payload = {
        "experiment": "feedforward_vs_recurrent",
        "runs": len(seeds),
        "seeds": seeds,
        "method": {
            "primary_value": "bias_corrected",
            "correction": "raw_cmi - permutation_null_mean (floored at 0)",
            "n_permutations": n_perm,
        },
        "results": {
            "feedforward_baseline": {
                "mean": float(np.mean(ff_corrected)),
                "std": float(np.std(ff_corrected)),
                "raw_mean": float(np.mean(ff_raw)),
                "raw_std": float(np.std(ff_raw)),
                "per_run": ff_runs,
            },
            "mrc_c_core": {
                "mean": float(np.mean(rnn_corrected)),
                "std": float(np.std(rnn_corrected)),
                "raw_mean": float(np.mean(rnn_raw)),
                "raw_std": float(np.std(rnn_raw)),
                "per_run": rnn_runs,
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
    print("Feedforward corrected mean:", out["results"]["feedforward_baseline"]["mean"])
    print("Recurrent corrected mean:", out["results"]["mrc_c_core"]["mean"])
