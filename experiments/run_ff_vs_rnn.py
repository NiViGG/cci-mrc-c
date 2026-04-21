import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from metrics.cci import CCIEstimator
from models.mrc_lstm import MRCLSTM
from models.reactive import ReactiveModel


def _estimate_with_bias_control(
    h_t: torch.Tensor,
    h_t1: torch.Tensor,
    e_t: torch.Tensor,
    seed: int,
    n_perm: int = 20,
):
    cci = CCIEstimator()
    raw = cci.compute(h_t, h_t1, e_t)
    used_jitter = cci.last_jitter

    g = torch.Generator()
    g.manual_seed(seed + 10_000)
    null_values = []
    for _ in range(n_perm):
        idx = torch.randperm(h_t1.size(0), generator=g)
        null_values.append(cci.compute(h_t, h_t1[idx], e_t))

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


def _single_run(
    mode: str, seed: int, seq_len: int = 1000, env_dim: int = 5, hidden_dim: int = 64
):
    torch.manual_seed(seed)
    np.random.seed(seed)

    env = torch.randn(seq_len, env_dim)

    if mode == "reactive":
        model = ReactiveModel(env_dim, hidden_dim)
        h_series = []
        for t in range(seq_len - 1):
            x_t = env[t].unsqueeze(0)
            h_series.append(model(x_t).detach())
    else:
        model = MRCLSTM(env_dim, hidden_dim)
        h = torch.zeros(1, hidden_dim)
        c = torch.zeros(1, hidden_dim)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
        h_series = []
        for t in range(seq_len - 1):
            x_t = env[t].unsqueeze(0)
            x_next = env[t + 1].unsqueeze(0)
            h_next, c_next, pred, h_recon = model(x_t, h, c)
            loss = nn.MSELoss()(pred, x_next) + 0.05 * nn.MSELoss()(h_recon, h)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            h_series.append(h_next.detach())
            h, c = h_next.detach(), c_next.detach()

    H = torch.cat(h_series, dim=0)
    H_t = H[:-1]
    H_t1 = H[1:]
    E_t = env[:-2]
    return _estimate_with_bias_control(H_t, H_t1, E_t, seed=seed)


def run():
    seeds = [42, 43, 44, 45, 46]
    ff_runs = [_single_run("reactive", seed) for seed in seeds]
    rnn_runs = [_single_run("recurrent", seed) for seed in seeds]
    ff_corrected = [r["bias_corrected"] for r in ff_runs]
    rnn_corrected = [r["bias_corrected"] for r in rnn_runs]
    ff_raw = [r["raw"] for r in ff_runs]
    rnn_raw = [r["raw"] for r in rnn_runs]

    result = {
        "experiment": "reactive_vs_recurrent",
        "runs": len(seeds),
        "seeds": seeds,
        "method": {
            "primary_value": "bias_corrected",
            "correction": "raw_cmi - permutation_null_mean (floored at 0)",
            "n_permutations": 20,
        },
        "results": {
            "feedforward_baseline": {
                "mean": float(np.mean(ff_corrected)),
                "std": float(np.std(ff_corrected)),
                "raw_mean": float(np.mean(ff_raw)),
                "raw_std": float(np.std(ff_raw)),
                "per_run": ff_runs,
            },
            "mrc_lstm_core": {
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
        json.dump(result, f, indent=2)

    print("Reactive corrected mean:", result["results"]["feedforward_baseline"]["mean"])
    print("Recurrent corrected mean:", result["results"]["mrc_lstm_core"]["mean"])


if __name__ == "__main__":
    run()
