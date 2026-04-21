import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from metrics.cci import CCIEstimator
from models.mrc_lstm import MRCLSTM
from models.reactive import ReactiveModel


def _single_run(mode: str, seed: int, seq_len: int = 1000, env_dim: int = 5, hidden_dim: int = 64):
    torch.manual_seed(seed)
    np.random.seed(seed)

    env = torch.randn(seq_len, env_dim)
    cci = CCIEstimator()

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
    return cci.compute(H_t, H_t1, E_t)


def run():
    seeds = [42, 43, 44, 45, 46]
    ff_vals = [_single_run("reactive", seed) for seed in seeds]
    rnn_vals = [_single_run("recurrent", seed) for seed in seeds]

    result = {
        "experiment": "reactive_vs_recurrent",
        "runs": len(seeds),
        "seeds": seeds,
        "results": {
            "feedforward_baseline": {
                "mean": float(np.mean(ff_vals)),
                "std": float(np.std(ff_vals)),
            },
            "mrc_lstm_core": {
                "mean": float(np.mean(rnn_vals)),
                "std": float(np.std(rnn_vals)),
            },
        },
        "unit": "bits (shannon)",
    }

    Path("results").mkdir(exist_ok=True)
    with open("results/cci_values.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print("Reactive mean:", result["results"]["feedforward_baseline"]["mean"])
    print("Recurrent mean:", result["results"]["mrc_lstm_core"]["mean"])


if __name__ == "__main__":
    run()
