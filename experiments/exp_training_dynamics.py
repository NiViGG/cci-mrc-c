import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from metrics.cci import CCIEstimator
from models.mrc_c import MRCCore


def run(
    seed: int = 42, seq_len: int = 900, env_dim: int = 5, hidden_dim: int = 64, window: int = 64
) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)
    env = torch.randn(seq_len, env_dim)
    model = MRCCore(env_dim=env_dim, hidden_dim=hidden_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    estimator = CCIEstimator()

    h_t, c_t = model.init_state(batch_size=1, device=env.device)
    history_h = []
    history_e = []
    cci_series = []
    jitter_series = []
    losses = []

    for t in range(seq_len - 1):
        x_t = env[t].unsqueeze(0)
        x_next = env[t + 1].unsqueeze(0)
        h_next, c_next, env_pred, h_recon = model(x_t, h_t, c_t)
        loss = nn.MSELoss()(env_pred, x_next) + 0.05 * nn.MSELoss()(h_recon, h_t)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(float(loss.item()))
        history_h.append(h_next.detach())
        history_e.append(x_t.detach())
        h_t, c_t = h_next.detach(), c_next.detach()

        if len(history_h) >= window:
            h_win = torch.cat(history_h[-window:], dim=0)
            e_win = torch.cat(history_e[-window:], dim=0)
            cci_value = estimator.compute(h_win[:-1], h_win[1:], e_win[:-1])
            cci_series.append(float(cci_value))
            jitter_series.append(estimator.last_jitter)

    payload = {
        "experiment": "training_dynamics",
        "seed": seed,
        "steps": seq_len - 1,
        "window": window,
        "cci_series": cci_series,
        "loss_series": losses,
        "jitter_series": jitter_series,
        "cci_tail_mean": float(np.mean(cci_series[-50:])) if cci_series else 0.0,
        "unit": "bits (shannon)",
    }
    Path("results").mkdir(exist_ok=True)
    with open("results/training_dynamics.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return {"payload": payload, "history_h": history_h}


if __name__ == "__main__":
    output = run()
    print("CCI points:", len(output["payload"]["cci_series"]))
