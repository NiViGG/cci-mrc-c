import json
from pathlib import Path

import torch

from metrics.cci import CCIEstimator
from models.mrc_c import MRCCore
from utils.seed import set_seed


def run(seed: int = 42, seq_len: int = 1000, env_dim: int = 5) -> dict:
    set_seed(seed)
    env = torch.randn(seq_len, env_dim)
    model = MRCCore(env_dim=env_dim, hidden_dim=64)
    h_t, c_t = model.init_state(batch_size=1, device=env.device)
    states = []

    for t in range(seq_len - 1):
        h_t, c_t, _, _ = model(env[t].unsqueeze(0), h_t, c_t)
        states.append(h_t.detach())

    h = torch.cat(states, dim=0)
    estimator = CCIEstimator()
    value = estimator.compute(h[:-1], h[1:], env[:-2])
    payload = {
        "experiment": "noise_only",
        "seed": seed,
        "results": {"cci": float(value), "jitter": estimator.last_jitter},
        "unit": "bits (shannon)",
    }
    Path("results").mkdir(exist_ok=True)
    with open("results/noise_only.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return payload


if __name__ == "__main__":
    out = run()
    print("CCI under white noise:", out["results"]["cci"])
