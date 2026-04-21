import json
from pathlib import Path

import numpy as np
import torch

from metrics.cci import CCIEstimator
from models.baseline import FeedForwardBaseline
from utils.seed import set_seed


def run(seeds: list[int] | None = None, seq_len: int = 700, env_dim: int = 5) -> dict:
    seeds = seeds or [42, 43, 44, 45, 46]
    values = []
    jitters = []

    for seed in seeds:
        set_seed(seed)
        env = torch.randn(seq_len, env_dim)
        model = FeedForwardBaseline(env_dim=env_dim, hidden_dim=64)
        states = []
        for t in range(seq_len - 1):
            h_t, _ = model(env[t].unsqueeze(0))
            states.append(h_t.detach())
        h = torch.cat(states, dim=0)
        estimator = CCIEstimator()
        values.append(estimator.compute(h[:-1], h[1:], env[:-2]))
        jitters.append(estimator.last_jitter)

    payload = {
        "experiment": "feedforward_baseline_only",
        "runs": len(seeds),
        "seeds": seeds,
        "results": {
            "feedforward_baseline": {
                "per_run": [float(v) for v in values],
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "jitters": jitters,
            }
        },
        "unit": "bits (shannon)",
    }
    Path("results").mkdir(exist_ok=True)
    with open("results/baseline.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return payload


if __name__ == "__main__":
    result = run()
    print("Baseline mean:", result["results"]["feedforward_baseline"]["mean"])
