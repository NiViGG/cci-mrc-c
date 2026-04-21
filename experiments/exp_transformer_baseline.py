import json
from pathlib import Path

import torch

from metrics.cci import CCIEstimator
from models.transformer import SimpleTransformer
from utils.seed import set_seed


def run(seed: int = 42, seq_len: int = 700, env_dim: int = 5):
    set_seed(seed)
    env = torch.randn(1, seq_len, env_dim)
    model = SimpleTransformer(env_dim=5, hidden_dim=64, heads=4)
    h, _ = model(env)
    h = h.squeeze(0)
    estimator = CCIEstimator()
    value = estimator.compute(h[:-1], h[1:], env.squeeze(0)[:-1])
    payload = {
        "experiment": "transformer_baseline",
        "seed": seed,
        "result": {"cci": float(value), "jitter": estimator.last_jitter},
        "unit": "bits (shannon)",
    }
    Path("results").mkdir(exist_ok=True)
    with open("results/transformer_baseline.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print("Transformer CCI:", value)
    return payload


if __name__ == "__main__":
    run()
