import torch
import json
from pathlib import Path

from metrics.cci import CCIEstimator
from models.transformer import SimpleTransformer


def run():
    seed = 42
    torch.manual_seed(seed)
    env = torch.randn(1, 1000, 5)
    model = SimpleTransformer(env_dim=5, hidden_dim=64, heads=4)
    h, _ = model(env)
    h = h.squeeze(0)
    h_t = h[:-1]
    h_t1 = h[1:]
    e_t = env.squeeze(0)[:-1]

    cci = CCIEstimator()
    raw = cci.compute(h_t, h_t1, e_t)
    jitter = cci.last_jitter

    g = torch.Generator()
    g.manual_seed(seed + 30_000)
    null_vals = []
    for _ in range(20):
        idx = torch.randperm(h_t1.size(0), generator=g)
        null_vals.append(cci.compute(h_t, h_t1[idx], e_t))
    null_mean = float(sum(null_vals) / len(null_vals))
    corrected = max(raw - null_mean, 0.0)

    result = {
        "experiment": "transformer_baseline",
        "seed": seed,
        "method": {
            "primary_value": "bias_corrected",
            "correction": "raw_cmi - permutation_null_mean (floored at 0)",
            "n_permutations": 20,
        },
        "raw": float(raw),
        "null_mean": null_mean,
        "bias_corrected": float(corrected),
        "used_jitter": jitter,
        "interpretation_note": (
            "A non-zero CCI value in attention models does not imply autonomy; "
            "interpret as conditional dependence magnitude under this benchmark."
        ),
    }
    Path("results").mkdir(exist_ok=True)
    with open("results/transformer_baseline.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print("Transformer corrected CCI:", corrected)


if __name__ == "__main__":
    run()
