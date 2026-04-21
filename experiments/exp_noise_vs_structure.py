import torch
import json
from pathlib import Path

from metrics.cci import CCIEstimator
from models.mrc_lstm import MRCLSTM


def _cci_on_env(env: torch.Tensor, seed: int) -> dict:
    model = MRCLSTM(env_dim=env.size(1), hidden_dim=64)
    h = torch.zeros(1, 64)
    c = torch.zeros(1, 64)
    states = []
    for t in range(env.size(0) - 1):
        h, c, _, _ = model(env[t].unsqueeze(0), h, c)
        states.append(h.detach())
    h_all = torch.cat(states, dim=0)
    h_t = h_all[:-1]
    h_t1 = h_all[1:]
    e_t = env[:-2]

    cci = CCIEstimator()
    raw = cci.compute(h_t, h_t1, e_t)
    jitter = cci.last_jitter

    g = torch.Generator()
    g.manual_seed(seed + 20_000)
    null_vals = []
    for _ in range(20):
        idx = torch.randperm(h_t1.size(0), generator=g)
        null_vals.append(cci.compute(h_t, h_t1[idx], e_t))
    null_mean = float(sum(null_vals) / len(null_vals))
    corrected = max(raw - null_mean, 0.0)
    return {
        "raw": float(raw),
        "null_mean": null_mean,
        "bias_corrected": float(corrected),
        "used_jitter": jitter,
    }


def run():
    seed = 42
    torch.manual_seed(seed)
    noise_env = torch.randn(1000, 5)
    structured_env = torch.sin(torch.linspace(0, 40, 1000)).unsqueeze(1).repeat(1, 5)

    cci_noise = _cci_on_env(noise_env, seed)
    cci_struct = _cci_on_env(structured_env, seed)
    result = {
        "experiment": "noise_vs_structured",
        "seed": seed,
        "method": {
            "primary_value": "bias_corrected",
            "correction": "raw_cmi - permutation_null_mean (floored at 0)",
            "n_permutations": 20,
        },
        "noise_input": cci_noise,
        "structured_input": cci_struct,
        "unit": "bits (shannon)",
        "interpretation_note": (
            "White-noise forcing can excite broader latent variance and yield larger raw CCI; "
            "compare corrected values and trends, not raw magnitudes alone."
        ),
    }

    Path("results").mkdir(exist_ok=True)
    with open("results/noise_vs_structure.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print("Noise corrected:", cci_noise["bias_corrected"])
    print("Structured corrected:", cci_struct["bias_corrected"])


if __name__ == "__main__":
    run()
