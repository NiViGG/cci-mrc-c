import json
from pathlib import Path

import torch

from metrics.cci import CCIEstimator
from models.mrc_c import MRCCore


def _cci_on_env(env: torch.Tensor) -> tuple[float, float | None]:
    model = MRCCore(env_dim=env.size(1), hidden_dim=64)
    h, c = model.init_state(batch_size=1, device=env.device)
    states = []
    for t in range(env.size(0) - 1):
        h, c, _, _ = model(env[t].unsqueeze(0), h, c)
        states.append(h.detach())
    h_all = torch.cat(states, dim=0)
    estimator = CCIEstimator()
    value = estimator.compute(h_all[:-1], h_all[1:], env[:-2])
    return value, estimator.last_jitter


def run():
    torch.manual_seed(42)
    noise_env = torch.randn(1000, 5)
    structured_env = torch.sin(torch.linspace(0, 40, 1000)).unsqueeze(1).repeat(1, 5)

    cci_noise, jitter_noise = _cci_on_env(noise_env)
    cci_struct, jitter_struct = _cci_on_env(structured_env)
    print("CCI noise:", cci_noise)
    print("CCI structured:", cci_struct)
    payload = {
        "experiment": "noise_vs_structure",
        "results": {
            "noise_input": {"cci": float(cci_noise), "jitter": jitter_noise},
            "structured_input": {"cci": float(cci_struct), "jitter": jitter_struct},
        },
        "unit": "bits (shannon)",
    }
    Path("results").mkdir(exist_ok=True)
    with open("results/noise_vs_structure.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return payload


if __name__ == "__main__":
    run()
