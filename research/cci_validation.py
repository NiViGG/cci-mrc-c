import json
from datetime import datetime, timezone
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.engine import CCIEngine
from core.models import MRCCore
from data.sample_eeg import generate_synthetic_eeg


def evaluate_mode(mode: str, seeds: list[int], seq_len: int = 512, channels: int = 16) -> dict:
    values = []
    jitters = []
    for seed in seeds:
        eeg = generate_synthetic_eeg(mode=mode, seq_len=seq_len, channels=channels, seed=seed)
        eeg_t = torch.tensor(eeg, dtype=torch.float32)
        model = MRCCore(input_dim=channels, hidden_dim=64, z_dim=16)
        hidden, _, _ = model.forward_sequence(eeg_t)
        engine = CCIEngine()
        result = engine.process_signal(hidden, eeg_t)
        values.append(result["cci_bits"])
        jitters.append(result["used_jitter"])

    return {
        "mode": mode,
        "seeds": seeds,
        "per_run": values,
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "jitters": jitters,
        "unit": "bits (shannon)",
    }


def main():
    seeds = [42, 43, 44, 45, 46]
    report = {
        "experiment": "eeg_extension_validation",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "notes": "Exploratory diagnostic dependence analysis. Not a consciousness proof.",
        "results": {
            "conscious-like": evaluate_mode("conscious-like", seeds),
            "anesthesia-like": evaluate_mode("anesthesia-like", seeds),
            "deep-noise": evaluate_mode("deep-noise", seeds),
        },
    }
    Path("results").mkdir(exist_ok=True)
    out = Path("results/eeg_validation.json")
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("EEG validation complete.")
    for mode, payload in report["results"].items():
        print(f"{mode}: mean={payload['mean']:.4f} std={payload['std']:.4f}")
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
