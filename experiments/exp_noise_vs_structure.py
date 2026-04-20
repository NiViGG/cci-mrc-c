"""
experiments/exp_noise_vs_structure.py
--------------------------------------
Experiment 3: Noise vs Structure — CCI Under Varying Input Regularity.

Evaluates CCI for a trained RecurrentNet as the input transitions from
fully random (white noise) to highly structured (low-rank sinusoidal).

The hypothesis: CCI is a property of the model's internal dynamics, not
the input statistics.  Under both noise conditions the same trained model
should exhibit similar CCI values.
"""

import numpy as np
import torch

from metrics.cci import compute_cci
from models.mrc_c import RecurrentNet

SEED = 7
SEQ_LEN = 300
INPUT_DIM = 16
HIDDEN_DIM = 64
OUTPUT_DIM = 16


def make_noise_input(
    seq_len: int,
    dim: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Fully random (white noise) input."""
    return rng.standard_normal((1, seq_len, dim)).astype(np.float32)


def make_structured_input(seq_len: int, dim: int) -> np.ndarray:
    """Low-rank sinusoidal input (structured)."""
    t = np.linspace(0, 4 * np.pi, seq_len)
    freqs = np.linspace(0.5, 2.0, dim)
    x = np.sin(np.outer(t, freqs)).astype(np.float32)
    return x[np.newaxis, :, :]   # (1, T, dim)


def run() -> dict[str, float]:
    rng = np.random.default_rng(SEED)
    torch.manual_seed(SEED)

    model = RecurrentNet(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM)
    model.eval()

    noise_input_np = make_noise_input(SEQ_LEN, INPUT_DIM, rng)
    structured_input_np = make_structured_input(SEQ_LEN, INPUT_DIM)

    noise_tensor = torch.from_numpy(noise_input_np)
    structured_tensor = torch.from_numpy(structured_input_np)

    with torch.no_grad():
        _, h_noise = model(noise_tensor)
        _, h_structured = model(structured_tensor)

    cci_noise = compute_cci(h_noise[0].numpy(), noise_input_np[0])
    cci_structured = compute_cci(h_structured[0].numpy(), structured_input_np[0])

    print(f"[Exp 3] CCI (noise input)     : {cci_noise:.6f}")
    print(f"[Exp 3] CCI (structured input): {cci_structured:.6f}")

    return {"cci_noise": cci_noise, "cci_structured": cci_structured}


if __name__ == "__main__":
    run()
