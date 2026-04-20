"""
experiments/exp_feedforward_vs_rnn.py
--------------------------------------
Experiment 1: Feedforward vs Recurrent CCI Comparison.

Runs a FeedforwardNet and a RecurrentNet on identical random trajectories
and computes CCI for each.  The expected outcome is:
  - FeedforwardNet  → CCI ≈ 0
  - RecurrentNet    → CCI > 0
"""

import numpy as np
import torch

from metrics.cci import compute_cci
from models.mrc_c import FeedforwardNet, RecurrentNet

SEED = 42
SEQ_LEN = 200
BATCH = 1
INPUT_DIM = 16
HIDDEN_DIM = 64
OUTPUT_DIM = 16


def run() -> dict[str, float]:
    rng = np.random.default_rng(SEED)
    torch.manual_seed(SEED)

    # Random input trajectory ─ identical for both models
    inputs_np = rng.standard_normal((BATCH, SEQ_LEN, INPUT_DIM)).astype(np.float32)
    inputs = torch.from_numpy(inputs_np)

    ff_model = FeedforwardNet(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM)
    rnn_model = RecurrentNet(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM)
    ff_model.eval()
    rnn_model.eval()

    with torch.no_grad():
        _, ff_hidden = ff_model(inputs)      # (1, T, hidden_dim)
        _, rnn_hidden = rnn_model(inputs)    # (1, T, hidden_dim)

    ff_h_np = ff_hidden[0].numpy()           # (T, hidden_dim)
    rnn_h_np = rnn_hidden[0].numpy()
    ext_np = inputs_np[0]                    # (T, input_dim)

    cci_ff = compute_cci(ff_h_np, ext_np)
    cci_rnn = compute_cci(rnn_h_np, ext_np)

    print(f"[Exp 1] Feedforward CCI : {cci_ff:.6f}")
    print(f"[Exp 1] Recurrent   CCI : {cci_rnn:.6f}")

    return {"feedforward_cci": cci_ff, "recurrent_cci": cci_rnn}


if __name__ == "__main__":
    run()
