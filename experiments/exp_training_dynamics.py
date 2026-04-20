"""
experiments/exp_training_dynamics.py
--------------------------------------
Experiment 2: CCI Dynamics During Training.

Trains a RecurrentNet on a sequence-copying task and records CCI at
regular checkpoints.  Demonstrates that CCI grows as the model learns
to maintain internal state for future predictions.
"""

import numpy as np
import torch
import torch.nn as nn

from metrics.cci import compute_cci
from models.mrc_c import RecurrentNet

SEED = 0
SEQ_LEN = 50
BATCH = 32
INPUT_DIM = 8
HIDDEN_DIM = 32
OUTPUT_DIM = 8
NUM_EPOCHS = 60
LR = 1e-3
EVAL_EVERY = 10
EVAL_SEQ_LEN = 200


def make_copy_batch(
    batch_size: int,
    seq_len: int,
    dim: int,
    rng: np.random.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate a copying task: target = input shifted one step."""
    x = rng.standard_normal((batch_size, seq_len, dim)).astype(np.float32)
    y = np.roll(x, shift=-1, axis=1)
    y[:, -1, :] = 0.0
    return torch.from_numpy(x), torch.from_numpy(y)


def run() -> list[dict]:
    rng = np.random.default_rng(SEED)
    torch.manual_seed(SEED)

    model = RecurrentNet(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()

    # Fixed evaluation trajectory
    x_eval_np = rng.standard_normal((1, EVAL_SEQ_LEN, INPUT_DIM)).astype(np.float32)
    x_eval = torch.from_numpy(x_eval_np)
    ext_eval = x_eval_np[0]

    records: list[dict] = []

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        x_tr, y_tr = make_copy_batch(BATCH, SEQ_LEN, INPUT_DIM, rng)
        optimizer.zero_grad()
        output, _ = model(x_tr)
        loss = criterion(output, y_tr)
        loss.backward()
        optimizer.step()

        if epoch % EVAL_EVERY == 0 or epoch == 1:
            model.eval()
            with torch.no_grad():
                _, hidden = model(x_eval)
            h_np = hidden[0].numpy()
            cci_val = compute_cci(h_np, ext_eval)
            record = {"epoch": epoch, "loss": loss.item(), "cci": cci_val}
            records.append(record)
            print(
                f"[Exp 2] Epoch {epoch:3d}  loss={loss.item():.4f}  "
                f"CCI={cci_val:.6f}"
            )

    return records


if __name__ == "__main__":
    run()
