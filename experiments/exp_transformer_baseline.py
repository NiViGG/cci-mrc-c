"""
experiments/exp_transformer_baseline.py
----------------------------------------
Experiment 4: Transformer Baseline CCI.

Evaluates CCI for a causal Transformer to understand how self-attention
compares to explicit recurrence in terms of internal temporal dependence.
"""

import numpy as np
import torch

from metrics.cci import compute_cci
from models.mrc_c import TransformerNet, RecurrentNet

SEED = 13
SEQ_LEN = 200
BATCH = 1
INPUT_DIM = 16
HIDDEN_DIM = 64
OUTPUT_DIM = 16
NUM_HEADS = 4
NUM_LAYERS = 2


def run() -> dict[str, float]:
    rng = np.random.default_rng(SEED)
    torch.manual_seed(SEED)

    inputs_np = rng.standard_normal((BATCH, SEQ_LEN, INPUT_DIM)).astype(np.float32)
    inputs = torch.from_numpy(inputs_np)
    ext_np = inputs_np[0]

    transformer = TransformerNet(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM, NUM_HEADS, NUM_LAYERS)
    rnn = RecurrentNet(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM)
    transformer.eval()
    rnn.eval()

    with torch.no_grad():
        _, h_transformer = transformer(inputs)
        _, h_rnn = rnn(inputs)

    cci_transformer = compute_cci(h_transformer[0].numpy(), ext_np)
    cci_rnn = compute_cci(h_rnn[0].numpy(), ext_np)

    print(f"[Exp 4] Transformer CCI : {cci_transformer:.6f}")
    print(f"[Exp 4] RNN         CCI : {cci_rnn:.6f}")

    return {"transformer_cci": cci_transformer, "rnn_cci": cci_rnn}


if __name__ == "__main__":
    run()
