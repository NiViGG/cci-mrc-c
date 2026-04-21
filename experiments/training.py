import numpy as np
import torch
import torch.nn as nn

from metrics.cci import CCIEstimator
from models.mrc_lstm import MRCLSTM


def run(seq_len: int = 1000, env_dim: int = 5, hidden_dim: int = 64, seed: int = 42):
    torch.manual_seed(seed)
    np.random.seed(seed)

    env = torch.randn(seq_len, env_dim)
    model = MRCLSTM(env_dim, hidden_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    cci = CCIEstimator()

    h = torch.zeros(1, hidden_dim)
    c = torch.zeros(1, hidden_dim)
    history_h = []
    history_e = []
    cci_series = []

    for t in range(seq_len - 1):
        x_t = env[t].unsqueeze(0)
        x_next = env[t + 1].unsqueeze(0)

        h_next, c_next, pred, h_recon = model(x_t, h, c)
        loss = nn.MSELoss()(pred, x_next) + 0.05 * nn.MSELoss()(h_recon, h)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        history_h.append(h_next.detach())
        history_e.append(x_t.detach())
        h, c = h_next.detach(), c_next.detach()

        if len(history_h) > 50:
            H = torch.cat(history_h[-50:], dim=0)
            E = torch.cat(history_e[-50:], dim=0)
            cci_series.append(cci.compute(H[:-1], H[1:], E[:-1]))

    return cci_series, history_h


if __name__ == "__main__":
    values, _ = run()
    print("Final CCI window mean:", float(np.mean(values[-50:])) if values else 0.0)
