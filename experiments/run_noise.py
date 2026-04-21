import torch

from metrics.cci import CCIEstimator
from models.mrc_lstm import MRCLSTM


def run(seq_len: int = 1000, env_dim: int = 5, hidden_dim: int = 64, seed: int = 42):
    torch.manual_seed(seed)
    env = torch.randn(seq_len, env_dim)
    model = MRCLSTM(env_dim, hidden_dim)

    h = torch.zeros(1, hidden_dim)
    c = torch.zeros(1, hidden_dim)
    states = []

    for t in range(seq_len - 1):
        x_t = env[t].unsqueeze(0)
        h, c, _, _ = model(x_t, h, c)
        states.append(h.detach())

    H = torch.cat(states, dim=0)
    value = CCIEstimator().compute(H[:-1], H[1:], env[:-2])
    print("CCI under white noise:", value)
    return value


if __name__ == "__main__":
    run()
