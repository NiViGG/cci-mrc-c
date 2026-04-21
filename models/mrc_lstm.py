import torch.nn as nn


class MRCLSTM(nn.Module):
    """
    Modular Recurrent Core.
    """

    def __init__(self, env_dim: int = 5, hidden_dim: int = 64, z_dim: int = 16):
        super().__init__()
        self.lstm = nn.LSTMCell(env_dim, hidden_dim)
        self.env_head = nn.Linear(hidden_dim, env_dim)
        self.m_bottleneck = nn.Sequential(
            nn.Linear(hidden_dim, z_dim),
            nn.ReLU(),
            nn.Linear(z_dim, hidden_dim),
        )

    def forward(self, env_t, h, c):
        h_next, c_next = self.lstm(env_t, (h, c))
        env_pred = self.env_head(h_next)
        h_recon = self.m_bottleneck(h_next)
        return h_next, c_next, env_pred, h_recon
