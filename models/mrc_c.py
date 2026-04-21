import torch
import torch.nn as nn


class MRCCore(nn.Module):
    """
    MRC-C recurrent core with predictive and self-model heads.
    """

    def __init__(self, env_dim: int = 5, hidden_dim: int = 64, z_dim: int = 16):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.cell = nn.LSTMCell(env_dim, hidden_dim)
        self.env_head = nn.Linear(hidden_dim, env_dim)
        self.self_model = nn.Sequential(
            nn.Linear(hidden_dim, z_dim),
            nn.ReLU(),
            nn.Linear(z_dim, hidden_dim),
        )

    def init_state(self, batch_size: int, device: torch.device | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        h = torch.zeros(batch_size, self.hidden_dim, device=device)
        c = torch.zeros(batch_size, self.hidden_dim, device=device)
        return h, c

    def forward(
        self, env_t: torch.Tensor, h_t: torch.Tensor, c_t: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        h_next, c_next = self.cell(env_t, (h_t, c_t))
        env_pred = self.env_head(h_next)
        h_recon = self.self_model(h_next)
        return h_next, c_next, env_pred, h_recon
