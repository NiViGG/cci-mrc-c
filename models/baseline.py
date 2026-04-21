import torch
import torch.nn as nn


class FeedForwardBaseline(nn.Module):
    """
    Feedforward baseline with no recurrent state.
    """

    def __init__(self, env_dim: int = 5, hidden_dim: int = 64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(env_dim, hidden_dim),
            nn.ReLU(),
        )
        self.predictor = nn.Linear(hidden_dim, env_dim)

    def encode(self, env_t: torch.Tensor) -> torch.Tensor:
        return self.encoder(env_t)

    def forward(self, env_t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h_t = self.encode(env_t)
        env_pred = self.predictor(h_t)
        return h_t, env_pred
