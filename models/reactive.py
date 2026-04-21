import torch.nn as nn


class ReactiveModel(nn.Module):
    def __init__(self, env_dim: int = 5, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(env_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, env_dim),
        )

    def forward(self, x):
        return self.net(x)
