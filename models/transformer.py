import torch.nn as nn


class SimpleTransformer(nn.Module):
    def __init__(self, env_dim: int = 5, hidden_dim: int = 64, heads: int = 4):
        super().__init__()
        self.in_proj = nn.Linear(env_dim, hidden_dim)
        self.attn = nn.MultiheadAttention(hidden_dim, heads, batch_first=True)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        h = self.in_proj(x)
        out, _ = self.attn(h, h, h)
        return self.out_proj(out)
