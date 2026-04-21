import torch.nn as nn


class SimpleTransformer(nn.Module):
    """
    Transformer encoder baseline for temporal dependence comparison.
    """

    def __init__(self, env_dim: int = 5, hidden_dim: int = 64, heads: int = 4):
        super().__init__()
        self.in_proj = nn.Linear(env_dim, hidden_dim)
        self.attn = nn.MultiheadAttention(hidden_dim, heads, batch_first=True, dropout=0.0)
        self.norm = nn.LayerNorm(hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, env_dim)

    def forward(self, x):
        h = self.in_proj(x)
        out, _ = self.attn(h, h, h)
        h_next = self.norm(h + out)
        env_pred = self.out_proj(h_next)
        return h_next, env_pred
