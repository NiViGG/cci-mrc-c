import torch
import torch.nn as nn


class MRCCore(nn.Module):
    """
    EEG-oriented recurrent core with predictive and self-modeling heads.
    """

    def __init__(self, input_dim: int = 16, hidden_dim: int = 64, z_dim: int = 16):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.cell = nn.LSTMCell(input_dim, hidden_dim)
        self.env_head = nn.Linear(hidden_dim, input_dim)
        self.self_head = nn.Sequential(
            nn.Linear(hidden_dim, z_dim),
            nn.ReLU(),
            nn.Linear(z_dim, hidden_dim),
        )

    def init_state(self, batch_size: int, device: torch.device | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        h = torch.zeros(batch_size, self.hidden_dim, device=device)
        c = torch.zeros(batch_size, self.hidden_dim, device=device)
        return h, c

    def forward_step(
        self, x_t: torch.Tensor, h_t: torch.Tensor, c_t: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        h_next, c_next = self.cell(x_t, (h_t, c_t))
        env_pred = self.env_head(h_next)
        h_recon = self.self_head(h_next)
        return h_next, c_next, env_pred, h_recon

    def forward_sequence(
        self, sequence: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            sequence: [time, channels]
        Returns:
            hidden_states: [time-1, hidden_dim]
            env_predictions: [time-1, channels]
            self_reconstructions: [time-1, hidden_dim]
        """
        if sequence.ndim != 2:
            raise ValueError("sequence must be 2D tensor [time, channels].")
        if sequence.size(0) < 2:
            raise ValueError("sequence must contain at least 2 time steps.")

        h_t, c_t = self.init_state(batch_size=1, device=sequence.device)
        h_hist = []
        env_hist = []
        recon_hist = []
        for t in range(sequence.size(0) - 1):
            x_t = sequence[t].unsqueeze(0)
            h_next, c_next, env_pred, h_recon = self.forward_step(x_t, h_t, c_t)
            h_hist.append(h_next.detach())
            env_hist.append(env_pred.detach())
            recon_hist.append(h_recon.detach())
            h_t, c_t = h_next, c_next

        return (
            torch.cat(h_hist, dim=0),
            torch.cat(env_hist, dim=0),
            torch.cat(recon_hist, dim=0),
        )
