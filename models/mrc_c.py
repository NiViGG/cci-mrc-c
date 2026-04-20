"""
models/mrc_c.py
---------------
MRC-C: Minimal Recurrent Core — Causal

Provides two model classes used across CCI experiments:
  - FeedforwardNet  : a standard MLP with no recurrent connections (CCI ≈ 0)
  - RecurrentNet    : an Elman RNN that develops non-trivial CCI during training
  - TransformerNet  : a causal (autoregressive) Transformer encoder baseline
"""

import torch
import torch.nn as nn


class FeedforwardNet(nn.Module):
    """
    A simple feedforward (MLP) network.

    Hidden-state at each step is computed purely from the current input;
    there is no recurrent connection, so CCI = 0 by construction.

    Parameters
    ----------
    input_dim  : int
    hidden_dim : int
    output_dim : int
    num_layers : int
    """

    def __init__(
        self,
        input_dim: int = 16,
        hidden_dim: int = 64,
        output_dim: int = 16,
        num_layers: int = 2,
    ) -> None:
        super().__init__()
        layers = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]
        for _ in range(num_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.net = nn.Sequential(*layers)
        self._hidden_dim = hidden_dim

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        x : torch.Tensor, shape (batch, seq_len, input_dim)

        Returns
        -------
        output : torch.Tensor, shape (batch, seq_len, output_dim)
        hidden : torch.Tensor, shape (batch, seq_len, hidden_dim)
            Intermediate representations from the penultimate layer.
        """
        B, T, D = x.shape
        x_flat = x.reshape(B * T, D)
        hidden = torch.relu(
            nn.functional.linear(
                x_flat,
                self.net[0].weight,
                self.net[0].bias,
            )
        )
        out = self.net(x_flat)
        return out.reshape(B, T, -1), hidden.reshape(B, T, -1)


class RecurrentNet(nn.Module):
    """
    A single-layer Elman RNN.

    Recurrent connections allow the hidden state at step t to influence the
    hidden state at step t+1, enabling non-zero CCI.

    Parameters
    ----------
    input_dim  : int
    hidden_dim : int
    output_dim : int
    """

    def __init__(
        self,
        input_dim: int = 16,
        hidden_dim: int = 64,
        output_dim: int = 16,
    ) -> None:
        super().__init__()
        self.rnn = nn.RNN(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            nonlinearity="tanh",
        )
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        self._hidden_dim = hidden_dim

    def forward(
        self,
        x: torch.Tensor,
        h0: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        x  : torch.Tensor, shape (batch, seq_len, input_dim)
        h0 : torch.Tensor or None, shape (1, batch, hidden_dim)

        Returns
        -------
        output : torch.Tensor, shape (batch, seq_len, output_dim)
        hidden : torch.Tensor, shape (batch, seq_len, hidden_dim)
        """
        hidden_seq, _ = self.rnn(x, h0)     # (batch, seq_len, hidden_dim)
        output = self.output_layer(hidden_seq)
        return output, hidden_seq


class TransformerNet(nn.Module):
    """
    A causal (autoregressive) Transformer encoder.

    Uses a causal attention mask so that position t can only attend to
    positions 0 … t.  Serves as a baseline for CCI comparison.

    Parameters
    ----------
    input_dim  : int
    hidden_dim : int
    output_dim : int
    num_heads  : int
    num_layers : int
    """

    def __init__(
        self,
        input_dim: int = 16,
        hidden_dim: int = 64,
        output_dim: int = 16,
        num_heads: int = 4,
        num_layers: int = 2,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        x : torch.Tensor, shape (batch, seq_len, input_dim)

        Returns
        -------
        output : torch.Tensor, shape (batch, seq_len, output_dim)
        hidden : torch.Tensor, shape (batch, seq_len, hidden_dim)
        """
        T = x.shape[1]
        causal_mask = nn.Transformer.generate_square_subsequent_mask(T, device=x.device)
        h = self.input_proj(x)
        hidden = self.transformer(h, mask=causal_mask, is_causal=True)
        output = self.output_layer(hidden)
        return output, hidden
