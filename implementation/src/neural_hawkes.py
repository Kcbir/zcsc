"""
Continuous-time Neural Hawkes Process with c-LSTM cells.

Each ticker node maintains a hidden state h(t) that:
  - decays exponentially between events
  - updates discontinuously when a relevant event arrives
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from src.config import N_TICKERS, EMBED_DIM, HIDDEN_DIM, LATENT_DIM, DECAY_INIT


class ContinuousTimeLSTMCell(nn.Module):
    """
    c-LSTM cell (Mei & Eisner, 2017).
    Between events the cell state decays toward a steady state c_bar.
    """
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        combined = input_dim + hidden_dim

        self.W_i = nn.Linear(combined, hidden_dim)
        self.W_f = nn.Linear(combined, hidden_dim)
        self.W_z = nn.Linear(combined, hidden_dim)
        self.W_o = nn.Linear(combined, hidden_dim)
        self.W_d = nn.Linear(combined, hidden_dim)

        self.c_bar_layer = nn.Linear(combined, hidden_dim)

    def forward(self, x, h_prev, c_prev, c_bar_prev, dt):
        """All inputs: (1, dim). Returns new (h, c, c_bar) each (1, dim)."""
        decay = F.softplus(self.W_d(torch.cat([x, h_prev], dim=-1)))
        c_decayed = c_bar_prev + (c_prev - c_bar_prev) * torch.exp(-decay * dt)
        h_decayed = torch.tanh(c_decayed) * torch.sigmoid(
            self.W_o(torch.cat([x, h_prev], dim=-1))
        )

        combined = torch.cat([x, h_decayed], dim=-1)
        i_gate = torch.sigmoid(self.W_i(combined))
        f_gate = torch.sigmoid(self.W_f(combined))
        z_gate = torch.tanh(self.W_z(combined))
        o_gate = torch.sigmoid(self.W_o(combined))

        c_new = f_gate * c_decayed + i_gate * z_gate
        c_bar_new = torch.tanh(self.c_bar_layer(combined))
        h_new = o_gate * torch.tanh(c_new)

        return h_new, c_new, c_bar_new


class BilinearAttention(nn.Module):
    """
    alpha = w^T * tanh(W_q h_i + W_k h_j + W_v v)
    """
    def __init__(self, hidden_dim: int, embed_dim: int, latent_dim: int):
        super().__init__()
        self.W_q = nn.Linear(hidden_dim, latent_dim, bias=False)
        self.W_k = nn.Linear(hidden_dim, latent_dim, bias=False)
        self.W_v = nn.Linear(embed_dim, latent_dim, bias=False)
        self.w = nn.Linear(latent_dim, 1, bias=False)

    def forward(self, h_i, h_j, v):
        """All (1, dim). Returns (1,1) non-negative scalar."""
        proj = torch.tanh(self.W_q(h_i) + self.W_k(h_j) + self.W_v(v))
        return F.softplus(self.w(proj))


class NeuralHawkesProcess(nn.Module):
    def __init__(
        self,
        n_nodes: int = N_TICKERS,
        embed_dim: int = EMBED_DIM,
        hidden_dim: int = HIDDEN_DIM,
        latent_dim: int = LATENT_DIM,
    ):
        super().__init__()
        self.n_nodes = n_nodes
        self.hidden_dim = hidden_dim
        self.embed_dim = embed_dim

        self.cell = ContinuousTimeLSTMCell(embed_dim, hidden_dim)
        self.attention = BilinearAttention(hidden_dim, embed_dim, latent_dim)

        self.mu = nn.Parameter(torch.rand(n_nodes) * 0.01)
        self.delta = nn.Parameter(torch.ones(n_nodes) * DECAY_INIT)
        self.embed_proj = nn.Linear(embed_dim, embed_dim)

        # Learnable adjacency mixture weights (co-mention, semantic, correlation)
        # Initialised so that softmax ≈ (0.40, 0.35, 0.25)
        self.adj_logits = nn.Parameter(
            torch.log(torch.tensor([0.40, 0.35, 0.25]))
        )

    def init_states(self, device: torch.device):
        """Return lists of per-node tensors (avoids in-place mutation issues)."""
        h = [torch.zeros(1, self.hidden_dim, device=device) for _ in range(self.n_nodes)]
        c = [torch.zeros(1, self.hidden_dim, device=device) for _ in range(self.n_nodes)]
        c_bar = [torch.zeros(1, self.hidden_dim, device=device) for _ in range(self.n_nodes)]
        return h, c, c_bar
