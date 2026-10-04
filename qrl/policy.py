"""Gate-selection policies: message-passing GNN over the pool-gate graph, and an MLP ablation."""
import torch
from torch import nn


class GatePolicy(nn.Module):
    def __init__(self, n_feat, hidden=64, layers=2, message_passing=True):
        super().__init__()
        self.mp = message_passing
        self.inp = nn.Linear(n_feat, hidden)
        self.self_lin = nn.ModuleList(nn.Linear(hidden, hidden) for _ in range(layers))
        self.nbr_lin = nn.ModuleList(nn.Linear(hidden, hidden) for _ in range(layers))
        self.out = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, x, adj, mask):
        h = torch.relu(self.inp(x))
        deg = adj.sum(-1, keepdim=True).clamp(min=1)
        for ls, ln in zip(self.self_lin, self.nbr_lin):
            m = (adj @ h) / deg if self.mp else torch.zeros_like(h)
            h = torch.relu(ls(h) + ln(m)) + h
        glob = h.mean(0, keepdim=True).expand_as(h)  # graph readout gives every node global context
        logits = self.out(torch.cat([h, glob], -1)).squeeze(-1)
        return logits.masked_fill(~mask, -1e9)
