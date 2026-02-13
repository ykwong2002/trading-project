"""
Shared context encoder (optional) + per-asset decision heads for the contextual bandit.
Train with bandit loss (reward-weighted log-likelihood / REINFORCE).
"""
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple

from config import ACTION_SPACE_SIZE


class SharedEncoder(nn.Module):
    """Maps raw context (sentiment + tech) to shared embedding."""

    def __init__(self, input_dim: int, hidden_dim: int = 64, out_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
            nn.ReLU(),
        )
        self.out_dim = out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DecisionHead(nn.Module):
    """Per-asset head: context embedding -> action logits."""

    def __init__(self, input_dim: int, num_actions: int = ACTION_SPACE_SIZE, hidden_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_actions),
        )
        self.num_actions = num_actions

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class RLVRPolicy(nn.Module):
    """
    Modular policy: shared encoder + one decision head per asset.
    Forward(obs, asset_id) -> logits for that asset.
    """

    def __init__(
        self,
        obs_dim: int,
        asset_keys: List[str],
        shared_encoder_dim: int = 32,
        encoder_hidden: int = 64,
        head_hidden: int = 32,
    ):
        super().__init__()
        self.asset_keys = list(asset_keys)
        self.encoder = SharedEncoder(obs_dim, encoder_hidden, shared_encoder_dim)
        self.heads = nn.ModuleDict({
            key: DecisionHead(shared_encoder_dim, ACTION_SPACE_SIZE, head_hidden)
            for key in self.asset_keys
        })

    def forward(self, obs: torch.Tensor, asset_key: str) -> torch.Tensor:
        emb = self.encoder(obs)
        return self.heads[asset_key](emb)

    def get_action(self, obs: np.ndarray, asset_key: str, deterministic: bool = False) -> Tuple[int, np.ndarray]:
        """Returns (action, log_probs or logits)."""
        with torch.no_grad():
            x = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
            logits = self.forward(x, asset_key).squeeze(0)
            probs = torch.softmax(logits, dim=-1)
            if deterministic:
                action = int(probs.argmax().item())
            else:
                action = int(torch.multinomial(probs, 1).item())
        return action, probs.cpu().numpy()


def bandit_loss(
    policy: RLVRPolicy,
    obs: torch.Tensor,
    asset_key: str,
    action: int,
    reward: float,
) -> torch.Tensor:
    """REINFORCE-style bandit loss: -log_prob * reward (baseline 0)."""
    logits = policy(obs, asset_key)
    log_probs = torch.log_softmax(logits, dim=-1)
    log_prob_a = log_probs[:, action]
    return -(log_prob_a * reward).mean()
