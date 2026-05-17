"""Reference PPO implementation."""

from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal

from algorithms.base import HomeworkAgent


def _mlp(in_dim: int, out_dim: int, hidden_dim: int = 256) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, out_dim),
    )


class PPOAgent(HomeworkAgent):
    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        lr: float = 3e-4,
        clip_range: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
    ):
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.clip_range = clip_range
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.actor = _mlp(observation_dim, action_dim).to(self.device)
        self.critic = _mlp(observation_dim, 1).to(self.device)
        self.log_std = nn.Parameter(torch.zeros(action_dim, device=self.device))
        self.optimizer = torch.optim.Adam(
            list(self.actor.parameters())
            + list(self.critic.parameters())
            + [self.log_std],
            lr=lr,
        )

    def _distribution(self, obs_t: torch.Tensor) -> Normal:
        mu = self.actor(obs_t)
        std = torch.exp(torch.clamp(self.log_std, -5.0, 1.0)).expand_as(mu)
        return Normal(mu, std)

    def _tensor(self, arr: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(arr, dtype=torch.float32, device=self.device)

    def _value(self, obs_t: torch.Tensor) -> torch.Tensor:
        return self.critic(obs_t).squeeze(-1)

    @torch.no_grad()
    def act(self, observation: np.ndarray, deterministic: bool = True) -> np.ndarray:
        obs_t = self._tensor(observation).unsqueeze(0)
        dist = self._distribution(obs_t)
        action = dist.mean if deterministic else dist.sample()
        return (
            torch.clamp(action, -1.0, 1.0).squeeze(0).cpu().numpy().astype(np.float32)
        )

    @torch.no_grad()
    def sample_action(self, observation: np.ndarray) -> Tuple[np.ndarray, float, float]:
        # TODO: Implement action sampling that returns the action, log probability, and value estimate for the given observation.
        # --- Student Implementation Start ---
        # Implement action sampling logic and value estimation here
        # --- Student Implementation End ---
        # return (
        #     action.squeeze(0).cpu().numpy().astype(np.float32),
        #     float(log_prob.item()),
        #     float(value.item()),
        # )
        raise NotImplementedError("Action sampling not implemented yet.")

    @torch.no_grad()
    def value(self, observation: np.ndarray) -> float:
        obs_t = self._tensor(observation).unsqueeze(0)
        return float(self._value(obs_t).item())

    def evaluate_actions(
        self, obs: np.ndarray, actions: np.ndarray
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        obs_t = self._tensor(obs)
        actions_t = self._tensor(actions)
        dist = self._distribution(obs_t)
        log_prob = dist.log_prob(actions_t).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        values = self._value(obs_t)
        return log_prob, entropy, values

    def train_step(self, batch: Dict[str, np.ndarray]) -> Dict[str, float]:
        required = ["obs", "actions", "returns", "advantages", "old_log_probs"]
        for key in required:
            if key not in batch:
                raise ValueError(f"Missing key '{key}' for PPO train_step.")

        old_log_probs = self._tensor(batch["old_log_probs"])
        returns_t = self._tensor(batch["returns"])
        advantages_t = self._tensor(batch["advantages"])

        # TODO: Normalize advantages here for training stability.
        # --- Student Implementation Start ---
        # Ensure your advantage matches the typical definitions.
        # Get current log probabilities, entropy, and value estimates for the batch actions and observations.

        # TODO: Compute the policy ratio (pi_theta / pi_theta_old)
        # HINT: You can use exponentiated log probabilities to get the ratio.

        # TODO: Compute the surrogate loss terms

        # TODO: Compute value loss and entropy loss

        # TODO: Combine losses using value_coef and entropy_coef
        # loss = (
        #     policy_loss
        #     + self.value_coef * value_loss
        #     + self.entropy_coef * entropy_loss
        # )  # <-- Verify or adjust this weighting

        # Optimize the combined loss and perform gradient clipping

        # TODO: Return training metrics such as total loss, policy loss, value loss, and entropy for logging purposes.
        # return {
        #     "loss": float(loss.item()),
        #     "policy_loss": float(policy_loss.item()),
        #     "value_loss": float(value_loss.item()),
        #     "entropy": float(entropy.mean().item()),
        # }
        # --- Student Implementation End ---
        raise NotImplementedError("PPO train_step not implemented yet.")

    def save(self, path: str) -> None:
        torch.save(
            {
                "actor": self.actor.state_dict(),
                "critic": self.critic.state_dict(),
                "log_std": self.log_std.detach().cpu(),
                "optimizer": self.optimizer.state_dict(),
            },
            path,
        )

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])
        with torch.no_grad():
            self.log_std.copy_(ckpt["log_std"].to(self.device))
        if "optimizer" in ckpt:
            self.optimizer.load_state_dict(ckpt["optimizer"])
