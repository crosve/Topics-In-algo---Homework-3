"""Reference SAC implementation."""

from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from algorithms.base import HomeworkAgent

LOG_STD_MIN = -20.0
LOG_STD_MAX = 2.0


def _mlp(in_dim: int, out_dim: int, hidden_dim: int = 256) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, out_dim),
    )


class SquashedGaussianActor(nn.Module):
    def __init__(self, observation_dim: int, action_dim: int):
        super().__init__()
        self.net = _mlp(observation_dim, 2 * action_dim)
        self.action_dim = action_dim

    def forward(self, obs_t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        out = self.net(obs_t)
        mu, log_std = torch.chunk(out, 2, dim=-1)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        return mu, log_std

    # TODO: Implement the sampling function for the Squashed Gaussian policy. Remember to apply the tanh squashing and adjust log probabilities accordingly.
    def sample(
        self, obs_t: torch.Tensor, deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Dim-0 is speed command in [0, 1], others remain in [-1, 1]
        # return action, log_prob
        raise NotImplementedError("Actor sampling not implemented yet.")


class Critic(nn.Module):
    def __init__(self, observation_dim: int, action_dim: int):
        super().__init__()
        self.q1 = _mlp(observation_dim + action_dim, 1)
        self.q2 = _mlp(observation_dim + action_dim, 1)

    def forward(
        self, obs_t: torch.Tensor, act_t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat([obs_t, act_t], dim=-1)
        return self.q1(x).squeeze(-1), self.q2(x).squeeze(-1)


class SACAgent(HomeworkAgent):
    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        actor_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        alpha_lr: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
    ):
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.actor = SquashedGaussianActor(observation_dim, action_dim).to(self.device)
        self.critic = Critic(observation_dim, action_dim).to(self.device)
        self.critic_target = Critic(observation_dim, action_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.log_alpha = torch.tensor(0.0, requires_grad=True, device=self.device)
        self.target_entropy = -float(action_dim)

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=alpha_lr)

    def _tensor(self, arr: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(arr, dtype=torch.float32, device=self.device)

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    @torch.no_grad()
    def act(self, observation: np.ndarray, deterministic: bool = True) -> np.ndarray:
        obs_t = self._tensor(observation).unsqueeze(0)
        action, _ = self.actor.sample(obs_t, deterministic=deterministic)
        return action.squeeze(0).cpu().numpy().astype(np.float32)

    def _soft_update(self):
        # TODO: Implement soft update of target critic parameters using self.tau
        # --- Student Implementation Start ---
        # Implement soft update logic here
        # --- Student Implementation End ---
        raise NotImplementedError("Soft update not implemented yet.")

    def train_step(self, batch: Dict[str, np.ndarray]) -> Dict[str, float]:
        required = ["obs", "actions", "rewards", "next_obs", "dones"]
        for key in required:
            if key not in batch:
                raise ValueError(f"Missing key '{key}' for SAC train_step.")

        obs = self._tensor(batch["obs"])
        actions = self._tensor(batch["actions"])
        rewards = self._tensor(batch["rewards"]).squeeze(-1)
        next_obs = self._tensor(batch["next_obs"])
        dones = self._tensor(batch["dones"]).squeeze(-1)

        # TODO: Compute Target Q-Values
        # HINT: Sample next_actions from the actor for next_obs
        # Calculate target_q = r + gamma * (1 - d) * (min. Q1/Q2_target - alpha * log_prob)
        with torch.no_grad():
            next_actions, next_logp = self.actor.sample(next_obs, deterministic=False)

            # --- Student Implementation Start ---
            # Implement target Q value logic here
            target = torch.zeros_like(rewards)  # Replace with true target calculation
            # --- Student Implementation End ---

        # TODO: Critic Update
        # HINT: Get current Q1 and Q2 estimates arrays from self.critic
        # Compute MSE loss vs your `target`
        # --- Student Implementation Start ---
        q1, q2 = self.critic(obs, actions)
        critic_loss = torch.tensor(
            0.0, requires_grad=True, device=self.device
        )  # Replace with MSE loss
        # --- Student Implementation End ---

        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        # TODO: Actor and Alpha update
        # HINT: Sample new actions from observation obs via self.actor
        # Minimize actor loss defined via the delayed critic Q-networks
        # Calculate actor_loss and alpha_loss

        # --- Student Implementation Start ---
        new_actions, logp = self.actor.sample(obs, deterministic=False)
        q1_pi, q2_pi = self.critic(obs, new_actions)
        q_pi = torch.min(q1_pi, q2_pi)

        actor_loss = torch.tensor(
            0.0, requires_grad=True, device=self.device
        )  # Replace
        # --- Student Implementation End ---

        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        # --- Student Implementation Start ---
        # Update alpha based on self.target_entropy
        alpha_loss = torch.tensor(
            0.0, requires_grad=True, device=self.device
        )  # Replace
        # --- Student Implementation End ---

        self.alpha_opt.zero_grad()
        alpha_loss.backward()
        self.alpha_opt.step()

        # Soft update target critics
        self._soft_update()
        return {
            "critic_loss": float(critic_loss.item()),
            "actor_loss": float(actor_loss.item()),
            "alpha_loss": float(alpha_loss.item()),
            "alpha": float(self.alpha.item()),
            "mean_q": float(q_pi.mean().item()),
        }

    def save(self, path: str) -> None:
        torch.save(
            {
                "actor": self.actor.state_dict(),
                "critic": self.critic.state_dict(),
                "critic_target": self.critic_target.state_dict(),
                "log_alpha": self.log_alpha.detach().cpu(),
                "actor_opt": self.actor_opt.state_dict(),
                "critic_opt": self.critic_opt.state_dict(),
                "alpha_opt": self.alpha_opt.state_dict(),
            },
            path,
        )

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])
        self.critic_target.load_state_dict(ckpt["critic_target"])
        with torch.no_grad():
            self.log_alpha.copy_(ckpt["log_alpha"].to(self.device))
        if "actor_opt" in ckpt:
            self.actor_opt.load_state_dict(ckpt["actor_opt"])
        if "critic_opt" in ckpt:
            self.critic_opt.load_state_dict(ckpt["critic_opt"])
        if "alpha_opt" in ckpt:
            self.alpha_opt.load_state_dict(ckpt["alpha_opt"])
