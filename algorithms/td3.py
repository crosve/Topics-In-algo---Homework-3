"""Reference TD3 implementation."""

from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from algorithms.base import HomeworkAgent


def _mlp(in_dim: int, out_dim: int, hidden_dim: int = 256) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, out_dim),
    )


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

    def q1_only(self, obs_t: torch.Tensor, act_t: torch.Tensor) -> torch.Tensor:
        return self.q1(torch.cat([obs_t, act_t], dim=-1)).squeeze(-1)


class TD3Agent(HomeworkAgent):
    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        actor_lr: float = 1e-3,
        critic_lr: float = 1e-3,
        gamma: float = 0.99,
        tau: float = 0.005,
        policy_noise: float = 0.2,
        noise_clip: float = 0.5,
        policy_delay: int = 2,
    ):
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_delay = policy_delay
        self.train_step_count = 0
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.actor = _mlp(observation_dim, action_dim).to(self.device)
        self.actor_target = _mlp(observation_dim, action_dim).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic = Critic(observation_dim, action_dim).to(self.device)
        self.critic_target = Critic(observation_dim, action_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

    def _tensor(self, arr: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(arr, dtype=torch.float32, device=self.device)

    def _transform_action(self, raw_action: torch.Tensor) -> torch.Tensor:
        bounded = torch.tanh(raw_action)
        first = 0.5 * (bounded[..., :1] + 1.0)  # speed command in [0, 1]
        rest = bounded[..., 1:]
        return torch.cat([first, rest], dim=-1)

    @torch.no_grad()
    def act(self, observation: np.ndarray, deterministic: bool = True) -> np.ndarray:
        obs_t = self._tensor(observation).unsqueeze(0)
        action = self._transform_action(self.actor(obs_t))
        if not deterministic:
            action = action + 0.1 * torch.randn_like(action)
        low = torch.tensor([0.0] + [-1.0] * (self.action_dim - 1), device=self.device)
        high = torch.tensor([1.0] + [1.0] * (self.action_dim - 1), device=self.device)
        action = torch.max(torch.min(action, high), low)
        return action.squeeze(0).cpu().numpy().astype(np.float32)

    # TODO: Implement the TD3 training step following the algorithm. You can refer to the TD3 paper for more details: https://arxiv.org/abs/1802.09477
    def _soft_update(self):
        # --- Student Implementation Start ---
        # Implement soft update logic here
        # --- Student Implementation End ---
        raise NotImplementedError("Soft update not implemented yet.")

    def train_step(self, batch: Dict[str, np.ndarray]) -> Dict[str, float]:
        required = ["obs", "actions", "rewards", "next_obs", "dones"]
        for key in required:
            if key not in batch:
                raise ValueError(f"Missing key '{key}' for TD3 train_step.")

        obs = self._tensor(batch["obs"])
        actions = self._tensor(batch["actions"])
        rewards = self._tensor(batch["rewards"]).squeeze(-1)
        next_obs = self._tensor(batch["next_obs"])
        dones = self._tensor(batch["dones"]).squeeze(-1)

        # TODO: Critic Update
        # TD3 adds clipped target policy noise to the target actions
        # Compute the target Q-value: r + gamma * (1-done) * min(Q1_target, Q2_target)
        # --- Student Code Below ---
        with torch.no_grad():
            next_actions = None  # Hint: sample from self.actor_target(next_obs), don't forget self._transform_action
            # Add clipped noise based on self.policy_noise and self.noise_clip
            # Target calculation goes here

        # Feed [obs, actions] to current critics

        # MSE Loss for Critic Q1 & Q2
        # Calculate MSE here

        # Optimize critic
        # return critic_loss in metrics for logging critic_loss, actor_loss and mean_q
        # metrics = {
        #     "critic_loss": float(critic_loss.item()),
        #     "actor_loss": 0.0,
        #     "mean_q": float(torch.min(q1, q2).mean().item()),
        # }
        # --- Student Code End ---

        # TODO: Delayed Policy (Actor) Update
        # Update the actor policy and target networks every `policy_delay` steps
        # if self.train_step_count % self.policy_delay == 0:
        # --- Student Code Below ---

        # 1. Infer actions through actor and compute deterministic actor loss
        # actor_actions = ?
        # actor_loss = ? # Maximize Q1(obs, actor_actions) -> Minimize -Q1(obs, actor_actions)

        # Optimize actor
        # --- Student Code End ---

        # Soft update of targets
        # Log actor_loss in metrics
        # metrics["actor_loss"] = float(actor_loss.item())
        # --- Student Code End ---

        # Increment train step count and return metrics for logging
        # self.train_step_count += 1
        # return metrics
        raise NotImplementedError("TD3 train_step not implemented yet.")

    def save(self, path: str) -> None:
        torch.save(
            {
                "actor": self.actor.state_dict(),
                "actor_target": self.actor_target.state_dict(),
                "critic": self.critic.state_dict(),
                "critic_target": self.critic_target.state_dict(),
                "actor_opt": self.actor_opt.state_dict(),
                "critic_opt": self.critic_opt.state_dict(),
                "train_step_count": self.train_step_count,
            },
            path,
        )

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(ckpt["actor"])
        self.actor_target.load_state_dict(ckpt["actor_target"])
        self.critic.load_state_dict(ckpt["critic"])
        self.critic_target.load_state_dict(ckpt["critic_target"])
        if "actor_opt" in ckpt:
            self.actor_opt.load_state_dict(ckpt["actor_opt"])
        if "critic_opt" in ckpt:
            self.critic_opt.load_state_dict(ckpt["critic_opt"])
        self.train_step_count = int(ckpt.get("train_step_count", 0))
