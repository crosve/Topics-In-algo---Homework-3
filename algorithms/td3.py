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
        # Update Critic Target Weights: Q_target = tau * Q_current + (1 - tau) * Q_target
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

        # Update Actor Target Weights: pi_target = tau * pi_current + (1 - tau) * pi_target
        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)
        # --- Student Implementation End ---

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
            # Generate the next action using target policy and structural constraints
            raw_next_actions = self.actor_target(next_obs)
            next_actions = self._transform_action(raw_next_actions)
            
            # Target Policy Smoothing: generate noise, clip it, and add to next action
            noise = torch.randn_like(next_actions) * self.policy_noise
            noise = torch.clamp(noise, -self.noise_clip, self.noise_clip)
            smoothed_next_actions = next_actions + noise
            
            # Enforce execution workspace constraints explicitly
            low_bounds = torch.tensor([0.0] + [-1.0] * (self.action_dim - 1), device=self.device)
            high_bounds = torch.tensor([1.0] + [1.0] * (self.action_dim - 1), device=self.device)
            smoothed_next_actions = torch.max(torch.min(smoothed_next_actions, high_bounds), low_bounds)
            
            # Evaluate using twin target critics and apply pessimistic estimation
            target_q1, target_q2 = self.critic_target(next_obs, smoothed_next_actions)
            min_target_q = torch.min(target_q1, target_q2)
            
            # Standard Bellman expectation step
            target_q = rewards + self.gamma * (1.0 - dones) * min_target_q

        # Evaluate current critic network performance
        q1, q2 = self.critic(obs, actions)
        critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)

        # Optimize current critic structures
        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        # Initialize tracking metrics with standard defaults
        metrics = {
            "critic_loss": float(critic_loss.item()),
            "actor_loss": 0.0,
            "mean_q": float(torch.min(q1, q2).mean().item()),
        }

        # ==========================================
        # 2. DELAYED POLICY & TARGET UPDATE
        # ==========================================
        if self.train_step_count % self.policy_delay == 0:
            # Deterministic Policy Gradient optimization: Maximize Q1 performance
            actor_actions = self._transform_action(self.actor(obs))
            actor_loss = -self.critic.q1_only(obs, actor_actions).mean()

            # Optimize primary actor network configurations
            self.actor_opt.zero_grad()
            actor_loss.backward()
            self.actor_opt.step()

            # Slowly update target structures toward current evaluation properties
            self._soft_update()
            
            # Overwrite defaults inside reporting structures
            metrics["actor_loss"] = float(actor_loss.item())

        # Step management mechanics
        self.train_step_count += 1
        return metrics
        # --- Student Code End ---

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
