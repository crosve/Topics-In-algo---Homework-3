"""Experience buffers for training."""

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np


@dataclass
class ReplayBuffer:
    obs: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    next_obs: np.ndarray
    dones: np.ndarray
    ptr: int = 0
    size: int = 0

    @classmethod
    def create(cls, capacity: int, obs_dim: int, act_dim: int) -> "ReplayBuffer":
        return cls(
            obs=np.zeros((capacity, obs_dim), dtype=np.float32),
            actions=np.zeros((capacity, act_dim), dtype=np.float32),
            rewards=np.zeros((capacity, 1), dtype=np.float32),
            next_obs=np.zeros((capacity, obs_dim), dtype=np.float32),
            dones=np.zeros((capacity, 1), dtype=np.float32),
        )

    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
    ) -> None:
        idx = self.ptr
        self.obs[idx] = obs
        self.actions[idx] = action
        self.rewards[idx] = reward
        self.next_obs[idx] = next_obs
        self.dones[idx] = float(done)
        self.ptr = (self.ptr + 1) % len(self.obs)
        self.size = min(self.size + 1, len(self.obs))

    def sample(self, batch_size: int) -> Dict[str, np.ndarray]:
        idx = np.random.randint(0, self.size, size=batch_size)
        return {
            "obs": self.obs[idx],
            "actions": self.actions[idx],
            "rewards": self.rewards[idx],
            "next_obs": self.next_obs[idx],
            "dones": self.dones[idx],
        }


@dataclass
class PPOReplayBuffer:
    obs: np.ndarray
    actions: np.ndarray
    returns: np.ndarray
    advantages: np.ndarray
    old_log_probs: np.ndarray
    ptr: int = 0
    size: int = 0

    @classmethod
    def create(cls, capacity: int, obs_dim: int, act_dim: int) -> "PPOReplayBuffer":
        return cls(
            obs=np.zeros((capacity, obs_dim), dtype=np.float32),
            actions=np.zeros((capacity, act_dim), dtype=np.float32),
            returns=np.zeros((capacity,), dtype=np.float32),
            advantages=np.zeros((capacity,), dtype=np.float32),
            old_log_probs=np.zeros((capacity,), dtype=np.float32),
        )

    def add_batch(self, batch: Dict[str, np.ndarray]) -> None:
        obs = np.asarray(batch["obs"], dtype=np.float32)
        actions = np.asarray(batch["actions"], dtype=np.float32)
        returns = np.asarray(batch["returns"], dtype=np.float32)
        advantages = np.asarray(batch["advantages"], dtype=np.float32)
        old_log_probs = np.asarray(batch["old_log_probs"], dtype=np.float32)

        batch_size = obs.shape[0]
        if batch_size == 0:
            return

        capacity = len(self.obs)
        if batch_size >= capacity:
            self.obs[:] = obs[-capacity:]
            self.actions[:] = actions[-capacity:]
            self.returns[:] = returns[-capacity:]
            self.advantages[:] = advantages[-capacity:]
            self.old_log_probs[:] = old_log_probs[-capacity:]
            self.ptr = 0
            self.size = capacity
            return

        end = self.ptr + batch_size
        if end <= capacity:
            self.obs[self.ptr : end] = obs
            self.actions[self.ptr : end] = actions
            self.returns[self.ptr : end] = returns
            self.advantages[self.ptr : end] = advantages
            self.old_log_probs[self.ptr : end] = old_log_probs
        else:
            first = capacity - self.ptr
            second = end - capacity
            self.obs[self.ptr :] = obs[:first]
            self.actions[self.ptr :] = actions[:first]
            self.returns[self.ptr :] = returns[:first]
            self.advantages[self.ptr :] = advantages[:first]
            self.old_log_probs[self.ptr :] = old_log_probs[:first]

            self.obs[:second] = obs[first:]
            self.actions[:second] = actions[first:]
            self.returns[:second] = returns[first:]
            self.advantages[:second] = advantages[first:]
            self.old_log_probs[:second] = old_log_probs[first:]

        self.ptr = end % capacity
        self.size = min(self.size + batch_size, capacity)

    def sample(self, batch_size: int) -> Dict[str, np.ndarray]:
        if self.size == 0:
            raise RuntimeError("Cannot sample from an empty PPO replay buffer.")
        idx = np.random.randint(0, self.size, size=batch_size)
        return {
            "obs": self.obs[idx],
            "actions": self.actions[idx],
            "returns": self.returns[idx],
            "advantages": self.advantages[idx],
            "old_log_probs": self.old_log_probs[idx],
        }


class RolloutBuffer:
    def __init__(self, capacity: int, obs_dim: int, act_dim: int):
        self.capacity = capacity
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, act_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity,), dtype=np.float32)
        self.terminated = np.zeros((capacity,), dtype=np.float32)
        self.episode_end = np.zeros((capacity,), dtype=np.float32)
        self.values = np.zeros((capacity,), dtype=np.float32)
        self.log_probs = np.zeros((capacity,), dtype=np.float32)
        self.next_values = np.zeros((capacity,), dtype=np.float32)
        self.ptr = 0

    def reset(self) -> None:
        self.ptr = 0

    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        terminated: bool,
        episode_end: bool,
        value: float,
        log_prob: float,
        next_value: float,
    ) -> None:
        if self.ptr >= self.capacity:
            raise RuntimeError("RolloutBuffer overflow. Increase n_steps.")
        self.obs[self.ptr] = obs
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.terminated[self.ptr] = float(terminated)
        self.episode_end[self.ptr] = float(episode_end)
        self.values[self.ptr] = value
        self.log_probs[self.ptr] = log_prob
        self.next_values[self.ptr] = next_value
        self.ptr += 1

    def compute_advantages(
        self, gamma: float, gae_lambda: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        size = self.ptr
        advantages = np.zeros((size,), dtype=np.float32)
        last_adv = 0.0
        for t in reversed(range(size)):
            bootstrap_mask = 1.0 - self.terminated[t]
            continue_mask = 1.0 - self.episode_end[t]
            delta = (
                self.rewards[t]
                + gamma * self.next_values[t] * bootstrap_mask
                - self.values[t]
            )
            last_adv = delta + gamma * gae_lambda * continue_mask * last_adv
            advantages[t] = last_adv
        returns = advantages + self.values[:size]
        return advantages, returns

    def as_batch(
        self, advantages: np.ndarray, returns: np.ndarray
    ) -> Dict[str, np.ndarray]:
        size = self.ptr
        return {
            "obs": self.obs[:size],
            "actions": self.actions[:size],
            "returns": returns.astype(np.float32),
            "advantages": advantages.astype(np.float32),
            "old_log_probs": self.log_probs[:size],
        }
