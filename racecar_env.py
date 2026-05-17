"""Racecar Gymnasium environment with moving obstacles."""

from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from config import EnvConfig
from controllers.pid import LowLevelController
from controllers.pure_pursuit import PurePursuitPlanner
from dynamics.vehicle_dynamics import VehicleDynamics
from dynamics.vehicle_params import VehicleParams
from utils import calculate_distance, normalize_angle, world_to_body_frame
from visualization import Renderer


class RaceCarEnv(gym.Env):
    """Goal-reaching racecar env with scenario-specific control interfaces."""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(
        self, render_mode: Optional[str] = None, config: Optional[EnvConfig] = None
    ):
        super().__init__()
        if config is None:
            config = EnvConfig()
        self.config = config

        self.render_mode = render_mode
        self.arena_size = config.arena_size
        self.goal_threshold = config.goal_threshold
        self.max_episode_steps = config.max_episode_steps
        self.dt = config.dt
        self.scenario = config.scenario

        self.params = VehicleParams(config.vehicle)
        self.dynamics = VehicleDynamics(self.params)
        self.controller = LowLevelController(self.params, self.dt, config.controller)
        self.pure_pursuit = PurePursuitPlanner(self.params.wheelbase)
        self.renderer = Renderer(self.arena_size, self.params)

        # Obstacles
        self.obstacles: List[Dict[str, Any]] = []
        self.obstacle_cfg = config.obstacles

        # Observation: dynamic size based on obstacles enabled
        base_obs_high = [
            self.params.max_velocity,
            5.0,
            5.0,
            self.params.max_steering_angle,
            10.0,
            100.0,
            self.arena_size,
            self.arena_size,
            self.arena_size * np.sqrt(2),
            np.pi,
            self.params.max_velocity,
            self.params.max_velocity,
        ]

        if config.obstacles.enabled:
            obs_high_list = base_obs_high + [
                self.arena_size,
                self.arena_size,
                self.obstacle_cfg.max_speed,
                self.obstacle_cfg.max_speed,
                self.arena_size,
                10.0,
            ]
        else:
            obs_high_list = base_obs_high

        obs_high = np.array(obs_high_list, dtype=np.float32)

        self.observation_space = spaces.Box(
            low=-obs_high, high=obs_high, dtype=np.float32
        )

        # Base environment always takes continuous [throttle_norm, steer_norm] in [-1, 1]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        self.state: Optional[np.ndarray] = None
        self.goal: Optional[np.ndarray] = None
        self.step_count: int = 0
        self.trajectory: List[Tuple[float, float]] = []
        self.prev_distance: float = 0.0
        self.path: List[Tuple[float, float]] = []

    def reset(
        self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)
        margin = self._sampling_margin()

        if self.config.fixed_start is not None:
            X, Y, psi = self.config.fixed_start
        else:
            X, Y = self._sample_arena_point(margin)
            psi = self.np_random.uniform(-np.pi, np.pi)

        self.state = VehicleDynamics.create_initial_state(X, Y, psi)

        if self.config.fixed_goal is not None:
            self.goal = np.array(self.config.fixed_goal, dtype=np.float64)
        else:
            min_goal_distance = self._minimum_goal_distance(margin)
            goal_x, goal_y = X, Y
            for _ in range(200):
                goal_x, goal_y = self._sample_arena_point(margin)
                dist = calculate_distance((X, Y), (goal_x, goal_y))
                if dist >= min_goal_distance:
                    break
            else:
                goal_x, goal_y = self._fallback_goal(X, Y, margin)

            self.goal = np.array([goal_x, goal_y], dtype=np.float64)

        self.controller.reset()
        self.step_count = 0
        self.trajectory = [(X, Y)]
        self.prev_distance = calculate_distance((X, Y), tuple(self.goal))
        self.path = [(X, Y), (float(self.goal[0]), float(self.goal[1]))]
        self._reset_obstacles()
        return self._get_observation(), self._get_info()

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        action = np.asarray(action, dtype=np.float32)

        tw_norm = float(np.clip(action[0], -1.0, 1.0))
        tst_norm = float(np.clip(action[1], -1.0, 1.0))
        tw = tw_norm * self.params.max_motor_torque
        tst = tst_norm * self.params.max_steering_torque

        self.state = self.dynamics.step(self.state, tw, tst, self.dt)
        self._update_obstacles()

        X, Y = self.state[0], self.state[1]
        self.trajectory.append((X, Y))
        self.step_count += 1

        observation = self._get_observation()
        reward = self._compute_reward()
        terminated, truncated = self._check_termination()
        info = self._get_info()
        self.prev_distance = calculate_distance((X, Y), tuple(self.goal))
        if self.render_mode == "human":
            self.render()
        return observation, reward, terminated, truncated, info

    def _sampling_margin(self) -> float:
        margin = min(2.0, max(0.5, 0.1 * self.arena_size))
        # Keep valid sampling bounds even for very small arenas.
        return min(margin, max(0.05, self.arena_size / 2.0 - 1e-3))

    def _sample_arena_point(self, margin: float) -> Tuple[float, float]:
        x = float(self.np_random.uniform(margin, self.arena_size - margin))
        y = float(self.np_random.uniform(margin, self.arena_size - margin))
        return x, y

    def _minimum_goal_distance(self, margin: float) -> float:
        usable_span = max(self.arena_size - 2.0 * margin, 1e-3)
        max_possible_dist = np.sqrt(2.0) * usable_span
        return float(np.clip(0.5 * max_possible_dist, 0.5, 3.0))

    def _fallback_goal(
        self, start_x: float, start_y: float, margin: float
    ) -> Tuple[float, float]:
        corners = np.array(
            [
                [margin, margin],
                [margin, self.arena_size - margin],
                [self.arena_size - margin, margin],
                [self.arena_size - margin, self.arena_size - margin],
            ],
            dtype=np.float64,
        )
        start = np.array([start_x, start_y], dtype=np.float64)
        dists = np.linalg.norm(corners - start[None, :], axis=1)
        idx = int(np.argmax(dists))
        return float(corners[idx, 0]), float(corners[idx, 1])

    def _reset_obstacles(self):
        self.obstacles = []
        if not self.obstacle_cfg.enabled:
            return

        margin = max(self.obstacle_cfg.radius + 0.1, self._sampling_margin())
        margin = min(margin, max(0.05, self.arena_size / 2.0 - 1e-3))
        clearance = min(2.0, max(0.75, 0.2 * self.arena_size))

        for _ in range(self.obstacle_cfg.count):
            accepted = False
            ox, oy = self._sample_arena_point(margin)
            for _ in range(150):
                ox, oy = self._sample_arena_point(margin)
                start_clear = (
                    calculate_distance((ox, oy), (self.state[0], self.state[1]))
                    > clearance
                )
                goal_clear = calculate_distance((ox, oy), tuple(self.goal)) > clearance
                others_clear = all(
                    calculate_distance((ox, oy), tuple(existing["pos"]))
                    > (existing["radius"] + self.obstacle_cfg.radius + 0.25)
                    for existing in self.obstacles
                )
                if start_clear and goal_clear and others_clear:
                    accepted = True
                    break

            if not accepted:
                continue

            angle = self.np_random.uniform(-np.pi, np.pi)
            speed = self.np_random.uniform(0.2, self.obstacle_cfg.max_speed)
            self.obstacles.append(
                {
                    "pos": np.array([ox, oy], dtype=np.float64),
                    "vel": np.array(
                        [speed * np.cos(angle), speed * np.sin(angle)], dtype=np.float64
                    ),
                    "radius": self.obstacle_cfg.radius,
                }
            )

    def _update_obstacles(self):
        if not self.obstacle_cfg.enabled:
            return
        margin = self.obstacle_cfg.bounce_margin
        for obs in self.obstacles:
            obs["vel"] += self.np_random.normal(0.0, 0.05, size=2)
            speed = np.linalg.norm(obs["vel"])
            if speed > self.obstacle_cfg.max_speed:
                obs["vel"] *= self.obstacle_cfg.max_speed / speed
            obs["pos"] += obs["vel"] * self.dt
            for axis in [0, 1]:
                if obs["pos"][axis] < margin:
                    obs["pos"][axis] = margin
                    obs["vel"][axis] *= -1.0
                if obs["pos"][axis] > self.arena_size - margin:
                    obs["pos"][axis] = self.arena_size - margin
                    obs["vel"][axis] *= -1.0

    def _nearest_obstacle_features(self) -> np.ndarray:
        if len(self.obstacles) == 0:
            return np.zeros(6, dtype=np.float32)
        x, y, psi = self.state[0], self.state[1], self.state[2]
        nearest = min(
            self.obstacles, key=lambda o: calculate_distance((x, y), tuple(o["pos"]))
        )
        dx_world, dy_world = nearest["pos"][0] - x, nearest["pos"][1] - y
        dx_body, dy_body = world_to_body_frame(dx_world, dy_world, psi)
        dist = np.sqrt(dx_world**2 + dy_world**2)
        return np.array(
            [
                dx_body,
                dy_body,
                nearest["vel"][0],
                nearest["vel"][1],
                dist,
                nearest["radius"],
            ],
            dtype=np.float32,
        )

    def _get_observation(self) -> np.ndarray:
        X, Y, psi, vx, vy, r, delta, omega_delta, omega_W = self.state
        dx_world = self.goal[0] - X
        dy_world = self.goal[1] - Y
        dx_body, dy_body = world_to_body_frame(dx_world, dy_world, psi)
        dist_goal = np.sqrt(dx_world**2 + dy_world**2)
        angle_to_goal = normalize_angle(np.arctan2(dy_world, dx_world) - psi)
        ideal_velocity = min(2.0, dist_goal * 0.5)
        vx_error = ideal_velocity - vx
        progress = self.prev_distance - dist_goal

        base_obs = [
            vx,
            vy,
            r,
            delta,
            omega_delta,
            omega_W,
            dx_body,
            dy_body,
            dist_goal,
            angle_to_goal,
            vx_error,
            progress,
        ]

        if self.config.obstacles.enabled:
            obstacle_features = self._nearest_obstacle_features().tolist()
            return np.array(base_obs + obstacle_features, dtype=np.float32)
        else:
            return np.array(base_obs, dtype=np.float32)

    def _get_info(self) -> Dict[str, Any]:
        X, Y, psi, vx, vy, r, delta, _, _ = self.state
        dist_to_goal = calculate_distance((X, Y), tuple(self.goal))
        nearest_obstacle_dist = (
            self._nearest_obstacle_features()[4] if self.obstacles else float("inf")
        )
        return {
            "position": (X, Y),
            "heading": psi,
            "velocity": (vx, vy),
            "steering_angle": delta,
            "distance_to_goal": dist_to_goal,
            "goal": self.goal.copy(),
            "step_count": self.step_count,
            "goal_reached": dist_to_goal < self.goal_threshold,
            "collision": self._has_collision(),
            "nearest_obstacle_distance": float(nearest_obstacle_dist),
            "scenario": self.scenario,
        }

    def _has_collision(self) -> bool:
        if not self.obstacle_cfg.enabled:
            return False
        x, y = self.state[0], self.state[1]
        for obs in self.obstacles:
            if calculate_distance((x, y), tuple(obs["pos"])) <= (obs["radius"] + 0.25):
                return True
        return False

    def _compute_reward(self) -> float:
        X, Y, psi, vx, vy, r, delta, omega_delta, _ = self.state
        rc = self.config.reward
        dist_to_goal = calculate_distance((X, Y), tuple(self.goal))
        reward_distance = rc.distance_weight * dist_to_goal
        reward_goal = rc.goal_bonus if dist_to_goal < self.goal_threshold else 0.0
        progress = self.prev_distance - dist_to_goal
        reward_progress = rc.progress_weight * progress
        angle_to_goal = np.arctan2(self.goal[1] - Y, self.goal[0] - X)
        heading_error = abs(normalize_angle(angle_to_goal - psi))
        reward_heading = rc.heading_weight * heading_error
        reward_velocity = (
            rc.velocity_weight * min(max(vx, 0.0), 2.0) if vx > 0.1 else -0.05
        )
        reward_smoothness = rc.smoothness_weight * (abs(r) + abs(omega_delta))
        reward_boundary = 0.0
        margin = rc.boundary_margin
        if X < margin or X > self.arena_size - margin:
            reward_boundary += rc.boundary_penalty
        if Y < margin or Y > self.arena_size - margin:
            reward_boundary += rc.boundary_penalty
        reward_time = rc.time_penalty
        nearest_obs_dist = (
            self._nearest_obstacle_features()[4] if self.obstacles else 999.0
        )
        obstacle_proximity_penalty = 0.0
        if nearest_obs_dist < rc.obstacle_safe_distance:
            obstacle_proximity_penalty = rc.obstacle_proximity_weight * (
                rc.obstacle_safe_distance - nearest_obs_dist
            )
        collision_penalty = rc.collision_penalty if self._has_collision() else 0.0
        return (
            reward_distance
            + reward_goal
            + reward_progress
            + reward_heading
            + reward_velocity
            + reward_smoothness
            + reward_boundary
            + reward_time
            + obstacle_proximity_penalty
            + collision_penalty
        )

    def _check_termination(self) -> Tuple[bool, bool]:
        X, Y = self.state[0], self.state[1]
        terminated = False
        truncated = False
        dist_to_goal = calculate_distance((X, Y), tuple(self.goal))
        if dist_to_goal < self.goal_threshold:
            terminated = True
        if X < 0 or X > self.arena_size or Y < 0 or Y > self.arena_size:
            terminated = True
        if self._has_collision():
            terminated = True
        if self.step_count >= self.max_episode_steps:
            truncated = True
        return terminated, truncated

    def render(self):
        if self.render_mode is None:
            return None
        return self.renderer.render(
            self.state,
            self.goal,
            self.trajectory,
            self.step_count,
            self.goal_threshold,
            self.obstacles,
        )

    def close(self):
        self.renderer.close()
