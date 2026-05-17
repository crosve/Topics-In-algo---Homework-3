import numpy as np
import gymnasium as gym
from utils import calculate_distance

# from controllers.pure_pursuit import PurePursuitPlanner


class BaselineWrapper(gym.ActionWrapper):
    """Overrides any action with Pure Pursuit + PID target speed."""

    def __init__(self, env):
        super().__init__(env)
        self.pure_pursuit = self.env.unwrapped.pure_pursuit
        self.controller = self.env.unwrapped.controller

    def action(self, action):
        state = self.env.unwrapped.state
        goal = self.env.unwrapped.goal
        X, Y, psi, vx = state[0], state[1], state[2], state[3]
        delta_actual = state[6]
        omega_delta = state[7]

        dist_to_goal = calculate_distance((X, Y), tuple(goal))
        speed_norm = float(np.clip(dist_to_goal / 2.0, 0.0, 0.8))
        vx_ref = speed_norm * self.env.unwrapped.params.max_velocity

        path = self.env.unwrapped.path
        ld = float(np.clip(dist_to_goal, 0.5, 1.5))
        heading_error = self.pure_pursuit.compute_heading_reference(
            (X, Y), psi, path, lookahead_distance=ld
        )
        delta_ref = self.pure_pursuit.heading_error_to_delta(heading_error, vx, ld)

        tw, tst = self.controller.compute_torques(
            vx_ref, delta_ref, vx, delta_actual, omega_delta
        )

        tw_norm = tw / self.env.unwrapped.params.max_motor_torque
        tst_norm = tst / self.env.unwrapped.params.max_steering_torque

        return np.array([tw_norm, tst_norm], dtype=np.float32)


class TD3HeadingWrapper(gym.Wrapper):
    """TD3 outputs low-level torques, but observes Pure Pursuit heading reference."""

    def __init__(self, env):
        super().__init__(env)
        self.pure_pursuit = self.env.unwrapped.pure_pursuit

        # Add heading error to observation
        low = np.append(self.env.observation_space.low, -np.pi)
        high = np.append(self.env.observation_space.high, np.pi)
        self.observation_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self._add_heading(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self._add_heading(obs), reward, terminated, truncated, info

    def _add_heading(self, obs):
        state = self.env.unwrapped.state
        X, Y, psi = state[0], state[1], state[2]
        path = self.env.unwrapped.path
        heading_error = self.pure_pursuit.compute_heading_reference(
            (X, Y), psi, path, lookahead_distance=1.5
        )
        return np.append(obs, [heading_error]).astype(np.float32)


class PPOPlannerWrapper(gym.ActionWrapper):
    """PPO outputs desired speed and heading reference, which PID drives."""

    def __init__(self, env):
        super().__init__(env)
        self.controller = self.env.unwrapped.controller

    def action(self, action):
        state = self.env.unwrapped.state
        psi_actual = state[2]
        vx_actual = state[3]
        delta_actual = state[6]
        omega_delta = state[7]

        # action[0] is throttle in [-1, 1], scaled to [0, max_vx]
        speed_norm = (float(np.clip(action[0], -1.0, 1.0)) + 1.0) / 2.0
        vx_ref = speed_norm * self.env.unwrapped.params.max_velocity

        # Interpret action[1] as change-in-heading (delta) in [-1,1], scaled to [-pi/2, pi/2]
        delta_heading = float(np.clip(action[1], -1.0, 1.0))  # * (np.pi)
        heading_error = delta_heading  # desired relative heading change

        # P-control for steering reference based on heading error
        delta_ref = np.clip(
            1.5 * heading_error,
            -self.env.unwrapped.params.max_steering_angle,
            self.env.unwrapped.params.max_steering_angle,
        )

        tw, tst = self.controller.compute_torques(
            vx_ref, delta_ref, vx_actual, delta_actual, omega_delta
        )

        tw_norm = tw / self.env.unwrapped.params.max_motor_torque
        tst_norm = tst / self.env.unwrapped.params.max_steering_torque

        return np.array([tw_norm, tst_norm], dtype=np.float32)


class SACEndToEndWrapper(gym.ActionWrapper):
    """SAC outputs low-level torques directly. No PID, no pure pursuit."""

    def action(self, action):
        return action


class CurriculumWrapper(gym.ActionWrapper):
    """Adjusts task difficulty over time."""

    def __init__(self, env, curriculum_config):
        super().__init__(env)
        self.curriculum_config = curriculum_config
        self.current_distance = (
            curriculum_config.initial_distance if curriculum_config else 3.0
        )
        self.success_count = 0

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if info.get("goal_reached", False):
            self.success_count += 1
            if self.success_count >= (
                self.curriculum_config.success_threshold
                if self.curriculum_config
                else 10
            ):
                self.current_distance = min(
                    self.current_distance
                    + (
                        self.curriculum_config.distance_increment
                        if self.curriculum_config
                        else 0.5
                    ),
                    (
                        self.curriculum_config.max_distance
                        if self.curriculum_config
                        else 15.0
                    ),
                )
                self.success_count = 0
        return obs, reward, terminated, truncated, info

    def action(self, action):
        return action
