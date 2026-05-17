"""
Configuration and Parameters
=============================

All adjustable parameters are collected in this file.
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple
import numpy as np


@dataclass
class VehicleConfig:
    """Vehicle physical parameters configuration"""

    # Basic dimensions
    wheelbase: float = 0.28  # m (front-rear axle distance)
    wheel_radius: float = 0.055  # m
    length: float = 0.44  # m
    width: float = 0.315  # m
    height: float = 0.2875  # m

    # Mass and Center of Gravity
    mass: float = 4.5  # kg
    cg_from_rear: float = 0.04  # m (distance from rear axle to CG)
    cg_height: float = 0.135  # m

    # Motor parameters
    motor_poles: int = 14
    gear_ratio: float = 2.769
    battery_cells: int = 3
    max_erpm: float = 30000
    max_motor_torque: float = 2.0  # Nm
    max_steering_torque: float = 0.5  # Nm

    # Steering and velocity limits
    max_steering_angle_deg: float = 29.85  # degrees
    max_velocity: float = 3.0  # m/s (approximately 10.8 km/h)

    # Tire parameters
    tire_Bf: float = 10.0  # Front tire stiffness
    tire_Br: float = 12.0  # Rear tire stiffness
    tire_Cf: float = 1.3  # Shape factor
    tire_Cr: float = 1.3
    mu: float = 0.8  # Friction coefficient

    # Damping coefficients
    Bst: float = 0.1  # Steering damping (Nm*s/rad)
    Kst: float = 2.0  # Steering stiffness (Nm/rad)
    BL: float = 0.01  # Drive damping (Nm*s/rad)

    # Inertia moments (None means automatic calculation)
    Iz: Optional[float] = None  # Yaw inertia
    Jw: Optional[float] = 0.001  # Wheel inertia
    Jst: float = 0.001  # Steering inertia

    # Resistance forces
    rolling_resistance: float = 0.015
    aero_coeff: float = 0.0

    # Physics constants
    gravity: float = 9.81

    @property
    def max_steering_angle(self) -> float:
        """Maximum steering angle (radians)"""
        return np.deg2rad(self.max_steering_angle_deg)

    @property
    def a(self) -> float:
        """Distance from CG to front axle"""
        return self.wheelbase - self.cg_from_rear

    @property
    def b(self) -> float:
        """Distance from CG to rear axle"""
        return self.cg_from_rear


@dataclass
class ControllerConfig:
    """PID controller parameters"""

    # Velocity controller
    kp_vel: float = 2.0
    ki_vel: float = 0.1
    kd_vel: float = 0.05
    vel_integral_limit: float = 1.0

    # Steering controller
    kp_steer: float = 10.0
    ki_steer: float = 0.5
    kd_steer: float = 0.1
    steer_integral_limit: float = 0.2


@dataclass
class RewardConfig:
    """Reward function parameters"""

    # Reward weights
    distance_weight: float = 0.0
    goal_bonus: float = 150.0
    progress_weight: float = 8.0
    heading_weight: float = -0.2
    velocity_weight: float = 0.05
    smoothness_weight: float = -0.005
    boundary_penalty: float = -15.0
    time_penalty: float = -0.002

    # Boundary margin
    boundary_margin: float = 0.5
    collision_penalty: float = -150.0
    obstacle_proximity_weight: float = -0.5
    obstacle_safe_distance: float = 1.0


@dataclass
class ObstacleConfig:
    """Dynamic obstacle settings."""

    enabled: bool = True
    count: int = 2
    radius: float = 0.25
    max_speed: float = 0.5
    bounce_margin: float = 0.6


@dataclass
class EnvConfig:
    """Environment configuration"""

    # Arena
    arena_size: float = 20.0  # m
    goal_threshold: float = 0.50  # m
    max_episode_steps: int = 1000

    # Simulation
    dt: float = 0.02  # s (50 Hz)
    scenario: str = (
        "sac_end_to_end"  # baseline, sac_end_to_end, td3_heading_control, ppo_trajectory_planner
    )

    # Start/Goal (None = random)
    fixed_goal: Optional[Tuple[float, float]] = None
    fixed_start: Optional[Tuple[float, float, float]] = None

    # Sub-configurations
    vehicle: VehicleConfig = field(default_factory=VehicleConfig)
    controller: ControllerConfig = field(default_factory=ControllerConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    obstacles: ObstacleConfig = field(default_factory=ObstacleConfig)

    # Render
    render_fps: int = 30


@dataclass
class TrainingConfig:
    """Training configuration"""

    # Algorithm
    algorithm: str = "PPO"  # PPO, SAC, TD3

    # Hyperparameters (PPO)
    learning_rate: float = 3e-4
    n_steps: int = 2048
    batch_size: int = 64
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.01
    ppo_replay_capacity: int = 50000

    # Baseline (no-RL) run
    baseline_episodes: int = 5

    # Training
    total_timesteps: int = 100000
    eval_freq: int = 5000
    save_freq: int = 10000

    # Directories
    log_dir: str = "./logs/"
    model_dir: str = "./models/"
    tensorboard_log: str = "./tensorboard/"


@dataclass
class CurriculumConfig:
    """Curriculum learning configuration"""

    initial_distance: float = 3.0
    max_distance: float = 15.0
    distance_increment: float = 0.5
    success_threshold: int = 10


# Default configurations
DEFAULT_ENV_CONFIG = EnvConfig()
DEFAULT_TRAINING_CONFIG = TrainingConfig()
DEFAULT_CURRICULUM_CONFIG = CurriculumConfig()
