"""
RaceCar RL environment and reference homework agents.

Kullanım:
    from racecar_rl import RaceCarEnv
    
    env = RaceCarEnv(render_mode="human")
    obs, info = env.reset()
    
    for _ in range(1000):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break
    
    env.close()
"""

from racecar_env import RaceCarEnv
from dynamics.vehicle_params import VehicleParams
from dynamics.vehicle_dynamics import VehicleDynamics
from dynamics.tire_model import TireModel
from controllers.pid import LowLevelController
from controllers.pure_pursuit import PurePursuitController
from wrappers import CurriculumWrapper
from config import EnvConfig, TrainingConfig

__version__ = "1.0.0"
__author__ = "Your Name"

__all__ = [
    "RaceCarEnv",
    "VehicleParams", 
    "VehicleDynamics",
    "TireModel",
    "LowLevelController",
    "PurePursuitController",
    "CurriculumWrapper",
    "EnvConfig",
    "TrainingConfig",
]


def register_env():
    """Environment'ı Gymnasium'a kaydet"""
    import gymnasium as gym
    
    gym.register(
        id='RaceCar-v0',
        entry_point='racecar_rl:RaceCarEnv',
        max_episode_steps=1000,
    )