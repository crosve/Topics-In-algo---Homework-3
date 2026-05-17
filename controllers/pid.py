"""PID based low-level controller."""

from typing import Tuple

import numpy as np

from config import ControllerConfig
from dynamics.vehicle_params import VehicleParams


class PIDController:
    def __init__(
        self, kp: float, ki: float, kd: float, integral_limit: float = float("inf")
    ):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integral_limit = integral_limit
        self.integral = 0.0
        self.prev_error = 0.0

    def reset(self):
        self.integral = 0.0
        self.prev_error = 0.0

    def compute(self, error: float, dt: float) -> float:
        self.integral += error * dt
        self.integral = np.clip(
            self.integral, -self.integral_limit, self.integral_limit
        )
        derivative = (error - self.prev_error) / dt if dt > 0.0 else 0.0
        self.prev_error = error
        return self.kp * error + self.ki * self.integral + self.kd * derivative


class LowLevelController:
    """Converts desired speed/steering to torques."""

    def __init__(
        self, params: VehicleParams, dt: float, config: ControllerConfig = None
    ):
        self.params = params
        self.dt = dt
        if config is None:
            config = ControllerConfig()
        self.vel_controller = PIDController(
            kp=config.kp_vel,
            ki=config.ki_vel,
            kd=config.kd_vel,
            integral_limit=config.vel_integral_limit,
        )
        self.steer_controller = PIDController(
            kp=config.kp_steer,
            ki=config.ki_steer,
            kd=config.kd_steer,
            integral_limit=config.steer_integral_limit,
        )

    def reset(self):
        self.vel_controller.reset()
        self.steer_controller.reset()

    def compute_torques(
        self,
        vx_ref: float,
        delta_ref: float,
        vx_actual: float,
        delta_actual: float,
        omega_delta: float = 0.0,
    ) -> Tuple[float, float]:
        tw = self.vel_controller.compute(vx_ref - vx_actual, self.dt)
        tst = self.steer_controller.compute(delta_ref - delta_actual, self.dt)
        tw = np.clip(tw, -self.params.max_motor_torque, self.params.max_motor_torque)
        tst = np.clip(
            tst, -self.params.max_steering_torque, self.params.max_steering_torque
        )
        return tw, tst
